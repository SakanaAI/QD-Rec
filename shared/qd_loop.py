"""Shared run loop for Novelty, Novelty+Evo and QD-Rec.

Novelty uses random items and zero-shot proposals. Novelty+Evo adds few-shot
context; QD-Rec also selects items by Thompson sampling with a diversity bonus.

Workers evaluate against a shared batch snapshot. Results are reconciled before
updating the bandit and saving a checkpoint. All completed evaluations count
toward the budget, including discards and replacements. Failures are retried
within the attempt cap in shared.evaluation_budget.

Each domain supplies its evaluation step, data loaders and settings via Domain.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from types import ModuleType

import numpy as np

from shared.archive_format import METHOD_NAMES, METHODS, archive_results, load_archive, run_config, write_archive
from shared.archive_records import active_entries, reconcile_batch, sample_context
from shared.bandit import (
    TS_DIVERSITY_BETA,
    TS_SIGMA,
    LinearThompsonSampling,
    ThompsonPool,
    scaled_reward,
    select_thompson_batch,
    update_posterior,
)
from shared.embeddings import require_item_embeddings
from shared.evaluation_budget import next_batch_size
from shared.paths import find_latest_archive, seed_suffix, unique_archive_path
from shared.seeding import seed_everything


@dataclass(frozen=True)
class Domain:
    """What differs between the domains' runs.

    ``step(profile_text, item, context, snapshot, method)`` evaluates one item
    without changing the archive: ``context`` is None for Novelty, else the
    few-shot examples; ``snapshot`` is the active archive the batch started from.
    It returns ``{"new_entry": ..., "action": "added" | "replaced" | "discarded"}``,
    plus ``replace_target_entry_id`` for a replacement.
    """

    app: str
    config: ModuleType
    profiles: ModuleType  # its profiles.py: DEFAULT_PROFILE, get_profile_text, profile_tag
    items: ModuleType  # its items.py: load_items, load_item_embeddings
    step: Callable
    quality: Callable[[dict], float]  # an entry's quality: judge score or reader rating
    quality_name: str
    quality_max: int  # the top of its scale; the bandit's reward is quality / quality_max
    revisit_items: bool  # whether a successfully evaluated item may be evaluated again
    replaced_reason: str


async def _run(d: Domain, args: argparse.Namespace) -> None:
    cfg = d.config
    profile, method, iterations, seed = args.profile, args.method, args.iterations, args.seed
    selection = "thompson" if method == "qd_rec" else "random"
    profile_text = d.profiles.get_profile_text(profile)
    tag = d.profiles.profile_tag(profile)

    items = d.items.load_items(profile)
    item_embeddings = d.items.load_item_embeddings(profile)
    dim = require_item_embeddings([it["item_id"] for it in items], item_embeddings, d.app)

    def item_embedding(x: dict):
        """Use the item's content embedding for the bandit; archive embeddings also include the hypothesis."""
        return item_embeddings.get(x["item_id"])

    def reward(entry: dict) -> float:
        return scaled_reward(d.quality(entry), d.quality_max)

    models = f"{cfg.PROPOSER_SUFFIX}{cfg.JUDGE_SUFFIX}{seed_suffix(seed)}"
    seed_file = find_latest_archive(cfg.ARCHIVES_DIR, tag, f"seed{models}")
    if seed_file is None:
        raise FileNotFoundError(
            f"No seed archive for {profile!r} in {cfg.ARCHIVES_DIR}. "
            f"Run `uv run python -m apps.{d.app}.seed --profile '{profile}'` first."
        )
    seed_archive = load_archive(seed_file)
    entries = seed_archive["entries"]
    if not entries:
        raise ValueError(f"Seed archive {seed_file} has no entries; rerun seed.py")
    seed_meta = seed_archive["metadata"]
    expected_seeds = seed_meta["config"]["num_seeds"]
    if not seed_meta.get("completed_at") or len(entries) != expected_seeds:
        raise ValueError(
            f"Seed archive {seed_file} is incomplete ({len(entries)}/{expected_seeds} seeds); "
            "rerun seed.py"
        )
    seed_entry_count = len(entries)

    pending = deque()
    if d.revisit_items:
        pool = items
        excluded: set[str] = set()
    else:
        # Only completed evaluations consume an item; failed items stay eligible.
        excluded = {e["item_id"] for e in entries}
        pool = [it for it in items if it["item_id"] not in excluded]
        if iterations > len(pool):
            raise ValueError(
                f"--iterations {iterations} exceeds the {len(pool)} items of {profile!r} not used "
                f"as seeds; {d.app} allows at most one completed evaluation per item"
            )
        random.shuffle(pool)
        if selection == "random":
            pending.extend(pool)

    print("=" * 60)
    print(f"{METHOD_NAMES[method]} | {d.app} | {profile}")
    print(f"proposer {cfg.PROPOSER_MODEL} | judge {cfg.JUDGE_MODEL} | selection {selection}")
    print(f"{len(pool)} items | {iterations} evaluations | batches of {cfg.NUM_PARALLEL_WORKERS}")
    print(f"seed archive {seed_file} ({seed_entry_count} entries)")
    print("=" * 60)

    started_at = datetime.now().isoformat()
    rng = np.random.default_rng(seed)
    ts = ts_pool = None
    warm = 0
    if selection == "thompson":
        ts = LinearThompsonSampling(dim, sigma=args.ts_sigma)
        ts_pool = ThompsonPool(pool, item_embedding)
        update_posterior(ts, entries, item_embedding, reward)
        warm = len(entries)
        print(f"[thompson] warm-started on {warm} seed entries (ts_sigma={args.ts_sigma})")

    os.makedirs(cfg.ARCHIVES_DIR, exist_ok=True)
    out_ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_tag = f"_{args.out_tag}" if args.out_tag else ""
    out_file = unique_archive_path(os.path.join(cfg.ARCHIVES_DIR, f"{out_ts}_{tag}_{method}{out_tag}{models}.json"))
    config = run_config(
        method=method,
        proposer_model=cfg.PROPOSER_MODEL,
        judge_model=cfg.JUDGE_MODEL,
        reasoning_effort=cfg.REASONING_EFFORT,
        num_iterations=iterations,
        num_context_samples=cfg.NUM_CONTEXT_SAMPLES,
        num_neighbors_novelty=cfg.NUM_NEIGHBORS_NOVELTY,
        dedup_similarity_threshold=cfg.DEDUP_SIMILARITY_THRESHOLD,
        num_parallel_workers=cfg.NUM_PARALLEL_WORKERS,
        allow_repick_active_targets=d.revisit_items,
        selection=selection,
        ts=ts,
        ts_diversity_beta=args.ts_diversity_beta,
        reward_scaling=f"{d.quality_name}/{d.quality_max}",
        ts_warm_start_observations=warm,
        seed=seed,
    )
    attempts = 0  # steps scheduled so far, failed ones included

    def save(final: bool = False) -> dict:
        summary = archive_results(seed_entry_count, entries, attempts, quality=d.quality)
        write_archive(
            out_file,
            profile=profile,
            method=method,
            created_at=started_at,
            completed=final and summary["evaluations_succeeded"] == iterations,
            config=config,
            results=summary,
            entries=entries,
        )
        return summary

    batch_num = 0
    while True:
        persisted = len(entries) - seed_entry_count
        n = next_batch_size(
            persisted=persisted, budget=iterations, scheduled=attempts, batch_size=cfg.NUM_PARALLEL_WORKERS,
        )
        if not n:
            break
        snapshot = active_entries(entries)
        if selection == "thompson":
            batch = select_thompson_batch(
                rng=rng, ts=ts, pool=ts_pool, k=n, archive=snapshot,
                item_embedding=item_embedding, beta=args.ts_diversity_beta, excluded_ids=excluded,
            )
        elif d.revisit_items:
            batch = [None] * n  # each slot draws its random item below
        else:
            batch = [pending.popleft() for _ in range(min(n, len(pending)))]
        if not batch:
            break

        tasks = []
        for item in batch:
            if item is None:
                item = random.choice(pool)
            context = None if method == "novelty" else sample_context(snapshot, cfg.NUM_CONTEXT_SAMPLES)
            tasks.append(d.step(profile_text, item, context, snapshot, method))

        batch_num += 1
        print(f"\n=== batch {batch_num} ({len(tasks)} evaluations; {persisted}/{iterations} persisted) ===")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        attempts += len(results)
        ok = []
        for i, r in enumerate(results, 1):
            if isinstance(r, Exception):
                print(f"  [{i}/{len(results)}] FAILED: {type(r).__name__}: {r}")
                if not d.revisit_items and selection == "random":
                    pending.append(batch[i - 1])
                continue
            entry = r["new_entry"]
            if not d.revisit_items:
                # A completed discard also consumes the item; a failed step does not.
                excluded.add(entry["item_id"])
            print(f"  [{i}/{len(results)}] {entry['item_id']}: {r['action']} ({d.quality_name} {d.quality(entry)})")
            ok.append(r)
        reconcile_batch(ok, entries, threshold=cfg.DEDUP_SIMILARITY_THRESHOLD, reason=d.replaced_reason)
        if ts is not None:
            update_posterior(ts, [r["new_entry"] for r in ok], item_embedding, reward)
        save()
        print(f"  archive: {len(entries)} entries, {len(active_entries(entries))} active")

    summary = save(final=True)
    if summary["evaluations_succeeded"] != iterations:
        raise RuntimeError(
            f"incomplete run: {summary['evaluations_succeeded']}/{iterations} evaluations "
            f"persisted; archive saved to {out_file}"
        )
    print("\n" + "=" * 60)
    print(f"{METHOD_NAMES[method]} finished: {summary['active_entries']} active entries, "
          f"mean {d.quality_name} {summary['mean_quality']}")
    print(f"Saved: {out_file}")


def run_qd_loop(d: Domain) -> None:
    """Command-line entry point of a domain's qd_rec.py."""
    ap = argparse.ArgumentParser(description=f"Novelty, Novelty+Evo or QD-Rec in the {d.app} domain")
    ap.add_argument("--profile", default=d.profiles.DEFAULT_PROFILE,
                    help="the profile to run (default: %(default)s)")
    ap.add_argument("--iterations", type=int, default=d.config.NUM_ITERATIONS,
                    help="evaluations after seeding (default: %(default)s)")
    ap.add_argument("--method", choices=METHODS, default="qd_rec",
                    help="qd_rec = QD-Rec (few-shot, Thompson selection); novelty_evo = Novelty+Evo "
                         "(few-shot, random selection); novelty = Novelty (zero-shot, random selection) "
                         "(default: %(default)s)")
    ap.add_argument("--ts-sigma", type=float, default=TS_SIGMA,
                    help="QD-Rec bandit: reward noise standard deviation (default: %(default)s)")
    ap.add_argument("--ts-diversity-beta", type=float, default=TS_DIVERSITY_BETA,
                    help="QD-Rec bandit: diversity bonus weight, score += beta * (1 - max cosine "
                         "similarity to the archive's items) (default: %(default)s)")
    ap.add_argument("--out-tag", default="",
                    help="optional suffix appended to the output archive filename")
    ap.add_argument("--seed", type=int, default=d.config.SEED,
                    help="random seed (default: %(default)s)")
    args = ap.parse_args()
    seed_everything(args.seed)
    asyncio.run(_run(d, args))
