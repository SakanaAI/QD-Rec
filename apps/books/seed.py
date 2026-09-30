"""
Seed the archive for a query user.

Samples NUM_SEEDS books from the reader's universe. For each, the LLM writes a memo
(stored as `hypothesis`), the reader's star rating becomes the quality (`rating`),
and the book plus memo is embedded as entry["embedding"].

Run:  uv run python -m apps.books.seed [--profile KEY] [--num-seeds N]
      (KEY is the reader: picture_books_reader or series_ya_reader)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from datetime import datetime

from shared.archive_format import archive_results, seed_config, write_archive
from shared.embeddings import get_embedding_async
from shared.paths import seed_suffix, unique_archive_path
from shared.seed_loop import evaluate_seeds
from shared.seeding import seed_everything

from .config import (
    REASONING_EFFORT,
    ARCHIVES_DIR,
    JUDGE_SUFFIX,
    PROPOSER_SUFFIX,
    EMBEDDING_DIMENSIONS,
    PROPOSER_MODEL,
    JUDGE_MODEL,
    NUM_PARALLEL_WORKERS,
    NUM_SEEDS,
    SEED,
)
from .items import archive_text, load_items, make_entry
from .profiles import DEFAULT_PROFILE, get_profile_text, get_user_name, profile_tag
from .proposer import generate_hypothesis_async


async def process_seed(
    book: dict,
    profile_text: str,
    sem: asyncio.Semaphore,
) -> dict:
    """Generate memo + archive embedding + attach rating, returning a seed entry."""
    async with sem:
        hypothesis = await generate_hypothesis_async(profile_text, book)
        emb = await get_embedding_async(
            archive_text(book, hypothesis),
            dimensions=EMBEDDING_DIMENSIONS,
        )
    return make_entry(book, hypothesis, emb, "seed")


async def main_async(profile: str, num_seeds: int, seed: int = SEED) -> str:
    started_at = datetime.now().isoformat()
    profile_text = get_profile_text(profile)
    universe = load_items(profile)

    print("=" * 60)
    print(f"Seeding Goodreads archive — {get_user_name(profile)} ({profile})")
    print(f"LLM: proposer={PROPOSER_MODEL} judge={JUDGE_MODEL}")
    print(f"Universe: {len(universe)} | target seeds: {num_seeds} | parallel: {NUM_PARALLEL_WORKERS}")
    print("=" * 60)

    if num_seeds > len(universe):
        raise ValueError(f"--num-seeds {num_seeds} exceeds the {len(universe)} books in the pool")
    sampled = random.sample(universe, num_seeds)
    sem = asyncio.Semaphore(NUM_PARALLEL_WORKERS)
    entries, attempts = await evaluate_seeds(
        sampled, lambda book, index: process_seed(book, profile_text, sem),
        num_parallel_workers=NUM_PARALLEL_WORKERS,
    )

    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_file = unique_archive_path(os.path.join(ARCHIVES_DIR, f"{ts}_{profile_tag(profile)}_seed{PROPOSER_SUFFIX}{JUDGE_SUFFIX}{seed_suffix(seed)}.json"))

    summary = archive_results(0, entries, attempts, quality=lambda e: e["rating"])
    write_archive(
        out_file,
        profile=profile,
        method="seed",
        created_at=started_at,
        completed=len(entries) == num_seeds,
        config=seed_config(
            proposer_model=PROPOSER_MODEL,
            judge_model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            num_seeds=num_seeds,
            num_parallel_workers=NUM_PARALLEL_WORKERS,
            seed=seed,
        ),
        results=summary,
        entries=entries,
    )
    if len(entries) != num_seeds:
        raise RuntimeError(
            f"incomplete seeding: {len(entries)}/{num_seeds} seeds persisted; archive saved to {out_file}"
        )

    print(f"\nDone. {len(entries)} entries (mean rating {summary['mean_quality']})")
    print(f"Saved: {out_file}")
    return out_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed a Goodreads archive for a query user")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="reader key (default: %(default)s)")
    ap.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    ap.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    args = ap.parse_args()
    seed_everything(args.seed)
    asyncio.run(main_async(args.profile, args.num_seeds, seed=args.seed))


if __name__ == "__main__":
    main()
