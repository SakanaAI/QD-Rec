"""
Seed archive for the papers domain: randomly sampled papers, each with a
zero-shot memo (stored as the entry's `hypothesis`), a judge score and an
archive embedding.

Usage:
  uv run python -m apps.papers.seed
  uv run python -m apps.papers.seed --num-seeds 10 --profile "OpenEndedness Researcher"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from datetime import datetime
from pathlib import Path

from shared.archive_format import archive_results, seed_config, write_archive
from shared.archive_records import new_entry_id
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
    PAPER_ABSTRACTS_FILE,
    SEED,
)
from .evaluation import judge_score_async
from .items import archive_text, load_items
from .profiles import DEFAULT_PROFILE, get_profile_text, get_researcher_name, profile_tag
from .proposer import generate_hypothesis_async


async def process_seed(
    paper: dict,
    researcher_profile: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict:
    async with semaphore:
        item_id = paper["item_id"]
        title = (paper.get("title") or "Unknown")[:80]
        print(f"[{index+1}/{total}] {title} (id={item_id})")

        abstract = (paper.get("abstract") or "").strip()

        hypothesis = await generate_hypothesis_async(researcher_profile, paper)

        judgment_task = judge_score_async(researcher_profile, paper, hypothesis)
        archive_embedding_task = get_embedding_async(
            archive_text(paper, hypothesis),
            dimensions=EMBEDDING_DIMENSIONS,
        )
        judgment, archive_embedding = await asyncio.gather(
            judgment_task, archive_embedding_task
        )

        entry = {
            "entry_id": new_entry_id(),
            "item_id": item_id,
            "status": "active",
            "method": "seed",
            "title": paper.get("title"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "abstract": abstract,
            "hypothesis": hypothesis,
            "judgment": judgment,
            # Archive embedding (paper + memo): novelty neighbor-search + diversity metric.
            "embedding": archive_embedding,
            "created_at": datetime.now().isoformat(),
        }
        return entry


async def main_async(profile: str, num_seeds: int, seed: int = SEED) -> str:
    started_at = datetime.now().isoformat()
    researcher_name = get_researcher_name(profile)
    researcher_profile = get_profile_text(profile)

    print("=" * 60)
    print("Seeding Abstract-only Archive")
    print("=" * 60)
    print(f"Researcher: {researcher_name}")
    print(f"LLM: proposer={PROPOSER_MODEL} judge={JUDGE_MODEL}")
    print(f"Target seeds: {num_seeds}")
    print(f"Parallel workers: {NUM_PARALLEL_WORKERS}")
    print()

    papers_path = Path(PAPER_ABSTRACTS_FILE)
    if not papers_path.exists():
        raise FileNotFoundError(f"Missing papers.jsonl: {papers_path}")

    papers = load_items(profile)
    if num_seeds > len(papers):
        raise ValueError(f"--num-seeds {num_seeds} exceeds the {len(papers)} papers in the pool")

    sampled = random.sample(papers, num_seeds)

    semaphore = asyncio.Semaphore(NUM_PARALLEL_WORKERS)
    entries, attempts = await evaluate_seeds(
        sampled, lambda paper, index: process_seed(paper, researcher_profile, semaphore, index, len(sampled)),
        num_parallel_workers=NUM_PARALLEL_WORKERS,
    )

    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_file = unique_archive_path(os.path.join(ARCHIVES_DIR, f"{ts}_{profile_tag(profile)}_seed{PROPOSER_SUFFIX}{JUDGE_SUFFIX}{seed_suffix(seed)}.json"))

    summary = archive_results(0, entries, attempts, quality=lambda e: e["judgment"]["score"])
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

    print()
    print("Done")
    print(f"Saved: {out_file}")
    print(f"Entries: {len(entries)} (mean score {summary['mean_quality']})")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed an abstract-only archive")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS, help="Number of seed papers to sample")
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help="Researcher profile to use (e.g., 'OpenEndedness Researcher', 'AI Startup Researcher')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    asyncio.run(main_async(args.profile, args.num_seeds, seed=args.seed))


if __name__ == "__main__":
    main()
