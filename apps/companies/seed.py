"""
Seed the archive for an acquirer: sample NUM_SEEDS companies from the universe and
write, judge and embed a hypothesis for each.

Usage:
    python -m apps.companies.seed
    python -m apps.companies.seed --profile DIS --num-seeds 10
"""

import os
import random
import asyncio
import argparse
from datetime import datetime

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
    PROPOSER_MODEL,
    JUDGE_MODEL,
    NUM_SEEDS,
    NUM_PARALLEL_WORKERS,
    SEED,
)
from .evaluation import judge_score_async
from .items import archive_text, load_items
from .profiles import DEFAULT_PROFILE, get_acquirer_name, get_profile_text, profile_tag
from .proposer import generate_hypothesis_async


def sample_companies(profile: str, num_samples: int) -> list[dict]:
    """Random sample of the universe's companies, excluding the acquirer `profile`."""
    companies = load_items(profile)

    if num_samples > len(companies):
        raise ValueError(f"--num-seeds {num_samples} exceeds the {len(companies)} companies in the pool")
    return random.sample(companies, num_samples)


async def process_seed(
    company: dict,
    acquirer_profile: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict:
    """One seed entry: hypothesis, judge score and archive embedding."""
    async with semaphore:
        company_name = company.get("CompanyName", "Unknown")
        item_id = company["item_id"]

        print(f"[Seed {index+1}/{total}] Processing: {company_name}")

        hypothesis = await generate_hypothesis_async(acquirer_profile, company)

        judgment_task = judge_score_async(acquirer_profile, company, hypothesis)
        embedding_task = get_embedding_async(archive_text(company, hypothesis))

        judgment, embedding = await asyncio.gather(judgment_task, embedding_task)

        entry = {
            "entry_id": new_entry_id(),
            "item_id": item_id,
            "status": "active",
            "method": "seed",
            **company,
            "hypothesis": hypothesis,
            "judgment": judgment,
            "embedding": embedding,
            "created_at": datetime.now().isoformat(),
        }

        print(
            f"[Seed {index+1}/{total}] Done: {company_name} - Score: {judgment.get('score', 'N/A')}/10"
        )
        return entry


async def main_async(
    profile: str = DEFAULT_PROFILE,
    num_seeds: int = NUM_SEEDS,
    seed: int = SEED,
):
    started_at = datetime.now().isoformat()
    print(f"Loading acquirer: {profile}")
    acquirer_profile = get_profile_text(profile)
    print(f"Acquirer: {get_acquirer_name(profile)}")
    print(f"Parallel workers: {NUM_PARALLEL_WORKERS}\n")

    semaphore = asyncio.Semaphore(NUM_PARALLEL_WORKERS)

    print(f"{'='*50}")
    print(f"Sampling seeds (num_seeds={num_seeds})")
    print(f"{'='*50}")

    companies = sample_companies(profile, num_seeds)
    print(
        f"Sampled {len(companies)} companies. Processing in parallel...\n"
    )

    archive, num_attempted = await evaluate_seeds(
        companies, lambda company, index: process_seed(company, acquirer_profile, semaphore, index, len(companies)),
        num_parallel_workers=NUM_PARALLEL_WORKERS,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_file = unique_archive_path(os.path.join(ARCHIVES_DIR, f"{timestamp}_{profile_tag(profile)}_seed{PROPOSER_SUFFIX}{JUDGE_SUFFIX}{seed_suffix(seed)}.json"))

    os.makedirs(ARCHIVES_DIR, exist_ok=True)

    write_archive(
        output_file,
        profile=profile,
        method="seed",
        created_at=started_at,
        completed=len(archive) == num_seeds,
        config=seed_config(
            proposer_model=PROPOSER_MODEL,
            judge_model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            num_seeds=num_seeds,
            num_parallel_workers=NUM_PARALLEL_WORKERS,
            seed=seed,
        ),
        results=archive_results(0, archive, num_attempted, quality=lambda e: e["judgment"]["score"]),
        entries=archive,
    )
    if len(archive) != num_seeds:
        raise RuntimeError(
            f"incomplete seeding: {len(archive)}/{num_seeds} seeds persisted; archive saved to {output_file}"
        )

    print(f"\nSaved {len(archive)} seeds to {output_file}")

    print("\n=== Summary ===")
    print(f"\nSeeds ({len(archive)}):")
    for entry in archive:
        name = entry.get("CompanyName", "Unknown")
        item_id = entry.get("item_id", "?")
        score = entry["judgment"].get("score", "N/A")
        print(f"  {name} ({item_id}): {score}/10")


def main():
    parser = argparse.ArgumentParser(
        description="Generate seed archive for M&A experiments"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help=f"Acquirer item ID (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--num-seeds",
        "-n",
        type=int,
        default=NUM_SEEDS,
        help=f"Number of seeds to generate (default: {NUM_SEEDS})",
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
