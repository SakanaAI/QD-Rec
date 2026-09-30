"""
Seed archive for the Marketing domain: randomly sampled personas, each with a
zero-shot hypothesis, a judge score and an archive embedding.
"""

import os
import asyncio
from datetime import datetime
from shared.archive_format import archive_results, seed_config, write_archive
from shared.archive_records import new_entry_id

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
from .items import archive_text, load_persona_table
from .profiles import DEFAULT_PROFILE, PRODUCTS, get_profile_text, profile_tag
from .proposer import generate_hypothesis_async
from shared.embeddings import get_embedding_async
from shared.paths import seed_suffix, unique_archive_path
from shared.seed_loop import evaluate_seeds
from shared.seeding import seed_everything


def sample_personas(num_samples: int) -> list[dict]:
    """Randomly sample personas (pandas draws from NumPy's RNG, seeded by seed_everything)."""
    df = load_persona_table()

    if num_samples > len(df):
        raise ValueError(f"--num-seeds {num_samples} exceeds the {len(df)} personas in the pool")
    sampled = df.sample(n=num_samples)
    return sampled.to_dict("records")


async def process_seed(
    persona: dict,
    target_product: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict:
    """Build one seed entry: zero-shot hypothesis, judge score and archive embedding."""
    async with semaphore:
        item_id = persona["item_id"]

        print(f"[{index+1}/{total}] Processing: {item_id}")

        hypothesis = await generate_hypothesis_async(target_product, persona)
        print(f"[{index+1}/{total}] Hypothesis for {item_id}: {hypothesis}")

        judgment_task = judge_score_async(target_product, persona, hypothesis)
        embedding_task = get_embedding_async(archive_text(persona, hypothesis))

        judgment, embedding = await asyncio.gather(judgment_task, embedding_task)

        entry = {
            "entry_id": new_entry_id(),
            "item_id": item_id,
            "status": "active",
            "method": "seed",
            **persona,
            "hypothesis": hypothesis,
            "judgment": judgment,
            "embedding": embedding,
            "created_at": datetime.now().isoformat(),
        }

        reasoning = judgment.get("reasoning", "")
        score = judgment.get("score", "N/A")
        print(f"[{index+1}/{total}] Done: {item_id}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Score: {score}/10")
        return entry


async def main_async(
    profile: str = DEFAULT_PROFILE,
    num_seeds: int = NUM_SEEDS,
    seed: int = SEED,
):
    started_at = datetime.now().isoformat()
    target_product = get_profile_text(profile)

    print("=" * 70)
    print("PERSONAS US - SEED GENERATION")
    print("=" * 70)
    print(f"Product: {profile}")
    print(f"Num seeds: {num_seeds}")
    print(f"Parallel workers: {NUM_PARALLEL_WORKERS}")
    print("=" * 70 + "\n")

    semaphore = asyncio.Semaphore(NUM_PARALLEL_WORKERS)

    print(f"Sampling {num_seeds} personas from dataset...")
    personas = sample_personas(num_seeds)
    print(f"Sampled {len(personas)} personas. Processing in parallel...\n")

    archive, attempts = await evaluate_seeds(
        personas, lambda persona, index: process_seed(persona, target_product, semaphore, index, len(personas)),
        num_parallel_workers=NUM_PARALLEL_WORKERS,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_file = unique_archive_path(os.path.join(
        ARCHIVES_DIR, f"{timestamp}_{profile_tag(profile)}_seed{PROPOSER_SUFFIX}{JUDGE_SUFFIX}{seed_suffix(seed)}.json"
    ))

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
        results=archive_results(0, archive, attempts, quality=lambda e: e["judgment"]["score"]),
        entries=archive,
    )
    if len(archive) != num_seeds:
        raise RuntimeError(
            f"incomplete seeding: {len(archive)}/{num_seeds} seeds persisted; archive saved to {output_file}"
        )

    print(f"\nSaved {len(archive)} seeds to {output_file}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    scores = [e["judgment"].get("score", 0) for e in archive if "judgment" in e]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Average score: {avg_score:.1f}/10")
        print(f"Score range: {min(scores)} - {max(scores)}")

    print("\nSample entries:")
    for entry in list(archive)[:5]:
        item_id = entry.get("item_id", "unknown")
        occupation = entry.get("occupation", "unknown")
        age = entry.get("age", "?")
        location = ", ".join([p for p in (entry.get("city", ""), entry.get("state", "")) if p])
        score = entry.get("judgment", {}).get("score", "N/A")
        print(f"  {item_id} ({age} yrs, {occupation}, {location}): {score}/10")

    if len(archive) > 5:
        print(f"  ... and {len(archive) - 5} more")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate seed personas for archive")
    parser.add_argument(
        "--num-seeds",
        "-n",
        type=int,
        default=NUM_SEEDS,
        help=f"Number of seeds to generate (default: {NUM_SEEDS} from config)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        choices=list(PRODUCTS.keys()),
        default=DEFAULT_PROFILE,
        help=f"Product to use (default: {DEFAULT_PROFILE})",
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
