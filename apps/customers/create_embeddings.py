"""
Compute the item embedding of every persona and write them to EMBEDDINGS_FILE.

Usage:
    python -m apps.customers.create_embeddings
"""

import asyncio
import argparse

import pandas as pd

from shared.embeddings import get_embedding_async
from .config import EMBEDDING_CONCURRENCY, EMBEDDING_DIMENSIONS, EMBEDDINGS_FILE, PERSONAS_FILE
from .items import embedding_text


async def process_persona(
    semaphore: asyncio.Semaphore,
    uuid: str,
    text: str,
    index: int,
    total: int,
) -> tuple[str, list[float]] | None:
    """Embed one persona; returns (uuid, embedding), or None on error."""
    async with semaphore:
        try:
            embedding = await get_embedding_async(text, dimensions=EMBEDDING_DIMENSIONS)
            if (index + 1) % 100 == 0 or index == 0:
                print(f"[{index + 1}/{total}] Processing...")
            return (uuid, embedding)
        except Exception as e:
            print(f"[{index + 1}/{total}] ERROR {uuid}: {e}")
            return None


async def main_async(input_file: str, output_file: str):
    print(f"Reading personas: {input_file}")

    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} personas")

    print("Creating text representations...")
    texts = []
    for _, row in df.iterrows():
        text = embedding_text(row.to_dict())
        texts.append(text)

    print("\n" + "=" * 60)
    print("Sample text representation:")
    print("=" * 60)
    print(texts[0])
    print("=" * 60 + "\n")

    print(f"Generating {EMBEDDING_DIMENSIONS}-dimensional embeddings...")
    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    tasks = [
        process_persona(semaphore, row["uuid"], text, i, len(df))
        for i, ((_, row), text) in enumerate(zip(df.iterrows(), texts))
    ]

    results = await asyncio.gather(*tasks)

    successful = [(uuid, emb) for result in results if result for uuid, emb in [result]]
    print(f"\nSuccessfully generated {len(successful):,} embeddings")

    embeddings_df = pd.DataFrame(successful, columns=["uuid", "embedding"])

    print(f"Saving to: {output_file}")
    embeddings_df.to_parquet(output_file, index=False)

    print(f"Done! Saved {len(embeddings_df):,} embeddings to {output_file}")
    return embeddings_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for California personas"
    )
    parser.parse_args()

    asyncio.run(main_async(PERSONAS_FILE, EMBEDDINGS_FILE))


if __name__ == "__main__":
    main()
