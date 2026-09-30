"""Embed company business descriptions and save them to config.EMBEDDINGS_FILE.

    uv run python -m apps.companies.create_embeddings
"""

import csv
import json
import asyncio
import argparse

from shared.embeddings import get_embedding_async
from .config import EMBEDDING_CONCURRENCY, EMBEDDING_DIMENSIONS, EMBEDDINGS_FILE, UNIVERSE_FILE
from .items import embedding_text


async def process_company(
    semaphore: asyncio.Semaphore,
    instrument: str,
    description: str,
    index: int,
    total: int,
) -> tuple[str, list[float]] | None:
    """Return (instrument, embedding), or None on failure."""
    async with semaphore:
        try:
            embedding = await get_embedding_async(
                description, dimensions=EMBEDDING_DIMENSIONS
            )
            print(f"[{index + 1}/{total}] {instrument}")
            return (instrument, embedding)
        except Exception as e:
            print(f"[{index + 1}/{total}] ERROR {instrument}: {e}")
            return None


async def main_async(universe_file: str, output_file: str):
    print(f"Reading universe: {universe_file}")

    companies = []
    with open(universe_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instrument = row.get("Instrument", "")
            description = embedding_text(row)
            if instrument and description:
                companies.append((instrument, description))

    print(f"Found {len(companies)} companies with descriptions")

    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
    tasks = [
        process_company(semaphore, inst, desc, i, len(companies))
        for i, (inst, desc) in enumerate(companies)
    ]

    results = await asyncio.gather(*tasks)

    embeddings = [(inst, emb) for result in results if result for inst, emb in [result]]

    print(f"\nSuccessfully generated {len(embeddings)} embeddings")

    print(f"Saving to: {output_file}")
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Instrument", "embedding"])
        for instrument, embedding in embeddings:
            writer.writerow([instrument, json.dumps(embedding)])

    print(f"Done! Saved {len(embeddings)} embeddings to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for universe companies"
    )
    parser.parse_args()

    asyncio.run(main_async(UNIVERSE_FILE, EMBEDDINGS_FILE))


if __name__ == "__main__":
    main()
