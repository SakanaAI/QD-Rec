"""
Precompute the content embedding (title + description) of each book in a reader's
universe and save it to data/<key>/book_embeddings.json as {item_id: [float, ...]}.

These are the Thompson bandit's item features. The archive embedding (book plus
memo, entry["embedding"]) is computed during the runs.

Run:  uv run python -m apps.books.create_embeddings [--profile picture_books_reader|series_ya_reader|all]
"""

from __future__ import annotations

import argparse
import asyncio
import json

from shared.embeddings import get_embedding_async

from .config import (
    EMBEDDING_CONCURRENCY,
    EMBEDDING_DIMENSIONS,
    book_embeddings_file,
)
from .items import embedding_text, load_items
from .profiles import get_user_name, list_user_keys


async def _embed_one(book: dict, sem: asyncio.Semaphore) -> tuple[str, list[float]]:
    async with sem:
        vec = await get_embedding_async(
            embedding_text(book),
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return book["item_id"], vec


async def build_user_async(key: str) -> None:
    universe = load_items(key)
    print(f"[{key}] embedding {len(universe)} books ({get_user_name(key)}) ...", flush=True)
    sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
    results = await asyncio.gather(*[_embed_one(b, sem) for b in universe])
    embeddings = {bid: vec for bid, vec in results}

    out = book_embeddings_file(key)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    print(f"[{key}] wrote {len(embeddings)} embeddings -> {out}", flush=True)


async def main_async(keys: list[str]) -> None:
    for key in keys:
        await build_user_async(key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute content embeddings for a user's universe")
    ap.add_argument("--profile", default="all", help="reader key or 'all' (default: all)")
    args = ap.parse_args()
    keys = list_user_keys() if args.profile == "all" else [args.profile]
    asyncio.run(main_async(keys))


if __name__ == "__main__":
    main()
