"""
Compute the item embedding of every paper in `papers.jsonl` and write them to a
JSON map {paper_id: embedding}. Papers already in the map are skipped.

Usage:
  uv run python -m apps.papers.create_embeddings
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from shared.embeddings import get_embedding_async as _get_embedding_async_base

from .config import (
    EMBEDDING_CONCURRENCY,
    EMBEDDING_DIMENSIONS,
    PAPER_ABSTRACTS_EMBEDDINGS_FILE,
    PAPER_ABSTRACTS_FILE,
)
from .items import embedding_text


def _iter_jsonl(path: Path) -> list[dict]:
    papers: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def load_embeddings(path: str) -> dict[str, list[float]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(path: str, embeddings: dict[str, list[float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)


async def _embed_one(paper: dict, semaphore: asyncio.Semaphore) -> tuple[str, list[float]]:
    async with semaphore:
        pid = (paper.get("id") or "").strip()
        text = embedding_text(paper)
        emb = await _get_embedding_async_base(
            text, dimensions=EMBEDDING_DIMENSIONS
        )
        return pid, emb


async def main_async(
    embeddings_file: str = PAPER_ABSTRACTS_EMBEDDINGS_FILE,
    papers_file: str = PAPER_ABSTRACTS_FILE,
) -> None:
    print("=" * 60)
    print("Creating Embeddings (Abstract-only)")
    print("=" * 60)
    print(f"Papers: {papers_file}")
    print(f"Embeddings out: {embeddings_file}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS} (model: shared/embeddings.py)")

    papers_path = Path(papers_file)
    if not papers_path.exists():
        raise FileNotFoundError(
            f"Missing papers.jsonl: {papers_path}. Place the dataset at this path before running."
        )

    papers = _iter_jsonl(papers_path)

    existing = load_embeddings(embeddings_file)
    to_process = [p for p in papers if (p.get("id") or "").strip() and (p.get("id") or "").strip() not in existing]

    print(f"Total papers loaded: {len(papers)}")
    print(f"Existing embeddings: {len(existing)}")
    print(f"Need embeddings: {len(to_process)}")
    print(f"Concurrent requests: {EMBEDDING_CONCURRENCY}")
    print("-" * 60)

    if not to_process:
        print("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
    started = datetime.now()

    tasks = [_embed_one(p, semaphore) for p in to_process]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    new_embeddings: dict[str, list[float]] = {}
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
            print(f"[ERROR] {r}")
            continue
        pid, emb = r
        if pid:
            new_embeddings[pid] = emb

    merged = {**existing, **new_embeddings}
    save_embeddings(embeddings_file, merged)

    elapsed = (datetime.now() - started).total_seconds()
    print("-" * 60)
    print("Done")
    print(f"New embeddings: {len(new_embeddings)}")
    print(f"Errors: {errors}")
    print(f"Total embeddings: {len(merged)}")
    print(f"Elapsed: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create embeddings for abstract-only papers")
    parser.parse_args()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


