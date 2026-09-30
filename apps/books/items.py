"""The reader's rated books, content embeddings and prompt text."""

from __future__ import annotations

import json
from datetime import datetime

from shared.archive_records import new_entry_id, normalize_source_item

from .config import book_embeddings_file, universe_file


def load_items(profile: str) -> list[dict]:
    """The reader's universe (data/<key>/universe.jsonl), in file order."""
    p = universe_file(profile)
    if not p.exists():
        raise FileNotFoundError(
            f"Universe not built for {profile!r}: {p}. Run `data_prep` first."
        )
    out: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(normalize_source_item(json.loads(line), "book_id"))
    return out


def item_label(book: dict) -> str:
    return book["title"]


def item_text(book: dict) -> str:
    """Content text of a book: title and description, or the title alone."""
    title = (book.get("title") or "").strip()
    desc = (book.get("description") or "").strip()
    return f"{title}\n\n{desc}" if desc else title


def archive_text(book: dict, memo: str) -> str:
    """Text embedded as entry["embedding"]: the book's content text plus the memo."""
    return f"{item_text(book)}\n\n---\n\nWhy this reader would like it:\n{memo}"


def embedding_text(book: dict) -> str:
    """Text of the per-item content embedding (create_embeddings.py): item_text."""
    return item_text(book)


def load_item_embeddings(profile: str) -> dict[str, list[float]]:
    """{item_id: content embedding} for the reader's books (create_embeddings.py)."""
    with open(book_embeddings_file(profile), encoding="utf-8") as f:
        return json.load(f)


def make_entry(
    book: dict,
    hypothesis: str,
    embedding: list[float],
    method: str,
) -> dict:
    """Assemble an archive entry; its quality is the reader's star rating (`rating`)."""
    rating = int(book["rating"])
    return {
        "entry_id": new_entry_id(),
        "item_id": book["item_id"],
        "status": "active",
        "method": method,
        "title": book.get("title", ""),
        "description": book.get("description", ""),
        "shelves": book.get("shelves", []),
        "rating": rating,
        "hypothesis": hypothesis,
        "embedding": embedding,  # archive embedding: novelty check + diversity metric
        "created_at": datetime.now().isoformat(),
    }
