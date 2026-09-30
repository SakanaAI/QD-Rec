"""Paper records, item embeddings and prompt text."""

from __future__ import annotations

import json
from pathlib import Path

from shared.archive_records import normalize_source_item

from .config import PAPER_ABSTRACTS_EMBEDDINGS_FILE, PAPER_ABSTRACTS_FILE


def load_items(profile: str) -> list[dict]:
    """Every paper of papers.jsonl, in file order (the same pool for every researcher)."""
    papers: list[dict] = []
    with open(Path(PAPER_ABSTRACTS_FILE), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(normalize_source_item(json.loads(line), "id"))
    return papers


def item_label(paper: dict) -> str:
    return paper["title"]


def item_text(paper: dict) -> str:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    return f"{title}\n\n{abstract}" if abstract else title


def archive_text(paper: dict, hypothesis: str) -> str:
    """Text embedded as entry['embedding']: paper (title+abstract) + memo."""
    return f"{item_text(paper)}\n\n---\n\nHypothesis/Memo:\n{hypothesis}"


def embedding_text(paper: dict) -> str:
    """Text of the per-item embedding (create_embeddings.py): title + abstract
    with its whitespace collapsed (unlike item_text, no title-only fallback)."""
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    abstract = " ".join(abstract.split())
    return f"{title}\n\n{abstract}"


def load_item_embeddings(profile: str) -> dict[str, list[float]]:
    """{item_id: embedding} from PAPER_ABSTRACTS_EMBEDDINGS_FILE (the same store for every researcher)."""
    with open(Path(PAPER_ABSTRACTS_EMBEDDINGS_FILE), "r", encoding="utf-8") as f:
        return json.load(f)
