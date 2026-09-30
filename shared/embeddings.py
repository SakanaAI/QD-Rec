"""Embedding requests through OpenRouter and cosine-similarity helpers."""

import numpy as np

from .llm import aembed

EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Shared output dimension for item and item–hypothesis embeddings.
DEFAULT_EMBEDDING_DIMENSIONS = 512


async def get_embedding_async(
    text: str,
    model: str = EMBEDDING_MODEL,
    dimensions: int | None = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    text_clean = text.replace("\n", " ")
    kwargs = {"input": [text_clean], "model": model}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    response = await aembed(**kwargs)
    return response.data[0].embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either is zero."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def nearest_neighbors(
    embedding: list[float], entries: list[dict], top_k: int
) -> list[tuple[float, dict]]:
    """Up to ``top_k`` (cosine similarity, entry) pairs over the active entries
    with an ``embedding``, most similar first (ties keep the entries' order)."""
    scored = [
        (cosine_similarity(embedding, entry["embedding"]), entry)
        for entry in entries
        if entry.get("status") == "active" and entry.get("embedding")
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def require_item_embeddings(item_ids, item_embeddings, app: str) -> int:
    """Validate every candidate embedding and return their common dimension.

    Missing, empty or mismatched vectors raise instead of shrinking the item pool.
    The expected dimension is the most common vector length among these items.
    """
    if not item_ids:
        raise ValueError(f"no {app} items loaded")
    vecs = [item_embeddings.get(str(i)) for i in item_ids]
    lengths = [len(v) for v in vecs if v is not None and len(v) > 0]
    dim = max(set(lengths), key=lengths.count) if lengths else None
    bad = sum(1 for v in vecs if v is None or len(v) == 0 or len(v) != dim)
    if bad:
        raise ValueError(
            f"{bad} of {len(vecs)} {app} items have no usable item embedding "
            f"(missing, empty or not {dim}-d). "
            f"Run `uv run python -m apps.{app}.create_embeddings` first."
        )
    return dim
