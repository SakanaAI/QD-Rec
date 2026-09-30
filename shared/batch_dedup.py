"""Deduplicate additions judged against the same batch snapshot.

Candidates within a batch have not been compared with each other. For each pair
above the cosine-similarity threshold, randomly discard one. Pair order is fixed
and the RNG is seeded per run.
"""

from __future__ import annotations

import random
from datetime import datetime

from shared.embeddings import cosine_similarity


def dedupe_added_batch(
    added: list[dict],
    discarded: list[dict],
    threshold: float,
    *,
    embedding_key: str = "embedding",
) -> list[dict]:
    """Return retained additions in order, appending dropped results to discarded.

    Discarded entries remain in the archive as completed evaluations.
    """
    kept = list(added)
    if len(kept) <= 1:
        return kept

    keep_mask = [True] * len(kept)
    for i in range(len(kept)):
        if not keep_mask[i]:
            continue
        emb_i = kept[i]["new_entry"].get(embedding_key)
        if not emb_i:
            continue
        for j in range(i + 1, len(kept)):
            if not keep_mask[j]:
                continue
            emb_j = kept[j]["new_entry"].get(embedding_key)
            if not emb_j:
                continue
            sim = cosine_similarity(emb_i, emb_j)
            if sim > threshold:
                drop_idx = random.choice([i, j])
                keep_mask[drop_idx] = False
                dropped = kept[drop_idx]
                dropped["new_entry"]["status"] = "discarded"
                dropped["new_entry"]["discarded_at"] = datetime.now().isoformat()
                dropped["new_entry"]["discarded_reason"] = (
                    "Batch deduplication (too similar to another entry)"
                )
                dropped["new_entry"]["dedup_similarity"] = sim
                discarded.append(dropped)
                if drop_idx == i:
                    break  # i is gone; move to the next i

    return [kept[i] for i in range(len(kept)) if keep_mask[i]]
