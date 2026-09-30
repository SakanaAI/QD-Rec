"""Novelty, Novelty+Evo and QD-Rec for Goodreads book recommendation.

Selection and batching are handled by shared.qd_loop; this module evaluates one
item–hypothesis pair against the active archive.

Quality uses the reader's star rating. Each book can complete one evaluation;
failed attempts remain eligible for retry.

    uv run python -m apps.books.qd_rec --profile PROFILE --method qd_rec
"""

from __future__ import annotations

from datetime import datetime

from shared.archive_records import set_replacement_target
from shared.embeddings import get_embedding_async, nearest_neighbors
from shared.qd_loop import Domain, run_qd_loop

from . import config, items, profiles
from .config import EMBEDDING_DIMENSIONS, NUM_NEIGHBORS_NOVELTY
from .evaluation import check_novelty_async, new_entry_beats_nearest_neighbor
from .items import archive_text, make_entry
from .proposer import generate_hypothesis_async, generate_hypothesis_fewshot_async


async def iteration_async(profile_text, book, context, snapshot, method) -> dict:
    """Evaluate one candidate against the batch snapshot without mutating the archive."""
    if method == "novelty":
        hypothesis = await generate_hypothesis_async(profile_text, book)
    else:
        hypothesis, evolution = await generate_hypothesis_fewshot_async(profile_text, book, context)
    emb = await get_embedding_async(
        archive_text(book, hypothesis),
        dimensions=EMBEDDING_DIMENSIONS,
    )
    new_entry = make_entry(book, hypothesis, emb, method)
    if method != "novelty":
        new_entry["parent_entry_ids"] = [e["entry_id"] for e in context]
        new_entry["evolution"] = evolution

    neighbors = nearest_neighbors(emb, snapshot, top_k=NUM_NEIGHBORS_NOVELTY)
    result = {"new_entry": new_entry}
    novelty = await check_novelty_async(profile_text, new_entry, neighbors)
    if novelty["is_novel"]:
        new_entry["admit_path"] = "novel"
        result["action"] = "added"
        return result
    _sim, nearest = neighbors[0]
    if new_entry_beats_nearest_neighbor(new_entry, nearest):
        new_entry["admit_path"] = "duel_win"
        result["action"] = "replaced"
        set_replacement_target(result, nearest)
    else:
        new_entry["status"] = "discarded"
        new_entry["discarded_at"] = datetime.now().isoformat()
        new_entry["admit_path"] = "duel_loss"
        new_entry["discarded_reason"] = "tie-or-lower rating vs nearest neighbor"
        new_entry["lost_to_entry_id"] = nearest["entry_id"]
        result["action"] = "discarded"
    return result


def main() -> None:
    run_qd_loop(Domain(
        app="books",
        config=config,
        profiles=profiles,
        items=items,
        step=iteration_async,
        quality=lambda e: e["rating"],
        quality_name="rating",
        quality_max=5,
        revisit_items=False,
        replaced_reason="Lost rating comparison",
    ))


if __name__ == "__main__":
    main()
