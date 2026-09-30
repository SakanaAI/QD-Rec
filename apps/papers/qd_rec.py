"""Novelty, Novelty+Evo and QD-Rec for paper recommendation.

Selection and batching are handled by shared.qd_loop; this module evaluates one
item–hypothesis pair against the active archive.

    uv run python -m apps.papers.qd_rec --profile PROFILE --method qd_rec
"""

from __future__ import annotations

from datetime import datetime

from shared.archive_records import new_entry_id, set_replacement_target
from shared.embeddings import get_embedding_async, nearest_neighbors
from shared.qd_loop import Domain, run_qd_loop

from . import config, items, profiles
from .config import EMBEDDING_DIMENSIONS, NUM_NEIGHBORS_NOVELTY
from .evaluation import check_novelty_async, judge_score_async, pairwise_duel_async
from .items import archive_text
from .proposer import generate_hypothesis_async, generate_hypothesis_fewshot_async


async def iteration_async(
    researcher_profile: str,
    paper: dict,
    context_entries: list[dict] | None,
    archive_snapshot: list[dict],
    method: str,
) -> dict:
    """Evaluate one candidate against the batch snapshot without mutating the archive."""
    if method == "novelty":
        hypothesis = await generate_hypothesis_async(researcher_profile, paper)
        lineage: dict = {}
    else:
        hypothesis, evolution = await generate_hypothesis_fewshot_async(
            researcher_profile, paper, context_entries
        )
        lineage = {
            "parent_entry_ids": [e["entry_id"] for e in context_entries],
            "evolution": evolution,
        }

    judgment = await judge_score_async(researcher_profile, paper, hypothesis)
    archive_embedding = await get_embedding_async(
        archive_text(paper, hypothesis),
        dimensions=EMBEDDING_DIMENSIONS,
    )

    new_entry: dict = {
        "entry_id": new_entry_id(),
        "item_id": paper["item_id"],
        "status": "active",
        "method": method,
        "title": paper.get("title", "Unknown"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "abstract": (paper.get("abstract") or "").strip(),
        "hypothesis": hypothesis,
        "judgment": judgment,
        "embedding": archive_embedding,
        **lineage,
        "created_at": datetime.now().isoformat(),
    }

    neighbors = nearest_neighbors(
        archive_embedding,
        archive_snapshot,
        top_k=NUM_NEIGHBORS_NOVELTY,
    )

    result: dict = {"new_entry": new_entry}

    novelty = await check_novelty_async(researcher_profile, new_entry, neighbors)
    _sim, nearest_entry = neighbors[0]

    if novelty["is_novel"]:
        new_entry["admit_path"] = "novel"
        result["action"] = "added"
        return result

    duel = await pairwise_duel_async(researcher_profile, new_entry, nearest_entry)

    if duel["new_wins"]:
        new_entry["admit_path"] = "duel_win"
        result["action"] = "replaced"
        set_replacement_target(result, nearest_entry)
        return result

    new_entry["status"] = "discarded"
    new_entry["discarded_at"] = datetime.now().isoformat()
    new_entry["admit_path"] = "duel_loss"
    new_entry["discarded_reason"] = "Lost blind pairwise duel"
    new_entry["lost_to_entry_id"] = nearest_entry["entry_id"]
    result["action"] = "discarded"
    return result


def main() -> None:
    run_qd_loop(Domain(
        app="papers",
        config=config,
        profiles=profiles,
        items=items,
        step=iteration_async,
        quality=lambda e: e["judgment"]["score"],
        quality_name="score",
        quality_max=10,
        revisit_items=True,
        replaced_reason="Lost blind pairwise duel",
    ))


if __name__ == "__main__":
    main()
