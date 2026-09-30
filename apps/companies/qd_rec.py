"""Novelty, Novelty+Evo and QD-Rec for M&A target discovery.

Selection and batching are handled by shared.qd_loop; this module evaluates one
item–hypothesis pair against the active archive.

    uv run python -m apps.companies.qd_rec --profile PROFILE --method qd_rec
"""

from datetime import datetime

from shared.archive_records import new_entry_id, set_replacement_target
from shared.embeddings import get_embedding_async, nearest_neighbors
from shared.qd_loop import Domain, run_qd_loop

from . import config, items, profiles
from .config import NUM_NEIGHBORS_NOVELTY
from .evaluation import check_novelty_async, judge_score_async, pairwise_duel_async
from .items import archive_text
from .proposer import generate_hypothesis_async, generate_hypothesis_fewshot_async


async def iteration_async(
    acquirer_profile: str,
    company: dict,
    context_entries: list[dict] | None,
    archive_snapshot: list[dict],
    method: str,
) -> dict:
    """Evaluate one candidate against the batch snapshot without mutating the archive."""
    if method == "novelty":
        hypothesis = await generate_hypothesis_async(acquirer_profile, company)
        lineage: dict = {}
    else:
        hypothesis, evolution = await generate_hypothesis_fewshot_async(
            acquirer_profile, company, context_entries
        )
        lineage = {
            "parent_entry_ids": [e["entry_id"] for e in context_entries],
            "evolution": evolution,
        }

    embedding = await get_embedding_async(archive_text(company, hypothesis))

    neighbors = nearest_neighbors(
        embedding,
        archive_snapshot,
        top_k=NUM_NEIGHBORS_NOVELTY,
    )

    new_entry = {
        "entry_id": new_entry_id(),
        "item_id": company["item_id"],
        "status": "active",
        "method": method,
        **company,
        "hypothesis": hypothesis,
        "embedding": embedding,
        **lineage,
        "created_at": datetime.now().isoformat(),
    }

    result: dict = {"new_entry": new_entry}

    judgment = await judge_score_async(acquirer_profile, company, hypothesis)
    new_entry["judgment"] = judgment

    novelty = await check_novelty_async(acquirer_profile, new_entry, neighbors)
    _sim, nearest_entry = neighbors[0]

    if novelty["is_novel"]:
        new_entry["admit_path"] = "novel"
        result["action"] = "added"
        return result

    duel = await pairwise_duel_async(acquirer_profile, new_entry, nearest_entry)

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
        app="companies",
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
