"""
LLM judge for the papers domain: the 0-10 score (from abstract + memo), the
novelty check and the blind pairwise duel.
"""

from __future__ import annotations

import random

from shared.llm import acreate, client, loads_lenient, parse_json_bool, parse_judge_score

from .config import JUDGE_MODEL, REASONING_EFFORT
from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    NOVELTY_CHECK_SYSTEM_PROMPT,
    NOVELTY_CHECK_USER_PROMPT,
    PAIRWISE_DUEL_SYSTEM_PROMPT,
    PAIRWISE_DUEL_USER_PROMPT,
)


async def judge_score_async(researcher_profile: str, paper: dict, hypothesis: str) -> dict:
    """Judge a research memo (the hypothesis) against the paper's abstract."""
    response = await acreate(client(),
        model=JUDGE_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_PROMPT.format(
                    researcher_profile=researcher_profile,
                    paper_abstract=(paper.get("abstract") or "").strip(),
                    research_memo=hypothesis,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    result = loads_lenient(response.choices[0].message.content)
    result["score"] = parse_judge_score(result.get("score"))
    return result


MAX_PARSE_ATTEMPTS = 3


def _format_entry_plain(entry: dict, header: str) -> str:
    """Format a recommendation entry with a free-form header (no scores)."""
    title = entry.get("title", "Unknown")
    abstract = entry.get("abstract", "") or ""
    memo = entry.get("hypothesis", "") or ""
    return f"""### {header}

**Paper:** {title} ({entry["venue"]} {entry["year"]})

**Abstract:**
{abstract}

**Hypothesis/Memo:**
{memo}
"""


async def check_novelty_async(
    researcher_profile: str,
    new_entry: dict,
    neighbors: list[tuple[float, dict]],
) -> dict:
    """Judge novelty against the nearest neighbors.

    Raise RuntimeError after MAX_PARSE_ATTEMPTS unusable replies.
    """
    existing_text = "\n".join(
        _format_entry_plain(nb, f"Existing recommendation {i+1}")
        for i, (_sim, nb) in enumerate(neighbors)
    )
    new_text = _format_entry_plain(new_entry, "New recommendation")

    for _attempt in range(MAX_PARSE_ATTEMPTS):
        response = await acreate(client(),
            model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": NOVELTY_CHECK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": NOVELTY_CHECK_USER_PROMPT.format(
                        researcher_profile=researcher_profile,
                        existing_text=existing_text,
                        new_text=new_text,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            result = loads_lenient(response.choices[0].message.content)
        except ValueError:
            continue
        is_novel = parse_json_bool(result.get("is_novel"))
        if is_novel is not None:
            return {
                "is_novel": is_novel,
                "novelty_reasoning": result.get("novelty_reasoning", ""),
            }
    raise RuntimeError(f"novelty check reply unreadable {MAX_PARSE_ATTEMPTS} times (model={JUDGE_MODEL})")


async def pairwise_duel_async(
    researcher_profile: str,
    new_entry: dict,
    nearest_entry: dict,
) -> dict:
    """Compare the new entry and its nearest neighbor under shuffled A/B labels.

    Hide scores and which entry is new. Raise RuntimeError after
    MAX_PARSE_ATTEMPTS unusable replies.
    """
    pair = [(new_entry, True), (nearest_entry, False)]
    random.shuffle(pair)
    new_label = "A" if pair[0][1] else "B"
    duel_text = "\n".join(
        _format_entry_plain(entry, f"Recommendation {label}")
        for label, (entry, _is_new) in zip("AB", pair)
    )

    for _attempt in range(MAX_PARSE_ATTEMPTS):
        response = await acreate(client(),
            model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": PAIRWISE_DUEL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PAIRWISE_DUEL_USER_PROMPT.format(
                        researcher_profile=researcher_profile,
                        duel_text=duel_text,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            result = loads_lenient(response.choices[0].message.content)
        except ValueError:
            continue
        winner = str(result.get("winner", "")).strip().upper()
        if winner in ("A", "B"):
            return {
                "new_wins": winner == new_label,
                "comparison_reasoning": result.get("comparison_reasoning", ""),
                "new_label": new_label,
            }
    raise RuntimeError(f"duel reply unreadable {MAX_PARSE_ATTEMPTS} times (model={JUDGE_MODEL})")

