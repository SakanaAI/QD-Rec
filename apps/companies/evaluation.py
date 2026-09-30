"""
The judge's 0-10 score, the novelty check and the blind pairwise duel.
"""

import random

from shared.llm import acreate, client, loads_lenient, parse_json_bool, parse_judge_score

from .config import JUDGE_MODEL, REASONING_EFFORT
from .items import item_text
from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    NOVELTY_CHECK_SYSTEM_PROMPT,
    NOVELTY_CHECK_USER_PROMPT,
    PAIRWISE_DUEL_SYSTEM_PROMPT,
    PAIRWISE_DUEL_USER_PROMPT,
)


async def judge_score_async(
    acquirer_profile: str, target_company: dict, hypothesis: str
) -> dict:
    """Judge the quality of an acquisition hypothesis (0-10 score)."""
    response = await acreate(client(),
        model=JUDGE_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_PROMPT.format(
                    acquirer_profile=acquirer_profile,
                    target_profile=item_text(target_company),
                    hypothesis=hypothesis,
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
    """Format a thesis entry with a free-form header (no scores)."""
    company_profile = item_text(entry)
    hypothesis = entry.get("hypothesis", "")
    return f"""### {header}

**Target Company:**
{company_profile}

**Hypothesis:**
{hypothesis}
"""


async def check_novelty_async(
    acquirer_profile: str,
    new_entry: dict,
    neighbors: list[tuple[float, dict]],
) -> dict:
    """Judge novelty against the nearest neighbors.

    Raise RuntimeError after MAX_PARSE_ATTEMPTS unusable replies.
    """
    existing_text = "\n".join(
        _format_entry_plain(nb, f"Existing thesis {i+1}")
        for i, (_sim, nb) in enumerate(neighbors)
    )
    new_text = _format_entry_plain(new_entry, "New thesis")

    for _attempt in range(MAX_PARSE_ATTEMPTS):
        response = await acreate(client(),
            model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": NOVELTY_CHECK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": NOVELTY_CHECK_USER_PROMPT.format(
                        acquirer_profile=acquirer_profile,
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
    acquirer_profile: str,
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
        _format_entry_plain(entry, f"Thesis {label}")
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
                        acquirer_profile=acquirer_profile,
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
