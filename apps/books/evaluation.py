"""
Novelty check and the rating-based replacement rule.

Quality is the reader's star rating, so replacement compares ratings directly.
The LLM only judges whether a new recommendation adds variety (novelty).
"""

from __future__ import annotations

import random

from shared.llm import acreate, client, loads_lenient, parse_json_bool

from .config import JUDGE_MODEL, REASONING_EFFORT
from .prompts import NOVELTY_SYSTEM_PROMPT, NOVELTY_USER_PROMPT

MAX_PARSE_ATTEMPTS = 3


def format_recommendation_for_novelty(entry: dict, label: str) -> str:
    """Format a book+memo entry for the novelty prompt (no rating shown)."""
    title = entry.get("title", "Unknown")
    description = (entry.get("description") or "").strip()
    memo = (entry.get("hypothesis") or "").strip()
    if len(description) > 600:
        description = description[:600] + " ..."
    return f"""### Recommendation {label}
**Book:** {title}

**Description:** {description}

**Memo:** {memo}
"""


async def check_novelty_async(
    profile: str,
    new_entry: dict,
    neighbors: list[tuple[float, dict]],
) -> dict:
    """Judge novelty against the nearest neighbors.

    Raise RuntimeError after MAX_PARSE_ATTEMPTS unusable replies.
    """
    entries = [new_entry] + [nb for _sim, nb in neighbors]
    order = list(range(len(entries)))
    random.shuffle(order)

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    recommendations_text = ""
    new_label = None
    for pos, idx in enumerate(order):
        label = labels[pos]
        entry = entries[idx]
        marker = " **(NEW)**" if idx == 0 else ""
        block = format_recommendation_for_novelty(entry, label)
        block = block.replace(f"### Recommendation {label}", f"### Recommendation {label}{marker}")
        recommendations_text += block + "\n"
        if idx == 0:
            new_label = label

    for _attempt in range(MAX_PARSE_ATTEMPTS):
        resp = await acreate(client(),
            model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": NOVELTY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": NOVELTY_USER_PROMPT.format(
                        profile=profile,
                        recommendations_text=recommendations_text,
                        new_recommendation_label=new_label,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            result = loads_lenient(resp.choices[0].message.content)
        except ValueError:
            continue
        is_novel = parse_json_bool(result.get("is_novel"))
        if is_novel is not None:
            return {
                "is_novel": is_novel,
                "novelty_reasoning": result.get("novelty_reasoning", ""),
            }
    raise RuntimeError(f"novelty check reply unreadable {MAX_PARSE_ATTEMPTS} times (model={JUDGE_MODEL})")


def new_entry_beats_nearest_neighbor(new_entry: dict, nearest_entry: dict) -> bool:
    """True if the new book is rated strictly higher (a tie keeps the incumbent)."""
    return float(new_entry.get("rating", 0)) > float(nearest_entry.get("rating", 0))
