"""Zero-shot and few-shot hypothesis generation, with lineage parsed by shared.lineage."""

from __future__ import annotations

from shared.lineage import fewshot_schema, parse_evolution
from shared.llm import acreate, agenerate_payload, client, json_schema_format, loads_lenient

from .config import PROPOSER_MODEL, REASONING_EFFORT
from .prompts import (
    MEMO_FEWSHOT_SYSTEM_PROMPT,
    MEMO_FEWSHOT_USER_PROMPT,
    MEMO_SYSTEM_PROMPT,
    MEMO_USER_PROMPT,
)

# Constrained output avoids malformed escapes in research notes, including LaTeX.
_MEMO_SCHEMA = {
    "type": "object",
    "properties": {"research_notes": {"type": "string"}},
    "required": ["research_notes"],
    "additionalProperties": False,
}
_FEWSHOT_SCHEMA = fewshot_schema("research_notes")


async def _generate_once(researcher_profile: str, paper: dict) -> dict:
    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "") or ""
    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": MEMO_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MEMO_USER_PROMPT.format(
                    researcher_profile=researcher_profile,
                    title=title,
                    abstract=abstract,
                ),
            },
        ],
        response_format=json_schema_format("abstracts_memo", _MEMO_SCHEMA),
    )
    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_async(researcher_profile: str, paper: dict) -> str:
    """Zero-shot memo, retrying a provider that omits the payload field."""
    _result, hypothesis = await agenerate_payload(
        lambda: _generate_once(researcher_profile, paper), "research_notes"
    )
    return hypothesis


def _format_context(entries: list[dict]) -> str:
    """Few-shot examples for the prompt: each entry's paper and memo, without its judge score."""
    if not entries:
        return "(none)"

    parts: list[str] = []
    for i, e in enumerate(entries, 1):
        title = e.get("title", "Unknown")
        year = e.get("year")
        abstract = (e.get("abstract") or "").strip()
        memo = (e.get("hypothesis") or "").strip()

        # Bound lengths to keep prompts sane.
        abstract = abstract[:2000]
        memo = memo[:2500]

        parts.append(
            f"""### Example {i}
Title: {title}
Year: {year}

Abstract:
{abstract}

**Research Memo (example):**
{memo}
"""
        )
    return "\n\n---\n\n".join(parts)


async def _generate_fewshot_once(researcher_profile: str, paper: dict, context_entries: list[dict]) -> dict:
    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "") or ""
    context_text = _format_context(context_entries)

    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": MEMO_FEWSHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MEMO_FEWSHOT_USER_PROMPT.format(
                    researcher_profile=researcher_profile,
                    existing_recommendations=context_text,
                    title=title,
                    abstract=abstract,
                ),
            },
        ],
        response_format=json_schema_format("abstracts_memo_fewshot", _FEWSHOT_SCHEMA),
    )
    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_fewshot_async(
    researcher_profile: str, paper: dict, context_entries: list[dict]
) -> tuple[str, dict]:
    """Return (hypothesis, evolution), including the operator and parent entry IDs."""
    result, hypothesis = await agenerate_payload(
        lambda: _generate_fewshot_once(researcher_profile, paper, context_entries),
        "research_notes",
    )
    return hypothesis, parse_evolution(result, context_entries)
