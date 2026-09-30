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

_FEWSHOT_SCHEMA = fewshot_schema("notes")


async def _generate_once(profile_text: str, book: dict) -> dict:
    resp = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": MEMO_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MEMO_USER_PROMPT.format(
                    profile=profile_text,
                    title=book.get("title", ""),
                    description=(book.get("description") or "").strip(),
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    return loads_lenient(resp.choices[0].message.content)


async def generate_hypothesis_async(profile_text: str, book: dict) -> str:
    """Zero-shot memo, retrying a provider that omits the payload field."""
    _result, text = await agenerate_payload(
        lambda: _generate_once(profile_text, book), "notes"
    )
    return text


def _format_context(entries: list[dict]) -> str:
    if not entries:
        return "(none)"
    parts = []
    for i, e in enumerate(entries, 1):
        desc = (e.get("description") or "").strip()[:600]
        memo = (e.get("hypothesis") or "").strip()[:1500]
        parts.append(f"### Example {i}\nBook: {e.get('title','')}\n\nDescription: {desc}\n\nMemo:\n{memo}")
    return "\n\n---\n\n".join(parts)


async def _generate_fewshot_once(profile_text: str, book: dict, context: list[dict]) -> dict:
    resp = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": MEMO_FEWSHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MEMO_FEWSHOT_USER_PROMPT.format(
                    profile=profile_text,
                    existing_recommendations=_format_context(context),
                    title=book.get("title", ""),
                    description=(book.get("description") or "").strip(),
                ),
            },
        ],
        response_format=json_schema_format("goodreads_memo_fewshot", _FEWSHOT_SCHEMA),
    )
    return loads_lenient(resp.choices[0].message.content)


async def generate_hypothesis_fewshot_async(
    profile_text: str, book: dict, context: list[dict]
) -> tuple[str, dict]:
    """Return (hypothesis, evolution), including the operator and parent entry IDs."""
    result, hypothesis = await agenerate_payload(
        lambda: _generate_fewshot_once(profile_text, book, context), "notes"
    )
    evolution = parse_evolution(result, context)
    return hypothesis, evolution
