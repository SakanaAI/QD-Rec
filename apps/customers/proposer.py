"""Zero-shot and few-shot hypothesis generation, with lineage parsed by shared.lineage."""

from shared.lineage import fewshot_schema, parse_evolution
from shared.llm import acreate, agenerate_payload, client, json_schema_format, loads_lenient

from .config import PROPOSER_MODEL, REASONING_EFFORT
from .items import item_text
from .prompts import (
    HYPOTHESIS_SYSTEM_PROMPT,
    HYPOTHESIS_USER_PROMPT,
    HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT,
    HYPOTHESIS_FEWSHOT_USER_PROMPT,
)

_FEWSHOT_SCHEMA = fewshot_schema("hypothesis")


async def _generate_once(target_product: str, persona_data: dict) -> dict:
    persona_profile = item_text(persona_data)

    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HYPOTHESIS_USER_PROMPT.format(
                    target_product=target_product,
                    persona_profile=persona_profile,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_async(target_product: str, persona_data: dict) -> str:
    """Generate a hypothesis for why a persona would be interested in the product (zero-shot)."""
    _result, hypothesis = await agenerate_payload(
        lambda: _generate_once(target_product, persona_data), "hypothesis"
    )
    return hypothesis


def _format_context(entries: list[dict]) -> str:
    """Format sampled entries into text for the prompt."""
    if not entries:
        return "(No existing personas yet)"

    text = ""
    for i, entry in enumerate(entries, 1):
        persona_profile = item_text(entry)
        hypothesis = entry.get("hypothesis", "")
        text += f"""### Persona {i}
**Persona profile:**
{persona_profile}

**Hypothesis:** {hypothesis}

"""
    return text


async def _generate_fewshot_once(
    target_product: str, persona_data: dict, context_entries: list[dict]
) -> dict:
    persona_profile = item_text(persona_data)
    context_text = _format_context(context_entries)

    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HYPOTHESIS_FEWSHOT_USER_PROMPT.format(
                    target_product=target_product,
                    existing_theses=context_text,
                    persona_profile=persona_profile,
                ),
            },
        ],
        response_format=json_schema_format("personas_hypothesis_fewshot", _FEWSHOT_SCHEMA),
    )

    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_fewshot_async(
    target_product: str, persona_data: dict, context_entries: list[dict]
) -> tuple[str, dict]:
    """Return (hypothesis, evolution), including the operator and parent entry IDs."""
    result, hypothesis = await agenerate_payload(
        lambda: _generate_fewshot_once(target_product, persona_data, context_entries), "hypothesis"
    )
    return hypothesis, parse_evolution(result, context_entries)
