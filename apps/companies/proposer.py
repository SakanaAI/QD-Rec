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


async def _generate_once(acquirer_profile: str, target_company: dict) -> dict:
    target_profile = item_text(target_company)

    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HYPOTHESIS_USER_PROMPT.format(
                    acquirer_profile=acquirer_profile,
                    target_profile=target_profile,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_async(acquirer_profile: str, target_company: dict) -> str:
    """Generate an acquisition hypothesis for a target company (zero-shot)."""
    _result, hypothesis = await agenerate_payload(
        lambda: _generate_once(acquirer_profile, target_company), "hypothesis"
    )
    return hypothesis


def _format_context(entries: list[dict]) -> str:
    """Format sampled entries into text for the prompt."""
    if not entries:
        return "(No existing theses yet)"

    text = ""
    for i, entry in enumerate(entries, 1):
        company_profile = item_text(entry)
        hypothesis = entry.get("hypothesis", "")
        text += f"""### Thesis {i}
**Company Profile:**
{company_profile}

**Hypothesis:** {hypothesis}

"""
    return text


async def _generate_fewshot_once(
    acquirer_profile: str, target_company: dict, context_entries: list[dict]
) -> dict:
    target_profile = item_text(target_company)
    context_text = _format_context(context_entries)

    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HYPOTHESIS_FEWSHOT_USER_PROMPT.format(
                    acquirer_profile=acquirer_profile,
                    existing_theses=context_text,
                    target_profile=target_profile,
                ),
            },
        ],
        response_format=json_schema_format("mna_hypothesis_fewshot", _FEWSHOT_SCHEMA),
    )

    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_fewshot_async(
    acquirer_profile: str, target_company: dict, context_entries: list[dict]
) -> tuple[str, dict]:
    """Return (hypothesis, evolution), including the operator and parent entry IDs."""
    result, hypothesis = await agenerate_payload(
        lambda: _generate_fewshot_once(acquirer_profile, target_company, context_entries), "hypothesis"
    )
    return hypothesis, parse_evolution(result, context_entries)
