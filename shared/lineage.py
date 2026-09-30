"""Map the proposer's self-reported evolution lineage to archive entry IDs.

Few-shot replies name an operation (mutate, crossover or explore) and 1-based
source_examples from the displayed context. Example i refers to context[i - 1];
explore has no parents.
"""

from __future__ import annotations

VALID_OPERATIONS = ("mutate", "crossover", "explore")


def fewshot_schema(payload_key: str) -> dict:
    """JSON schema for a hypothesis under payload_key and its evolution lineage."""
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": list(VALID_OPERATIONS)},
            "source_examples": {"type": "array", "items": {"type": "integer"}},
            "evolution_note": {"type": "string"},
            payload_key: {"type": "string"},
        },
        "required": ["operation", "source_examples", "evolution_note", payload_key],
        "additionalProperties": False,
    }


def _coerce_indices(raw, n: int) -> list[int]:
    """The integers in [1, n] from the LLM's `source_examples`, deduplicated in
    order; anything malformed or out of range is dropped."""
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    seen: set[int] = set()
    out: list[int] = []
    for item in raw:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def parse_evolution(
    result: dict,
    context: list[dict],
) -> dict:
    """Validate the reported operation and map source examples to parent entry IDs.

    context follows the prompt's example order. Invalid operations are inferred
    from valid parents: none -> explore, one -> mutate, multiple -> crossover.
    """
    n = len(context)
    indices = _coerce_indices(result.get("source_examples"), n)

    op = result.get("operation")
    op = op.strip().lower() if isinstance(op, str) else ""
    if op not in VALID_OPERATIONS:
        op = "explore" if not indices else ("mutate" if len(indices) == 1 else "crossover")

    # An unresolved mutation/crossover becomes a new lineage root.
    if op == "explore":
        indices = []
    elif not indices:
        op = "explore"

    evo_parent_entry_ids = [context[i - 1].get("entry_id") for i in indices]
    if any(not isinstance(parent_id, str) or not parent_id for parent_id in evo_parent_entry_ids):
        raise ValueError("lineage context entry has no entry_id")
    return {
        "operation": op,
        "source_examples": indices,
        "evo_parent_entry_ids": evo_parent_entry_ids,
        "evolution_note": (result.get("evolution_note") or "").strip(),
    }
