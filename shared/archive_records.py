"""Archive identity and replacement operations.

item_id identifies a source item; entry_id identifies an evaluated item–hypothesis
pair. Replacements use entry_id because one item can have several active hypotheses.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from shared.batch_dedup import dedupe_added_batch

ENTRY_METHODS = ("seed", "novelty", "novelty_evo", "qd_rec")
ENTRY_STATUSES = ("active", "replaced", "discarded")


def new_entry_id() -> str:
    """Return a unique, opaque ID for one evaluated item--hypothesis pair."""
    return str(uuid4())


def normalize_source_item(record: dict, source_id_key: str) -> dict:
    """Copy a source record, replacing its domain-specific ID field with item_id."""
    item = dict(record)
    raw_id = item.pop(source_id_key, None)
    if raw_id is None or not str(raw_id).strip():
        raise ValueError(f"source record is missing {source_id_key!r}")
    item["item_id"] = str(raw_id)
    return item


def validate_archive_entries(entries: list[dict]) -> None:
    """Require the common entry schema and valid entry-level references."""
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        item_id = entry.get("item_id")
        entry_id = entry.get("entry_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"archive entry {index} has no item_id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"archive entry {index} has no entry_id")
        if entry_id in seen:
            raise ValueError(f"duplicate entry_id in archive: {entry_id}")
        if entry.get("method") not in ENTRY_METHODS:
            raise ValueError(f"archive entry {entry_id} has method {entry.get('method')!r}")
        if entry.get("status") not in ENTRY_STATUSES:
            raise ValueError(f"archive entry {entry_id} has status {entry.get('status')!r}")
        if not isinstance(entry.get("hypothesis"), str):
            raise ValueError(f"archive entry {entry_id} has no hypothesis")
        seen.add(entry_id)

    by_entry_id = {entry["entry_id"]: entry for entry in entries}
    for entry in entries:
        entry_id = entry["entry_id"]
        status = entry.get("status", "active")
        if status == "replaced" and not entry.get("replaced_by_entry_id"):
            raise ValueError(f"replaced entry {entry_id} has no replacer entry ID")
        if status == "discarded" and entry.get("replaces_entry_id"):
            raise ValueError(f"discarded entry {entry_id} claims a replacement")
        for field in (
            "replaces_entry_id",
            "replaced_by_entry_id",
            "lost_to_entry_id",
        ):
            target = entry.get(field)
            if target is not None and target not in seen:
                raise ValueError(f"{entry_id}.{field} references unknown entry {target}")
        for field in ("parent_entry_ids",):
            for target in entry.get(field) or []:
                if target not in seen:
                    raise ValueError(
                        f"{entry_id}.{field} references unknown entry {target}"
                    )
        evolution = entry.get("evolution") or {}
        for target in evolution.get("evo_parent_entry_ids") or []:
            if target not in seen:
                raise ValueError(
                    f"{entry_id}.evolution.evo_parent_entry_ids references "
                    f"unknown entry {target}"
                )
        replaced = entry.get("replaces_entry_id")
        if replaced and (
            by_entry_id[replaced].get("status") != "replaced"
            or by_entry_id[replaced].get("replaced_by_entry_id") != entry_id
        ):
            raise ValueError(
                f"replacement link {entry_id} -> {replaced} is not reciprocal"
            )
        replacer = entry.get("replaced_by_entry_id")
        if replacer and (
            by_entry_id[replacer].get("status") == "discarded"
            or by_entry_id[replacer].get("replaces_entry_id") != entry_id
        ):
            raise ValueError(
                f"replacement link {entry_id} <- {replacer} is not reciprocal"
            )


def set_replacement_target(result: dict, incumbent: dict) -> None:
    """Record the exact incumbent pair targeted by a successful duel."""
    target = incumbent.get("entry_id")
    if not isinstance(target, str) or not target:
        raise ValueError("replacement target has no entry_id")
    result["replace_target_entry_id"] = target


def resolve_replacement_collisions(
    replaced: list[dict],
    discarded: list[dict],
    *,
    choose: Callable[[list[dict]], dict] = random.choice,
    timestamp: Callable[[], str] = lambda: datetime.now().isoformat(),
) -> list[dict]:
    """Keep one proposal when several workers target the same archive entry.

    Proposals targeting different hypotheses of the same item are intentionally
    kept in different groups.
    """
    groups: dict[str, list[dict]] = {}
    for result in replaced:
        target = result.get("replace_target_entry_id")
        if not isinstance(target, str) or not target:
            raise ValueError("replacement result has no replace_target_entry_id")
        groups.setdefault(target, []).append(result)

    winners: list[dict] = []
    for target, group in groups.items():
        winner = choose(group) if len(group) > 1 else group[0]
        winners.append(winner)
        for result in group:
            if result is winner:
                continue
            entry = result["new_entry"]
            entry["status"] = "discarded"
            entry["discarded_at"] = timestamp()
            entry["discarded_reason"] = (
                f"Batch collision: another proposal replaced entry {target}"
            )
            discarded.append(result)
    return winners


def apply_replacement(
    archive: list[dict],
    result: dict,
    *,
    reason: str,
    timestamp: Callable[[], str] = lambda: datetime.now().isoformat(),
) -> dict:
    """Mark the exact targeted incumbent as replaced and return it.

    Missing or duplicate targets are fatal because silently appending the winner
    would create an archive inconsistent with the recorded duel.
    """
    target_id = result.get("replace_target_entry_id")
    new_entry = result.get("new_entry")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("replacement result has no replace_target_entry_id")
    if not isinstance(new_entry, dict):
        raise ValueError("replacement result has no new_entry")
    replacement_id = new_entry.get("entry_id")
    if not isinstance(replacement_id, str) or not replacement_id:
        raise ValueError("replacement entry has no entry_id")

    matches = [
        entry
        for entry in archive
        if entry.get("status", "active") == "active"
        and entry.get("entry_id") == target_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one active replacement target {target_id}, found {len(matches)}"
        )

    incumbent = matches[0]
    incumbent["status"] = "replaced"
    incumbent["replaced_by_entry_id"] = replacement_id
    incumbent["replaced_at"] = timestamp()
    incumbent["replaced_reason"] = reason
    new_entry["replaces_entry_id"] = target_id
    return incumbent


def active_entries(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if entry.get("status") == "active"]


def sample_context(entries: list[dict], k: int) -> list[dict]:
    """Sample up to k active entries without replacement."""
    active = active_entries(entries)
    if not active or k <= 0:
        return []
    return active if len(active) <= k else random.sample(active, k)


def reconcile_batch(
    results: list[dict],
    archive: list[dict],
    *,
    threshold: float,
    reason: str,
) -> None:
    """Apply worker results to the archive in place.

    Deduplicate additions judged against the same snapshot, then randomly resolve
    replacements targeting the same entry. Preserve discarded entries for accounting.
    """
    added = [r for r in results if r.get("action") == "added"]
    replaced = [r for r in results if r.get("action") == "replaced"]
    discarded = [r for r in results if r.get("action") == "discarded"]

    kept_added = dedupe_added_batch(added, discarded, threshold)
    kept_replaced = resolve_replacement_collisions(replaced, discarded)

    for r in kept_added:
        archive.append(r["new_entry"])
    for r in kept_replaced:
        apply_replacement(archive, r, reason=reason)
        archive.append(r["new_entry"])
    for r in discarded:
        archive.append(r["new_entry"])
