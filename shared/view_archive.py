"""Write a self-contained archive viewer with a PCA map, progress curves and lineage.

    uv run python -m shared.view_archive                       # newest run
    uv run python -m shared.view_archive <archive.json> --open

The HTML is saved beside the archive. Its parent directory names the domain
(archives/<app>/), which supplies the item display functions.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import webbrowser
from pathlib import Path

import numpy as np

from shared.archive_format import METHOD_NAMES

GRID = 25
TEMPLATE = Path(__file__).with_name("view_archive.html")
RUN_ARCHIVE = re.compile(r"_(qd_rec|novelty_evo|novelty)[_.]")


def quality_of(app: str):
    """Return the domain quality function, maximum score and display name."""
    if app == "books":
        return (lambda e: float(e["rating"])), 5, "rating"
    return (lambda e: float(e["judgment"]["score"])), 10, "score"


def newest_run_archive(root: str) -> str:
    """The most recently written Novelty, Novelty+Evo or QD-Rec archive under root."""
    files = [f for f in glob.glob(os.path.join(root, "*", "*.json")) if RUN_ARCHIVE.search(os.path.basename(f))]
    if not files:
        raise SystemExit(f"no run archive found under {root}/; pass an archive path")
    return max(files, key=os.path.getmtime)


def replay_order(entries: list[dict]) -> list[dict]:
    """Seed entries first, then every evaluation in the order it was created."""
    seeds = [e for e in entries if e["method"] == "seed"]
    evals = sorted((e for e in entries if e["method"] != "seed"), key=lambda e: e["created_at"])
    return seeds + evals


def replay(order: list[dict], quality) -> list[dict]:
    """Replay admissions and replacements to recover active-archive size and mean quality."""
    active: dict[str, float] = {}
    curve = []
    n_seed = sum(1 for e in order if e["method"] == "seed")
    for k, e in enumerate(order):
        if e["status"] != "discarded":
            target = e.get("replaces_entry_id")
            if target:
                if target not in active:
                    raise ValueError(f"entry {e['entry_id']} replaces {target}, which is not in the archive at that point")
                del active[target]
            active[e["entry_id"]] = quality(e)
        if k + 1 >= n_seed:
            curve.append({"n": k + 1 - n_seed, "q": round(float(np.mean(list(active.values()))), 4) if active else None,
                          "size": len(active)})
    final = {e["entry_id"] for e in order if e["status"] == "active"}
    if set(active) != final:
        raise ValueError("replaying the archive does not end with its active entries")
    return curve


def project(entries: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """2D PCA of the L2-normalized entry embeddings, fitted on the active entries,
    and each entry's cell on the GRID x GRID map over the active entries' extent."""
    vecs = np.array([e["embedding"] for e in entries], dtype=float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    active = np.array([e["status"] == "active" for e in entries])
    mean = vecs[active].mean(axis=0)
    _, _, vt = np.linalg.svd(vecs[active] - mean, full_matrices=False)
    xy = (vecs - mean) @ vt[:2].T
    lo, hi = xy[active].min(axis=0), xy[active].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    cells = np.clip(np.floor((xy - lo) / span * GRID), 0, GRID - 1).astype(int)
    active_vecs = vecs[active]
    sims = active_vecs @ active_vecs.T
    m = len(active_vecs)
    diversity = 1.0 - (sims.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
    coverage = len({tuple(c) for c in cells[active]}) / GRID ** 2
    return xy, cells, {"diversity": round(float(diversity), 4), "coverage": round(float(coverage), 4)}


def build(archive_path: str) -> dict:
    data = json.loads(Path(archive_path).read_text())
    meta, entries = data["metadata"], data["entries"]
    app = Path(archive_path).resolve().parent.name
    items = importlib.import_module(f"apps.{app}.items")
    quality, qmax, qname = quality_of(app)

    order = replay_order(entries)
    curve = replay(order, quality)
    number = {}
    n_seed = sum(1 for e in order if e["method"] == "seed")
    for k, e in enumerate(order):
        number[e["entry_id"]] = 0 if e["method"] == "seed" else k + 1 - n_seed
    xy, cells, spread = project(order)

    children: dict[str, list[str]] = {}
    for e in order:
        for p in (e.get("evolution") or {}).get("evo_parent_entry_ids", []):
            children.setdefault(p, []).append(e["entry_id"])

    rows = []
    for e, (x, y), (cx, cy) in zip(order, xy, cells):
        evo = e.get("evolution") or {}
        judgment = e.get("judgment") or {}
        rows.append({
            "id": e["entry_id"],
            "n": number[e["entry_id"]],
            "label": items.item_label(e),
            "text": items.item_text(e),
            "hyp": e["hypothesis"],
            "q": quality(e),
            "reason": judgment.get("reasoning", ""),
            "status": e["status"],
            "admit": e.get("admit_path", "seed" if e["method"] == "seed" else ""),
            "op": evo.get("operation", "seed" if e["method"] == "seed" else "zero-shot"),
            "note": evo.get("evolution_note", ""),
            "used": evo.get("evo_parent_entry_ids", []),
            "shown": e.get("parent_entry_ids", []),
            "children": children.get(e["entry_id"], []),
            "replaces": e.get("replaces_entry_id"),
            "replacedBy": e.get("replaced_by_entry_id"),
            "lostTo": e.get("lost_to_entry_id"),
            "why": e.get("discarded_reason", ""),
            "x": round(float(x), 4), "y": round(float(y), 4), "cx": int(cx), "cy": int(cy),
        })

    cfg = meta.get("config", {})
    active = [r for r in rows if r["status"] == "active"]
    return {
        "meta": {
            "file": os.path.basename(archive_path), "app": app, "profile": meta["profile"],
            "method": METHOD_NAMES.get(meta["method"], meta["method"]),
            "proposer": cfg.get("proposer_model"), "judge": cfg.get("judge_model"),
            "created": meta.get("created_at"), "completed": meta.get("completed_at"),
            "qname": qname, "qmax": qmax, "grid": GRID, "evaluations": len(order) - n_seed, "seeds": n_seed,
        },
        "stats": {
            "active": len(active),
            "replaced": sum(r["status"] == "replaced" for r in rows),
            "discarded": sum(r["status"] == "discarded" for r in rows),
            "quality": round(float(np.mean([r["q"] for r in active])), 3),
            **spread,
        },
        "entries": rows,
        "curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Write an HTML page for exploring one archive.")
    ap.add_argument("archive", nargs="?", help="archive JSON (default: the newest run archive under archives/)")
    ap.add_argument("--out", help="output HTML (default: next to the archive)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()
    archive = args.archive or newest_run_archive(os.environ.get("QD_ARCHIVES_ROOT", "archives"))
    out = Path(args.out or os.path.splitext(archive)[0] + ".html")
    payload = json.dumps(build(archive), ensure_ascii=False).replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text().replace("/*ARCHIVE_DATA*/null", payload))
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
