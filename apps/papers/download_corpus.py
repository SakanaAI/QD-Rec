"""
Download the paper pool: the accepted papers of ICLR 2026, ICML 2026 and
NeurIPS 2025 from the community-maintained `papercopilot/paperlists` GitHub repo
(https://github.com/papercopilot/paperlists), which mirrors OpenReview listings.

Output: `papers.jsonl`, one JSON object per line, with fields `id`, `title`,
`abstract`, `year` and `venue` ("ICLR" | "ICML" | "NeurIPS").

Usage:
  uv run python -m apps.papers.download_corpus
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import requests

from .config import PAPER_ABSTRACTS_FILE

# Pinned revisions preserve the paper's 17,511-item pool (ICLR 5,358, ICML 6,341,
# NeurIPS 5,812). Each venue maps to (URL, year).
# ICLR uses the media endpoint because its raw URL returns a Git LFS pointer.
SOURCES: dict[str, tuple[str, int]] = {
    "ICLR": (
        "https://media.githubusercontent.com/media/papercopilot/paperlists/"
        "875df9d6550849ec07c9c6feba24986be2dd4551/iclr/iclr2026.json",
        2026,
    ),
    "ICML": (
        "https://raw.githubusercontent.com/papercopilot/paperlists/"
        "9fcfd1234879a8e5630b5116dde411ec143dd763/icml/icml2026.json",
        2026,
    ),
    "NeurIPS": (
        "https://raw.githubusercontent.com/papercopilot/paperlists/"
        "e687edf6748206aedac80f5c17c0221d3c97f788/nips/nips2025.json",
        2025,
    ),
}

# Statuses counted as "accepted". Excludes Reject / Withdraw / Desk Reject.
ACCEPTED_STATUSES = {
    "Oral",
    "Poster",
    "Spotlight",
    "ICLR 2026 ConditionalPoster",
    "ICLR 2026 ConditionalOral",
}


def download_venue(venue: str, url: str, timeout: int = 120) -> list[dict]:
    """Fetch one venue's paperlist JSON and return the raw records (unfiltered)."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    records = resp.json()
    if not isinstance(records, list):
        raise ValueError(f"{venue}: expected a JSON array from {url}, got {type(records)}")
    return records


def filter_and_normalize(records: list[dict], venue: str, year: int) -> list[dict]:
    """Keep accepted papers only, mapped to the output schema."""
    papers: list[dict] = []
    for rec in records:
        if rec.get("status") not in ACCEPTED_STATUSES:
            continue
        pid = (rec.get("id") or "").strip()
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()
        if not pid or not title or not abstract:
            continue
        papers.append(
            {
                "id": pid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "venue": venue,
            }
        )
    return papers


def dedupe_by_id(papers: list[dict]) -> tuple[list[dict], int]:
    """Drop later duplicates by `id`; return (deduped, num_dupes_found)."""
    seen: set[str] = set()
    deduped: list[dict] = []
    dupes = 0
    for p in papers:
        pid = p["id"]
        if pid in seen:
            dupes += 1
            continue
        seen.add(pid)
        deduped.append(p)
    return deduped, dupes


def main(output: str = PAPER_ABSTRACTS_FILE) -> None:
    output_path = Path(output)

    print("=" * 60)
    print("Downloading Multi-Venue Paper-Abstracts Corpus")
    print("=" * 60)

    all_papers: list[dict] = []
    per_venue_counts: dict[str, int] = {}
    for venue, (url, year) in SOURCES.items():
        print(f"\n[{venue} {year}] downloading {url}")
        records = download_venue(venue, url)
        print(f"[{venue} {year}] {len(records)} total records")
        accepted = filter_and_normalize(records, venue, year)
        print(f"[{venue} {year}] {len(accepted)} accepted papers kept")
        per_venue_counts[venue] = len(accepted)
        all_papers.extend(accepted)

    all_papers, num_dupes = dedupe_by_id(all_papers)
    if num_dupes:
        print(f"\n[WARN] Dropped {num_dupes} duplicate id(s) across venues")
    else:
        print("\nNo duplicate ids found across venues")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print("-" * 60)
    print("Done")
    for venue in SOURCES:
        print(f"  {venue}: {per_venue_counts[venue]}")
    print(f"  TOTAL: {len(all_papers)}")
    print(f"Wrote {output_path}")
    print(f"Downloaded: {date.today().isoformat()}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Download the multi-venue (ICLR/ICML/NeurIPS) paper-abstracts corpus"
    )
    parser.parse_args()
    main()


if __name__ == "__main__":
    cli()
