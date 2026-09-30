"""Build each reader's recommendation pool and profile from Goodreads data.

Use rated books (rating >= 1). Reserve REVIEWS_PER_STAR median-length reviews
per star level for the profile and exclude those books from recommendations.
Write data/<key>/{universe.jsonl, profile_items.json, profile.txt}.

    uv run python -m apps.books.data_prep --profile all
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict

from .config import (
    MIN_RATING,
    MIN_REVIEW_LEN,
    RAW_BOOKS,
    RAW_INTERACTIONS,
    RAW_REVIEWS,
    REVIEWS_PER_STAR,
    profile_file,
    profile_items_file,
    universe_file,
    user_data_dir,
)
from .profiles import (
    get_user_id,
    get_user_name,
    list_user_keys,
)

# Shelves about reading status / ownership / format, not content.
BLOCK = {
    "to-read", "currently-reading", "owned", "owned-books", "books-i-own",
    "favorites", "favourites", "to-buy", "wish-list", "wishlist", "default",
    "library", "my-library", "kindle", "ebook", "ebooks", "e-book", "audio",
    "audiobook", "audiobooks", "audio-books", "have", "re-read", "reread",
    "dnf", "abandoned", "did-not-finish", "unfinished", "my-books", "books",
    "all-books", "to-read-owned", "owned-tbr", "i-own", "series",
    "children", "childrens", "childrens-books", "children-s-books", "kids",
    "kid", "juvenile", "juvenile-fiction", "read",
}
_YEAR = re.compile(r"(^read-in|^read-20|\b(19|20)\d\d\b)")


def content_shelves(popular_shelves: list[dict], top: int = 10) -> list[str]:
    """Top content-genre shelf names (count-sorted input), filtering process shelves."""
    out: list[str] = []
    for s in popular_shelves:
        name = s.get("name", "")
        if name in BLOCK or _YEAR.search(name):
            continue
        out.append(name)
        if len(out) >= top:
            break
    return out


def collect_ratings(user_ids: set[str]) -> dict[str, dict[str, int]]:
    """user_id -> {book_id: rating} for rating >= MIN_RATING."""
    print(f"[1/3] scanning interactions for {len(user_ids)} user(s) ...", flush=True)
    out: dict[str, dict[str, int]] = {u: {} for u in user_ids}
    with gzip.open(RAW_INTERACTIONS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            u = r["user_id"]
            if u in user_ids and r["rating"] >= MIN_RATING:
                out[u][r["book_id"]] = r["rating"]
    return out


def collect_reviews(user_ids: set[str]) -> dict[str, dict[str, str]]:
    """user_id -> {book_id: review_text} (full text, stripped)."""
    print(f"[2/3] scanning reviews for {len(user_ids)} user(s) ...", flush=True)
    out: dict[str, dict[str, str]] = {u: {} for u in user_ids}
    with gzip.open(RAW_REVIEWS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            u = r["user_id"]
            if u in user_ids:
                out[u][r["book_id"]] = (r.get("review_text") or "").strip()
    return out


def collect_book_meta(book_ids: set[str]) -> dict[str, dict]:
    """book_id -> {title, description, shelves}."""
    print(f"[3/3] scanning books for {len(book_ids):,} title(s) ...", flush=True)
    out: dict[str, dict] = {}
    with gzip.open(RAW_BOOKS, "rt") as fh:
        for line in fh:
            b = json.loads(line)
            bid = b["book_id"]
            if bid in book_ids:
                out[bid] = {
                    "title": b.get("title") or "",
                    "description": (b.get("description") or "").strip(),
                    "shelves": content_shelves(b.get("popular_shelves") or []),
                }
    return out


def select_profile_books(
    rated: dict[str, int], reviews: dict[str, str]
) -> list[tuple[int, str]]:
    """
    Per star level, the REVIEWS_PER_STAR median-length substantive reviews.
    Returns list of (star, book_id), deterministic.
    """
    by_star: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for bid, rating in rated.items():
        txt = reviews.get(bid, "")
        if len(txt) >= MIN_REVIEW_LEN:
            by_star[rating].append((len(txt), bid))

    picks: list[tuple[int, str]] = []
    for star in range(1, 6):
        cands = sorted(by_star.get(star, []))  # ascending by review length
        if not cands:
            print(f"    [warn] no substantive review at {star}*; skipping that star")
            continue
        mid = len(cands) // 2
        start = max(0, mid - REVIEWS_PER_STAR // 2)
        for _len, bid in cands[start : start + REVIEWS_PER_STAR]:
            picks.append((star, bid))
    return picks


def build_profile_text(items: list[dict]) -> str:
    """Deterministic template (no LLM) wrapping the selected books and reviews."""
    lines = [
        "Below are books this reader previously read, one per star level. For each,",
        "the book's title and description are shown, followed by the star rating the",
        "reader gave (out of 5) and the review they wrote.",
        "",
    ]
    for it in items:
        desc = it.get("description") or "(no description available)"
        lines.append(f"Title: {it['title']}")
        lines.append(f"Description: {desc}")
        lines.append(f"Reader's rating: {it['rating']}/5")
        lines.append(f"Reader's review: {it['review_text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_user(key: str, ratings: dict[str, int], reviews: dict[str, str], meta: dict[str, dict]) -> None:
    picks = select_profile_books(ratings, reviews)
    profile_ids = {bid for _star, bid in picks}

    profile_items = []
    for star, bid in sorted(picks):
        m = meta.get(bid, {})
        profile_items.append({
            "book_id": bid,
            "rating": star,
            "title": m.get("title", ""),
            "description": m.get("description", ""),
            "review_text": reviews.get(bid, ""),
        })

    universe = []
    for bid, rating in ratings.items():
        if bid in profile_ids:
            continue
        m = meta.get(bid)
        if m is None:
            continue
        universe.append({
            "book_id": bid,
            "title": m["title"],
            "description": m["description"],
            "shelves": m["shelves"],
            "rating": rating,
            "review_text": reviews.get(bid, ""),
        })

    d = user_data_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    with open(universe_file(key), "w", encoding="utf-8") as f:
        for rec in universe:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(profile_items_file(key), "w", encoding="utf-8") as f:
        json.dump(profile_items, f, ensure_ascii=False, indent=2)
    profile_file(key).write_text(build_profile_text(profile_items), encoding="utf-8")

    print(f"\n=== {key}  ({get_user_name(key)}, id={get_user_id(key)[:8]}) ===")
    print(f"  rated books: {len(ratings)}  ->  universe: {len(universe)}  (profile removed: {len(profile_ids)})")
    print("  profile picks (median-length review per star):")
    for it in profile_items:
        print(f"    {it['rating']}*  [{len(it['review_text'])} chars]  {it['title'][:70]}")
    print(f"  wrote: {universe_file(key).name}, {profile_items_file(key).name}, {profile_file(key).name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-user Goodreads-Children dataset")
    ap.add_argument("--profile", default="all", help="reader key or 'all' (default: all)")
    args = ap.parse_args()

    keys = list_user_keys() if args.profile == "all" else [args.profile]
    user_ids = {get_user_id(k) for k in keys}

    ratings_by_uid = collect_ratings(user_ids)
    reviews_by_uid = collect_reviews(user_ids)
    all_book_ids = {bid for uid in user_ids for bid in ratings_by_uid[uid]}
    meta = collect_book_meta(all_book_ids)

    for key in keys:
        uid = get_user_id(key)
        build_user(key, ratings_by_uid[uid], reviews_by_uid[uid], meta)


if __name__ == "__main__":
    main()
