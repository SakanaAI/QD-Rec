#!/usr/bin/env bash
# Download the UCSD Goodreads Book Graph "Children" subset (Wan & McAuley)
# into apps/books/data/. See the top-level README.md for the dataset's terms.
#
# The UCSD server tends to reset long connections, so each file is fetched in
# a resume loop (curl -C -) and verified with gunzip -t before moving on.
# Expected sizes: books ~98 MB, reviews ~164 MB, interactions ~488 MB.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data
cd data

BASE="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre"
FILES="goodreads_books_children.json.gz goodreads_reviews_children.json.gz goodreads_interactions_children.json.gz"

for f in $FILES; do
  if [ -f "$f" ] && gunzip -t "$f" 2>/dev/null; then
    echo "== $f already complete, skipping"
    continue
  fi
  echo "== downloading $f"
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    curl -fL -C - --retry 5 --retry-delay 3 --retry-all-errors -o "$f" "$BASE/$f" && break
    if [ "$attempt" -ge 50 ]; then
      echo "ERROR: $f failed after $attempt resume attempts" >&2
      exit 1
    fi
    echo "   connection dropped (attempt $attempt); resuming in 3s..."
    sleep 3
  done
  if ! gunzip -t "$f"; then
    echo "ERROR: $f is corrupt after download" >&2
    exit 1
  fi
  echo "   OK: $f ($(du -h "$f" | cut -f1))"
done

echo "All files downloaded and verified."
