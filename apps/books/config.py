"""
Configuration for the Goodreads-Children QD recommender.

Human ground-truth ratings (1-5) replace the LLM quality signal used by the
other apps; the universe is each query user's own rated books.
"""

import os
from pathlib import Path

from shared.paths import model_tag

_APP_DIR = Path(__file__).parent
_PROJECT_ROOT = _APP_DIR.parent.parent
DATA_DIR = _APP_DIR / "data"

# Raw UCSD Goodreads *Children* subset (gitignored; fetch via download.sh)
RAW_BOOKS = DATA_DIR / "goodreads_books_children.json.gz"
RAW_REVIEWS = DATA_DIR / "goodreads_reviews_children.json.gz"
RAW_INTERACTIONS = DATA_DIR / "goodreads_interactions_children.json.gz"

# Query users (the profiles): the paper's two readers. Key -> Goodreads user ID and name.
USERS: dict[str, dict] = {
    "picture_books_reader": {
        "id": "929bfe22df70aacffe15fa2478d96177",
        "name": "Picture-book reader",
    },
    "series_ya_reader": {
        "id": "c2cb4a54cd109a3c683e7efdffba2e0f",
        "name": "Series/YA reader",
    },
}


def require_user(key: str) -> dict:
    """The USERS entry of a query user; KeyError for an unknown key."""
    if key not in USERS:
        raise KeyError(f"Unknown user key {key!r}; valid keys: {list(USERS)}")
    return USERS[key]


# Per-user data, written by data_prep.py and create_embeddings.py
def user_data_dir(key: str) -> Path:
    require_user(key)
    return DATA_DIR / key


def universe_file(key: str) -> Path:
    return user_data_dir(key) / "universe.jsonl"


def profile_items_file(key: str) -> Path:
    return user_data_dir(key) / "profile_items.json"


def profile_file(key: str) -> Path:
    return user_data_dir(key) / "profile.txt"


def book_embeddings_file(key: str) -> Path:
    """Per-reader item-embedding store."""
    return user_data_dir(key) / "book_embeddings.json"


# QD_ARCHIVES_ROOT redirects both seed lookup and run output.
ARCHIVES_DIR = str(_PROJECT_ROOT / (os.getenv("QD_ARCHIVES_ROOT") or "archives") / "books")

# Embeddings
EMBEDDING_DIMENSIONS = 512
EMBEDDING_CONCURRENCY = 50  # concurrent embedding requests in create_embeddings.py

# OpenRouter model IDs; see README for the paper's model configuration.
_DEFAULT_LLM_MODEL = "openai/gpt-6-luna"
_DEFAULT_JUDGE_MODEL = "openai/gpt-6-luna"
PROPOSER_MODEL = os.getenv("PROPOSER_MODEL", _DEFAULT_LLM_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
# Set to an empty string to disable reasoning effort.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high") or None
PROPOSER_SUFFIX = "_prop-" + model_tag(PROPOSER_MODEL)
JUDGE_SUFFIX = "_judge-" + model_tag(JUDGE_MODEL)

# Seeds Python and NumPy randomness.
SEED = int(os.getenv("SEED", "0"))

# Seed / QD loop
NUM_SEEDS = 10
NUM_PARALLEL_WORKERS = int(os.getenv("NUM_PARALLEL_WORKERS", "20"))
NUM_ITERATIONS = 1000
NUM_NEIGHBORS_NOVELTY = int(os.getenv("NUM_NEIGHBORS_NOVELTY", "3"))
NUM_CONTEXT_SAMPLES = int(os.getenv("NUM_CONTEXT_SAMPLES", "3"))

# Cosine-similarity cutoff for deduplicating additions within a batch.
DEDUP_SIMILARITY_THRESHOLD = 0.9

# Universe + profile construction
MIN_RATING = 1        # rating 0 == shelved-but-unrated; exclude from universe
REVIEWS_PER_STAR = 1  # profile reviews per star level
MIN_REVIEW_LEN = 120  # chars; only substantive reviews qualify for the profile
