"""
Configuration for the paper-abstract exploration domain.
"""

import os

from shared.paths import model_tag
from pathlib import Path


_APP_DIR = Path(__file__).parent
_PROJECT_ROOT = _APP_DIR.parent.parent

# Candidate papers (from download_corpus.py) and their item embeddings
# (from create_embeddings.py)
PAPER_ABSTRACTS_FILE = str(_APP_DIR / "data" / "paper-abstracts" / "papers.jsonl")
PAPER_ABSTRACTS_EMBEDDINGS_FILE = str(_APP_DIR / "data" / "paper-abstracts" / "paper_embeddings.json")

# QD_ARCHIVES_ROOT redirects both seed lookup and run output.
ARCHIVES_DIR = str(_PROJECT_ROOT / (os.getenv("QD_ARCHIVES_ROOT") or "archives") / "papers")

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

# Seed
NUM_SEEDS = 10
NUM_PARALLEL_WORKERS = int(os.getenv("NUM_PARALLEL_WORKERS", "20"))

# QD loop (Novelty, Novelty+Evo, QD-Rec)
NUM_ITERATIONS = 1000
NUM_NEIGHBORS_NOVELTY = int(os.getenv("NUM_NEIGHBORS_NOVELTY", "3"))
NUM_CONTEXT_SAMPLES = int(os.getenv("NUM_CONTEXT_SAMPLES", "3"))

# Cosine-similarity cutoff for deduplicating additions within a batch.
DEDUP_SIMILARITY_THRESHOLD = 0.9

