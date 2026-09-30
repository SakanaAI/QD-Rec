"""
Configuration for the Marketing domain (California personas from Nemotron-Personas-USA).
"""

import os

from shared.paths import model_tag
from pathlib import Path


_APP_DIR = Path(__file__).parent
_PROJECT_ROOT = _APP_DIR.parent.parent

# Data paths
PERSONAS_FILE = str(_APP_DIR / "data" / "nemotron_personas_CA.parquet")
EMBEDDINGS_FILE = str(_APP_DIR / "data" / "personas_with_embeddings_CA.parquet")

# QD_ARCHIVES_ROOT redirects both seed lookup and run output.
ARCHIVES_DIR = str(_PROJECT_ROOT / (os.getenv("QD_ARCHIVES_ROOT") or "archives") / "customers")

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

# QD loop (Novelty, Novelty+Evo, QD-Rec)
# Archive entries shown to the few-shot proposer (Novelty+Evo, QD-Rec).
NUM_CONTEXT_SAMPLES = int(os.getenv("NUM_CONTEXT_SAMPLES", "3"))

# Nearest neighbors the novelty check compares against.
NUM_NEIGHBORS_NOVELTY = int(os.getenv("NUM_NEIGHBORS_NOVELTY", "3"))

NUM_ITERATIONS = 1000

NUM_SEEDS = 10

# Parallelism
NUM_PARALLEL_WORKERS = int(os.getenv("NUM_PARALLEL_WORKERS", "20"))  # Concurrent evaluations

# Embeddings
EMBEDDING_DIMENSIONS = 512
EMBEDDING_CONCURRENCY = 50  # concurrent embedding requests in create_embeddings.py

# Cosine-similarity cutoff for deduplicating additions within a batch.
DEDUP_SIMILARITY_THRESHOLD = 0.9

# Persona fields used in the profile text (items.item_text)
PERSONA_FIELDS = [
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
]

CONTEXT_FIELDS = [
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
]

DEMOGRAPHIC_FIELDS = [
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "city",
    "state",
]
