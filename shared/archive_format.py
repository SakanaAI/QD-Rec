"""The archive file every domain writes and reads: ``{"metadata": ..., "entries": [...]}``.

``metadata`` has the same keys, in the same order, in every domain and stage:

    profile        the value passed to --profile (reader, acquirer, product or researcher)
    method         "seed", "novelty", "novelty_evo" or "qd_rec"
    created_at     when the stage started
    checkpoint_at  when this file was written; runs rewrite it after every batch
    completed_at   when the stage finished with its whole budget persisted, else None
    config         the stage's settings: seed_config() or run_config()
    results        its entry and evaluation counts: archive_results()

``entries`` are the evaluated item--hypothesis pairs (see archive_records.py).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime

from shared.archive_records import validate_archive_entries
from shared.bandit import LinearThompsonSampling
from shared.embeddings import DEFAULT_EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from shared.llm import reasoning_effort_for
from shared.paths import write_json_atomic

METHODS = ("qd_rec", "novelty_evo", "novelty")
METHOD_NAMES = {"qd_rec": "QD-Rec", "novelty_evo": "Novelty+Evo", "novelty": "Novelty"}


def _environment_config(seed: int) -> dict:
    return {
        "seed": seed,
        "temperature": None,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
    }


def seed_config(
    *,
    proposer_model: str,
    judge_model: str,
    reasoning_effort: str | None,
    num_seeds: int,
    num_parallel_workers: int,
    seed: int,
) -> dict:
    """``metadata.config`` of a seed archive."""
    return {
        "proposer_model": proposer_model,
        "proposer_reasoning_effort": reasoning_effort_for(proposer_model, reasoning_effort),
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort_for(judge_model, reasoning_effort),
        "num_seeds": num_seeds,
        "num_parallel_workers": num_parallel_workers,
        **_environment_config(seed),
    }


def run_config(
    *,
    method: str,
    proposer_model: str,
    judge_model: str,
    reasoning_effort: str | None,
    num_iterations: int,
    num_context_samples: int,
    num_neighbors_novelty: int,
    dedup_similarity_threshold: float,
    num_parallel_workers: int,
    allow_repick_active_targets: bool,
    selection: str,
    ts: LinearThompsonSampling | None,
    ts_diversity_beta: float,
    reward_scaling: str,
    ts_warm_start_observations: int,
    seed: int,
) -> dict:
    """Build run metadata, including context and bandit settings only for methods that use them."""
    return {
        "proposer_model": proposer_model,
        "proposer_reasoning_effort": reasoning_effort_for(proposer_model, reasoning_effort),
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort_for(judge_model, reasoning_effort),
        "num_iterations": num_iterations,
        **({"num_context_samples": num_context_samples} if method != "novelty" else {}),
        "num_neighbors_novelty": num_neighbors_novelty,
        "dedup_similarity_threshold": dedup_similarity_threshold,
        "num_parallel_workers": num_parallel_workers,
        "allow_repick_active_targets": allow_repick_active_targets,
        "selection": selection,
        **({
            "ts_prior_lambda": ts.lam,
            "ts_sigma": ts.sigma,
            "ts_diversity_beta": ts_diversity_beta,
            "reward_scaling": reward_scaling,
            "ts_warm_start_observations": ts_warm_start_observations,
        } if method == "qd_rec" else {}),
        **_environment_config(seed),
    }


def archive_results(
    initial_entries: int,
    entries: list[dict],
    attempts: int,
    quality: Callable[[dict], float],
) -> dict:
    """Count entries, evaluation attempts and mean quality over active entries.

    Each completed evaluation adds an entry, even if discarded or later replaced.
    Failures add none; initial_entries excludes the seeds from the run's count.
    """
    succeeded = len(entries) - initial_entries
    statuses = Counter(e["status"] for e in entries)
    active_quality = [quality(e) for e in entries if e["status"] == "active"]
    return {
        "initial_entries": initial_entries,
        "total_entries": len(entries),
        "active_entries": statuses["active"],
        "replaced_entries": statuses["replaced"],
        "discarded_entries": statuses["discarded"],
        "evaluations_attempted": attempts,
        "evaluations_succeeded": succeeded,
        "evaluations_failed": attempts - succeeded,
        "mean_quality": sum(active_quality) / len(active_quality) if active_quality else None,
    }


def write_archive(
    path: str,
    *,
    profile: str,
    method: str,
    created_at: str,
    completed: bool,
    config: dict,
    results: dict,
    entries: list[dict],
) -> None:
    """Validate ``entries`` and write the archive to ``path`` atomically."""
    now = datetime.now().isoformat()
    archive = {
        "metadata": {
            "profile": profile,
            "method": method,
            "created_at": created_at,
            "checkpoint_at": now,
            "completed_at": now if completed else None,
            "config": config,
            "results": results,
        },
        "entries": entries,
    }
    validate_archive_entries(entries)
    write_json_atomic(path, archive, indent=2, ensure_ascii=False)


def load_archive(path: str) -> dict:
    """Read an archive written by write_archive and validate its entries."""
    with open(path, "r", encoding="utf-8") as f:
        archive = json.load(f)
    validate_archive_entries(archive["entries"])
    return archive
