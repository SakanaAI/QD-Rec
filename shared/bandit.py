"""QD-Rec's item selection: linear Thompson sampling with a diversity bonus.

Every candidate item has a fixed, L2-normalized item embedding x. The reward of an
evaluated item, its score scaled to [0, 1], is modelled as linear in x:

    r = x^T theta + eps,  eps ~ N(0, sigma^2),  prior theta ~ N(0, lambda^-1 I).

The posterior is N(mu, A^-1) with A = lambda I + sigma^-2 sum x x^T and
mu = A^-1 sigma^-2 sum r x. It is warm-started from the seed archive and updated
with every new evaluation.

A batch of k items is picked greedily. Each pick draws theta from the posterior and
takes the remaining candidate with the highest

    x^T theta + beta * (1 - max cosine similarity of x to the reference set),

where the reference set holds the archive's items and the items already picked for
the batch (beta = 0 is the plain bandit). Ties go to the candidate listed first in
the pool.
"""
from __future__ import annotations

from collections.abc import Callable, Collection

import numpy as np

TS_PRIOR_LAMBDA = 1.0  # prior precision lambda
TS_SIGMA = 0.1  # reward noise standard deviation sigma
TS_DIVERSITY_BETA = 1.0  # diversity bonus weight beta


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(x))
    if denom <= 0.0:
        return x
    return x / denom


class LinearThompsonSampling:
    """The posterior N(mu, A^-1) over theta, kept as A and b = sigma^-2 sum r x."""

    def __init__(self, dim: int, *, lam: float = TS_PRIOR_LAMBDA, sigma: float = TS_SIGMA) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if lam <= 0:
            raise ValueError("lam must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.dim = dim
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.sigma2 = float(sigma * sigma)
        self.A = np.eye(dim, dtype=np.float64) * self.lam
        self.b = np.zeros(dim, dtype=np.float64)
        self.num_observations = 0

    def update(self, x: np.ndarray, r: float) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {x.shape}")
        rr = float(r)
        self.A += np.outer(x, x) / self.sigma2
        self.b += x * (rr / self.sigma2)
        self.num_observations += 1

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        # theta = mu + L^-T z with A = L L^T and z ~ N(0, I), so Cov(theta) = A^-1.
        L = np.linalg.cholesky(self.A)
        mu = np.linalg.solve(self.A, self.b)
        z = rng.standard_normal(self.dim)
        delta = np.linalg.solve(L.T, z)
        return mu + delta


def scaled_reward(score, top: float) -> float:
    return float(score) / top


def update_posterior(
    ts: LinearThompsonSampling,
    entries: list[dict],
    item_embedding: Callable,
    reward: Callable,
) -> None:
    """Update from evaluated entries using normalized item embeddings and scaled rewards."""
    for e in entries:
        ts.update(l2_normalize(np.asarray(item_embedding(e), dtype=np.float64)), reward(e))


class ThompsonPool:
    """Cache normalized candidate embeddings in pool order, which determines tie-breaking."""

    def __init__(self, items: list[dict], item_embedding: Callable):
        self.items = list(items)
        self.ids = [it["item_id"] for it in self.items]
        self.matrix = np.vstack([
            l2_normalize(np.asarray(item_embedding(it), dtype=np.float64)) for it in self.items
        ])


def select_thompson_batch(
    *,
    rng: np.random.Generator,
    ts: LinearThompsonSampling,
    pool: ThompsonPool,
    k: int,
    archive: list[dict],
    item_embedding: Callable,
    beta: float,
    excluded_ids: Collection[str] = (),
) -> list[dict]:
    """Pick up to k pool items (see the module docstring). The reference set starts
    as the item embeddings of the archive entries; items in excluded_ids are never
    picked."""
    M = pool.matrix
    alive = np.fromiter((i not in excluded_ids for i in pool.ids),
                        dtype=bool, count=len(pool.ids))
    ref: list[np.ndarray] = [l2_normalize(np.asarray(item_embedding(e), dtype=np.float64))
                             for e in archive]

    selected: list[dict] = []
    for _ in range(k):
        if not alive.any():
            break
        theta = ts.sample_theta(rng)
        scores = M @ theta
        if beta:
            sims = M @ np.vstack(ref).T       # (n, r)
            scores = scores + beta * (1.0 - sims.max(axis=1))
        scores[~alive] = -np.inf
        idx = int(np.argmax(scores))
        alive[idx] = False
        selected.append(pool.items[idx])
        ref.append(M[idx])
    return selected
