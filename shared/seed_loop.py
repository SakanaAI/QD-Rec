"""Evaluate a fixed seed sample with bounded retries of failed items only."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable

from shared.evaluation_budget import next_batch_size


async def evaluate_seeds(
    items: list[dict],
    evaluate: Callable[[dict, int], Awaitable[dict]],
    *,
    num_parallel_workers: int,
) -> tuple[list[dict], int]:
    """Return successful entries in sample order and the total attempt count.

    evaluate(item, index) receives the original sample index. Retry only failures,
    using the main run's attempt cap. The caller saves partial results and raises.
    """
    if not items:
        raise ValueError("seeding requires at least one item")
    if num_parallel_workers <= 0:
        raise ValueError("num_parallel_workers must be positive")
    entries: list[dict | None] = [None] * len(items)
    pending = deque(range(len(items)))
    succeeded = attempts = 0
    while True:
        n = next_batch_size(persisted=succeeded, budget=len(items), scheduled=attempts,
                            batch_size=num_parallel_workers)
        if not n:
            break
        batch = [pending.popleft() for _ in range(min(n, len(pending)))]
        results = await asyncio.gather(
            *(evaluate(items[i], i) for i in batch), return_exceptions=True,
        )
        attempts += len(results)
        for index, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"[WARN] Seed {index + 1}/{len(items)} failed: {type(result).__name__}: {result}")
                pending.append(index)
            else:
                entries[index] = result
                succeeded += 1
    return [entry for entry in entries if entry is not None], attempts
