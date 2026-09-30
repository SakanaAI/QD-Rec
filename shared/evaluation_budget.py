"""The bounded evaluation budget used by seeding and the main run loop."""

import math


EXTRA_ATTEMPTS_FRACTION = 0.05


def next_batch_size(*, persisted: int, budget: int, scheduled: int, batch_size: int) -> int:
    """Steps to schedule next, or 0 once the budget or attempt cap is reached.

    Failed evaluations persist nothing. Allow up to one extra worker batch or
    5% of the requested budget, whichever is larger, to make up the shortfall.
    The caller saves its partial archive and raises if the cap is exhausted.
    """
    if persisted >= budget:
        return 0
    cap = budget + max(batch_size, math.ceil(budget * EXTRA_ATTEMPTS_FRACTION))
    if scheduled >= cap:
        print(
            f"[WARN] {persisted}/{budget} evaluations persisted after scheduling {scheduled} "
            f"steps (cap {cap}): failures are systematic, giving up"
        )
        return 0
    return min(batch_size, budget - persisted, cap - scheduled)
