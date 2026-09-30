"""Seed Python and NumPy randomness once per command-line run."""

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed stdlib ``random`` and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
