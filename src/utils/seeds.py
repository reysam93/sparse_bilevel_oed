"""Deterministic seeding utilities (binding convention 0.8).

Independent child streams are spawned from a single base seed so that adding a
method or changing the number of restarts never changes the generated data or
the designs of other methods.
"""

from __future__ import annotations

import numpy as np

# Fixed order of named streams. New stream names must be appended at the end;
# reordering or inserting would silently change all downstream randomness.
STREAM_NAMES = ("data", "random_baseline", "restarts")


def spawn_streams(base_seed: int) -> dict[str, np.random.Generator]:
    """Return independent named generators spawned from ``base_seed``."""
    children = np.random.SeedSequence(base_seed).spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children)
    }


# Fixed stream tags for per-grid-point generators (never reuse or reorder).
_GRIDPOINT_STREAM_TAGS = {"random_baseline": 1715, "restarts": 2716}


def gridpoint_rng(name: str, base_seed: int, mu: float,
                  lambda_ratio: float) -> np.random.Generator:
    """Deterministic generator for one (seed, mu, lambda_ratio) grid point.

    Unlike a shared stream consumed sequentially, this makes the random
    baseline's designs (and restart draws) independent of the execution
    granularity/order of grid points - a batch loop and single-point
    resumable runs produce identical designs (decision 2026-07-05).
    """
    entropy = [
        _GRIDPOINT_STREAM_TAGS[name],
        int(base_seed),
        int(round(mu * 10**9)),
        int(round(lambda_ratio * 10**9)),
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))
