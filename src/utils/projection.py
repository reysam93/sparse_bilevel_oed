"""Projection onto the design set.

BUDGET CONVENTION (author decision 2026-07-04, second audit pass; overrides
the inequality wording of plan Section 0.2 / AGENTS.md — recorded in
docs/implementation_notes.md): budgeted simulation designs use the EQUALITY
capped simplex by default,

    W_eq = { w in [0, 1]^M : sum_i w_i = M0 },

projected via w_i = clip(v_i - tau, 0, 1) with tau found by bisection so that
sum(w) = M0. Binary designs must satisfy sum_i s_i = M0 exactly.

The inequality projection/check are kept ONLY as internal utilities for
diagnostics that explicitly ask for them; they must not be used as the
default in E0/E1 experiments.
"""

from __future__ import annotations

import numpy as np


def _bisect_shift(v: np.ndarray, budget: float, lo: float, hi: float,
                  tol: float, max_iter: int) -> float:
    """Solve sum(clip(v - tau, 0, 1)) = budget for tau (nonincreasing in tau)."""
    for _ in range(max_iter):
        t = 0.5 * (lo + hi)
        if np.clip(v - t, 0.0, 1.0).sum() > budget:
            lo = t
        else:
            hi = t
        if hi - lo < tol:
            break
    return hi


def project_capped_simplex_equality(
    v: np.ndarray, budget: float, tol: float = 1e-12, max_iter: int = 200
) -> np.ndarray:
    """Project ``v`` onto {w in [0,1]^M : sum(w) = budget} (the default)."""
    v = np.asarray(v, dtype=float)
    M = v.size
    if not 0.0 <= budget <= M:
        raise ValueError(f"equality budget must be in [0, {M}], got {budget}")
    # tau may be negative (raising entries) when clip(v,0,1) sums below budget.
    lo = float(np.min(v)) - 1.0    # sum = M  >= budget
    hi = float(np.max(v))          # sum = 0  <= budget
    tau = _bisect_shift(v, budget, lo, hi, tol, max_iter)
    w = np.clip(v - tau, 0.0, 1.0)
    return w


def project_capped_simplex_inequality(
    v: np.ndarray, budget: float, tol: float = 1e-12, max_iter: int = 200
) -> np.ndarray:
    """Project onto {w in [0,1]^M : sum(w) <= budget}.

    INTERNAL / diagnostic use only — not the default design set (see module
    docstring).
    """
    v = np.asarray(v, dtype=float)
    if budget < 0:
        raise ValueError(f"budget must be nonnegative, got {budget}")
    w = np.clip(v, 0.0, 1.0)
    if w.sum() <= budget + tol:
        return w
    tau = _bisect_shift(v, budget, 0.0, float(np.max(v)), tol, max_iter)
    return np.clip(v - tau, 0.0, 1.0)


def check_feasible(
    w: np.ndarray,
    budget: float,
    tol: float = 1e-8,
    binary: bool = False,
    equality: bool = True,
) -> None:
    """Raise ValueError if ``w`` violates the design set.

    Default is the EQUALITY budget sum(w) = budget; pass ``equality=False``
    only for explicitly inequality-based diagnostics.
    """
    w = np.asarray(w, dtype=float)
    if np.any(w < -tol) or np.any(w > 1.0 + tol):
        raise ValueError("design violates box constraints [0, 1]")
    total = w.sum()
    if equality:
        if abs(total - budget) > tol * max(1.0, budget):
            raise ValueError(
                f"design violates equality budget: sum={total:.10g} != {budget}"
            )
    elif total > budget + tol:
        raise ValueError(f"design violates budget: sum={total:.6g} > {budget}")
    if binary:
        if not np.all((np.abs(w) < tol) | (np.abs(w - 1.0) < tol)):
            raise ValueError("design is not binary")
