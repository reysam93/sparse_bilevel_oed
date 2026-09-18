"""Weighted ridge: closed-form solution of the smooth lower-level problem.

    min_beta 0.5 * || D_w^{1/2} R^{-1/2} (y - X beta) ||^2 + 0.5 * mu * ||beta||^2

with explicit diagonal noise R (binding convention 0.3). Solution:

    beta = (X^T D_w R^{-1} X + mu I)^{-1} X^T D_w R^{-1} y
"""

from __future__ import annotations

import numpy as np


def weighted_ridge(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    mu: float,
    R_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form weighted ridge estimate. ``y`` may be (M,) or (N, M)."""
    M = X.shape[0]
    if R_diag is None:
        R_diag = np.ones(M)
    if mu <= 0:
        raise ValueError("weighted ridge requires mu > 0")
    d = np.asarray(w, dtype=float) / np.asarray(R_diag, dtype=float)   # (M,)
    A = X.T @ (d[:, None] * X) + mu * np.eye(X.shape[1])
    if y.ndim == 1:
        b = X.T @ (d * y)
    else:
        b = (y * d) @ X                                                # (N, D)
        return np.linalg.solve(A, b.T).T
    return np.linalg.solve(A, b)
