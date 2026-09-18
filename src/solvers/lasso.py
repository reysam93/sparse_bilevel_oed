"""Weighted Lasso / elastic-net lower-level solver (FISTA).

Problem (binding convention 0.3, explicit diagonal noise R):

    min_beta g_s(w, beta) + lambda * ||beta||_1
    g_s(w, beta) = 0.5 * || D_w^{1/2} R^{-1/2} (y - X beta) ||^2
                   + 0.5 * mu * ||beta||^2

g_s is mu-strongly convex for mu > 0 uniformly in w >= 0. FISTA on the smooth
part with entrywise soft-thresholding; supports warm starts, over-solving for
diagnostics, and returns solver diagnostics.

For lambda = 0 the solution must match ``weighted_ridge`` (required test).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LassoResult:
    beta: np.ndarray
    objective: float
    primal_residual: float      # relative change of beta at the last step
    iterations: int
    runtime: float
    converged: bool


def objective_value(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, beta: np.ndarray,
    lam: float, mu: float, R_diag: np.ndarray,
) -> float:
    r = y - X @ beta
    d = w / R_diag
    return float(
        0.5 * np.dot(d * r, r) + 0.5 * mu * np.dot(beta, beta)
        + lam * np.abs(beta).sum()
    )


def smooth_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, beta: np.ndarray,
    mu: float, R_diag: np.ndarray,
) -> np.ndarray:
    d = w / R_diag
    return X.T @ (d * (X @ beta - y)) + mu * beta


def smooth_lipschitz(X: np.ndarray, w: np.ndarray, mu: float, R_diag: np.ndarray) -> float:
    """L = lambda_max(X^T D_w R^{-1} X) + mu (exact, D x D eigendecomposition)."""
    d = w / R_diag
    H = X.T @ (d[:, None] * X)
    return float(np.linalg.eigvalsh(H)[-1]) + mu


def soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


@dataclass
class BatchLassoResult:
    B: np.ndarray               # (N, D) solutions
    objective: np.ndarray       # (N,) per-instance objectives
    primal_residual: float      # max relative iterate change at the last step
    iterations: int
    runtime: float
    converged: bool


def solve_weighted_lasso_batch(
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    lam: float,
    mu: float,
    R_diag: np.ndarray | None = None,
    B0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
    active_row_reduction: bool = True,
) -> BatchLassoResult:
    """Batched FISTA over N instances sharing (X, w, lam, mu, R).

    Mathematically identical to running ``solve_weighted_lasso`` per row of
    ``Y`` (same iterate map, same momentum schedule, which is data
    independent); iterations continue until EVERY instance meets ``tol``.
    Rows with w_i = 0 contribute nothing to the weighted loss, so with
    ``active_row_reduction`` the problem is reduced to the selected rows
    (exact, not an approximation) - the main saving for binary designs.
    """
    t_start = time.perf_counter()
    M, D = X.shape
    Y = np.atleast_2d(Y)
    N = Y.shape[0]
    if R_diag is None:
        R_diag = np.ones(M)
    if mu <= 0:
        raise ValueError("solver requires mu > 0 (strong convexity)")
    w = np.asarray(w, dtype=float)

    Xa, Ya, wa, Ra = X, Y, w, np.asarray(R_diag, dtype=float)
    if active_row_reduction:
        active = w > 0.0
        if active.sum() < M:
            Xa, Ya, wa, Ra = X[active], Y[:, active], w[active], Ra[active]

    d = wa / Ra
    H = Xa.T @ (d[:, None] * Xa)                 # (D, D), shared across instances
    L = float(np.linalg.eigvalsh(H)[-1]) + mu
    step = 1.0 / L
    XtDy = (Ya * d) @ Xa                          # (N, D)

    B = np.zeros((N, D)) if B0 is None else np.array(B0, dtype=float, copy=True)
    Z = B.copy()
    t_mom = 1.0
    rel = np.full(N, np.inf)
    it = 0
    for it in range(1, max_iter + 1):
        grad = Z @ H + mu * Z - XtDy
        B_new = soft_threshold(Z - step * grad, step * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom**2))
        Z = B_new + ((t_mom - 1.0) / t_new) * (B_new - B)
        rel = (np.linalg.norm(B_new - B, axis=1)
               / (1.0 + np.linalg.norm(B_new, axis=1)))
        B = B_new
        t_mom = t_new
        if np.all(rel < tol):
            break

    resid = Ya - B @ Xa.T
    objective = (0.5 * (resid**2 @ d) + 0.5 * mu * np.sum(B**2, axis=1)
                 + lam * np.abs(B).sum(axis=1))
    return BatchLassoResult(
        B=B,
        objective=objective,
        primal_residual=float(np.max(rel)),
        iterations=it,
        runtime=time.perf_counter() - t_start,
        converged=bool(np.all(rel < tol)),
    )


def solve_weighted_lasso(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    lam: float,
    mu: float,
    R_diag: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> LassoResult:
    """FISTA for the weighted Lasso/elastic-net.

    ``tol`` is on the relative iterate change; ``max_iter`` large values can be
    used to over-solve for diagnostics.
    """
    t_start = time.perf_counter()
    M, D = X.shape
    if R_diag is None:
        R_diag = np.ones(M)
    if mu <= 0:
        raise ValueError("solver requires mu > 0 (strong convexity)")
    w = np.asarray(w, dtype=float)
    beta = np.zeros(D) if beta0 is None else np.asarray(beta0, dtype=float).copy()

    L = smooth_lipschitz(X, w, mu, R_diag)
    step = 1.0 / L
    z = beta.copy()
    t_mom = 1.0
    d = w / R_diag
    XtDy = X.T @ (d * y)
    H = X.T @ (d[:, None] * X)          # (D, D), reused every iteration

    converged = False
    rel_change = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        grad = H @ z - XtDy + mu * z
        beta_new = soft_threshold(z - step * grad, step * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom**2))
        z = beta_new + ((t_mom - 1.0) / t_new) * (beta_new - beta)
        rel_change = float(
            np.linalg.norm(beta_new - beta) / (1.0 + np.linalg.norm(beta_new))
        )
        beta = beta_new
        t_mom = t_new
        if rel_change < tol:
            converged = True
            break

    return LassoResult(
        beta=beta,
        objective=objective_value(X, y, w, beta, lam, mu, R_diag),
        primal_residual=rel_change,
        iterations=it,
        runtime=time.perf_counter() - t_start,
        converged=converged,
    )
