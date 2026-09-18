"""Evaluate a design with the deployed lower-level estimator.

Fair-evaluation rule: every method's design is scored by solving the SAME
deployed estimator (weighted Lasso/elastic-net; closed-form ridge for
lambda = 0) on the same test instances, with lambda and mu fixed from
training data (binding conventions 0.4 and 5.2 of the plan).
"""

from __future__ import annotations

import time

import numpy as np

from src.evaluation import metrics as mx
from src.solvers.lasso import solve_weighted_lasso, solve_weighted_lasso_batch
from src.solvers.ridge import weighted_ridge


def solve_deployed(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    lam: float,
    mu: float,
    R_diag: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Deployed estimate for one instance (ridge closed form if lam == 0)."""
    if lam == 0.0:
        return weighted_ridge(X, y, w, mu, R_diag)
    return solve_weighted_lasso(
        X, y, w, lam, mu, R_diag, max_iter=max_iter, tol=tol
    ).beta


def evaluate_design(
    design: np.ndarray,
    X: np.ndarray,
    Y_test: np.ndarray,
    beta_test: np.ndarray,
    lam: float,
    mu: float,
    R_diag: np.ndarray,
    tau_supp: float,
    rho: float,
    solver_max_iter: int = 2000,
    solver_tol: float = 1e-10,
    deadline: float | None = None,
    regularizer: str = "l1",
) -> dict:
    """Median/IQR-aggregated test metrics for one design.

    ``regularizer``: "l1" (default; weighted Lasso/elastic-net) or "tv"
    (E6 deployed estimator; support_F1 is then a value-support placeholder,
    not the plan's jump-F1 - see the E6 scaffold notes).

    Returns a dict with NMSE, prediction_error, support_F1 (medians; extra
    *_q25/_q75/_mean diagnostic columns), covariance metrics, n_selected and
    a status flag ('zero_beta_hat' if any deployed estimate collapsed to 0).
    """
    N_test = Y_test.shape[0]
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("evaluation exceeded wall-clock timeout")
    # Batched deployed solves: identical estimator, one vectorized FISTA over
    # all instances (rows with zero weight dropped exactly); ridge closed
    # form for lam = 0.
    if lam == 0.0:
        beta_hats = weighted_ridge(X, Y_test, design, mu, R_diag)
    elif regularizer == "tv":
        from src.solvers.tv import solve_weighted_tv_batch

        beta_hats = solve_weighted_tv_batch(
            X, Y_test, design, lam, mu, R_diag,
            max_iter=solver_max_iter, tol=solver_tol,
        )
    else:
        beta_hats = solve_weighted_lasso_batch(
            X, Y_test, design, lam, mu, R_diag,
            max_iter=solver_max_iter, tol=solver_tol,
        ).B
    n_zero = int(np.sum(~np.any(beta_hats, axis=1)))
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("evaluation exceeded wall-clock timeout")

    per_nmse = [mx.nmse(beta_hats[n], beta_test[n]) for n in range(N_test)]
    per_pe = [mx.prediction_error(X, beta_hats[n], Y_test[n]) for n in range(N_test)]
    per_f1 = [mx.support_f1(beta_hats[n], beta_test[n], tau_supp) for n in range(N_test)]

    agg_nmse = mx.median_iqr(per_nmse)
    agg_pe = mx.median_iqr(per_pe)
    agg_f1 = mx.median_iqr(per_f1)
    cov = mx.error_covariance_metrics(beta_hats, beta_test, rho)

    result = {
        "NMSE": agg_nmse["median"],
        "NMSE_q25": agg_nmse["q25"],
        "NMSE_q75": agg_nmse["q75"],
        "NMSE_mean": agg_nmse["mean"],
        "prediction_error": agg_pe["median"],
        "prediction_error_q25": agg_pe["q25"],
        "prediction_error_q75": agg_pe["q75"],
        "support_F1": agg_f1["median"],
        "support_F1_q25": agg_f1["q25"],
        "support_F1_q75": agg_f1["q75"],
        "n_selected": int(np.sum(design > 0.5)),
        "n_zero_beta_hat": n_zero,
        "status": "ok" if n_zero == 0 else "zero_beta_hat",
    }
    result.update(cov)
    return result
