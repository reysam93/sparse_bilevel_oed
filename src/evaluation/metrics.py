"""Evaluation metrics (binding conventions 0.5-0.7).

Per-instance metrics are aggregated over test instances with the median;
dispersion is the IQR (25-75 percentiles). Means may be stored as extra
diagnostic columns but paper figures use medians.
"""

from __future__ import annotations

import numpy as np


def nmse(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    """Per-instance NMSE: ||beta_hat - beta_true||^2 / ||beta_true||^2."""
    denom = float(np.dot(beta_true, beta_true))
    if denom == 0.0:
        raise ValueError("beta_true is identically zero; NMSE undefined")
    diff = beta_hat - beta_true
    return float(np.dot(diff, diff) / denom)


def prediction_error(X: np.ndarray, beta_hat: np.ndarray, y: np.ndarray) -> float:
    """Per-instance prediction error over the full M candidate rows:
    0.5 * ||X beta_hat - y||^2 (convention 0.7, same rows for all methods)."""
    r = X @ beta_hat - y
    return float(0.5 * np.dot(r, r))


def estimated_support(beta_hat: np.ndarray, tau_supp: float) -> np.ndarray:
    """supp(beta_hat) = {j : |beta_hat_j| > tau_supp * ||beta_hat||_inf}."""
    scale = float(np.max(np.abs(beta_hat)))
    if scale == 0.0:
        return np.zeros(beta_hat.shape, dtype=bool)
    return np.abs(beta_hat) > tau_supp * scale


def support_f1(beta_hat: np.ndarray, beta_true: np.ndarray, tau_supp: float) -> float:
    est = estimated_support(beta_hat, tau_supp)
    true = beta_true != 0.0
    tp = int(np.sum(est & true))
    if tp == 0:
        return 0.0
    precision = tp / int(np.sum(est))
    recall = tp / int(np.sum(true))
    return float(2.0 * precision * recall / (precision + recall))


def error_covariance_metrics(
    beta_hats: np.ndarray, beta_trues: np.ndarray, rho: float
) -> dict:
    """trace / log-det / max-eigenvalue of C_hat = mean_n e_n e_n^T + rho I."""
    E = beta_hats - beta_trues                    # (N, D)
    N, D = E.shape
    C = (E.T @ E) / N + rho * np.eye(D)
    eigvals = np.linalg.eigvalsh(C)
    return {
        "traceC": float(np.trace(C)),
        "logdetC": float(np.sum(np.log(eigvals))),
        "lmaxC": float(eigvals[-1]),
    }


def binarity_metrics(w: np.ndarray) -> dict:
    """Obin(w) = sum w_i (1 - w_i);  theta(w) = sum min(w_i, 1 - w_i)."""
    w = np.asarray(w, dtype=float)
    return {
        "Obin": float(np.sum(w * (1.0 - w))),
        "theta": float(np.sum(np.minimum(w, 1.0 - w))),
    }


def median_iqr(values: np.ndarray | list) -> dict:
    """Median and IQR (q25, q75) aggregation of per-instance values."""
    v = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(v, [25.0, 50.0, 75.0])
    return {
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "mean": float(np.mean(v)),
    }
