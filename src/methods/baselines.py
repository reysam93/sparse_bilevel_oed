"""Mandatory design baselines: random budgeted, leverage/row-norm, D-opt greedy.

All baselines expose the same interface: they return a ``BaselineDesign`` with
a binary design vector s selecting EXACTLY M0 sensors (equality budget
convention, sum s = M0) and the selected indices. Evaluation with the deployed estimator happens elsewhere so
that every method is scored identically (binding convention: fair evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.projection import check_feasible


@dataclass
class BaselineDesign:
    method: str
    s: np.ndarray                    # (M,) binary design
    selected_indices: np.ndarray     # (n_selected,) int
    n_selected: int
    diagnostics: dict = field(default_factory=dict)


def _design_from_indices(method: str, M: int, indices: np.ndarray, **diag) -> BaselineDesign:
    s = np.zeros(M)
    s[indices] = 1.0
    check_feasible(s, budget=len(indices), binary=True)
    return BaselineDesign(
        method=method,
        s=s,
        selected_indices=np.sort(np.asarray(indices, dtype=int)),
        n_selected=int(len(indices)),
        diagnostics=dict(diag),
    )


def random_budgeted(rng: np.random.Generator, M: int, M0: int) -> BaselineDesign:
    """One random design: M0 sensors uniformly at random without replacement.

    Repeats are drawn by the caller from the dedicated random-baseline stream
    (n_random_repeats in the YAML) and aggregated with the median.
    """
    indices = rng.choice(M, size=M0, replace=False)
    return _design_from_indices("random", M, indices)


def leverage_row_norm(X: np.ndarray, M0: int, R_diag: np.ndarray) -> BaselineDesign:
    """Top-M0 sensors by ||x_i||^2 / R_ii (plan 3.1)."""
    scores = np.sum(X**2, axis=1) / R_diag
    indices = np.argsort(-scores)[:M0]
    return _design_from_indices("leverage", X.shape[0], indices)


def dopt_greedy(X: np.ndarray, M0: int, mu: float, R_diag: np.ndarray) -> BaselineDesign:
    """Greedy D-optimal selection maximizing log det(mu I + sum x_i x_i^T / R_ii).

    Greedy gain of adding sensor i is log(1 + x_i^T A^{-1} x_i / R_ii);
    A^{-1} is updated by Sherman-Morrison. Selects exactly M0 sensors.
    """
    M, D = X.shape
    if mu <= 0:
        raise ValueError("dopt_greedy requires mu > 0 for a well-posed matrix")
    A_inv = (1.0 / mu) * np.eye(D)
    selected: list[int] = []
    available = np.ones(M, dtype=bool)
    gains: list[float] = []
    for _ in range(M0):
        AX = X @ A_inv                                  # (M, D)
        quad = np.einsum("md,md->m", AX, X) / R_diag    # x_i^T A^{-1} x_i / R_ii
        quad[~available] = -np.inf
        i = int(np.argmax(quad))
        gains.append(float(np.log1p(quad[i])))
        selected.append(i)
        available[i] = False
        u = A_inv @ X[i]
        A_inv -= np.outer(u, u) / (R_diag[i] + X[i] @ u)
    return _design_from_indices(
        "dopt_greedy", M, np.array(selected), greedy_gains=gains
    )


def aopt_greedy(X: np.ndarray, M0: int, mu: float, R_diag: np.ndarray) -> BaselineDesign:
    """Greedy A-optimal selection minimizing tr((mu I + sum x_i x_i^T / R_ii)^{-1}).

    Pre-registered implementation choice (notes 12.2): greedy with rank-1
    Sherman-Morrison updates of A^{-1}, symmetric to dopt_greedy. Adding row i
    reduces the trace by ||A^{-1} x_i||^2 / R_ii / (1 + x_i^T A^{-1} x_i / R_ii).
    Selects exactly M0 sensors.
    """
    M, D = X.shape
    if mu <= 0:
        raise ValueError("aopt_greedy requires mu > 0 for a well-posed matrix")
    A_inv = (1.0 / mu) * np.eye(D)
    selected: list[int] = []
    available = np.ones(M, dtype=bool)
    gains: list[float] = []
    for _ in range(M0):
        AX = X @ A_inv                                   # (M, D) rows: A^{-1} x_i
        quad = np.einsum("md,md->m", AX, X) / R_diag     # x^T A^{-1} x / R
        reduction = np.sum(AX**2, axis=1) / R_diag / (1.0 + quad)
        reduction[~available] = -np.inf
        i = int(np.argmax(reduction))
        gains.append(float(reduction[i]))
        selected.append(i)
        available[i] = False
        u = A_inv @ X[i]
        A_inv -= np.outer(u, u) / (R_diag[i] + X[i] @ u)
    return _design_from_indices(
        "aopt_greedy", M, np.array(selected), greedy_gains=gains
    )


def full_design(M: int) -> BaselineDesign:
    """All sensors active. DIAGNOSTIC reference only: not budget-feasible
    (violates sum s = M0); reported as method 'diagnostic_full_design'."""
    return _design_from_indices("diagnostic_full_design", M, np.arange(M))
