from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def tiny_problem(rng):
    """Small weighted-regression instance shared by solver tests."""
    M, D = 15, 6
    X = rng.standard_normal((M, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    beta_true = np.zeros(D)
    beta_true[[1, 4]] = [1.5, -2.0]
    y = X @ beta_true + 0.05 * rng.standard_normal(M)
    w = rng.uniform(0.1, 1.0, size=M)
    R_diag = rng.uniform(0.5, 2.0, size=M)
    return X, y, w, R_diag, beta_true
