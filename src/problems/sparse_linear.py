"""Sparse linear synthetic problem generator.

E0 variant (approved decision, 2026-07-04): iid Gaussian sensing rows with
row normalization and k-sparse ground-truth vectors. This is deliberately
simpler than the structured block generator that E1 requires (plan section
E1); E1 must NOT reuse this generator.

Noise convention (documented per implementation notes 4.3):
    SNR_dB = 10 * log10(signal_power / sigma^2),
with signal_power the mean of (X beta_dagger)^2 over all M candidate rows and
all generated instances (train + val + test). R = sigma^2 * I is passed
explicitly as R_diag (binding convention 0.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SparseLinearData:
    X: np.ndarray                 # (M, D) candidate sensing rows, unit norm
    beta_train: np.ndarray        # (N_train, D) ground truth
    beta_val: np.ndarray          # (N_val, D)
    beta_test: np.ndarray         # (N_test, D)
    Y_train: np.ndarray           # (N_train, M) noisy measurements
    Y_val: np.ndarray             # (N_val, M)
    Y_test: np.ndarray            # (N_test, M)
    R_diag: np.ndarray            # (M,) noise variances, sigma^2 * ones
    sigma: float
    lambda_max: float             # training-only, uniform feasible design
    meta: dict = field(default_factory=dict)
    # Second, independent noisy acquisition of the TRAINING scenes (same X,
    # same beta_train, fresh noise). It is the validation measurement
    # y_{v,n} of the paper's UL criterion (ICASSP eq. 4): the LL sees
    # Y_train, the UL scores the free copies on Y_train_ul. Drawn AFTER the
    # train/val/test noise so those splits are unchanged (2026-09-18).
    Y_train_ul: np.ndarray | None = None   # (N_train, M)


def _sparse_signals(rng: np.random.Generator, n: int, D: int, s: int) -> np.ndarray:
    """k-sparse signals: random support, amplitudes +/- U[1, 2]."""
    B = np.zeros((n, D))
    for i in range(n):
        support = rng.choice(D, size=s, replace=False)
        signs = rng.choice([-1.0, 1.0], size=s)
        B[i, support] = signs * rng.uniform(1.0, 2.0, size=s)
    return B


def compute_lambda_max(
    X: np.ndarray, Y_train: np.ndarray, R_diag: np.ndarray, M0: int
) -> float:
    """Binding convention 0.4: training only, uniform feasible design, with R.

    lambda_max_n = || X^T D_{w_u} R^{-1} y_n ||_inf,  w_u = (M0/M) * ones,
    lambda_max   = median over training instances n.
    """
    M = X.shape[0]
    w_u = (M0 / M) * np.ones(M)
    coef = w_u / R_diag                                   # D_{w_u} R^{-1} diagonal
    per_instance = np.max(np.abs(Y_train * coef @ X), axis=1)
    return float(np.median(per_instance))


def _finalize(rng, X, beta_train, beta_val, beta_test, M0, snr_db, meta):
    """Shared noise/SNR/lambda_max assembly (documented convention above)."""
    clean = {
        "train": beta_train @ X.T,
        "val": beta_val @ X.T,
        "test": beta_test @ X.T,
    }
    signal_power = float(
        np.mean(np.concatenate([c.ravel() for c in clean.values()]) ** 2)
    )
    sigma = float(np.sqrt(signal_power / 10.0 ** (snr_db / 10.0)))
    noisy = {
        split: c + sigma * rng.standard_normal(c.shape)
        for split, c in clean.items()
    }
    R_diag = sigma**2 * np.ones(X.shape[0])
    lambda_max = compute_lambda_max(X, noisy["train"], R_diag, M0)
    return SparseLinearData(
        X=X,
        beta_train=beta_train, beta_val=beta_val, beta_test=beta_test,
        Y_train=noisy["train"], Y_val=noisy["val"], Y_test=noisy["test"],
        R_diag=R_diag, sigma=sigma, lambda_max=lambda_max,
        meta={**meta, "signal_power": signal_power},
    )


def generate_sparse_linear(
    rng: np.random.Generator,
    D: int,
    M: int,
    M0: int,
    N_train: int,
    N_val: int,
    N_test: int,
    sparsity: int,
    snr_db: float,
) -> SparseLinearData:
    X = rng.standard_normal((M, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    beta_train = _sparse_signals(rng, N_train, D, sparsity)
    beta_val = _sparse_signals(rng, N_val, D, sparsity)
    beta_test = _sparse_signals(rng, N_test, D, sparsity)

    meta = {
        "generator": "sparse_linear_iid_gaussian_e0",
        "D": D, "M": M, "M0": M0,
        "N_train": N_train, "N_val": N_val, "N_test": N_test,
        "sparsity": sparsity, "snr_db": snr_db,
    }
    return _finalize(rng, X, beta_train, beta_val, beta_test, M0, snr_db, meta)


# ----------------------------------------------------------------------------
# E1 structured block generator (plan section 6-E1)
# ----------------------------------------------------------------------------

def _blocks(D: int, n_blocks: int) -> list[np.ndarray]:
    """Split coordinates 0..D-1 into n_blocks contiguous blocks."""
    return [np.asarray(b) for b in np.array_split(np.arange(D), n_blocks)]


def _structured_support_signals(
    rng: np.random.Generator, n: int, D: int, s: int,
    frequent_coords: np.ndarray, frac_frequent: float,
) -> np.ndarray:
    """Sparse signals with structured support (plan E1): ~frac_frequent of the
    active coordinates in the frequent blocks, the rest uniform over the whole
    domain. Amplitudes +/- U[1, 2]."""
    B = np.zeros((n, D))
    k_freq = int(round(frac_frequent * s))
    for i in range(n):
        support = list(rng.choice(frequent_coords, size=k_freq, replace=False))
        remaining = np.setdiff1d(np.arange(D), support)
        support += list(rng.choice(remaining, size=s - k_freq, replace=False))
        support = np.asarray(support)
        signs = rng.choice([-1.0, 1.0], size=s)
        B[i, support] = signs * rng.uniform(1.0, 2.0, size=s)
    return B


def generate_sparse_linear_blocks(
    rng: np.random.Generator,
    D: int,
    M: int,
    M0: int,
    N_train: int,
    N_val: int,
    N_test: int,
    sparsity: int,
    snr_db: float,
    n_blocks: int = 4,
    block_gain: float = 5.0,
    frequent_blocks: tuple[int, ...] = (0, 1),
    frac_frequent: float = 0.7,
) -> SparseLinearData:
    """E1 generator (plan 6-E1): 4 coordinate blocks; M rows split into
    n_blocks sensor families, family f drawn N(0, Sigma_f) with std 1 on its
    own block and 1/block_gain elsewhere, then row-normalized. Signals have
    structured support: ~70% of actives in the two frequent blocks.
    """
    blocks = _blocks(D, n_blocks)
    rows_per_family = np.array_split(np.arange(M), n_blocks)
    X = np.empty((M, D))
    for f, rows in enumerate(rows_per_family):
        std = np.full(D, 1.0 / block_gain)
        std[blocks[f]] = 1.0
        X[rows] = rng.standard_normal((len(rows), D)) * std
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    frequent_coords = np.concatenate([blocks[b] for b in frequent_blocks])
    beta_train = _structured_support_signals(
        rng, N_train, D, sparsity, frequent_coords, frac_frequent)
    beta_val = _structured_support_signals(
        rng, N_val, D, sparsity, frequent_coords, frac_frequent)
    beta_test = _structured_support_signals(
        rng, N_test, D, sparsity, frequent_coords, frac_frequent)

    meta = {
        "generator": "sparse_linear_structured_blocks_e1",
        "D": D, "M": M, "M0": M0,
        "N_train": N_train, "N_val": N_val, "N_test": N_test,
        "sparsity": sparsity, "snr_db": snr_db,
        "n_blocks": n_blocks, "block_gain": block_gain,
        "frequent_blocks": list(frequent_blocks),
        "frac_frequent": frac_frequent,
    }
    return _finalize(rng, X, beta_train, beta_val, beta_test, M0, snr_db, meta)


# ----------------------------------------------------------------------------
# E2 anisotropic generator (plan 6-E2 scaffold)
# ----------------------------------------------------------------------------

def generate_sparse_linear_e2_anisotropic(
    rng: np.random.Generator,
    D: int, M: int, M0: int,
    N_train: int, N_val: int, N_test: int,
    sparsity: int, snr_db: float,
    n_blocks: int = 4,
    block_gain: float = 5.0,
    freq_group_size: int = 3,
    rare_group_size: int = 3,
    p_freq: float = 0.8,
    p_rare: float = 0.2,
    amp_moderate: tuple[float, float] = (1.0, 2.0),
    amp_large: tuple[float, float] = (3.0, 6.0),
) -> SparseLinearData:
    """E2 variant (plan 6-E2): sensor families as in E1, but anisotropic
    signals designed to induce dominant error modes: a FREQUENT coordinate
    group appears in ~p_freq of the signals with moderate amplitudes, a RARE
    group in ~p_rare with large amplitudes; remaining active coordinates are
    uniform with moderate amplitudes. Simplifications vs the loosely
    specified plan text are recorded in the implementation notes."""
    blocks = _blocks(D, n_blocks)
    rows_per_family = np.array_split(np.arange(M), n_blocks)
    X = np.empty((M, D))
    for f, rows in enumerate(rows_per_family):
        std = np.full(D, 1.0 / block_gain)
        std[blocks[f]] = 1.0
        X[rows] = rng.standard_normal((len(rows), D)) * std
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    freq_group = blocks[0][:freq_group_size]
    rare_group = blocks[-1][:rare_group_size]

    def _signals(n: int) -> np.ndarray:
        B = np.zeros((n, D))
        for i in range(n):
            active: list[int] = []
            if rng.random() < p_freq:
                B[i, freq_group] = (rng.choice([-1.0, 1.0], len(freq_group))
                                    * rng.uniform(*amp_moderate, len(freq_group)))
                active += list(freq_group)
            if rng.random() < p_rare:
                B[i, rare_group] = (rng.choice([-1.0, 1.0], len(rare_group))
                                    * rng.uniform(*amp_large, len(rare_group)))
                active += list(rare_group)
            n_extra = max(0, sparsity - len(active))
            if n_extra:
                # Random extras never fall in the two groups, so group
                # presence probabilities are exactly p_freq / p_rare.
                remaining = np.setdiff1d(
                    np.arange(D), np.concatenate([freq_group, rare_group]))
                extra = rng.choice(remaining, size=n_extra, replace=False)
                B[i, extra] = (rng.choice([-1.0, 1.0], n_extra)
                               * rng.uniform(*amp_moderate, n_extra))
        return B

    meta = {
        "generator": "sparse_linear_e2_anisotropic",
        "D": D, "M": M, "M0": M0,
        "N_train": N_train, "N_val": N_val, "N_test": N_test,
        "sparsity": sparsity, "snr_db": snr_db,
        "n_blocks": n_blocks, "block_gain": block_gain,
        "freq_group": [int(j) for j in freq_group],
        "rare_group": [int(j) for j in rare_group],
        "p_freq": p_freq, "p_rare": p_rare,
    }
    return _finalize(rng, X, _signals(N_train), _signals(N_val),
                     _signals(N_test), M0, snr_db, meta)


def _generate_digits_dct(rng, **kwargs):
    from src.problems.digits_dct import generate_digits_dct

    return generate_digits_dct(rng, **kwargs)


def _generate_tv_1d(rng, **kwargs):
    from src.problems.tv_1d import generate_tv_1d

    return generate_tv_1d(rng, **kwargs)


def _generate_traffic_pca(rng, **kwargs):
    from src.problems.traffic import generate_traffic_pca

    return generate_traffic_pca(rng, **kwargs)


GENERATORS = {
    "iid_e0": generate_sparse_linear,
    "structured_blocks": generate_sparse_linear_blocks,
    "e2_anisotropic": generate_sparse_linear_e2_anisotropic,
    "digits_dct": _generate_digits_dct,
    "tv_1d": _generate_tv_1d,
    "traffic_pca": _generate_traffic_pca,
}


def make_problem_data(rng: np.random.Generator, problem: dict) -> SparseLinearData:
    """Factory: build the dataset described by a config's ``problem`` block."""
    kind = problem.get("generator", "iid_e0")
    if kind not in GENERATORS:
        raise ValueError(f"unknown generator '{kind}'; known: {sorted(GENERATORS)}")
    kwargs = {k: problem[k] for k in
              ["D", "M", "M0", "N_train", "N_val", "N_test", "sparsity", "snr_db"]}
    kwargs.update(problem.get("generator_options", {}))
    return GENERATORS[kind](rng, **kwargs)
