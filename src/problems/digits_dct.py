"""R2 problem adapter (scaffold): sklearn digits with a 2D-DCT representation.

Plan 6-R2: candidate measurements are PIXELS; to avoid the trivial X = I
setting, signals are represented in an orthonormal 2D DCT basis:
x = B theta with B the 2D IDCT. Measuring pixel i gives y_i = e_i^T B theta,
so the sensing matrix is X = B (M = D = 64 for 8x8 digits) and the deployed
estimator is the standard weighted Lasso on the DCT coefficients theta -
the ENTIRE sparse-linear pipeline (solvers, proposed methods, evaluation,
equality budget, rounding) is reused unchanged.

Notes:
- beta_dagger = exact DCT coefficients of each image: compressible, not
  sparse, so support F1 is NOT meaningful here (kept as a column, ignore).
- NMSE on theta equals image-domain NMSE by orthonormality of B.
- The seed controls the train/val/test SPLIT shuffle and the measurement
  noise; the images themselves are the fixed sklearn digits.
"""

from __future__ import annotations

import numpy as np

from src.problems.sparse_linear import SparseLinearData, compute_lambda_max


def _idct2_matrix(side: int) -> np.ndarray:
    """Orthonormal 2D IDCT synthesis matrix B: pixels = B @ dct_coeffs."""
    from scipy.fft import idctn

    D = side * side
    B = np.empty((D, D))
    for j in range(D):
        c = np.zeros((side, side))
        c[j // side, j % side] = 1.0
        B[:, j] = idctn(c, norm="ortho").ravel()
    return B


def _load_fashion_mnist() -> np.ndarray:
    """Fashion-MNIST images, (70000, 28, 28) float in [0, 1].

    Prefers the local snapshot data/raw/fashion_mnist/fashion_mnist_28x28_u8.npz
    (key "images", uint8) so runs never need the network; falls back to
    sklearn's fetch_openml (~30 MB, cached by scikit-learn) otherwise.
    """
    from pathlib import Path

    local = (Path(__file__).resolve().parents[2] / "data" / "raw"
             / "fashion_mnist" / "fashion_mnist_28x28_u8.npz")
    if local.exists():
        with np.load(local) as z:
            return z["images"].astype(float) / 255.0
    from sklearn.datasets import fetch_openml

    d = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    return d.data.astype(float).reshape(-1, 28, 28) / 255.0


def _pool2d(images: np.ndarray, pool: int) -> np.ndarray:
    """Non-overlapping ``pool``x``pool`` average pooling of square images."""
    if pool <= 1:
        return images
    n, side0, _ = images.shape
    s = side0 // pool
    return images[:, :s * pool, :s * pool].reshape(
        n, s, pool, s, pool).mean(axis=(2, 4))


def generate_digits_dct(
    rng: np.random.Generator,
    D: int, M: int, M0: int,
    N_train: int, N_val: int, N_test: int,
    sparsity: int,          # unused (real data); kept for interface parity
    snr_db: float,
    dataset: str = "digits",
    pool: int = 1,
) -> SparseLinearData:
    from scipy.fft import dctn

    if dataset == "digits":
        from sklearn.datasets import load_digits

        images = load_digits().images / 16.0      # (n, 8, 8) in [0, 1]
    elif dataset == "fashion_mnist":
        images = _load_fashion_mnist()            # (n, 28, 28) in [0, 1]
    else:
        raise ValueError(f"unknown dataset '{dataset}'")
    images = _pool2d(images, int(pool))
    side = images.shape[1]
    if side * side != D or M != D:
        raise ValueError(
            f"digits_dct requires D = M = side^2 = {side * side} for "
            f"dataset '{dataset}' with pool {pool}"
        )
    n_total = N_train + N_val + N_test
    if n_total > len(images):
        raise ValueError(f"requested {n_total} instances, digits has "
                         f"{len(images)}")
    idx = rng.permutation(len(images))[:n_total]
    theta = np.stack([dctn(images[i], norm="ortho").ravel() for i in idx])

    X = _idct2_matrix(side)                        # (M, D), orthonormal
    beta_train = theta[:N_train]
    beta_val = theta[N_train:N_train + N_val]
    beta_test = theta[N_train + N_val:]

    clean = {"train": beta_train @ X.T, "val": beta_val @ X.T,
             "test": beta_test @ X.T}
    signal_power = float(np.mean(
        np.concatenate([c.ravel() for c in clean.values()]) ** 2))
    sigma = float(np.sqrt(signal_power / 10.0 ** (snr_db / 10.0)))
    noisy = {k: c + sigma * rng.standard_normal(c.shape)
             for k, c in clean.items()}
    R_diag = sigma**2 * np.ones(M)
    lambda_max = compute_lambda_max(X, noisy["train"], R_diag, M0)
    return SparseLinearData(
        X=X, beta_train=beta_train, beta_val=beta_val, beta_test=beta_test,
        Y_train=noisy["train"], Y_val=noisy["val"], Y_test=noisy["test"],
        R_diag=R_diag, sigma=sigma, lambda_max=lambda_max,
        meta={"generator": "digits_dct", "dataset": dataset,
              "pool": int(pool), "D": D, "M": M, "M0": M0,
              "N_train": N_train, "N_val": N_val, "N_test": N_test,
              "snr_db": snr_db, "signal_power": signal_power,
              "support_F1_meaningful": False},
    )
