"""Tests for the C1 sparsity-promoting design variant (design_mode=box_l1).

Additive ICASSP-companion machinery: box projection, l1 price semantics,
exact zeros, cardinality monotonicity in the price, and the fashion/pool
options of the digits adapter (pooling tested offline; the fashion download
itself is exercised only if the local snapshot or cache exists).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.methods.proposed import _gradients, _objective, make_criterion, run_proposed
from src.problems.digits_dct import _pool2d


def _tiny_instance(seed=0, D=12, M=30, N=6, snr_db=20.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M, D))
    X[M // 2:] *= 0.2      # heterogeneous rows: low-energy half should die first
    beta = np.zeros((N, D))
    for n in range(N):
        idx = rng.choice(D, size=3, replace=False)
        beta[n, idx] = rng.standard_normal(3) * 3.0
    clean = beta @ X.T
    sigma = float(np.sqrt(np.mean(clean**2) / 10 ** (snr_db / 10)))
    Y = clean + sigma * rng.standard_normal(clean.shape)
    Yv = clean + sigma * rng.standard_normal(clean.shape)
    R = sigma**2 * np.ones(M)
    return X, Y, Yv, beta, R


def _params(eta_c, iters=12):
    return {
        "gamma_schedule": [0.3, 3.0],
        "outer_iters_per_stage": iters,
        "alpha_init": 0.1,
        "backtracking": True,
        "inner": {"mode": "log", "T0": 5, "c": 3.0},
        "diag_inner_steps": 50,
        "diag_every": 6,
        "init_solve_steps": 100,
        "step_rule": "per_block",
        "gap_scale_mode": "noise_normalized",
        "eta_l1_c": eta_c,
        "n_restarts": 1,
    }


def test_objective_and_gradient_box_l1_penalty():
    X, Y, Yv, beta, R = _tiny_instance()
    M = X.shape[0]
    w = np.linspace(0.1, 0.9, M)
    B = np.zeros_like(beta)
    crit = make_criterion("ivb", X, Y)
    F_box, _, _ = _objective(X, Y, w, B, B, 0.1, 0.1, R, 1.0, 2.0, 1.0,
                             crit, design_mode="box_l1")
    F_bud, _, _ = _objective(X, Y, w, B, B, 0.1, 0.1, R, 1.0, 2.0, 1.0,
                             crit, design_mode="budget_equality")
    # Same state, different penalties: sum(w) vs sum(w(1-w)).
    assert np.isclose(F_box - F_bud,
                      2.0 * (np.sum(w) - np.sum(w * (1 - w))))
    gw_box, _ = _gradients(X, Y, w, B, B, 0.1, 0.1, R, 1.0, 2.0, 1.0,
                           crit, design_mode="box_l1")
    gw_bud, _ = _gradients(X, Y, w, B, B, 0.1, 0.1, R, 1.0, 2.0, 1.0,
                           crit, design_mode="budget_equality")
    assert np.allclose(gw_box - gw_bud, 2.0 * np.ones(M) - 2.0 * (1 - 2 * w))


def test_box_l1_stays_in_box_and_produces_exact_zeros():
    X, Y, Yv, beta, R = _tiny_instance()
    M = X.shape[0]
    res = run_proposed(X, Y, R, lam=0.05, mu=0.1, M0=M, params=_params(60.0),
                       w0=np.ones(M), criterion="ivb", Y_val=Yv,
                       design_mode="box_l1")
    w = res.w_relaxed
    assert np.all(w >= 0.0) and np.all(w <= 1.0)
    assert np.sum(w == 0.0) > 0, "large price should switch sensors exactly off"
    assert np.all(np.isfinite(w))


def test_box_l1_cardinality_decreases_with_price():
    X, Y, Yv, beta, R = _tiny_instance()
    M = X.shape[0]
    cards = []
    for eta_c in [0.1, 30.0]:
        res = run_proposed(X, Y, R, lam=0.05, mu=0.1, M0=M,
                           params=_params(eta_c), w0=np.ones(M),
                           criterion="ivb", Y_val=Yv, design_mode="box_l1")
        cards.append(int(np.sum(res.w_relaxed > 1e-3)))
    assert cards[1] < cards[0], f"cardinality should shrink with price: {cards}"


def test_budget_mode_unchanged_by_patch():
    # The budget path must still satisfy the equality budget exactly.
    X, Y, Yv, beta, R = _tiny_instance()
    M, M0 = X.shape[0], 10
    params = _params(1.0)
    params.pop("eta_l1_c")
    params["eta_c_schedule"] = [0.01]
    res = run_proposed(X, Y, R, lam=0.05, mu=0.1, M0=M0, params=params,
                       w0=np.full(M, M0 / M), criterion="ivb", Y_val=Yv)
    assert np.isclose(res.w_relaxed.sum(), M0, atol=1e-6)


def test_pool2d_average_pooling():
    imgs = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    pooled = _pool2d(imgs, 2)
    assert pooled.shape == (2, 2, 2)
    assert np.isclose(pooled[0, 0, 0], np.mean(imgs[0, :2, :2]))
    assert np.isclose(pooled[1, 1, 1], np.mean(imgs[1, 2:, 2:]))


def test_generators_provide_paired_ul_acquisition():
    """Y_train_ul: same clean signal as Y_train, independent noise of the
    same level, and the train/val/test splits are deterministic."""
    from src.problems.sparse_linear import generate_sparse_linear_blocks

    kw = dict(D=20, M=60, M0=10, N_train=32, N_val=4, N_test=4, sparsity=3,
              snr_db=20.0)
    d1 = generate_sparse_linear_blocks(np.random.default_rng(3), **kw)
    d2 = generate_sparse_linear_blocks(np.random.default_rng(3), **kw)
    assert np.array_equal(d1.Y_train, d2.Y_train)
    assert np.array_equal(d1.Y_test, d2.Y_test)
    assert d1.Y_train_ul.shape == d1.Y_train.shape
    e_ll = d1.Y_train - d1.beta_train @ d1.X.T
    e_ul = d1.Y_train_ul - d1.beta_train @ d1.X.T
    assert not np.allclose(e_ll, e_ul)
    assert abs(e_ul.std() / d1.sigma - 1.0) < 0.1
    assert abs(np.corrcoef(e_ll.ravel(), e_ul.ravel())[0, 1]) < 0.1


def test_blocks_generator_v2_row_gain_and_cluster_correlation():
    """v2 options: per-sensor gains spread over gain_range_db, rows within a
    cluster correlated ~rho, rows of different clusters ~uncorrelated; the
    default options leave the original generator untouched."""
    from src.problems.sparse_linear import generate_sparse_linear_blocks

    kw = dict(D=40, M=120, M0=20, N_train=4, N_val=4, N_test=4, sparsity=3,
              snr_db=20.0)
    base = generate_sparse_linear_blocks(np.random.default_rng(1), **kw)
    same = generate_sparse_linear_blocks(np.random.default_rng(1), **kw,
                                         row_corr="none", row_gain="none")
    assert np.array_equal(base.X, same.X) and np.array_equal(base.Y_test,
                                                             same.Y_test)
    assert np.allclose(np.linalg.norm(base.X, axis=1), 1.0)

    v2 = generate_sparse_linear_blocks(
        np.random.default_rng(1), **kw, row_corr="cluster", rho=0.7,
        cluster_size=5, row_gain="loguniform", gain_range_db=20.0)
    energy_db = 20 * np.log10(np.linalg.norm(v2.X, axis=1))
    assert energy_db.max() - energy_db.min() > 15.0       # ~20 dB spread
    assert abs(energy_db.max()) <= 10.0 + 1e-9 and abs(energy_db.min()) <= 10.0 + 1e-9
    Xn = v2.X / np.linalg.norm(v2.X, axis=1, keepdims=True)
    fam0 = Xn[:30]                                        # family 0: 6 clusters of 5
    G = fam0 @ fam0.T
    within = [G[i, j] for i in range(30) for j in range(i + 1, 30)
              if i // 5 == j // 5]
    between = [G[i, j] for i in range(30) for j in range(i + 1, 30)
               if i // 5 != j // 5]
    assert 0.5 < np.mean(within) < 0.9
    assert abs(np.mean(between)) < 0.15
    assert v2.meta["row_corr"] == "cluster" and v2.meta["row_gain"] == "loguniform"
    with pytest.raises(ValueError):
        generate_sparse_linear_blocks(np.random.default_rng(1), **kw,
                                      row_corr="ar1")


def test_run_proposed_ul_uses_paired_measurements():
    """With Y_ul the UL criterion is evaluated on Y_ul (not on the LL data),
    the design changes, and a shape mismatch is rejected."""
    from src.methods.proposed import _inner_solve_all, _phi_pred

    X, Y, Yv, beta, R = _tiny_instance()
    M = X.shape[0]
    params = _params(1.0, iters=4)
    res_train = run_proposed(X, Y, R, lam=0.05, mu=0.1, M0=M, params=params,
                             w0=np.ones(M), criterion="ivb",
                             design_mode="box_l1")
    res_pair = run_proposed(X, Y, R, lam=0.05, mu=0.1, M0=M, params=params,
                            w0=np.ones(M), criterion="ivb",
                            design_mode="box_l1", Y_ul=Yv)
    # Phi logged at the first iteration is evaluated at the initial free
    # copies B0 = beta*(w0), so it must equal the prediction risk on Y_ul.
    B0 = _inner_solve_all(X, Y, np.ones(M), 0.05, 0.1, R,
                          np.zeros((Y.shape[0], X.shape[1])),
                          params["init_solve_steps"])
    assert np.isclose(res_pair.curves[0]["Phi"], _phi_pred(X, Yv, B0))
    assert np.isclose(res_train.curves[0]["Phi"], _phi_pred(X, Y, B0))
    assert not np.allclose(res_train.w_relaxed, res_pair.w_relaxed)
    with pytest.raises(ValueError):
        run_proposed(X, Y, R, lam=0.05, mu=0.1, M0=M, params=params,
                     w0=np.ones(M), criterion="ivb", design_mode="box_l1",
                     Y_ul=Yv[:-1])


def test_config_rejects_unknown_ul_measurements():
    from pathlib import Path

    import yaml

    from src.config.loading import validate_config

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "c1_tiny_debug.yaml")
        .read_text(encoding="utf-8"))
    assert cfg["proposed"]["ul_measurements"] == "paired"
    validate_config(cfg)
    cfg["proposed"]["ul_measurements"] = "validation"
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_digits_dataset_option_default_matches_legacy():
    from src.problems.digits_dct import generate_digits_dct

    rng = np.random.default_rng(0)
    data = generate_digits_dct(rng, D=64, M=64, M0=16, N_train=8, N_val=8,
                               N_test=16, sparsity=0, snr_db=30.0)
    assert data.X.shape == (64, 64)
    assert data.meta["dataset"] == "digits"
