"""Proposed method: single-loop value-function penalty proximal-gradient.

Milestone 2 scope: criterion IV-B (validation prediction risk) for the sparse
linear Lasso/elastic-net lower level. The penalized single-level objective is
(losses normalized by the number N of stacked LL instances, plan 4.5):

    F_gamma(w, B) = Phi_pred(B) + eta * Obin(w)
                    + (gamma / N) * sum_n [ g_n(w, B_n) - v_n(w) ]

with, for each LL instance n (training instances; see the E0 adaptation
decision in docs/implementation_notes.md):

    g_n(w, b) = 0.5 ||D_w^{1/2} R^{-1/2} (y_n - X b)||^2 + 0.5 mu ||b||^2
                + lam ||b||_1
    Phi_pred(B) = (1/N) sum_n 0.5 || X B_n - y_n ||^2      (all M rows)

The value-function gradient uses the residual formula (paper eq. 12):

    [grad v_n(w)]_i = (y_{n,i} - x_i^T beta_n^*(w))^2 / (2 R_ii),

with beta_n^*(w) replaced by an inexact inner FISTA estimate (T_k steps, warm
started). The nonsmooth outer prox is the capped-simplex projection for w and
entrywise soft-thresholding (threshold alpha * gamma * lam / N) for B.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.evaluation.metrics import binarity_metrics, median_iqr, nmse
from src.solvers.lasso import soft_threshold
from src.utils.projection import check_feasible, project_capped_simplex_equality


@dataclass
class IVBResult:
    w_relaxed: np.ndarray
    w_stages: list            # (stage_label, w copy) per continuation stage
    beta_hat: np.ndarray      # final inner estimates of beta*_n(w), (N, D)
    curves: list              # per-outer-iteration log rows
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Lower-level pieces (design-weighted elastic net for one w, all N instances)
# ----------------------------------------------------------------------------

def _ll_matrices(X: np.ndarray, w: np.ndarray, R_diag: np.ndarray):
    """H_w = X^T D_w R^{-1} X (without mu) and its largest eigenvalue."""
    d = w / R_diag
    H = X.T @ (d[:, None] * X)
    L = float(np.linalg.eigvalsh(H)[-1])
    return d, H, L


def _make_regularizer(name: str) -> dict:
    """Nonsmooth LL regularizer strategy (design-independent, paper eq. 7).

    - "l1" (default): exact historical Lasso path - prox is
      ``soft_threshold`` with the identical threshold arguments.
    - "tv": 1D total variation; exact per-row Condat prox (E6).
    Both expose prox(V, t) accepting (D,) or (N, D), and penalty(B)
    returning the per-instance nonsmooth value at lam = 1.
    """
    if name == "l1":
        return {
            "name": "l1",
            "prox": soft_threshold,
            "penalty": lambda B: np.abs(B).sum(axis=-1),
        }
    if name == "tv":
        from src.solvers.tv import tv_prox

        def prox(V, t):
            V2 = np.atleast_2d(V)
            out = np.stack([tv_prox(V2[n], t) for n in range(V2.shape[0])])
            return out[0] if np.ndim(V) == 1 else out

        return {
            "name": "tv",
            "prox": prox,
            "penalty": lambda B: np.abs(np.diff(np.atleast_2d(B),
                                                axis=-1)).sum(axis=-1),
        }
    raise ValueError(f"unknown regularizer '{name}'")


_L1 = None  # created lazily; module-level default strategy


def _default_reg() -> dict:
    global _L1
    if _L1 is None:
        _L1 = _make_regularizer("l1")
    return _L1


def _fista(H, XtDy, mu, lam, beta0, n_steps, L, reg=None):
    """n_steps FISTA iterations on 0.5 b^T (H+muI) b - XtDy^T b + lam*g_n(b).

    Vectorized over instances: ``XtDy`` and ``beta0`` may be (D,) or (N, D);
    the N problems share H (same design w), so one matrix iteration solves all.
    """
    prox = (reg or _default_reg())["prox"]
    step = 1.0 / (L + mu)
    beta = np.array(beta0, dtype=float, copy=True)
    z = beta.copy()
    t = 1.0
    for _ in range(n_steps):
        grad = z @ H + mu * z - XtDy         # H symmetric
        beta_new = prox(z - step * grad, step * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_new + ((t - 1.0) / t_new) * (beta_new - beta)
        beta, t = beta_new, t_new
    return beta


def _inner_solve_all(X, Y, w, lam, mu, R_diag, B0, n_steps, return_L=False,
                     reg=None):
    """Warm-started inner solves for all N instances at design w."""
    d, H, L = _ll_matrices(X, w, R_diag)
    XtDy = (Y * d) @ X                      # (N, D) rows: X^T D_w R^{-1} y_n
    B = _fista(H, XtDy, mu, lam, B0, n_steps, L, reg=reg)
    return (B, L) if return_L else B


def _g_values(X, Y, w, lam, mu, R_diag, B, reg=None):
    """g_n(w, B_n) for all n. Residuals shape (N, M)."""
    resid = Y - B @ X.T
    d = w / R_diag
    smooth = 0.5 * (resid**2 @ d) + 0.5 * mu * np.sum(B**2, axis=1)
    return smooth + lam * (reg or _default_reg())["penalty"](B)


def _phi_pred(X, Y, B):
    """IV-B prediction risk over the full M rows, normalized by N."""
    resid = B @ X.T - Y
    return float(0.5 * np.sum(resid**2) / B.shape[0])


def make_criterion(name: str, X: np.ndarray, Y: np.ndarray,
                   beta_dagger: np.ndarray | None = None,
                   rho: float = 0.0,
                   tau: float = 50.0) -> dict:
    """UL criterion bundle: value(B), grad_B(B), Lipschitz constant of the
    B-gradient, and a label.

    - "ivb" (IV-B): validation prediction risk over the full M rows; no
      ground truth needed.
    - "ivc_trace" (IV-C-A): empirical error-covariance trace
      tr(C_hat) = (1/N) sum_n ||B_n - beta_dagger_n||^2 + rho * D, requiring
      the ground truth of the LL (training) instances; spectral gradient
      G = I, so grad_{B_n} = (2/N) (B_n - beta_dagger_n) (paper eq. 21d).
    """
    N = Y.shape[0]
    if name == "ivb":
        return {
            "label": "IV-B",
            "value": lambda B: _phi_pred(X, Y, B),
            "grad_B": lambda B: ((B @ X.T - Y) @ X) / N,
            "L_B": float(np.linalg.eigvalsh(X.T @ X)[-1]) / N,
        }
    if name == "ivc_trace":
        if beta_dagger is None:
            raise ValueError("ivc_trace requires beta_dagger of the LL instances")
        D = beta_dagger.shape[1]
        return {
            "label": "IV-C-A",
            "value": lambda B: float(np.sum((B - beta_dagger) ** 2) / N
                                     + rho * D),
            "grad_B": lambda B: (2.0 / N) * (B - beta_dagger),
            "L_B": 2.0 / N,
        }
    if name in ("ivc_logdet", "ivc_maxeig"):
        # Spectral error-covariance criteria (paper eq. 22): both have
        # grad_B = (2/N) E G with E = B - beta_dagger and a spectral matrix
        # G evaluated at C_hat = E^T E / N + rho I. Nonconvex in B; the
        # rho I floor makes gradients Lipschitz on the bounded region, and
        # the B-step uses a LOCAL curvature bound recomputed each iteration
        # (L_B is a callable; conservative Frobenius-norm bound).
        if beta_dagger is None:
            raise ValueError(f"{name} requires beta_dagger of the LL instances")
        D = beta_dagger.shape[1]
        if rho <= 0:
            raise ValueError(f"{name} requires a positive spectral rho")

        def _C(B):
            E = B - beta_dagger
            return E, (E.T @ E) / N + rho * np.eye(D)

        if name == "ivc_logdet":
            def value(B):
                _, C = _C(B)
                return float(np.linalg.slogdet(C)[1])

            def grad_B(B):
                E, C = _C(B)
                return (2.0 / N) * E @ np.linalg.inv(C)

            def L_B(B):
                E, _ = _C(B)
                e2 = float(np.sum(E**2))                 # >= ||E||_op^2
                return 2.0 / (N * rho) + 4.0 * e2 / (N * N * rho * rho)

            return {"label": "IV-C-D", "value": value, "grad_B": grad_B,
                    "L_B": L_B}

        def value(B):
            _, C = _C(B)
            nu = np.linalg.eigvalsh(C)
            m = nu[-1]
            return float(m + np.log(np.sum(np.exp(tau * (nu - m)))) / tau)

        def grad_B(B):
            E, C = _C(B)
            nu, U = np.linalg.eigh(C)
            wgt = np.exp(tau * (nu - nu[-1]))
            wgt /= wgt.sum()                              # softmax, trace 1
            G = (U * wgt) @ U.T
            return (2.0 / N) * E @ G

        def L_B(B):
            E, _ = _C(B)
            e2 = float(np.sum(E**2))
            return 2.0 / N + 8.0 * tau * e2 / (N * N)

        return {"label": "IV-C-E", "value": value, "grad_B": grad_B,
                "L_B": L_B}
    raise ValueError(f"unknown criterion '{name}'")


def value_function_gradient(X, y, beta_star, R_diag):
    """Residual formula for grad v(w) of one instance (paper eq. 12)."""
    resid = y - X @ beta_star
    return resid**2 / (2.0 * R_diag)


def lower_level_value(X, y, w, lam, mu, R_diag, n_steps=3000, beta0=None,
                      regularizer="l1"):
    """Over-solved v(w) = min_b g(w, b) for one instance (diagnostics/tests)."""
    reg = _make_regularizer(regularizer)
    d, H, L = _ll_matrices(X, w, R_diag)
    Xty = X.T @ (d * y)
    b0 = np.zeros(X.shape[1]) if beta0 is None else beta0
    beta = _fista(H, Xty, mu, lam, b0, n_steps, L, reg=reg)
    return _g_values(X, y[None, :], w, lam, mu, R_diag, beta[None, :],
                     reg=reg)[0], beta


# ----------------------------------------------------------------------------
# Penalized objective and gradients
# ----------------------------------------------------------------------------

def gap_scale_from_R(R_diag: np.ndarray) -> float:
    """Loss normalization of the penalty gap term (plan 4.5).

    The LL losses g_n carry the R^{-1} whitening, so their natural scale is
    1/sigma^2 while Phi_pred is O(1); without normalization the same gamma
    grid means different penalty strengths at different noise levels. We
    scale the gap term by 1/mean(R^{-1}) (= sigma^2 for homoscedastic noise),
    which is exactly a reparametrization gamma' = gamma * sigma^2 of the
    paper's objective. Recorded in docs/implementation_notes.md.
    """
    return float(1.0 / np.mean(1.0 / R_diag))


def _objective(X, Y, w, B, beta_hat, lam, mu, R_diag, gamma, eta, gap_scale,
               crit, reg=None, design_mode="budget_equality",
               eta_bin=0.0, price_kind="l1", price_theta=0.1):
    """Approximate F_gamma using v_n(w) ~= g_n(w, beta_hat_n).

    ``design_mode``: "budget_equality" (default; eta multiplies the binarity
    penalty Obin) or "box_l1" (ICASSP sparsity variant; eta is the l1 price
    and multiplies sum(w), linear on the box).
    """
    N = B.shape[0]
    phi = crit["value"](B)
    if design_mode == "box_l1":
        # Sparsity price: linear ("l1") or concave ("concave",
        # sum w/(w+theta), reweighted-l1 style: cheap for large weights,
        # steep near zero, so it selects instead of shrinking uniformly).
        # Plus an optional late binarity term anchoring survivors at 1.
        if price_kind == "concave":
            obin = float(np.sum(w / (w + price_theta)))
        else:
            obin = float(np.sum(w))
        pen_extra = eta_bin * float(np.sum(w * (1.0 - w)))
    else:
        obin = float(np.sum(w * (1.0 - w)))
        pen_extra = 0.0
    gap = gap_scale * float(
        np.sum(_g_values(X, Y, w, lam, mu, R_diag, B, reg=reg)
               - _g_values(X, Y, w, lam, mu, R_diag, beta_hat, reg=reg))
    ) / N
    return phi + eta * obin + pen_extra + gamma * gap, phi, gap


def _gradients(X, Y, w, B, beta_hat, lam, mu, R_diag, gamma, eta, gap_scale,
               crit, design_mode="budget_equality", eta_bin=0.0,
               price_kind="l1", price_theta=0.1):
    """Smooth gradients of F_gamma w.r.t. w and B (lam-term goes to the prox)."""
    N, _ = B.shape
    c = gamma * gap_scale / N
    resid_B = Y - B @ X.T                    # (N, M)
    resid_hat = Y - beta_hat @ X.T
    # w-gradient: eta*(1-2w) + c * sum_n (l_ni(B) - l_ni(beta_hat)) / R_ii
    # (the UL criteria used here have no explicit w-dependence).
    if design_mode == "box_l1":
        if price_kind == "concave":
            price_grad = eta * price_theta / (w + price_theta) ** 2
        else:
            price_grad = eta * np.ones_like(w)
        pen_grad = price_grad + eta_bin * (1.0 - 2.0 * w)
    else:
        pen_grad = eta * (1.0 - 2.0 * w)
    grad_w = pen_grad + c * (
        np.sum(resid_B**2 - resid_hat**2, axis=0) / (2.0 * R_diag)
    )
    # B-gradient: criterion term + gap smooth term.
    d = w / R_diag
    grad_B = crit["grad_B"](B) + c * ((-(resid_B * d) @ X) + mu * B)
    return grad_w, grad_B


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def run_proposed(
    X: np.ndarray,
    Y: np.ndarray,                    # (N, M) LL instance measurements
    R_diag: np.ndarray,
    lam: float,
    mu: float,
    M0: int,
    params: dict,
    w0: np.ndarray,
    criterion: str = "ivb",
    beta_dagger: np.ndarray | None = None,   # LL-instance ground truth
    rho: float = 0.0,                         # spectral reg. of C_hat (IV-C)
    Y_val: np.ndarray | None = None,          # val instances for monitoring
    beta_dagger_val: np.ndarray | None = None,
    logger=None,
    deadline: float | None = None,
    regularizer: str = "l1",
    design_mode: str = "budget_equality",
) -> IVBResult:
    """Single-loop value-function penalty method (criteria: IV-B, IV-C-A,
    IV-C-D, IV-C-E; LL regularizer: l1 default, tv for E6)."""
    t0 = time.perf_counter()
    N, M = Y.shape
    crit = make_criterion(criterion, X, Y, beta_dagger=beta_dagger, rho=rho,
                          tau=float(params.get("tau_maxeig", 50.0)))
    reg = _make_regularizer(regularizer)
    if design_mode not in ("budget_equality", "box_l1"):
        raise ValueError(f"unsupported design_mode '{design_mode}'")
    box_l1 = design_mode == "box_l1"
    price_kind = str(params.get("price_kind", "l1"))
    price_theta = float(params.get("price_theta", 0.1))
    if box_l1:
        def _proj(v):
            return np.clip(v, 0.0, 1.0)
    else:
        def _proj(v):
            return project_capped_simplex_equality(v, M0)

    gamma_schedule = list(params["gamma_schedule"])
    iters_per_stage = int(params["outer_iters_per_stage"])
    alpha_w = float(params["alpha_init"])       # design-block step (backtracked)
    alpha_w_max = alpha_w
    backtracking = bool(params.get("backtracking", True))
    max_halvings = int(params.get("max_halvings", 30))
    inner = params.get("inner", {"mode": "log", "T0": 5, "c": 3.0})
    diag_steps = int(params.get("diag_inner_steps", 500))
    diag_every = int(params.get("diag_every", 5))
    init_solve_steps = int(params.get("init_solve_steps", 1000))

    w = _proj(np.asarray(w0, dtype=float))
    # beta^0 = beta*(w^0) via a reasonably precise LL solve (plan 4.9).
    beta_hat, L_H = _inner_solve_all(
        X, Y, w, lam, mu, R_diag, np.zeros((N, X.shape[1])), init_solve_steps,
        return_L=True, reg=reg,
    )
    B = beta_hat.copy()
    beta_val_hat = None
    if Y_val is not None:
        beta_val_hat = np.zeros((Y_val.shape[0], X.shape[1]))
    # Approved implementation adaptations (2026-07-05): recorded explicitly in
    # the config so every run documents them; unsupported values fail fast.
    step_rule = params.get("step_rule", "per_block")
    if step_rule != "per_block":
        raise ValueError(
            f"unsupported step_rule '{step_rule}' (only 'per_block' is implemented)"
        )
    gap_mode = params.get("gap_scale_mode", "noise_normalized")
    if gap_mode == "noise_normalized":
        gap_scale = gap_scale_from_R(R_diag)
    elif gap_mode == "none":                 # paper-exact scaling, for ablation
        gap_scale = 1.0
    else:
        raise ValueError(f"unsupported gap_scale_mode '{gap_mode}'")

    # Curvature of the criterion term w.r.t. the free copies B. Constant for
    # IV-B / IV-C-A; a callable local bound for the spectral criteria.
    L_phi_spec = crit["L_B"]

    phi0 = crit["value"](B)
    # Stage tuples: (kind, gamma, eta, n_iters).
    stages = [("gamma", g, 0.0, iters_per_stage) for g in gamma_schedule]
    gamma_final = gamma_schedule[-1]
    # Late binarity continuation (plan 4.6): either absolute eta values
    # ("eta_schedule", normalized-losses option) or scaled by |Phi0|/M
    # ("eta_c_schedule", phase-B option).
    if "eta_schedule" in params:
        stages += [("eta", gamma_final, float(e), iters_per_stage)
                   for e in params["eta_schedule"]]
    else:
        stages += [
            ("eta", gamma_final, c * abs(phi0) / M, iters_per_stage)
            for c in params.get("eta_c_schedule", [])
        ]
    # Optional polish phase: after binarization, boost gamma at fixed eta so
    # the free copies re-converge to beta*(w) and the endpoint LL-gap /
    # tracking diagnostics reflect a bilevel-feasible point (see notes).
    if stages:
        eta_final_val = stages[-1][2]
        stages += [
            ("polish", gamma_final * float(f), eta_final_val, iters_per_stage)
            for f in params.get("polish_gamma_factors", [])
        ]
    # Optional short eta re-pin after polish: re-tighten binarity at the
    # final (polished) gamma to shrink the rounding gap that polish opens.
    if params.get("post_polish_eta_repin", False):
        stages.append((
            "repin",
            stages[-1][1],
            float(params.get("post_polish_eta_value", 1.0)),
            int(params.get("post_polish_iters", 25)),
        ))

    if box_l1:
        # ICASSP sparsity variant: no binarity/repin stages. The l1 price
        # replaces eta; "eta_l1" is absolute, "eta_l1_c" scales |Phi0|/M
        # (mirroring the eta_c_schedule convention). With
        # l1_scale_with_gamma (default), the price grows proportionally to
        # gamma along the continuation so the price/value-pressure ratio of
        # the keep-or-discard rule stays constant across stages.
        if "eta_l1" in params:
            eta_price = float(params["eta_l1"])
        elif "eta_l1_c" in params:
            # Anchor the price to the initial residual-sensitivity scale of
            # the value-function coupling (the value side of the
            # keep-or-discard rule), so that eta_l1_c ~ O(1) prices a sensor
            # at the typical per-sensor coupling magnitude at gamma_final
            # and the same grid transfers across problems and noise levels.
            coup0 = gap_scale / N * np.sum(
                (Y - beta_hat @ X.T) ** 2 / (2.0 * R_diag), axis=0)
            eta_price = (float(params["eta_l1_c"]) * gamma_final
                         * float(np.median(coup0)))
        else:
            raise ValueError(
                "design_mode='box_l1' requires params['eta_l1'] or "
                "params['eta_l1_c']"
            )
        # Price homotopy (decision 2026-08-31, ICASSP variant): the gamma
        # continuation runs PRICE-FREE so the value-function coupling is
        # formed on a rich, well-fit design; the price is then ramped in at
        # gamma_final, pruning sensors while the reconstructions (and hence
        # the keep-or-discard signals) remain meaningful. Ramping the price
        # during the gamma continuation instead (the first implementation)
        # crushes all weights uniformly before the selection signal exists
        # and picks poor supports at low cardinality.
        price_ramp = list(params.get("price_ramp", [0.1, 0.3, 1.0]))

        price_iters = int(params.get("price_iters_per_stage",
                                     iters_per_stage))
        stages = [("gamma", g, 0.0, iters_per_stage)
                  for g in gamma_schedule]
        stages += [("price", gamma_final, eta_price * float(r),
                    price_iters) for r in price_ramp]
        stage_eta_bin = [0.0] * len(stages)
        # Late binarity continuation (optional): pushes surviving weights
        # toward 1 so the deployed binary support matches the design the
        # optimizer actually scored (same |Phi0|/M convention as the
        # budget-mode eta_c_schedule).
        for c_bin in params.get("eta_bin_c_schedule", []):
            stages.append(("binz", gamma_final, eta_price, iters_per_stage))
            stage_eta_bin.append(float(c_bin) * abs(phi0) / M)
        eta_bin_final = stage_eta_bin[-1]
        for f in params.get("polish_gamma_factors", []):
            stages.append(
                ("polish", gamma_final * float(f), eta_price,
                 iters_per_stage))
            stage_eta_bin.append(eta_bin_final)
    else:
        stage_eta_bin = [0.0] * len(stages)

    curves: list[dict] = []
    w_stages: list[tuple[str, np.ndarray]] = []
    total_inner = N * init_solve_steps
    k_global = 0
    val_nmse_last = np.nan
    val_pred_last = np.nan

    def _T_k(k_stage: int) -> int:
        if inner.get("mode") == "fixed":
            return int(inner["T"])
        return int(np.ceil(inner["T0"] + inner["c"] * np.log(k_stage + 1)))

    # Stage agenda. For box_l1 price stages an adaptive bisection guards
    # against pruning avalanches: if one price step removes more than
    # avalanche_frac of the active sensors (or empties the design), the
    # stage is rolled back and re-run at the geometric mean of the previous
    # and current prices, so the homotopy resolves the informative segment
    # of the cardinality frontier instead of jumping across it.
    agenda = [(kind, gamma, eta, n_iters, stage_eta_bin[i])
              for i, (kind, gamma, eta, n_iters) in enumerate(stages)]
    agenda.reverse()
    stage_idx = -1
    prev_price = 0.0
    gamma_last, eta_last = stages[-1][1], stages[-1][2]
    avalanche_frac = float(params.get("avalanche_frac", 0.4))
    bisect_budget = int(params.get("max_price_bisect", 12))

    def _n_active(v):
        return int(np.sum(v > 1e-3))

    while agenda:
        kind, gamma, eta, n_iters, eta_bin = agenda.pop()
        stage_idx += 1
        stage_label = f"{stage_idx}_{kind}_g{gamma:g}_e{eta:g}"
        is_price = box_l1 and kind == "price"
        if is_price:
            snap = (w.copy(), B.copy(), beta_hat.copy(), alpha_w)
            act_before = _n_active(w)
        for k_stage in range(1, n_iters + 1):
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError(
                    f"proposed IV-B exceeded timeout at stage {stage_label}, "
                    f"iteration {k_stage}"
                )
            k_global += 1
            T_k = _T_k(k_stage)
            beta_hat, L_H = _inner_solve_all(
                X, Y, w, lam, mu, R_diag, beta_hat, T_k, return_L=True, reg=reg
            )
            total_inner += N * T_k

            F_curr, phi_curr, gap_curr = _objective(
                X, Y, w, B, beta_hat, lam, mu, R_diag, gamma, eta, gap_scale,
                crit, reg=reg, design_mode=design_mode, eta_bin=eta_bin,
                price_kind=price_kind, price_theta=price_theta,
            )
            grad_w, grad_B = _gradients(
                X, Y, w, B, beta_hat, lam, mu, R_diag, gamma, eta, gap_scale,
                crit, design_mode=design_mode, eta_bin=eta_bin,
                price_kind=price_kind, price_theta=price_theta,
            )

            # Per-block steps: the B-block uses the exact majorizer step
            # 1/L_B (its curvature ~ gamma/(N sigma^2) would otherwise starve
            # the w-block through a shared step); the w-block smooth part is
            # linear-plus-concave in w, so alpha_w is backtracked on the
            # approximate penalized objective (plan 4.7; deviation from the
            # paper's single-alpha analysis recorded in implementation notes).
            L_phi = L_phi_spec(B) if callable(L_phi_spec) else L_phi_spec
            alpha_B = 1.0 / (L_phi + (gamma * gap_scale / N) * (L_H + mu))
            B_new = reg["prox"](
                B - alpha_B * grad_B, alpha_B * gamma * gap_scale * lam / N
            )
            n_halved = 0
            while True:
                w_new = _proj(w - alpha_w * grad_w)
                if not backtracking:
                    break
                F_new, _, _ = _objective(
                    X, Y, w_new, B_new, beta_hat, lam, mu, R_diag, gamma, eta,
                    gap_scale, crit, reg=reg, design_mode=design_mode,
                    eta_bin=eta_bin, price_kind=price_kind,
                    price_theta=price_theta,
                )
                if F_new <= F_curr + 1e-12 or n_halved >= max_halvings:
                    if n_halved >= max_halvings and logger is not None:
                        logger.warning(
                            "backtracking exhausted at iter %d (alpha_w=%.3g)",
                            k_global, alpha_w,
                        )
                    break
                alpha_w *= 0.5
                n_halved += 1
            # Optional oscillation guard: during eta (binarization) stages,
            # do not regrow alpha_w (candidate fix for the theta oscillation
            # observed in the first calibration run; see implementation notes).
            freeze = kind == "eta" and bool(
                params.get("eta_freeze_alpha_growth", False)
            )
            if n_halved == 0 and not freeze:
                alpha_w = min(alpha_w * 1.2, alpha_w_max)

            pg_norm = float(np.sqrt(
                np.sum(((w - w_new) / alpha_w) ** 2)
                + np.sum(((B - B_new) / alpha_B) ** 2)
            ))
            w, B = w_new, B_new

            if not (np.all(np.isfinite(w)) and np.all(np.isfinite(B))):
                raise FloatingPointError(
                    f"non-finite iterate at stage {stage_label}, iter {k_stage}"
                )
            # Equality budget must hold after EVERY projected update
            # (budget mode only; the box_l1 design set is just [0,1]^M).
            if not box_l1:
                check_feasible(w, M0, tol=1e-6)

            # Diagnostics: long LL solve for LL gap / tracking / val metrics.
            row = {
                "iteration": k_global,
                "continuation_stage": stage_label,
                "gamma": gamma,
                "eta": eta,
                "objective": F_curr,
                "Phi": phi_curr,
                "PG_norm": pg_norm,
                **binarity_metrics(w),
                "alpha": alpha_w,
                "alpha_B": alpha_B,
                "T_k": T_k,
                "LL_gap": np.nan,
                "tracking_error": np.nan,
                "val_NMSE": val_nmse_last,
                "val_pred_risk": val_pred_last,
                "runtime_cumulative": time.perf_counter() - t0,
            }
            is_diag = (k_stage % diag_every == 0) or (k_stage == n_iters)
            if is_diag:
                beta_star = _inner_solve_all(
                    X, Y, w, lam, mu, R_diag, beta_hat, diag_steps, reg=reg
                )
                total_inner += N * diag_steps
                # LL gap in the same normalized units as the objective, so
                # the E0 criterion LL_gap/(1+|Phi|) < 1e-3 is scale-free.
                gap_diag = gap_scale * float(np.mean(
                    _g_values(X, Y, w, lam, mu, R_diag, B, reg=reg)
                    - _g_values(X, Y, w, lam, mu, R_diag, beta_star, reg=reg)
                ))
                row["LL_gap"] = gap_diag
                row["tracking_error"] = float(np.mean(
                    np.linalg.norm(B - beta_star, axis=1)
                    / (1.0 + np.linalg.norm(beta_star, axis=1))
                ))
                if Y_val is not None:
                    beta_val_hat = _inner_solve_all(
                        X, Y_val, w, lam, mu, R_diag, beta_val_hat, diag_steps,
                        reg=reg,
                    )
                    total_inner += Y_val.shape[0] * diag_steps
                    val_pred_last = _phi_pred(X, Y_val, beta_val_hat)
                    row["val_pred_risk"] = val_pred_last
                    if beta_dagger_val is not None:
                        val_nmse_last = median_iqr([
                            nmse(beta_val_hat[n], beta_dagger_val[n])
                            for n in range(Y_val.shape[0])
                        ])["median"]
                        row["val_NMSE"] = val_nmse_last
            curves.append(row)
        if is_price:
            act_after = _n_active(w)
            dropped = act_before - act_after
            avalanche = (dropped > max(5, avalanche_frac * act_before)
                         or act_after == 0)
            mid = (float(np.sqrt(prev_price * eta)) if prev_price > 0.0
                   else 0.5 * eta)
            resolvable = (prev_price == 0.0 or eta / prev_price > 1.05)
            if avalanche and bisect_budget > 0 and resolvable:
                w, B, beta_hat, alpha_w = (snap[0].copy(), snap[1].copy(),
                                           snap[2].copy(), snap[3])
                agenda.append((kind, gamma, eta, n_iters, eta_bin))
                agenda.append((kind, gamma, mid, n_iters, eta_bin))
                bisect_budget -= 1
                if logger is not None:
                    logger.info(
                        "price stage %s pruned %d->%d: rolled back, "
                        "bisecting price to %g", stage_label, act_before,
                        act_after, mid)
                continue
            prev_price = eta
        gamma_last, eta_last = gamma, eta
        if not box_l1:
            check_feasible(w, M0)
        w_stages.append((stage_label, w.copy()))
        if logger is not None:
            logger.info(
                "stage %s done: obj=%.6g Phi=%.6g LL_gap=%.3g Obin=%.3g "
                "theta=%.3g val_pred=%.4g alpha=%.3g",
                stage_label, curves[-1]["objective"], curves[-1]["Phi"],
                curves[-1]["LL_gap"], curves[-1]["Obin"], curves[-1]["theta"],
                val_pred_last, alpha_w,
            )

    return IVBResult(
        w_relaxed=w,
        w_stages=w_stages,
        beta_hat=beta_hat,
        curves=curves,
        diagnostics={
            "gamma_final": gamma_last,
            "eta_final": eta_last,
            "alpha_final": alpha_w,
            "outer_iters": k_global,
            "total_inner_steps": total_inner,
            "phi0": phi0,
            "val_pred_risk": val_pred_last,
            "val_NMSE": val_nmse_last,
            "LL_gap_final": curves[-1]["LL_gap"],
            "PG_norm_final": curves[-1]["PG_norm"],
        },
    )


# Backward-compatible alias (criterion defaults to IV-B).
run_ivb = run_proposed
