"""C1 sparsity-promoting design pipeline (ICASSP companion experiments).

Budget-free design variant: the design lives in the box [0,1]^M and each
acquisition weight pays a sparsity price (``design_mode="box_l1"`` of
``src/methods/proposed.py``; concave price by default). ONE run per seed:
the gamma continuation runs price-free, then a geometric price micro-ramp
prunes the design progressively at gamma_final; every price-stage snapshot
is a frontier point, deployed as the thresholded support supp_tau(w) -- no
equality budget, no top-M0 rounding -- so the cardinality is an OUTCOME.
Classical baselines are evaluated at a grid of cardinalities
(``baseline_M0_grid``) to draw the comparison curves.

Everything here is ADDITIVE: the journal pipelines (E0/E1/E3/R2/R3) and
their equality-budget conventions are untouched. Seeding follows the
gridpoint convention with NEW stream tags (never reuse the E0/E1 ones).
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from src.evaluation.evaluate import evaluate_design
from src.evaluation.metrics import binarity_metrics
from src.methods import baselines as bl
from src.methods.proposed import run_proposed
from src.problems.sparse_linear import make_problem_data
from src.utils import io as uio
from src.utils.logging import close_run_logger, get_run_logger
from src.utils.seeds import spawn_streams

BASELINE_METHODS = {"random", "leverage", "dopt_greedy", "aopt_greedy",
                    "diagnostic_full_design"}

# New per-grid-point stream tags (do NOT collide with src/utils/seeds.py).
_C1_STREAM_TAGS = {"random_baseline_c1": 3717, "restarts_c1": 3718}

_RANDOM_MEDIAN_COLS = [
    "NMSE", "NMSE_mean", "prediction_error", "support_F1",
    "traceC", "logdetC", "lmaxC",
]


def _c1_rng(name: str, base_seed: int, mu: float, lambda_ratio: float,
            extra: float) -> np.random.Generator:
    entropy = [
        _C1_STREAM_TAGS[name], int(base_seed),
        int(round(mu * 10**9)), int(round(lambda_ratio * 10**9)),
        int(round(float(extra) * 10**9)),
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _eval_kwargs(est: dict, deadline: float) -> dict:
    return dict(
        tau_supp=est["tau_supp"], rho=est["rho"],
        solver_max_iter=est.get("solver_max_iter", 2000),
        solver_tol=est.get("solver_tol", 1e-10),
        deadline=deadline,
        regularizer=est.get("regularizer", "l1"),
    )


def _base_row(config: dict, seed: int, method: str, mu: float, lam: float,
              lambda_ratio: float) -> dict:
    p = config["problem"]
    return {
        "experiment": config["experiment"],
        "seed": seed,
        "method": method,
        "criterion": "",
        "dataset": config["dataset"],
        "D": p["D"], "M": p["M"], "M0": p["M0"],
        "N_train": p["N_train"], "N_val": p["N_val"], "N_test": p["N_test"],
        "lambda": lam, "lambda_ratio": lambda_ratio, "mu": mu,
        "rounding": "native_binary",
        "budget_convention": "box_l1",
        "eta_l1_c": np.nan,
        "status": "ok",
        "error_message": "",
    }


def run(config: dict, results_root: Path) -> list[Path]:
    p = config["problem"]
    est = config["estimator"]
    batch = uio.new_batch_tag()
    timeout = float(config["timeout_seconds"])
    run_dirs: list[Path] = []
    m0_grid = list(config.get("baseline_M0_grid", []))

    for seed in config["seeds"]:
        streams = spawn_streams(seed)
        data = make_problem_data(streams["data"], p)
        for mu in est["mu_grid"]:
            for lambda_ratio in est["lambda_ratio_grid"]:
                lam = lambda_ratio * data.lambda_max
                if "proposed_ivb_l1" in config["methods"]:
                    run_id = (
                        f"{batch}_ivbl1_seed{seed}_mu{mu:g}"
                        f"_lr{lambda_ratio:g}"
                    ).replace(".", "p")
                    run_dir = uio.make_run_dir(
                        results_root, config["experiment"], run_id)
                    run_dirs.append(run_dir)
                    _run_proposed_l1(
                        config, data, seed, mu, lambda_ratio, run_dir,
                        timeout,
                        _c1_rng("restarts_c1", seed, mu, lambda_ratio, 0.0),
                    )
                for j, M0 in enumerate(m0_grid):
                    for method in config["methods"]:
                        if method not in BASELINE_METHODS:
                            continue
                        # The full-design reference has no budget; run once.
                        if method == "diagnostic_full_design" and j > 0:
                            continue
                        run_id = (
                            f"{batch}_{method}_seed{seed}_mu{mu:g}"
                            f"_lr{lambda_ratio:g}_M{M0}"
                        ).replace(".", "p")
                        run_dir = uio.make_run_dir(
                            results_root, config["experiment"], run_id)
                        run_dirs.append(run_dir)
                        _run_baseline(
                            config, data, seed, method, mu, lam, lambda_ratio,
                            M0, run_dir, timeout,
                            _c1_rng("random_baseline_c1", seed, mu,
                                    lambda_ratio, M0)
                            if method == "random" else None,
                        )
    return run_dirs


def _run_baseline(config, data, seed, method, mu, lam, lambda_ratio, M0,
                  run_dir: Path, timeout: float, rng_random) -> None:
    logger = get_run_logger(run_dir.name, run_dir / "log.txt")
    est = config["estimator"]
    t0 = time.perf_counter()
    deadline = t0 + timeout
    row = _base_row(config, seed, method, mu, lam, lambda_ratio)
    row["M0"] = int(M0)
    repeat_rows: list[dict] = []
    try:
        uio.save_config(run_dir, {k: v for k, v in config.items()
                                  if not k.startswith("_")})
        logger.info("baseline %s seed=%s mu=%g lr=%g M0=%d",
                    method, seed, mu, lambda_ratio, M0)
        M = config["problem"]["M"]
        if method == "random":
            designs = [bl.random_budgeted(rng_random, M, M0)
                       for _ in range(config["n_random_repeats"])]
        elif method == "leverage":
            designs = [bl.leverage_row_norm(data.X, M0, data.R_diag)]
        elif method == "dopt_greedy":
            designs = [bl.dopt_greedy(data.X, M0, mu, data.R_diag)]
        elif method == "aopt_greedy":
            designs = [bl.aopt_greedy(data.X, M0, mu, data.R_diag)]
        elif method == "diagnostic_full_design":
            designs = [bl.full_design(M)]
            row["M0"] = int(M)
        else:
            raise ValueError(f"unknown baseline '{method}'")

        per_design = []
        for rep, d in enumerate(designs):
            m = evaluate_design(d.s, data.X, data.Y_test, data.beta_test,
                                lam, mu, data.R_diag,
                                **_eval_kwargs(est, deadline))
            per_design.append(m)
            if len(designs) > 1:
                repeat_rows.append({**row, "restart_id": rep, **m})
        if len(designs) == 1:
            row.update(per_design[0])
            uio.save_design(run_dir, "s_design", designs[0].s)
        else:
            for col in _RANDOM_MEDIAN_COLS:
                vals = [m[col] for m in per_design]
                row[col] = float(np.median(vals))
                row[f"{col}_repeats_iqr"] = float(
                    np.percentile(vals, 75) - np.percentile(vals, 25))
            row["n_selected"] = int(np.median(
                [m["n_selected"] for m in per_design]))
            statuses = {m["status"] for m in per_design}
            row["status"] = "ok" if statuses == {"ok"} else "zero_beta_hat"
    except TimeoutError as exc:
        row["status"] = "timeout"
        row["error_message"] = str(exc)
        logger.warning("timeout: %s", exc)
    except (KeyboardInterrupt, SystemExit) as exc:
        row["status"] = "interrupted"
        row["error_message"] = type(exc).__name__
        logger.warning("interrupted")
        raise
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        logger.error("run failed:\n%s", traceback.format_exc())
    finally:
        row["runtime"] = time.perf_counter() - t0
        if row["status"] == "interrupted":
            uio.save_metrics_rows(run_dir, [row],
                                  filename="metrics_interrupted.csv")
            (run_dir / "INCOMPLETE.txt").write_text("interrupted\n")
        else:
            uio.save_metrics_rows(run_dir, [row])
        if repeat_rows:
            uio.save_metrics_rows(run_dir, repeat_rows,
                                  filename="metrics_repeats.csv")
        logger.info("status=%s runtime=%.2fs", row["status"], row["runtime"])
        close_run_logger(logger)


def _run_proposed_l1(config, data, seed, mu, lambda_ratio,
                     run_dir: Path, timeout: float, rng_restarts) -> None:
    logger = get_run_logger(run_dir.name, run_dir / "log.txt")
    est = config["estimator"]
    d1 = config["design_l1"]
    params = dict(config["proposed"])
    params["eta_l1_c"] = float(d1.get("eta_c", 1.0))
    params["price_kind"] = d1.get("price_kind", "concave")
    params["price_theta"] = float(d1.get("price_theta", 0.1))
    r0 = float(d1.get("price_ramp_start", 0.05))
    ratio = float(d1.get("price_ramp_ratio", 1.18))
    n_ramp = int(d1.get("price_ramp_len", 40))
    params["price_ramp"] = [r0 * ratio**j for j in range(n_ramp)]
    if "price_iters_per_stage" in d1:
        params["price_iters_per_stage"] = int(d1["price_iters_per_stage"])
    tau_w = float(d1.get("tau_w", 1e-3))
    lam = lambda_ratio * data.lambda_max
    t0 = time.perf_counter()
    deadline = t0 + timeout
    rows: list[dict] = []
    base = _base_row(config, seed, "proposed_ivb_l1", mu, lam, lambda_ratio)
    # UL measurements: "paired" = independent second acquisition of the
    # training scenes (paper eq. 4); "train" = the LL measurements themselves.
    ul_measurements = str(params.get("ul_measurements", "train"))
    base.update({"criterion": "IV-B",
                 "ul_measurements": ul_measurements,
                 "eta_l1_c": params["eta_l1_c"],
                 "price_kind": params["price_kind"],
                 "price_theta": params["price_theta"],
                 "gamma_schedule": str(params["gamma_schedule"]),
                 "alpha": params["alpha_init"]})
    try:
        uio.save_config(run_dir, {k: v for k, v in config.items()
                                  if not k.startswith("_")})
        logger.info("proposed_ivb_l1 ramp seed=%s mu=%g lr=%g lambda=%g "
                    "ramp=[%g..%g]x%d", seed, mu, lambda_ratio, lam,
                    params["price_ramp"][0], params["price_ramp"][-1],
                    n_ramp)
        M = data.X.shape[0]
        # The price homotopy starts from the full design (canonical init);
        # restarts are not used along the ramp (n_restarts is ignored here).
        res = run_proposed(
            data.X, data.Y_train, data.R_diag, lam, mu,
            config["problem"]["M0"], params, np.ones(M),
            criterion="ivb",
            Y_val=data.Y_val, beta_dagger_val=data.beta_val,
            Y_ul=(data.Y_train_ul if ul_measurements == "paired" else None),
            logger=logger, deadline=deadline,
            regularizer=est.get("regularizer", "l1"),
            design_mode="box_l1",
        )
        uio.save_training_curves(run_dir, res.curves)
        uio.save_design(run_dir, "w_final", res.w_relaxed)

        # Frontier: evaluate every price-stage snapshot with a new deployed
        # cardinality (validation AND test; test used only for reporting).
        prev_k = None
        n_pts = 0
        for label, w_stage in res.w_stages:
            if "_price_" not in f"_{label}_" and "price" not in label:
                continue
            s_dep = (w_stage > tau_w).astype(float)
            k = int(s_dep.sum())
            if k == 0 or k == prev_k:
                continue
            prev_k = k
            uio.save_design(run_dir, f"w_stage{label}", w_stage)
            m_val = evaluate_design(
                s_dep, data.X, data.Y_val, data.beta_val, lam, mu,
                data.R_diag, **_eval_kwargs(est, deadline))
            m_test = evaluate_design(
                s_dep, data.X, data.Y_test, data.beta_test, lam, mu,
                data.R_diag, **_eval_kwargs(est, deadline))
            rows.append({
                **base,
                "rounding": "threshold_support",
                "stage": label,
                "eta_price_stage": float(label.split("_e")[-1])
                if "_e" in label else np.nan,
                "M0": k,
                "val_pred_rounded": m_val["prediction_error"],
                "val_NMSE_rounded": m_val["NMSE"],
                **binarity_metrics(w_stage),
                "n_exact_zero": int(np.sum(w_stage == 0.0)),
                **m_test,
            })
            n_pts += 1
            logger.info("frontier point k=%d (stage %s): test_NMSE=%.4g",
                        k, label, m_test["NMSE"])
        base_diag = {
            "LL_gap": res.diagnostics["LL_gap_final"],
            "PG_norm": res.diagnostics["PG_norm_final"],
            "outer_iters": res.diagnostics["outer_iters"],
            "total_inner_steps": res.diagnostics["total_inner_steps"],
        }
        for r_ in rows:
            r_.update(base_diag)
        if not rows:
            raise RuntimeError("price ramp produced no nonempty frontier "
                               "points (check the ramp range)")
        logger.info("frontier points: %d", n_pts)
    except TimeoutError as exc:
        rows.append({**base, "status": "timeout", "error_message": str(exc)})
        logger.warning("timeout: %s", exc)
    except (KeyboardInterrupt, SystemExit) as exc:
        rows.append({**base, "status": "interrupted",
                     "error_message": type(exc).__name__})
        logger.warning("interrupted")
        uio.save_metrics_rows(run_dir, rows,
                              filename="metrics_interrupted.csv")
        (run_dir / "INCOMPLETE.txt").write_text("interrupted\n")
        close_run_logger(logger)
        raise
    except Exception as exc:  # noqa: BLE001
        rows.append({**base, "status": "failed",
                     "error_message": f"{type(exc).__name__}: {exc}"})
        logger.error("run failed:\n%s", traceback.format_exc())
    finally:
        if rows and rows[-1].get("status") != "interrupted":
            for r_ in rows:
                r_["runtime"] = time.perf_counter() - t0
            uio.save_metrics_rows(run_dir, rows)
            logger.info("status=%s runtime=%.2fs",
                        rows[-1].get("status", "ok"), rows[-1]["runtime"])
            close_run_logger(logger)
