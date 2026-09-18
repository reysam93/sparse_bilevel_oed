"""YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_KEYS = [
    "experiment",   # results subdirectory / canonical experiment name
    "pipeline",     # which runner to dispatch to (e.g. "e0_baselines")
    "dataset",
    "seeds",
    "problem",      # D, M, M0, N_train, N_val, N_test, sparsity, snr_db
    "estimator",    # mu_grid, lambda_ratio_grid, tau_supp, rho, solver opts
    "methods",
    "timeout_seconds",
]

REQUIRED_PROBLEM_KEYS = ["D", "M", "M0", "N_train", "N_val", "N_test", "sparsity", "snr_db"]
REQUIRED_ESTIMATOR_KEYS = ["mu_grid", "lambda_ratio_grid", "tau_supp", "rho"]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} did not parse to a mapping")
    validate_config(config)
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict) -> None:
    # Derived pipelines with their own schema.
    if config.get("pipeline") == "e3_rounding":
        for key in ["experiment", "pipeline", "source_experiment",
                    "timeout_seconds"]:
            if key not in config:
                raise ValueError(f"e3_rounding config missing key: {key}")
        return
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    problem = config["problem"]
    missing = [k for k in REQUIRED_PROBLEM_KEYS if k not in problem]
    if missing:
        raise ValueError(f"config['problem'] missing keys: {missing}")
    estimator = config["estimator"]
    missing = [k for k in REQUIRED_ESTIMATOR_KEYS if k not in estimator]
    if missing:
        raise ValueError(f"config['estimator'] missing keys: {missing}")
    if problem["M"] == "auto":
        # R3-only: M is inferred from the processed file's kept-sensor
        # count at run time (src/problems/traffic.resolve_auto_M); the
        # M0 <= M check happens there against the actual count.
        if config.get("pipeline") != "r3_traffic":
            raise ValueError("problem.M: auto is only supported by the "
                             "r3_traffic pipeline")
        if (problem.get("generator_options", {}).get("source")
                != "processed"):
            raise ValueError("problem.M: auto requires generator_options."
                             "source: processed")
    elif problem["M0"] > problem["M"]:
        raise ValueError("M0 must not exceed M")
    if not config["seeds"]:
        raise ValueError("config['seeds'] must be a non-empty list")
    if any(r < 0 for r in estimator["lambda_ratio_grid"]):
        raise ValueError("lambda ratios must be nonnegative")
    if "random" in config["methods"] and "n_random_repeats" not in config:
        raise ValueError("random baseline requires config['n_random_repeats']")
    needs_proposed = config["pipeline"] == "e0_proposed" or (
        config["pipeline"] == "e1_sparse_linear"
        and any(m.startswith("proposed") for m in config["methods"])
    )
    if needs_proposed:
        proposed = config.get("proposed")
        if not isinstance(proposed, dict):
            raise ValueError(
                f"pipeline {config['pipeline']} with proposed methods "
                "requires a 'proposed' block"
            )
        for key in ["gamma_schedule", "outer_iters_per_stage", "alpha_init"]:
            if key not in proposed:
                raise ValueError(f"config['proposed'] missing key: {key}")
    if config["pipeline"] == "e1_sparse_linear":
        if config["problem"].get("generator") != "structured_blocks":
            raise ValueError(
                "E1 must use the structured block generator "
                "(problem.generator: structured_blocks), never the E0 iid one"
            )
    if config["pipeline"] == "e2_criteria":
        if config["problem"].get("generator") != "e2_anisotropic":
            raise ValueError(
                "E2 must use the anisotropic generator "
                "(problem.generator: e2_anisotropic)"
            )
    if config["pipeline"] == "e6_tv":
        if config["problem"].get("generator") != "tv_1d":
            raise ValueError("E6 must use problem.generator: tv_1d")
        if config["estimator"].get("regularizer") != "tv":
            raise ValueError("E6 requires estimator.regularizer: tv")
        not_allowed = [m for m in config["methods"]
                       if m.startswith("proposed") and m != "proposed_ivb"]
        if not_allowed:
            raise ValueError(
                f"proposed methods {not_allowed} are not yet validated for "
                "the TV lower level; only proposed_ivb is enabled "
                "(results/e6_prox_beta_hook_plan.md)"
            )
    if config["pipeline"] == "c1_sparsity":
        d = config.get("design_l1")
        if not isinstance(d, dict):
            raise ValueError(
                "c1_sparsity requires a 'design_l1' block (price_kind, "
                "eta_c, price_ramp_*, tau_w; all optional with defaults)"
            )
        if "proposed_ivb_l1" in config["methods"]:
            proposed = config.get("proposed")
            if not isinstance(proposed, dict):
                raise ValueError("c1_sparsity with proposed_ivb_l1 requires "
                                 "a 'proposed' block")
            for key in ["gamma_schedule", "outer_iters_per_stage",
                        "alpha_init"]:
                if key not in proposed:
                    raise ValueError(
                        f"config['proposed'] missing key: {key}")
            ul = proposed.get("ul_measurements", "train")
            if ul not in ("train", "paired"):
                raise ValueError(
                    "config['proposed']['ul_measurements'] must be 'train' "
                    f"or 'paired', got {ul!r}")
        baseline = [m for m in config["methods"]
                    if m not in ("proposed_ivb_l1",)]
        if baseline and "baseline_M0_grid" not in config:
            raise ValueError("c1_sparsity with baseline methods requires "
                             "'baseline_M0_grid'")
        if config["problem"].get("generator") not in (
                "structured_blocks", "digits_dct", "tv_1d"):
            raise ValueError("c1_sparsity supports generators "
                             "structured_blocks, digits_dct and tv_1d")
        # tv_1d added 2026-09-05 for the preliminary "price route on the TV
        # lower level" study (E6 revisited); the price path is regularizer
        # agnostic because run_proposed takes `regularizer` and `design_mode`
        # independently.  Revert this tuple to drop the study.
    if config["pipeline"] == "r2_digits":
        if config["problem"].get("generator") != "digits_dct":
            raise ValueError(
                "R2 must use the digits DCT adapter "
                "(problem.generator: digits_dct)"
            )
    if config["pipeline"] == "r3_traffic":
        if config["problem"].get("generator") != "traffic_pca":
            raise ValueError(
                "R3 must use the traffic PCA adapter "
                "(problem.generator: traffic_pca)"
            )
