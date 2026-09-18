"""Result-saving conventions.

Each run writes to a unique directory ``results/<experiment>/<run_id>/`` and is
never overwritten: run directories carry a batch tag chosen once per
invocation of the run script, so re-running a config creates new directories.
``results/all_results.csv`` is regenerated from per-run directories by
``scripts/aggregate_results.py`` and never appended to during runs.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Required columns of the aggregated results file (AGENTS.md / plan 8.6).
RESULT_COLUMNS = [
    "experiment", "seed", "method", "criterion", "dataset",
    "D", "M", "M0", "N_train", "N_val", "N_test",
    "lambda", "lambda_ratio", "mu",
    "gamma_schedule", "gamma_final", "eta_schedule", "eta_final",
    "alpha", "inner_schedule", "rounding",
    "restart_id", "selected_restart",
    "NMSE", "prediction_error", "support_F1",
    "traceC", "logdetC", "lmaxC",
    "Obin", "theta", "rounding_gap_phi", "rounding_gap_NMSE",
    "LL_gap", "PG_norm",
    "runtime", "outer_iters", "total_inner_steps",
    "n_selected", "status", "error_message",
]


def new_batch_tag() -> str:
    """Timestamp tag making run directories of this invocation unique."""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def make_run_dir(results_root: Path, experiment: str, run_id: str) -> Path:
    """Create a fresh run directory; refuse to reuse an existing one."""
    run_dir = Path(results_root) / experiment / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists (results are never overwritten): {run_dir}"
        )
    run_dir.mkdir(parents=True)
    return run_dir


def save_config(run_dir: Path, config: dict) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_metrics_rows(run_dir: Path, rows: list[dict], filename: str = "metrics.csv") -> None:
    """Save result rows, padding to the full required column set."""
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    extra = [c for c in df.columns if c not in RESULT_COLUMNS]
    df = df[RESULT_COLUMNS + extra]
    df.to_csv(run_dir / filename, index=False)


def save_design(run_dir: Path, name: str, design: np.ndarray) -> None:
    np.save(run_dir / f"{name}.npy", np.asarray(design))


def save_selected_indices(run_dir: Path, indices_by_key: dict) -> None:
    serializable = {
        key: [int(i) for i in idx] for key, idx in indices_by_key.items()
    }
    with open(run_dir / "selected_indices.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_training_curves(
    run_dir: Path, rows: list[dict], filename: str = "training_curves.csv"
) -> None:
    pd.DataFrame(rows).to_csv(run_dir / filename, index=False)
