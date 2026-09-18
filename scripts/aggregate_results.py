"""Regenerate results/all_results.csv from per-run directories.

Never appended to during runs (binding convention): this script scans
results/<experiment>/<run_id>/metrics.csv files and rebuilds the aggregate
from scratch. metrics_repeats.csv diagnostics are not aggregated.

Usage:
    python scripts/aggregate_results.py --results-dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import RESULT_COLUMNS  # noqa: E402


# Runs parked here (deeper nesting) are never aggregated; see the
# budget-convention audit entry in docs/implementation_notes.md.
OBSOLETE_DIR = "obsolete_inequality_budget"


def aggregate(results_dir: Path) -> pd.DataFrame:
    frames = []
    for metrics_file in sorted(results_dir.glob("*/*/metrics.csv")):
        rel = metrics_file.parent.relative_to(results_dir)
        if rel.parts[0] == OBSOLETE_DIR:
            continue
        df = pd.read_csv(metrics_file)
        df["run_dir"] = str(rel)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS + ["run_dir"])
    all_df = pd.concat(frames, ignore_index=True)
    for col in RESULT_COLUMNS:
        if col not in all_df.columns:
            all_df[col] = pd.NA
    # Rows written before the equality-budget audit lack the column; binary
    # baseline designs always selected exactly M0, so they are equality-valid.
    if "budget_convention" not in all_df.columns:
        all_df["budget_convention"] = pd.NA
    all_df["budget_convention"] = all_df["budget_convention"].fillna(
        "equality_valid_binary_baseline"
    )
    extra = [c for c in all_df.columns if c not in RESULT_COLUMNS]
    return all_df[RESULT_COLUMNS + extra]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="results root")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results directory not found: {results_dir}")
    all_df = aggregate(results_dir)
    out = results_dir / "all_results.csv"
    all_df.to_csv(out, index=False)
    print(f"wrote {len(all_df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
