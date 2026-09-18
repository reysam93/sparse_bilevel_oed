"""Run an ICASSP experiment from a YAML config (single, non-resumable pass).

Usage:
    python scripts/run_experiment.py --config configs/c1_tiny_debug.yaml

This is a trimmed copy of the main repository's runner that registers only
the ``c1_sparsity`` pipeline used by the ICASSP paper. For the full 10-seed
batches prefer the time-budgeted, resumable ``scripts/run_c1_chunk.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loading import load_config  # noqa: E402
from src.pipelines import c1_sparsity  # noqa: E402

PIPELINES = {"c1_sparsity": c1_sparsity.run}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument(
        "--results-root", default=None,
        help="results directory (default: <this folder>/results)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    results_root = (Path(args.results_root) if args.results_root
                    else repo_root / "results")

    config = load_config(args.config)
    pipeline = config["pipeline"]
    if pipeline not in PIPELINES:
        raise SystemExit(
            f"unknown pipeline '{pipeline}'; available: {sorted(PIPELINES)}"
        )
    run_dirs = PIPELINES[pipeline](config, results_root)
    print(f"completed {len(run_dirs)} runs under "
          f"{results_root / config['experiment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
