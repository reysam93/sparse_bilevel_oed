"""Time-budgeted resumable runner for the c1_sparsity pipeline.

Runs grid points of a c1_sparsity config one at a time in a deterministic
order, SKIPPING points that already have a completed metrics.csv (matched
by run-id tail, ignoring the batch tag), and exits cleanly when the time
budget is about to be exceeded. Re-invoke until it prints ALL DONE.

Usage:
    python scripts/run_c1_chunk.py --config configs/c1_synth_frontier_10seed.yaml \
        [--max-seconds 150] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.loading import load_config                     # noqa: E402
from src.pipelines import c1_sparsity as c1                    # noqa: E402
from src.problems.sparse_linear import make_problem_data       # noqa: E402
from src.utils import io as uio                                # noqa: E402
from src.utils.seeds import spawn_streams                      # noqa: E402


def _tail(method, seed, mu, lr, M0=None):
    t = f"_{method}_seed{seed}_mu{mu:g}_lr{lr:g}"
    if M0 is not None:
        t += f"_M{M0}"
    return t.replace(".", "p")


def _completed(exp_dir: Path, tail: str) -> bool:
    if not exp_dir.exists():
        return False
    for d in exp_dir.iterdir():
        if d.is_dir() and d.name.endswith(tail) and (d / "metrics.csv").exists():
            return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-seconds", type=float, default=150.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="run every stride-th todo point (parallel workers)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    config = load_config(args.config)
    assert config["pipeline"] == "c1_sparsity"
    results_root = REPO_ROOT / "results"
    exp_dir = results_root / config["experiment"]
    est = config["estimator"]
    m0_grid = list(config.get("baseline_M0_grid", []))
    timeout = float(config["timeout_seconds"])

    # Deterministic grid enumeration: per seed, proposed first, then
    # baselines (method-major so the expensive random repeats spread out).
    points = []
    for seed in config["seeds"]:
        for mu in est["mu_grid"]:
            for lr in est["lambda_ratio_grid"]:
                if "proposed_ivb_l1" in config["methods"]:
                    points.append(("ivbl1", seed, mu, lr, None))
                for j, M0 in enumerate(m0_grid):
                    for method in config["methods"]:
                        if method not in c1.BASELINE_METHODS:
                            continue
                        if method == "diagnostic_full_design" and j > 0:
                            continue
                        points.append((method, seed, mu, lr, M0))

    todo = [pt for pt in points
            if not _completed(exp_dir, _tail(*pt) if pt[4] is not None
                              else _tail(pt[0], pt[1], pt[2], pt[3]))]
    todo = todo[args.offset::max(1, args.stride)]
    print(f"{len(points)} grid points, {len(todo)} to do in this worker")
    if args.dry_run:
        return 0
    if not todo:
        print("ALL DONE")
        return 0

    batch = uio.new_batch_tag()
    data_cache = {}
    n_done = 0
    for pt in todo:
        if time.perf_counter() - t0 > args.max_seconds:
            break
        method, seed, mu, lr, M0 = pt
        if seed not in data_cache:
            data_cache[seed] = make_problem_data(
                spawn_streams(seed)["data"], config["problem"])
        data = data_cache[seed]
        lam = lr * data.lambda_max
        tail = _tail(*pt) if M0 is not None else _tail(method, seed, mu, lr)
        run_dir = uio.make_run_dir(results_root, config["experiment"],
                                   batch + tail)
        if method == "ivbl1":
            c1._run_proposed_l1(config, data, seed, mu, lr, run_dir, timeout,
                                c1._c1_rng("restarts_c1", seed, mu, lr, 0.0))
        else:
            c1._run_baseline(config, data, seed, method, mu, lam, lr, M0,
                             run_dir, timeout,
                             c1._c1_rng("random_baseline_c1", seed, mu, lr,
                                        M0)
                             if method == "random" else None)
        n_done += 1
    remaining = len(todo) - n_done
    print(f"chunk done: {n_done} points in {time.perf_counter()-t0:.0f}s, "
          f"{remaining} remaining")
    if remaining == 0:
        print("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
