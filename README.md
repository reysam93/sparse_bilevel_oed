# ICASSP experiments: "Sparse Experimental Design for Nonsmooth Estimators via Bilevel Optimization"

Self-contained code, configurations, data and results for the two numerical
experiments of the ICASSP companion paper (A. G. Marques and S. Rey). This
folder is a curated copy of the parts of the main repository that these two
experiments use; it runs on its own, without access to the rest of the
repository.

Everything is driven by two YAML configs:

| Paper item | Config | Results folder (shipped) |
|---|---|---|
| Fig. 1(a), sparse linear test case | `configs/c1_synth_frontier_10seed.yaml` | `results/c1_synth_frontier_10seed/` |
| Fig. 1(b)-(c) and Table 1, Fashion-MNIST | `configs/rf_fashion_frontier_lr01_10seed.yaml` | `results/rf_fashion_frontier_lr01_10seed/` |

`configs/c1_tiny_debug.yaml` and `configs/rf_fashion_tiny_debug.yaml` are
small smoke tests (seconds) with the same schema.

## 1. Setup

Python 3.10+ and the packages in `requirements.txt`:

    pip install -r requirements.txt

No network access is needed: Fashion-MNIST is shipped as a local snapshot
(`data/raw/fashion_mnist/`, see the README there for provenance and license),
and the loader prefers it over any download.

Quick check (6 tests, a few seconds):

    python -m pytest tests/

## 2. Regenerating the paper figures from the shipped results

The `results/` folders contain the raw per-run outputs of the 10-seed batches
used in the paper (one directory per grid point with `config.yaml`,
`metrics.csv`, training curves and the saved designs). To rebuild the figures
and the numbers quoted in the text:

    python scripts/make_c1_figures.py --experiment c1_synth_frontier_10seed
    python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts'); from make_c1_figures import fashion_main; fashion_main('rf_fashion_frontier_lr01_10seed')"

Outputs go to `figures/<experiment>/`:

* `nmse_vs_cardinality.{pdf,png}` + `_data.csv`: Fig. 1(a) / 1(b) and the
  curves behind them.
* `improvement_vs_dopt_paired.csv`: paired per-draw improvement of the
  proposed designs over greedy D-optimal, interpolated to the same
  cardinality (the "66/69"-type statistics of Sec. 5 are computed the same
  way for every baseline; the script prints them, binned by cardinality).
* `fashion_masks.{pdf,png}`: Fig. 1(c) (masks and reconstructions at
  M_1 = 40, split 0). This step re-solves a few elastic nets and needs the
  Fashion-MNIST snapshot.

`python scripts/aggregate_results.py` builds a single `results/all_results.csv`
from all per-run directories (regenerated, never appended to).

## 3. Re-running the experiments from scratch

Both batches are "one price-homotopy run per seed" plus baselines on a grid
of cardinalities. Use the time-budgeted, resumable runner: it processes grid
points in a deterministic order, skips the ones that already have a
`metrics.csv`, and stops cleanly when the time budget is about to be
exceeded. Re-invoke it until it prints `ALL DONE`:

    python scripts/run_c1_chunk.py --config configs/c1_synth_frontier_10seed.yaml --max-seconds 600
    python scripts/run_c1_chunk.py --config configs/rf_fashion_frontier_lr01_10seed.yaml --max-seconds 600

Two workers can share a batch with `--stride 2 --offset 0` and
`--stride 2 --offset 1` (set `OMP_NUM_THREADS=1` per worker so they do not
compete for cores). Cost: the `runtime` column of each `metrics.csv` records
the measured time per grid point; summed over the shipped results, the
synthetic batch is under one CPU-hour and the Fashion-MNIST batch about two
CPU-hours (the baselines, which need one solve per cardinality, take most of
it; the 10 proposed price-homotopy runs take about 20 minutes per batch).

To re-run from scratch move or delete the shipped `results/<experiment>/`
first; otherwise the runner will find the completed points and skip them.
`scripts/run_experiment.py` is a plain single-pass alternative (no resume),
kept for the smoke tests:

    python scripts/run_experiment.py --config configs/c1_tiny_debug.yaml

## 4. What is where

    configs/    the four YAML configs (two paper runs, two smoke tests)
    data/       Fashion-MNIST snapshot (70000 x 28 x 28 uint8) + provenance README
    results/    raw outputs of the two paper batches (about 21 MB)
    figures/    created by scripts/make_c1_figures.py
    scripts/    run_c1_chunk.py (resumable runner), run_experiment.py (single pass),
                make_c1_figures.py (figures + paired statistics), aggregate_results.py
    src/        the modules the pipeline needs (imported as `src.*`)
    tests/      test_c1_sparsity.py: price semantics, box feasibility, exact zeros,
                cardinality monotone in the price, budget mode unchanged, pooling

Main code paths, in the order they are used:

* `src/pipelines/c1_sparsity.py`: the experiment pipeline. For each seed it
  generates the data, runs the baselines at every cardinality of
  `baseline_M0_grid`, runs the proposed price homotopy once, and evaluates
  every price stage whose deployed cardinality changed (validation and
  test), with the same deployed elastic net for all methods.
* `src/methods/proposed.py`: the single-loop value-function penalty method
  (`run_proposed`). The ICASSP variant is `design_mode="box_l1"` with
  `price_kind="concave"` (Alg. 1 of the paper: gamma continuation price-free
  from w = 1, then geometric price ramp with the anti-avalanche rollback).
  `design_mode="budget_equality"` is the journal's budgeted variant and is
  not used here.
* `src/methods/baselines.py`: random, leverage (row norm), greedy D- and
  A-optimal selection.
* `src/problems/sparse_linear.py`: data generators and the `lambda_max`
  convention (`generate_sparse_linear_blocks` is the synthetic test case);
  `src/problems/digits_dct.py`: Fashion-MNIST in the 2-D DCT basis with 2x2
  average pooling (14 x 14, D = M = 196).
* `src/solvers/lasso.py`, `src/solvers/ridge.py`: weighted elastic-net (FISTA,
  batched) and ridge solvers used by the lower level and by the evaluation.
* `src/evaluation/evaluate.py`, `src/evaluation/metrics.py`: deployed
  evaluation on test instances (median NMSE etc.; `status = zero_beta_hat`
  flags runs where the deployed estimate collapsed to zero on some test
  instance, which are kept in the aggregation).
* `src/config/loading.py`, `src/utils/*`: config validation, seeding
  (`SeedSequence` streams for data / random baseline / restarts), result
  I/O and logging.

## 5. Conventions worth knowing

* Seeds 0-9 are the 10 "draws" (synthetic) / "splits" (Fashion-MNIST) of the
  paper. Data generation, the random baseline and the proposed method use
  independent random streams, so adding a method does not change the data.
* Medians over test instances per run; medians and IQR over seeds in the
  figures.
* `lambda_max` is computed on training data only, with the uniform design
  `w = (M0/M) * 1` of the config's nominal `M0`; `lambda = lambda_ratio * lambda_max`.
* The deployed design of a price stage is `1[w > tau_w]` (exact zeros come
  from the box projection); baselines are evaluated at the cardinalities in
  `baseline_M0_grid` and interpolated (log-linearly) to the proposed
  cardinalities for the paired comparisons.

Package assembled on 2026-09-07 from the main repository; the module copies
are verbatim (only `scripts/run_experiment.py` was trimmed to register the
`c1_sparsity` pipeline alone).
