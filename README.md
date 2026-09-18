# Sparse Experimental Design for Nonsmooth Estimators via Bilevel Optimization

Code for the numerical experiments of

> A. G. Marques and S. Rey, *Sparse Experimental Design for Nonsmooth Estimators via Bilevel Optimization*, submitted to ICASSP 2027.

The repository is self-contained: it runs with a standard scientific Python stack, needs no network access, and reproduces the two experiments of the paper (a synthetic sparse linear inverse problem and pixel selection on Fashion-MNIST) from two YAML configuration files.

## 1. What the code does

Optimal experimental design (OED) chooses which of `M` candidate measurements to acquire. Classical OED scores a design through the Fisher information matrix of a linear-Gaussian model, which ignores the estimator that will actually process the data. This code treats OED as a **bilevel optimization problem** whose lower level is the *exact* nonsmooth estimator that is deployed (a design-weighted elastic net / Lasso) and whose upper level scores the resulting reconstructions.

Main ingredients, with the corresponding modules:

* **Lower level (LL).** For each training instance `n`, the design-weighted elastic net
  `beta_n*(w) = argmin_beta  1/2 sum_i (w_i / R_ii) (y_{n,i} - x_i^T beta)^2 + mu/2 ||beta||^2 + lambda ||beta||_1`,
  where `w in [0,1]^M` are continuous acquisition weights. Solved by FISTA (`src/solvers/lasso.py`).
* **Upper level (UL).** The validation prediction risk of the reconstructions: for every training scene, an independent second acquisition of its `M` candidate measurements (same `X`, fresh noise) is generated, the LL only sees the first one and the UL scores the reconstructions on the second (`proposed.ul_measurements: paired` in the configs). To this a **concave sparsity price** `eta * sum_i w_i / (w_i + theta)` on the weights is added. No cardinality budget is imposed: a measurement survives only if its estimator-aware value exceeds its price, so *how many* measurements are kept is an outcome of the optimization.
* **Single-loop algorithm.** A value-function penalty reformulation replaces the LL optimality constraint by a penalized optimality gap. The resulting single-level problem is solved by proximal gradient on the design `w` and on free reconstruction copies `B`, using only inexact warm-started LL solves; no hypergradient or differentiation through the nonsmooth solution map is needed (`src/methods/proposed.py`, function `run_proposed` with `design_mode="box_l1"`).
* **Price homotopy.** A continuation in the penalty parameter `gamma` is run price-free from the full design, then the price `eta` is ramped geometrically. Every price stage whose deployed cardinality changes yields one design, so a single run traces the whole cardinality-vs-accuracy frontier. An adaptive safeguard rolls back and bisects the price when a stage prunes too many measurements at once.
* **Baselines.** Random selection, leverage (row-norm) sampling, and greedy D- and A-optimal selection on the information matrix (`src/methods/baselines.py`).
* **Fair evaluation.** Every design, from every method, is evaluated with the same deployed elastic net on the same independent test instances (`src/evaluation/`).

## 2. Repository layout

```
configs/      YAML configurations (two paper runs, two small smoke tests)
data/         Fashion-MNIST snapshot (70000 x 28 x 28 uint8) and its provenance README
scripts/      command-line entry points (see Sections 4 and 5)
src/          library code, imported as `src.*`
  config/       YAML loading and validation
  problems/     data generators: synthetic structured-block problem, Fashion-MNIST in a 2-D DCT basis
  solvers/      weighted elastic-net (FISTA, batched over instances) and ridge solvers
  methods/      proposed price-homotopy method and the classical baselines
  pipelines/    the experiment pipeline `c1_sparsity` (data -> baselines -> proposed -> evaluation)
  evaluation/   deployed evaluation on test instances and metrics
  figures/      shared plotting conventions
  utils/        seeding, projections, result I/O, logging
tests/        pytest smoke tests
results/      created by the runners (not versioned)
figures/      created by the figure script (not versioned)
```

## 3. Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
python -m pytest tests/          # 6 tests, a few seconds
```

Dependencies: numpy, scipy, scikit-learn, pandas, PyYAML, matplotlib (pytest for the tests).

Fashion-MNIST is shipped as a local snapshot in `data/raw/fashion_mnist/` (MIT-licensed data from [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist), see the README in that folder). The data loader uses it and never downloads anything.

## 4. Quick start

A small end-to-end run (one seed, reduced sizes, about a minute) that exercises the whole pipeline:

```bash
python scripts/run_experiment.py --config configs/c1_tiny_debug.yaml
python scripts/run_experiment.py --config configs/rf_fashion_tiny_debug.yaml
```

Each run writes one directory per method and cardinality under `results/<experiment>/`, containing `config.yaml`, `metrics.csv` (one row per evaluated design), the saved designs (`*.npy`), a `log.txt` and, for the proposed method, `training_curves.csv`.

## 5. Reproducing the paper experiments

| Paper item | Config | Experiment name |
|---|---|---|
| Fig. 1(a): synthetic sparse linear problem | `configs/c1_synth_frontier_10seed.yaml` | `c1_synth_frontier_10seed` |
| Fig. 1(b)-(c) and Table 1: Fashion-MNIST pixel selection | `configs/rf_fashion_frontier_lr01_10seed.yaml` | `rf_fashion_frontier_lr01_10seed` |

Both batches are "one price-homotopy run per seed" plus the baselines on a grid of cardinalities, over 10 seeds. Use the time-budgeted, resumable runner: it processes grid points in a deterministic order, skips those that already have a `metrics.csv`, and stops cleanly when the time budget is about to be exceeded. Re-invoke it until it prints `ALL DONE`:

```bash
python scripts/run_c1_chunk.py --config configs/c1_synth_frontier_10seed.yaml --max-seconds 600
python scripts/run_c1_chunk.py --config configs/rf_fashion_frontier_lr01_10seed.yaml --max-seconds 600
```

Two workers can share a batch with `--stride 2 --offset 0` and `--stride 2 --offset 1` (set `OMP_NUM_THREADS=1` per worker so they do not compete for cores).

Approximate cost on a laptop CPU: under one CPU-hour for the synthetic batch and about two CPU-hours for Fashion-MNIST. Most of it goes to the baselines, which need one evaluation per cardinality; the ten price-homotopy runs take about 3 minutes (synthetic) and 18 minutes (Fashion-MNIST) in total.

Once a batch is complete, rebuild the figures and the paired statistics quoted in the paper:

```bash
python scripts/make_c1_figures.py --experiment c1_synth_frontier_10seed
python -c "import sys; sys.path[:0]=['.','scripts']; from make_c1_figures import fashion_main; fashion_main('rf_fashion_frontier_lr01_10seed')"
```

Outputs go to `figures/<experiment>/`:

* `nmse_vs_cardinality.{pdf,png}` and `nmse_vs_cardinality_data.csv`: median test NMSE versus deployed cardinality for every method (Fig. 1(a) / 1(b)) and the curves behind them.
* `improvement_vs_dopt_paired.csv`: paired per-seed improvement of the proposed designs over greedy D-optimal selection, interpolated to the same cardinality.
* `fashion_masks.{pdf,png}`: selected pixels and reconstructions of a few test images (Fig. 1(c)).

All numbers quoted in Sec. 5 of the paper (number of frontier designs and their cardinality range, median paired NMSE reduction with IQR and win counts per cardinality bin for every baseline, minimum cardinality reached, zero-estimate collapses, Table 1) are printed by

```bash
python scripts/paper_numbers.py --experiment c1_synth_frontier_10seed
python scripts/paper_numbers.py --experiment rf_fashion_frontier_lr01_10seed
```

`python scripts/aggregate_results.py` builds a single `results/all_results.csv` from all per-run directories.

## 6. Configuration reference

The YAML files are validated by `src/config/loading.py`. The most relevant blocks:

| Block | Keys | Meaning |
|---|---|---|
| `problem` | `generator`, `D`, `M`, `N_train`, `N_val`, `N_test`, `sparsity`, `snr_db`, `generator_options` | Data generator (`structured_blocks` or `digits_dct`), dimensions, number of instances per split, SNR |
| `estimator` | `mu_grid`, `lambda_ratio_grid`, `solver_max_iter`, `solver_tol` | Elastic-net parameters; `lambda = lambda_ratio * lambda_max` |
| `methods` | list | Any of `random`, `leverage`, `dopt_greedy`, `aopt_greedy`, `diagnostic_full_design`, `proposed_ivb_l1` |
| `baseline_M0_grid` | list | Cardinalities at which the baselines are evaluated |
| `design_l1` | `price_kind`, `price_theta`, `eta_c`, `price_ramp_start`, `price_ramp_ratio`, `price_ramp_len`, `price_iters_per_stage`, `tau_w` | Concave price (`theta`), geometric price ramp, iterations per price stage, support threshold of the deployed design |
| `proposed` | `gamma_schedule`, `outer_iters_per_stage`, `alpha_init`, `inner`, `avalanche_frac`, `max_price_bisect`, ... | Penalty continuation, proximal-gradient steps, inner-solve schedule `T_k = T0 + c log(k+1)`, anti-avalanche safeguard |
| `proposed.ul_measurements` | `paired` (paper) or `train` | Measurements scored by the UL: an independent paired acquisition of the training scenes, or the LL measurements themselves |
| `seeds` | list | One independent draw / split per seed |

`design_mode="budget_equality"` in `src/methods/proposed.py` is a budgeted variant with an equality cardinality constraint that is not used in this paper.

## 7. Conventions

* **Seeds.** Data generation, the random baseline and the proposed method use independent random streams derived from the seed, so adding a method never changes the data or the other methods' designs.
* **Noise.** `SNR_dB = 10 log10(signal_power / sigma^2)`, with `signal_power` the mean of `(X beta)^2` over all candidate rows and all instances. `R = sigma^2 I` is passed explicitly.
* **Regularization level.** `lambda_max` is computed on training data with the uniform design `w = (M0/M) 1` of the config's nominal `M0`, as the median over training instances of `||X^T D_w R^{-1} y_n||_inf`.
* **Deployed design.** A price stage is deployed as the binary support `1[w > tau_w]`; the box projection produces exact zeros. Baselines select exactly `M0` measurements.
* **Metrics.** Per-instance NMSE `||beta_hat - beta||^2 / ||beta||^2` on test instances, aggregated by the median; medians and interquartile ranges over seeds in the figures. Runs where the deployed estimate collapsed to zero on some test instances are flagged `status = zero_beta_hat` and kept.
* **Results are never overwritten.** Each invocation creates new run directories tagged with a timestamp; aggregates are always rebuilt from the per-run files.

## 8. Citation

```bibtex
@inproceedings{marques2027sparse,
  title     = {Sparse Experimental Design for Nonsmooth Estimators via Bilevel Optimization},
  author    = {Marques, Antonio G. and Rey, Samuel},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

Work supported by the Spanish AEI (grants PID2022-136887NB-I00 and PID2025-170000NB-I00) and the Community of Madrid (IDEA-CM TEC-2024/COM-89, URJC-F1180, Ellis Madrid Unit). Claude AI was used to assist in coding the simulations.
