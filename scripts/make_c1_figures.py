"""Figures for the ICASSP companion experiments (C1 synthetic frontier and
RF Fashion-MNIST frontier + masks), generated ONLY from saved results.

Usage:
    python scripts/make_c1_figures.py --experiment c1_synth_frontier_10seed \
        [--fashion rf_fashion_frontier_10seed]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from src.figures.plotting_utils import (method_color, method_label,  # noqa: E402
                                        save_figure)

PROPOSED = "proposed_ivb_l1"
PROPOSED_LABEL = "Proposed (price homotopy)"
PROPOSED_COLOR = "#D55E00"
BASELINES = ["random", "leverage", "dopt_greedy", "aopt_greedy"]


def load_rows(results_root: Path, experiment: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(results_root / experiment / "*" / "metrics.csv")):
        if "_quarantine" in f:
            continue
        rows.append(pd.read_csv(f))
    if not rows:
        raise SystemExit(f"no metrics.csv under results/{experiment}")
    df = pd.concat(rows, ignore_index=True)
    # zero_beta_hat rows are valid evaluations (the deployed estimator
    # collapsed on some instances -- a real, reportable outcome, flagged as
    # in the journal experiments); only failed/timeout/interrupted are
    # excluded.
    return df[df["status"].isin(["ok", "zero_beta_hat"])]


def frontier_per_seed(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per-seed proposed frontier (k, NMSE), lower envelope, sorted by k."""
    out = {}
    d = df[(df["method"] == PROPOSED)
           & (df["rounding"] == "threshold_support")]
    for seed, g in d.groupby("seed"):
        g = g.sort_values("M0")
        # lower envelope per cardinality (several stages can share one k)
        g = g.groupby("M0", as_index=False)["NMSE"].min()
        out[int(seed)] = g
    return out


def frontier_band(fronts: dict[int, pd.DataFrame], k_grid: np.ndarray,
                  prox: float = 3.0):
    """Interpolate each seed's frontier onto k_grid; median + IQR.

    A grid point only counts for a seed when that seed has an ACTUAL
    frontier design within ``prox`` measurements, so the band never spans
    cardinality gaps the homotopy jumped across.
    """
    curves = []
    for g in fronts.values():
        k, v = g["M0"].values.astype(float), g["NMSE"].values
        if len(k) < 2:
            continue
        vals = np.interp(k_grid, k, v, left=np.nan, right=np.nan)
        vals[(k_grid < k.min()) | (k_grid > k.max())] = np.nan
        near = np.min(np.abs(k_grid[:, None] - k[None, :]), axis=1) <= prox
        vals[~near] = np.nan
        curves.append(np.log(vals))
    A = np.vstack(curves)
    med = np.exp(np.nanmedian(A, axis=0))
    q25 = np.exp(np.nanpercentile(A, 25, axis=0))
    q75 = np.exp(np.nanpercentile(A, 75, axis=0))
    n = np.sum(~np.isnan(A), axis=0)
    return med, q25, q75, n


def fig_frontier(df: pd.DataFrame, figures_dir: Path, name: str,
                 min_seeds: int = 5):
    fig, ax = plt.subplots(figsize=(2.6, 2.45))
    table_rows = []
    # Baselines: median/IQR at each M0 of the grid.
    for m in BASELINES:
        g = df[(df["method"] == m)]
        if g.empty:
            continue
        agg = g.groupby("M0")["NMSE"].agg(
            median="median", q25=lambda x: x.quantile(.25),
            q75=lambda x: x.quantile(.75), n="count").reset_index()
        ax.plot(agg["M0"], agg["median"], "-o", ms=3, lw=1.2,
                color=method_color(m), label=method_label(m))
        ax.fill_between(agg["M0"], agg["q25"], agg["q75"], alpha=.15,
                        color=method_color(m), lw=0)
        agg["method"] = m
        table_rows.append(agg)
    # Proposed frontier band.
    fronts = frontier_per_seed(df)
    ks = np.concatenate([g["M0"].values for g in fronts.values()])
    kmax_data = np.percentile(ks, 95)
    m0_max = df[df["method"].isin(BASELINES)]["M0"].max()
    k_grid = np.arange(max(2, ks.min()), min(kmax_data, m0_max) + 1)
    med, q25, q75, n = frontier_band(fronts, k_grid)
    ok = n >= min_seeds
    ax.plot(k_grid[ok], med[ok], "-", lw=2.0, color=PROPOSED_COLOR,
            label=PROPOSED_LABEL)
    ax.fill_between(k_grid[ok], q25[ok], q75[ok], alpha=.2,
                    color=PROPOSED_COLOR, lw=0)
    # Individual frontier designs outside the dense band (the homotopy's
    # per-seed landing points at moderate cardinality).
    dense_ks = set(k_grid[ok].astype(int))
    outs = [(r["M0"], r["NMSE"]) for g in fronts.values()
            for _, r in g.iterrows()
            if not any(abs(r["M0"] - dk) <= 1 for dk in dense_ks)]
    if outs:
        xk, xv = zip(*outs)
        ax.scatter(xk, xv, s=12, marker="D", color=PROPOSED_COLOR,
                   alpha=.65, lw=0, label="Proposed (single runs)")
    table_rows.append(pd.DataFrame(
        {"M0": k_grid[ok], "median": med[ok], "q25": q25[ok],
         "q75": q75[ok], "n": n[ok], "method": PROPOSED}))
    ax.set_yscale("log")
    ax.set_xlim(0.8 * k_grid.min(), 1.06 * m0_max)
    ax.set_xlabel("deployed measurements $M_1$", fontsize=8)
    ax.set_ylabel("test NMSE (median)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, handlelength=1.6,
              borderaxespad=0.2, labelspacing=0.3)
    ax.grid(alpha=.25, which="both")
    return save_figure(fig, figures_dir, name, pd.concat(table_rows))


def improvement_table(df: pd.DataFrame, ref: str = "dopt_greedy"):
    """Paired per-seed improvement of the proposed frontier over a baseline
    interpolated to the same cardinality."""
    base = df[df["method"] == ref].groupby(["seed", "M0"])["NMSE"].median()
    rows = []
    for seed, g in frontier_per_seed(df).items():
        if (seed,) not in {tuple([s]) for s in
                           base.index.get_level_values(0)}:
            continue
        b = base.loc[seed]
        bk, bv = b.index.values.astype(float), b.values
        for _, r in g.iterrows():
            k = r["M0"]
            if k < bk.min() or k > bk.max():
                continue
            ref_v = float(np.exp(np.interp(k, bk, np.log(bv))))
            rows.append({"seed": seed, "k": k, "NMSE": r["NMSE"],
                         "ref_NMSE": ref_v,
                         "improvement": 1.0 - r["NMSE"] / ref_v})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)
    results_root = REPO_ROOT / "results"
    figures_dir = Path(args.figdir) if args.figdir else (
        REPO_ROOT / "figures" / args.experiment)
    df = load_rows(results_root, args.experiment)
    paths = fig_frontier(df, figures_dir, "nmse_vs_cardinality")
    imp = improvement_table(df)
    imp.to_csv(figures_dir / "improvement_vs_dopt_paired.csv", index=False)
    for ref in ("dopt_greedy", "aopt_greedy", "leverage", "random"):
        t = improvement_table(df, ref)
        if t.empty:
            continue
        med = t.groupby(pd.cut(t["k"], bins=[0, 25, 35, 50, 1000],
                               labels=["k<=25", "25<k<=35", "35<k<=50",
                                       "k>50"]),
                        observed=True)["improvement"].median()
        print(f"median paired improvement vs {ref}:")
        print((100 * med).round(1).to_string())
    print("figures:", [str(p) for p in paths])


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Fashion-MNIST masks + reconstructions figure (RF experiment)
# ---------------------------------------------------------------------------

def _closest_stage_design(run_dir: Path, k_target: int):
    """Saved price-stage design whose deployed cardinality is closest to
    k_target. Returns (s_binary, k)."""
    best = None
    for f in glob.glob(str(run_dir / "w_stage*_price_*.npy")):
        w = np.load(f)
        s = (w > 1e-3).astype(float)
        k = int(s.sum())
        if k == 0:
            continue
        if best is None or abs(k - k_target) < abs(best[1] - k_target):
            best = (s, k)
    return best


def fig_fashion_masks(results_root: Path, experiment: str, figures_dir: Path,
                      seed: int = 0, k_target: int = 40, n_examples: int = 3):
    from src.config.loading import load_config
    from src.problems.sparse_linear import make_problem_data
    from src.solvers.lasso import solve_weighted_lasso_batch
    from src.utils.seeds import spawn_streams

    exp_dir = results_root / experiment
    prop_dirs = [d for d in exp_dir.iterdir()
                 if d.is_dir() and d.name.endswith(f"_ivbl1_seed{seed}_mu0p01_lr0p1")]
    assert prop_dirs, "no proposed run dir found"
    s_prop, k_prop = _closest_stage_design(sorted(prop_dirs)[-1], k_target)
    cfg = load_config(REPO_ROOT / "configs" / "rf_fashion_frontier_lr01_10seed.yaml")
    data = make_problem_data(spawn_streams(seed)["data"], cfg["problem"])
    from src.methods.baselines import dopt_greedy
    s_dopt = dopt_greedy(data.X, k_prop, cfg["estimator"]["mu_grid"][0],
                         data.R_diag).s
    lam = cfg["estimator"]["lambda_ratio_grid"][0] * data.lambda_max
    mu = cfg["estimator"]["mu_grid"][0]
    side = int(np.sqrt(data.X.shape[0]))

    # pick visually diverse test images (highest-energy few)
    energies = np.sum(data.beta_test**2, axis=1)
    idx = np.argsort(-energies)[:n_examples]
    Yt, Bt = data.Y_test[idx], data.beta_test[idx]
    rec = {}
    for name, sdes in (("proposed", s_prop), ("dopt", s_dopt)):
        res = solve_weighted_lasso_batch(data.X, Yt, sdes, lam, mu,
                                         data.R_diag, max_iter=3000)
        B = res.B if hasattr(res, "B") else res.beta
        rec[name] = np.stack([np.clip((data.X @ b).reshape(side, side), 0, 1)
                              for b in B])
    truth = np.stack([np.clip((data.X @ b).reshape(side, side), 0, 1)
                      for b in Bt])

    ncol = 1 + n_examples
    fig, axes = plt.subplots(3, ncol, figsize=(2.45, 2.15))
    # column 0: mean image (row 0) and masks (selected pixels in orange over
    # the mean image) for the two designs (rows 1-2)
    meanimg = np.clip((data.X @ data.beta_train.mean(axis=0))
                      .reshape(side, side), 0, 1)
    axes[0, 0].imshow(meanimg, cmap="gray", vmin=0, vmax=1)
    for r, (name, sdes) in enumerate((("proposed", s_prop), ("dopt", s_dopt)),
                                     start=1):
        img = np.stack([meanimg * 0.55] * 3, axis=-1)
        mask = sdes.reshape(side, side) > 0
        img[mask] = [0.84, 0.37, 0.0]
        axes[r, 0].imshow(img)
    # columns 1..: truth / proposed recon / dopt recon
    for j in range(n_examples):
        axes[0, 1 + j].imshow(truth[j], cmap="gray", vmin=0, vmax=1)
        axes[1, 1 + j].imshow(rec["proposed"][j], cmap="gray", vmin=0, vmax=1)
        axes[2, 1 + j].imshow(rec["dopt"][j], cmap="gray", vmin=0, vmax=1)
    # column headers (top row) and row labels (left column)
    fs = 7
    axes[0, 0].set_title("mean / mask", fontsize=fs, pad=2)
    for j in range(n_examples):
        axes[0, 1 + j].set_title(f"image {j + 1}", fontsize=fs, pad=2)
    for r, lab in enumerate(("truth", "Proposed", "D-opt")):
        axes[r, 0].set_ylabel(lab, fontsize=fs, labelpad=2)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.91, bottom=0.01,
                        wspace=.05, hspace=.05)
    return save_figure(fig, figures_dir, "fashion_masks")


def fashion_main(experiment: str):
    results_root = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "figures" / experiment
    df = load_rows(results_root, experiment)
    fig_frontier(df, figures_dir, "nmse_vs_cardinality")
    for ref in ("dopt_greedy", "aopt_greedy", "leverage", "random"):
        t = improvement_table(df, ref)
        if t.empty:
            continue
        med = t.groupby(pd.cut(t["k"], bins=[0, 32, 48, 64, 1000],
                               labels=["k<=32", "32<k<=48", "48<k<=64",
                                       "k>64"]), observed=True)
        print(f"median paired improvement vs {ref} (%):")
        print((100 * med["improvement"].median()).round(1).to_string())
        wins = med["improvement"].apply(lambda s: f"{(s>0).sum()}/{len(s)}")
        print("  paired wins:", wins.to_dict())
    fig_fashion_masks(results_root, experiment, figures_dir)
