"""Print every number quoted in Sec. 5 of the ICASSP paper from saved results.

Usage:
    python scripts/paper_numbers.py --experiment c1_synth_frontier_10seed
    python scripts/paper_numbers.py --experiment rf_fashion_frontier_lr01_10seed

Reuses the loading / frontier / paired-comparison code of make_c1_figures.py
so that the printed statistics are exactly the ones behind the figures. The
cardinality bins are the paper's: [<=40, >40] for the synthetic case and
[<=32, 32-48, 48-64, >64] for Fashion-MNIST (Table 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_c1_figures import (BASELINES, PROPOSED, improvement_table,  # noqa: E402
                             load_rows)

BINS = {
    "c1_synth_frontier_10seed": ([0, 40, 10**6], ["M1<=40", "M1>40"]),
    "rf_fashion_frontier_lr01_10seed": (
        [0, 32, 48, 64, 10**6], ["M1<=32", "32<M1<=48", "48<M1<=64", "M1>64"]),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args(argv)
    df = load_rows(REPO_ROOT / "results", args.experiment)
    M = int(df["M"].iloc[0])
    bins, labels = BINS.get(args.experiment, ([0, 40, 10**6], ["M1<=40", "M1>40"]))

    prop = df[(df["method"] == PROPOSED) & (df["rounding"] == "threshold_support")]
    print(f"=== {args.experiment}: {prop['seed'].nunique()} seeds, M = {M}")
    if "ul_measurements" in prop.columns:
        print("UL measurements:", sorted(prop["ul_measurements"].dropna().unique()))

    # --- frontier designs (the unpruned first point M1 = M is not counted) ---
    pruned = prop[prop["M0"] < M]
    per_seed = pruned.groupby("seed")["M0"].agg(["count", "min", "max"])
    print("\n-- Frontier designs per seed (excluding the unpruned design M1 = M)")
    print(per_seed.T.to_string())
    print(f"designs per seed: {per_seed['count'].min()} to {per_seed['count'].max()}; "
          f"total {len(pruned)}; M1 in [{int(pruned['M0'].min())}, {int(pruned['M0'].max())}]")
    for b in bins[1:-1]:
        print(f"designs with M1 <= {b}: {int((pruned['M0'] <= b).sum())}")
    print("minimum M1 per seed:", per_seed["min"].astype(int).tolist())

    # --- paired comparison against every baseline, paper bins ---
    print("\n-- Median paired test-NMSE reduction (%) w.r.t. each baseline "
          "[IQR] (wins / paired cases)")
    rows = []
    for ref in BASELINES:
        t = improvement_table(df, ref)
        if t.empty:
            continue
        g = t.groupby(pd.cut(t["k"], bins=bins, labels=labels), observed=True)["improvement"]
        for lab, s in g:
            rows.append({"baseline": ref, "bin": lab,
                         "median_%": 100 * s.median(),
                         "q25_%": 100 * s.quantile(.25), "q75_%": 100 * s.quantile(.75),
                         "wins": int((s > 0).sum()), "cases": len(s)})
    tab = pd.DataFrame(rows)
    for ref, g in tab.groupby("baseline", sort=False):
        cells = [f"{r['bin']}: {r['median_%']:.0f}% [{r['q25_%']:.0f}-{r['q75_%']:.0f}] "
                 f"({r['wins']}/{r['cases']})" for _, r in g.iterrows()]
        print(f"{ref:12s} " + " | ".join(cells))

    # --- zero-estimate collapses of the deployed estimator ---
    z = df[df["n_zero_beta_hat"] > 0]
    if not z.empty:
        print("\n-- Test instances with an all-zero deployed estimate "
              "(median over seeds of n_zero_beta_hat, per method and M1)")
        zz = z.groupby(["method", "M0"])["n_zero_beta_hat"].agg(
            n_runs="count", median="median", max="max")
        print(zz.to_string())
    else:
        print("\n-- No zero-estimate collapses in any run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
