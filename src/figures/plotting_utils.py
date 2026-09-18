"""Shared plotting conventions: consistent method names, colors, and saving.

Figures are generated ONLY from saved result files and are written as both
.pdf and .png, with the processed table saved next to them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Consistent labels/colors across all figures (colorblind-safe palette).
METHOD_STYLE = {
    "random": ("Random", "#999999"),
    "leverage": ("Leverage", "#E69F00"),
    "dopt_greedy": ("D-opt greedy", "#0072B2"),
    "aopt_greedy": ("A-opt greedy", "#56B4E9"),
    "proposed_ivb": ("Proposed IV-B", "#D55E00"),
    "proposed_ivb_relaxed": ("Proposed IV-B (relaxed)", "#CC79A7"),
    "proposed_ivc_trace": ("Proposed IV-C trace", "#009E73"),
    "proposed_ivc_logdet": ("Proposed IV-C log-det", "#F0E442"),
    "proposed_ivc_maxeig": ("Proposed IV-C max-eig", "#882255"),
    "diagnostic_full_design": ("Full design (diagnostic)", "#000000"),
}


def method_label(method: str) -> str:
    return METHOD_STYLE.get(method, (method, None))[0]


def method_color(method: str):
    return METHOD_STYLE.get(method, (method, "#333333"))[1]


def save_figure(fig, figures_dir: Path, name: str,
                table: pd.DataFrame | None = None) -> list[Path]:
    """Save <name>.pdf and <name>.png (and the processed table as CSV)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        out = figures_dir / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        paths.append(out)
    if table is not None:
        table.to_csv(figures_dir / f"{name}_data.csv", index=False)
    plt.close(fig)
    return paths


def stage_boundaries(curves: pd.DataFrame) -> list[tuple[int, str]]:
    """(first iteration, stage label) of each continuation stage."""
    firsts = curves.groupby("continuation_stage", sort=False)["iteration"].min()
    return list(zip(firsts.values, firsts.index))
