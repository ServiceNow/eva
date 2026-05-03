"""Accuracy-vs-experience scatter plots with bootstrap CI error bars."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_config import ARCH_ORDER, ModelEntry, PaperConfig, sort_models
from paper_tables import lookup_pooled, _is_missing

ARCH_MARKER = {"cascade": "o", "hybrid": "D", "s2s": "s"}
ARCH_COLOR = {"cascade": "#3a86ff", "hybrid": "#2a9d8f", "s2s": "#7b2cbf"}
ARCH_DISPLAY = {
    "cascade": "Cascade (STT + LLM + TTS)",
    "hybrid": "Hybrid (AudioLLM + TTS)",
    "s2s": "Speech-to-speech (S2S)",
}


@dataclass(frozen=True)
class ScatterPoint:
    label: str
    arch: str
    x: float
    x_lo: float
    x_hi: float
    y: float
    y_lo: float
    y_hi: float


def asymmetric_err(
    points: list[float],
    los: list[float],
    his: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point - lo, hi - point)`` for matplotlib's xerr/yerr asymmetric form."""
    p = np.asarray(points, dtype=float)
    lo = np.asarray(los, dtype=float)
    hi = np.asarray(his, dtype=float)
    if np.any(lo > p) or np.any(hi < p):
        raise ValueError("ci_lower must be <= point <= ci_upper for every point")
    return p - lo, hi - p


def build_scatter_points(
    pooled_df: pd.DataFrame,
    cfg: PaperConfig,
    x_metric: str,
    y_metric: str,
) -> list[ScatterPoint]:
    """One ScatterPoint per system that has pooled CIs for both axes."""
    out: list[ScatterPoint] = []
    for m in sort_models(cfg.models):
        x, x_lo, x_hi = lookup_pooled(pooled_df, m.label, x_metric)
        y, y_lo, y_hi = lookup_pooled(pooled_df, m.label, y_metric)
        if any(_is_missing(v) for v in (x, x_lo, x_hi, y, y_lo, y_hi)):
            continue
        out.append(ScatterPoint(
            label=m.label, arch=m.arch,
            x=x, x_lo=x_lo, x_hi=x_hi,
            y=y, y_lo=y_lo, y_hi=y_hi,
        ))
    return out


def pareto_frontier_indices(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of points on the upper-right (higher-is-better) Pareto frontier."""
    n = len(points)
    keep: list[int] = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def write_scatter(
    pooled_df: pd.DataFrame,
    cfg: PaperConfig,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> int:
    """Write a 1-pt-per-system scatter PDF with asymmetric CI error bars.

    Returns the number of points drawn. If zero, no file is written.
    """
    points = build_scatter_points(pooled_df, cfg, x_metric, y_metric)
    if not points:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Pareto frontier on point estimates
    xy = [(p.x, p.y) for p in points]
    frontier_idx = pareto_frontier_indices(xy)
    frontier_drawn = False
    if len(frontier_idx) >= 2:
        frontier_pts = sorted([(points[i].x, points[i].y) for i in frontier_idx])
        ax.plot(
            [fx for fx, _ in frontier_pts],
            [fy for _, fy in frontier_pts],
            linestyle="--", color="grey", alpha=0.7, zorder=1,
        )
        frontier_drawn = True

    # errorbar plotting; do NOT pass label=, so the legend is built manually below
    # using marker-only proxies (otherwise legend entries include the cross-shaped
    # error bars and all archs look visually identical at small size).
    seen_archs: list[str] = []
    for arch in ARCH_ORDER:
        arch_points = [p for p in points if p.arch == arch]
        if not arch_points:
            continue
        seen_archs.append(arch)
        xs = [p.x for p in arch_points]
        ys = [p.y for p in arch_points]
        x_err = asymmetric_err(xs, [p.x_lo for p in arch_points], [p.x_hi for p in arch_points])
        y_err = asymmetric_err(ys, [p.y_lo for p in arch_points], [p.y_hi for p in arch_points])
        ax.errorbar(
            xs, ys, xerr=x_err, yerr=y_err,
            fmt=ARCH_MARKER[arch], color=ARCH_COLOR[arch],
            ecolor=ARCH_COLOR[arch], elinewidth=1.0, capsize=2,
            markersize=8, alpha=0.9, zorder=3,
        )

    from matplotlib.lines import Line2D
    handles: list[Line2D] = []
    if frontier_drawn:
        handles.append(Line2D([0], [0], linestyle="--", color="grey", alpha=0.7,
                              label="Pareto frontier"))
    for arch in seen_archs:
        handles.append(Line2D([0], [0], marker=ARCH_MARKER[arch], color=ARCH_COLOR[arch],
                              linestyle="", markersize=8, label=ARCH_DISPLAY[arch]))
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return len(points)
