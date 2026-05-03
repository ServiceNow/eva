"""LaTeX writers for the paper's domain-collapsed accuracy and experience tables."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from paper_config import (
    ACCENT_PALETTE, TEAL_PALETTE, PINK_PALETTE,
    ARCH_ORDER, LIGHT_TEXT_THRESHOLD_INDEX,
    ModelEntry, PaperConfig, sort_models,
)

MISSING_CELL = "--"


def _is_missing(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def format_cell(point: Optional[float], ci_lower: Optional[float], ci_upper: Optional[float]) -> str:
    """Render a pooled cell as ``"<point> ±<halfwidth>"`` or ``"--"`` if missing.

    halfwidth = max(point - ci_lower, ci_upper - point) so asymmetric
    percentile CIs aren't undersold.
    """
    if _is_missing(point) or _is_missing(ci_lower) or _is_missing(ci_upper):
        return MISSING_CELL
    half = max(point - ci_lower, ci_upper - point)
    return f"{point:.3f} $\\pm${half:.3f}"


def shade_index(value: float, lo: float, hi: float, n_steps: int) -> int:
    """Return the palette bucket index in [0, n_steps-1] for ``value``."""
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    idx = int(frac * n_steps)
    if idx >= n_steps:
        idx = n_steps - 1
    return idx


def lookup_pooled(
    pooled_df: pd.DataFrame,
    model_label: str,
    metric: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (point, ci_lower, ci_upper) for one (model, metric) pooled row.

    Returns (None, None, None) when the row is absent.
    """
    mask = (
        (pooled_df["domain"] == "pooled")
        & (pooled_df["model_label"] == model_label)
        & (pooled_df["metric"] == metric)
    )
    sub = pooled_df.loc[mask]
    if sub.empty:
        return (None, None, None)
    row = sub.iloc[0]
    return (float(row["point_estimate"]), float(row["ci_lower"]), float(row["ci_upper"]))
