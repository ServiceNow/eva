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
ARCH_DISPLAY = {"cascade": "Cascade", "hybrid": "Hybrid (AudioLLM + TTS)", "s2s": "S2S"}


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
