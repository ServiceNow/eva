"""Pareto-frontier JSON writer with bootstrap CIs for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_config import PaperConfig
from paper_plots import build_scatter_points, pareto_frontier_indices


def _round(x: float, places: int = 6) -> float:
    return float(round(x, places))


def write_frontier_json(pooled_df: pd.DataFrame, cfg: PaperConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[dict]] = {}
    for variant_key, axes in cfg.scatter.items():
        points = build_scatter_points(pooled_df, cfg, axes["x"], axes["y"])
        if not points:
            payload[variant_key] = []
            continue
        xy = [(p.x, p.y) for p in points]
        frontier = sorted(
            (points[i] for i in pareto_frontier_indices(xy)),
            key=lambda p: p.x,
        )
        payload[variant_key] = [
            {
                "system": p.label,
                "system_type": p.arch,
                "eva_a": {
                    "point":   _round(p.x),
                    "ci_low":  _round(p.x_lo),
                    "ci_high": _round(p.x_hi),
                },
                "eva_x": {
                    "point":   _round(p.y),
                    "ci_low":  _round(p.y_lo),
                    "ci_high": _round(p.y_hi),
                },
            }
            for p in frontier
        ]
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
