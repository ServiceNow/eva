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
    return f"{point:.3f} {{\\scriptsize $\\pm${half:.3f}}}"


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


ARCH_DISPLAY = {"cascade": "Cascade", "hybrid": "Hybrid", "s2s": "S2S"}

# Sub-header macros for the aggregate sub-columns. Defined in the paper preamble.
AGG_KEY_MACROS = {
    "pass_at_1":    "\\passatone",
    "pass_at_k":    "\\passatk",
    "pass_power_k": "\\passpowerk",
}

# Equal-width centered column for every metric cell, with consistent inter-group
# spacing and a faint vrule between groups (matches the original aggregate tables).
_METRIC_COL = ">{\\centering\\arraybackslash}p{1.8cm}"
_SEP_OUTER = "@{\\hskip 8pt}"
_SEP_BETWEEN = "@{\\hskip 8pt}!{\\color{black!25}\\vrule}@{\\hskip 8pt}"
# Breathing room between the pass-rate sub-columns inside the aggregate group.
# 4pt + zero-width !{} marker + 4pt = 8pt total visible gap. The !{} marker is
# essential: \cellcolor bleeds through @{} separators, so without an !{} break
# adjacent cellcolored cells merge into one continuous color block (which is what
# made the aggregate cols look like they had no whitespace between them).
_SEP_INTRA_AGG = "@{\\hskip 4pt}!{\\color{white}\\vrule width 0pt}@{\\hskip 4pt}"


def _build_col_spec(n_agg: int, n_sub: int) -> str:
    """Identity (ll), then one aggregate group of n_agg cols, then n_sub one-col groups,
    with rule+spacing between groups and equal-width centered metric columns."""
    parts = ["ll", _SEP_OUTER]
    # Aggregate cols joined by a small hskip so the three pass-rate columns aren't
    # crammed together visually.
    parts.append(_SEP_INTRA_AGG.join([_METRIC_COL] * n_agg))
    for _ in range(n_sub):
        parts.append(_SEP_BETWEEN)
        parts.append(_METRIC_COL)
    return "".join(parts)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _shaded_cell(palette: list[tuple[str, str]], idx: int, body: str) -> str:
    name, _ = palette[idx]
    text_color = "white" if idx >= LIGHT_TEXT_THRESHOLD_INDEX else "black"
    return f"\\cellcolor{{{name}}}\\textcolor{{{text_color}}}{{{body}}}"


def _column_shading_bounds(
    pooled_df: pd.DataFrame,
    models: list[ModelEntry],
    metric: str,
) -> tuple[float, float]:
    """Min/max of point estimates for one metric across the configured models.

    Returns (0.0, 0.0) if no values are present so shade_index returns 0.
    """
    vals = []
    for m in models:
        p, _, _ = lookup_pooled(pooled_df, m.label, metric)
        if not _is_missing(p):
            vals.append(p)
    if not vals:
        return (0.0, 0.0)
    return (min(vals), max(vals))


def _shared_shading_bounds(
    pooled_df: pd.DataFrame,
    models: list[ModelEntry],
    metrics: list[str],
) -> tuple[float, float]:
    """Min/max across the union of point estimates from multiple metrics.

    Used so the pass-rate sub-columns under EVA-{A,X} share one shading scale
    (pass^k values are typically much smaller than pass@k, so independent
    per-column scales make their gradients incomparable).
    """
    vals: list[float] = []
    for metric in metrics:
        for m in models:
            p, _, _ = lookup_pooled(pooled_df, m.label, metric)
            if not _is_missing(p):
                vals.append(p)
    if not vals:
        return (0.0, 0.0)
    return (min(vals), max(vals))


def _write_table(
    pooled_df: pd.DataFrame,
    cfg: PaperConfig,
    out_path: Path,
    aggregate: dict[str, str],
    submetrics: dict[str, str],
    aggregate_palette: list[tuple[str, str]],
    submetric_palette: list[tuple[str, str]],
    aggregate_header: str,
    table_label: str,
    caption: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg_keys = list(aggregate.keys())          # e.g. pass_at_1, pass_at_k, pass_power_k
    agg_metrics = [aggregate[k] for k in agg_keys]
    sub_keys = list(submetrics.keys())
    sub_display = [submetrics[k] for k in sub_keys]

    # Shading bounds: aggregate pass-rate columns share one scale across the three
    # variants; submetric columns each get their own scale.
    models_list = list(cfg.models.values())
    agg_bounds = _shared_shading_bounds(pooled_df, models_list, agg_metrics)
    bounds: dict[str, tuple[float, float]] = {m: agg_bounds for m in agg_metrics}
    for m in sub_keys:
        bounds[m] = _column_shading_bounds(pooled_df, models_list, m)

    # Color macro definitions for the preamble
    palette_defs = []
    for name, hex_ in aggregate_palette + submetric_palette:
        palette_defs.append(f"\\definecolor{{{name}}}{{HTML}}{{{hex_}}}")

    n_agg = len(agg_keys)
    n_sub = len(sub_keys)
    col_spec = _build_col_spec(n_agg, n_sub)

    sorted_models = sort_models(cfg.models)

    # Group into arches preserving sort_models order.
    arch_groups: dict[str, list[ModelEntry]] = {}
    for m in sorted_models:
        arch_groups.setdefault(m.arch, []).append(m)

    lines: list[str] = []
    lines.append("% Auto-generated by analysis/eva-bench-stats/run_paper.py")
    lines.append("% Requires: xcolor, colortbl, booktabs, multirow, array")
    lines.extend(palette_defs)
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    # Header row 1: aggregate group label spans n_agg cols; each submetric gets its own header.
    top_cells = ["", "", f"\\multicolumn{{{n_agg}}}{{c}}{{\\textbf{{{aggregate_header}}}}}"]
    for name in sub_display:
        top_cells.append(f"\\textbf{{{_latex_escape(name)}}}")
    lines.append(" & ".join(top_cells) + " \\\\")

    # cmidrules: one for the aggregate group, one per submetric column. No (lr) trim
    # so each bar spans the natural column width and lines up with the cell shading
    # underneath.
    cmid_parts = [f"\\cmidrule{{3-{2 + n_agg}}}"]
    for i in range(n_sub):
        col = 3 + n_agg + i
        cmid_parts.append(f"\\cmidrule{{{col}-{col}}}")
    lines.append(" ".join(cmid_parts))

    # Header row 2: arch/system labels, the three pass-rate macros under EVA-{A,X},
    # and "Mean" under each submetric so the per-cell semantics are explicit.
    sub_label_cells = ["\\textbf{Arch.}", "\\textbf{System}"]
    for k in agg_keys:
        macro = AGG_KEY_MACROS.get(k, _latex_escape(k))
        sub_label_cells.append(f"\\textbf{{{macro}}}")
    for _ in sub_keys:
        sub_label_cells.append("\\textbf{Mean}")
    lines.append(" & ".join(sub_label_cells) + " \\\\")
    lines.append("\\midrule")

    for arch in ARCH_ORDER:
        members = arch_groups.get(arch, [])
        if not members:
            continue
        first = True
        for m in members:
            row_cells: list[str] = []
            arch_cell = f"\\multirow{{{len(members)}}}{{*}}{{{ARCH_DISPLAY[arch]}}}" if first else ""
            row_cells.append(arch_cell)
            row_cells.append(_latex_escape(m.label))

            for metric in agg_metrics:
                point, lo, hi = lookup_pooled(pooled_df, m.label, metric)
                body = format_cell(point, lo, hi)
                if body == MISSING_CELL or _is_missing(point):
                    row_cells.append(body)
                else:
                    bmin, bmax = bounds[metric]
                    idx = shade_index(point, bmin, bmax, n_steps=len(aggregate_palette))
                    row_cells.append(_shaded_cell(aggregate_palette, idx, body))
            for sub_key in sub_keys:
                point, lo, hi = lookup_pooled(pooled_df, m.label, sub_key)
                body = format_cell(point, lo, hi)
                if body == MISSING_CELL or _is_missing(point):
                    row_cells.append(body)
                else:
                    bmin, bmax = bounds[sub_key]
                    idx = shade_index(point, bmin, bmax, n_steps=len(submetric_palette))
                    row_cells.append(_shaded_cell(submetric_palette, idx, body))

            lines.append(" & ".join(row_cells) + " \\\\")
            first = False
        lines.append("\\midrule")

    # Replace trailing midrule with bottomrule.
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{6pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def write_accuracy_table(pooled_df: pd.DataFrame, cfg: PaperConfig, out_path: Path) -> None:
    _write_table(
        pooled_df, cfg, out_path,
        aggregate=cfg.accuracy_aggregate,
        submetrics=cfg.accuracy_submetrics,
        aggregate_palette=ACCENT_PALETTE,
        submetric_palette=TEAL_PALETTE,
        aggregate_header="EVA-A",
        table_label="tab:accuracy-metrics",
        caption=(
            "Accuracy metrics for all evaluated systems under clean-audio conditions, "
            "pooled equal-weighted across the three EVA domains. Each cell shows the "
            "pooled point estimate $\\pm$ the percentile bootstrap CI half-width "
            "($\\alpha = 0.05$). The three pass-rate columns share a single shading scale (so \\passatone vs.\\ \\passatk vs.\\ \\passpowerk are visually comparable); each submetric column is scaled independently. Darker = higher point estimate."
        ),
    )


def write_experience_table(pooled_df: pd.DataFrame, cfg: PaperConfig, out_path: Path) -> None:
    _write_table(
        pooled_df, cfg, out_path,
        aggregate=cfg.experience_aggregate,
        submetrics=cfg.experience_submetrics,
        aggregate_palette=ACCENT_PALETTE,
        submetric_palette=PINK_PALETTE,
        aggregate_header="EVA-X",
        table_label="tab:experience-metrics",
        caption=(
            "Experience metrics for all evaluated systems under clean-audio conditions, "
            "pooled equal-weighted across the three EVA domains. Each cell shows the "
            "pooled point estimate $\\pm$ the percentile bootstrap CI half-width "
            "($\\alpha = 0.05$). The three pass-rate columns share a single shading scale (so \\passatone vs.\\ \\passatk vs.\\ \\passpowerk are visually comparable); each submetric column is scaled independently. Darker = higher point estimate."
        ),
    )
