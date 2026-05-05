"""Plots and display tables for perturbation analysis.

All functions take DataFrames and return Plotly figures or display-ready DataFrames.
No Streamlit calls, no file I/O.

Tables pipeline: stats_perturbations computes exact values →
this module formats them for display (rounding, significance markers,
column names suitable for rendering).
"""

import pandas as pd
import plotly.graph_objects as go
from plots_utils import empty_fig as _empty_fig
from plots_utils import fmt_p as _fmt_p

_CONDITION_DISPLAY = {
    "accent": "Accent",
    "background_noise": "Background noise",
    "both": "Accent + Background noise",
}
_CONDITION_ORDER = ["Accent", "Background noise", "Accent + Background noise"]
# Okabe-Ito colorblind-safe palette
_COLORS = {
    "Accent": "#0072B2",
    "Background noise": "#E69F00",
    "Accent + Background noise": "#009E73",
}
_DOMAIN_DISPLAY = {
    "airline": "Airline",
    "itsm": "ITSM",
    "medical_hr": "Medical HR",
    "pooled": "Pooled (all domains)",
}
# 11-color qualitative palette for per-metric coloring in overview plots
# Fallback palette for metrics not in _METRIC_COLOR_MAP (e.g. aggregate EVA scores)
_METRIC_COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#44BB99",
    "#FFAABB",
    "#77AADD",
    "#99DDFF",
]
# Semantic color families for the per-model overview "group by condition" view.
# Warm (reds/oranges/pinks): EVA-A + its individual metrics.
# Cool (blues/greens/yellows): EVA-X + its individual metrics.
_METRIC_COLOR_MAP = {
    # Warm family: reds, dark → light
    "EVA-A_pass": "#881111",
    "task_completion": "#BB3322",
    "agent_speech_fidelity": "#DD6655",
    "faithfulness": "#FFAA99",
    # Cool family: blues, dark → light
    "EVA-X_pass": "#003388",
    "turn_taking": "#2266BB",
    "conciseness": "#5599CC",
    "conversation_progression": "#99CCEE",
}


def _ast_tier(corrected_p: float) -> str:
    """Return asterisk tier string for a corrected p-value. Only call when reject=True."""
    if corrected_p < 0.001:
        return "***"
    if corrected_p < 0.01:
        return "**"
    return "*"


def _prettify_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["perturbation_condition"] = df["perturbation_condition"].map(lambda x: _CONDITION_DISPLAY.get(x, x))
    return df


def perturbation_delta_plot(
    results_df: pd.DataFrame,
    metric: str,
    title: str | None = None,
    y_range: tuple[float, float] | None = None,
    model_order: list[str] | None = None,
    group_boundary: int | list[int] | None = None,
    group_labels: tuple[str, ...] | list[str] | None = None,
    cld_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Bar chart of mean deltas per model per perturbation condition with 95% CI error bars.

    Bars marked with * where the Holm-Bonferroni corrected p-value is significant.
    Asterisks are positioned precisely at CI tips using explicit numeric x coordinates.

    Args:
        results_df: Output of stats_perturbations.run_analysis (pre-filtered to one domain).
        metric: Which metric to plot.
        title: Plot title. Defaults to the metric name.
        y_range: (min, max) for the y-axis. Pass the same value across plots for comparability.
        model_order: If provided, sets x-axis order (models absent from df are skipped).
        group_boundary: If provided, draws a dashed vertical line after this many models
            (e.g. pass the number of cascade models to separate cascade from s2s).
        group_labels: If provided alongside group_boundary, small labels shown at the top-left
            of each section, e.g. ("Cascade", "S2S").
        cld_df: Optional CLD lookup DataFrame (output of stats_perturbations._build_cld_lookup).
            If provided, compact letter display annotations are added to the plot.
    """
    df = _prettify_conditions(results_df[results_df["metric"] == metric])
    if df.empty:
        return _empty_fig(f"No data for metric: {metric}")

    present = set(df["model_label"].unique())
    if model_order is not None:
        models = [m for m in model_order if m in present]
    else:
        models = sorted(present)
    conditions = [c for c in _CONDITION_ORDER if c in df["perturbation_condition"].unique()]
    n_models = len(models)
    n_conds = len(conditions)

    # Explicit numeric x positioning so asterisk scatter trace aligns with bar centers.
    # Models map to integer positions 0..n_models-1. Bars within each group are spaced
    # evenly across 80% of that unit, with a 20% inter-group gap.
    group_width = 0.8
    bar_w = group_width / n_conds

    # Build per-condition lookup
    cond_data: dict[str, dict] = {}
    for cond in conditions:
        cond_df = df[df["perturbation_condition"] == cond]
        cond_data[cond] = {r["model_label"]: r for _, r in cond_df.iterrows()}

    fig = go.Figure()

    for j, cond in enumerate(conditions):
        by_model = cond_data[cond]
        offset = (j - (n_conds - 1) / 2) * bar_w

        x_vals, y_vals, err_up, err_down = [], [], [], []
        for i, model in enumerate(models):
            row = by_model.get(model)
            x_vals.append(i + offset)
            if row is None:
                y_vals.append(None)
                err_up.append(0)
                err_down.append(0)
            else:
                y_vals.append(row["observed_mean_delta"])
                err_up.append(max(row["ci_upper"] - row["observed_mean_delta"], 0))
                err_down.append(max(row["observed_mean_delta"] - row["ci_lower"], 0))

        fig.add_trace(
            go.Bar(
                name=cond,
                x=x_vals,
                y=y_vals,
                width=bar_w * 0.92,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": err_up,
                    "arrayminus": err_down,
                    "visible": True,
                    "color": "#555",
                    "thickness": 1.5,
                    "width": 3,
                },
                marker_color=_COLORS.get(cond, "#888888"),
                marker_line_width=0,
            )
        )

    # Asterisks: scatter trace placed beyond CI tips with explicit clearance so they
    # are never obscured by error bar caps. Clearance = 6% of the visible y span.
    if y_range is not None:
        y_span = y_range[1] - y_range[0]
    else:
        all_ci = df[["ci_lower", "ci_upper"]].stack()
        y_span = max(all_ci.max() - all_ci.min(), 0.05)
    ast_clearance = y_span * 0.06

    sig_x, sig_y, sig_textpos, sig_text = [], [], [], []
    for j, cond in enumerate(conditions):
        by_model = cond_data[cond]
        offset = (j - (n_conds - 1) / 2) * bar_w
        for i, model in enumerate(models):
            row = by_model.get(model)
            if row is not None and row["reject"]:
                delta = row["observed_mean_delta"]
                sig_x.append(i + offset)
                sig_text.append(_ast_tier(row["corrected_p"]))
                if delta < 0:
                    sig_y.append(row["ci_lower"] - ast_clearance)
                    sig_textpos.append("middle center")
                else:
                    sig_y.append(row["ci_upper"] + ast_clearance)
                    sig_textpos.append("middle center")

    if sig_x:
        # Size to fit "***" (worst case) within the bar.
        # Estimate: figure plot area ≈ 830px (900px figure minus l=60, r=10 margins).
        # Each asterisk is ~0.55 × font_size px wide; cap between 8 and 13.
        pixel_bar_w = 830 / n_models * bar_w
        ast_font_size = max(8, min(13, int(pixel_bar_w / (3 * 0.55))))
        fig.add_trace(
            go.Scatter(
                x=sig_x,
                y=sig_y,
                mode="text",
                text=sig_text,
                textposition=sig_textpos,
                textfont={"size": ast_font_size, "color": "#111"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if cld_df is not None and not cld_df.empty:
        cld_pretty = _prettify_conditions(cld_df)
        cld_lookup = {
            (row["model_label"], row["perturbation_condition"]): row["cld_letter"] for _, row in cld_pretty.iterrows()
        }
        cld_clearance = y_span * 0.04
        cld_x, cld_y, cld_text = [], [], []
        for j, cond in enumerate(conditions):
            by_model = cond_data[cond]
            offset = (j - (n_conds - 1) / 2) * bar_w
            for i, model in enumerate(models):
                row = by_model.get(model)
                letter = cld_lookup.get((model, cond), "")
                if row is not None and letter:
                    delta = row["observed_mean_delta"]
                    if delta < 0:
                        y_ast = row["ci_lower"] - ast_clearance
                        cld_y.append(y_ast - cld_clearance)
                    else:
                        y_ast = row["ci_upper"] + ast_clearance
                        cld_y.append(y_ast + cld_clearance)
                    cld_x.append(i + offset)
                    cld_text.append(letter)
        if cld_x:
            fig.add_trace(
                go.Scatter(
                    x=cld_x,
                    y=cld_y,
                    mode="text",
                    text=cld_text,
                    textposition="middle center",
                    textfont={"size": 11, "color": "#666"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    boundaries = (
        [group_boundary]
        if isinstance(group_boundary, int)
        else (list(group_boundary) if group_boundary is not None else [])
    )
    boundaries = [b for b in boundaries if 0 < b < n_models]
    for b in boundaries:
        fig.add_shape(
            type="line",
            x0=b - 0.5,
            x1=b - 0.5,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": "#aaa", "width": 1.5, "dash": "dash"},
        )
    if boundaries and group_labels is not None:
        # Place one label per region. label_xs gives the left edge of each region.
        ann_y = y_range[1] if y_range is not None else 0.97
        ann_yref = "y" if y_range is not None else "paper"
        label_style = {
            "xref": "x",
            "yref": ann_yref,
            "y": ann_y,
            "yanchor": "top",
            "showarrow": False,
            "font": {"size": 12, "color": "#888"},
        }
        label_xs = [-0.5] + [b - 0.45 for b in boundaries]
        for x, label in zip(label_xs, group_labels):
            fig.add_annotation(x=x, xanchor="left", text=label, **label_style)

    yaxis: dict = {
        "zeroline": True,
        "zerolinecolor": "#bbb",
        "zerolinewidth": 1.5,
        "gridcolor": "#ececec",
        "dtick": 0.1,
        "tick0": 0,
    }
    if y_range is not None:
        yaxis["range"] = y_range

    fig.update_layout(
        title={"text": title or metric, "font": {"size": 17}, "x": 0, "xanchor": "left"},
        xaxis={
            "tickvals": list(range(n_models)),
            "ticktext": models,
            "tickangle": 35,
            "tickfont": {"size": 10},
        },
        yaxis_title="Mean Δ",
        barmode="overlay",
        legend_title="Condition",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=yaxis,
        margin={"t": 55, "b": 130, "l": 60, "r": 10},
        height=400,
    )
    return fig


def perturbation_overview_plot(
    results_df: pd.DataFrame,
    model: str,
    group_by: str = "metric",
    y_range: tuple[float, float] | None = None,
    metric_order: list[str] | None = None,
) -> go.Figure:
    """Per-model overview: mean deltas across all metrics × conditions (pooled data).

    Args:
        results_df: Pooled run_analysis output (all models, all metrics).
        model: Which model to show.
        group_by: "metric" → x=metrics, one bar per condition (3 colors).
                  "condition" → x=conditions, one bar per metric (many colors).
        y_range: Shared y-axis range. Pass the same value across model plots.
        metric_order: If provided, sets the metric display order (metrics absent from df are skipped).
    """
    df = _prettify_conditions(results_df[results_df["model_label"] == model])
    if df.empty:
        return _empty_fig(f"No data for model: {model}")

    conditions = [c for c in _CONDITION_ORDER if c in df["perturbation_condition"].unique()]
    available = df["metric"].unique()
    if metric_order is not None:
        metrics = [m for m in metric_order if m in available]
    else:
        metrics = sorted(available)

    fig = go.Figure()

    def _err(df_indexed, keys, col_up, col_down):
        up, down = [], []
        for k in keys:
            if k in df_indexed.index:
                r = df_indexed.loc[k]
                up.append(max(r[col_up], 0))
                down.append(max(r[col_down], 0))
            else:
                up.append(0)
                down.append(0)
        return up, down

    if group_by == "metric":
        for cond in conditions:
            cond_df = df[df["perturbation_condition"] == cond].set_index("metric")
            cond_df = cond_df.assign(
                err_up=cond_df["ci_upper"] - cond_df["observed_mean_delta"],
                err_down=cond_df["observed_mean_delta"] - cond_df["ci_lower"],
            )
            y_vals = [cond_df.loc[m, "observed_mean_delta"] if m in cond_df.index else None for m in metrics]
            err_up, err_down = _err(cond_df, metrics, "err_up", "err_down")
            fig.add_trace(
                go.Bar(
                    name=cond,
                    x=metrics,
                    y=y_vals,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": err_up,
                        "arrayminus": err_down,
                        "visible": True,
                        "color": "#555",
                        "thickness": 1.5,
                        "width": 3,
                    },
                    marker_color=_COLORS.get(cond, "#888888"),
                )
            )

    else:  # group_by == "condition"
        for k, metric in enumerate(metrics):
            metric_df = df[df["metric"] == metric].set_index("perturbation_condition")
            metric_df = metric_df.assign(
                err_up=metric_df["ci_upper"] - metric_df["observed_mean_delta"],
                err_down=metric_df["observed_mean_delta"] - metric_df["ci_lower"],
            )
            y_vals = [metric_df.loc[c, "observed_mean_delta"] if c in metric_df.index else None for c in conditions]
            err_up, err_down = _err(metric_df, conditions, "err_up", "err_down")
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=conditions,
                    y=y_vals,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": err_up,
                        "arrayminus": err_down,
                        "visible": True,
                        "color": "#555",
                        "thickness": 1.5,
                        "width": 3,
                    },
                    marker_color=_METRIC_COLOR_MAP.get(metric, _METRIC_COLORS[k % len(_METRIC_COLORS)]),
                )
            )

    yaxis: dict = {
        "zeroline": True,
        "zerolinecolor": "#bbb",
        "zerolinewidth": 1.5,
        "gridcolor": "#ececec",
    }
    if y_range is not None:
        yaxis["range"] = y_range

    fig.update_layout(
        title={"text": model, "font": {"size": 15}, "x": 0, "xanchor": "left"},
        xaxis={"tickangle": 35, "tickfont": {"size": 10}},
        yaxis_title="Mean Δ (pooled)",
        barmode="group",
        legend_title="Condition" if group_by == "metric" else "Metric",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=yaxis,
        margin={"t": 50, "b": 120, "l": 60, "r": 10},
        height=380,
    )
    return fig


def perturbation_results_table(
    results_df: pd.DataFrame,
    metric: str,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """Paper-style results table: mean delta [95% CI] with significance markers.

    One row per model, one column per condition. Cells formatted as
    "+0.12 [+0.07, +0.17]*" where * = corrected p < alpha.
    """
    df = _prettify_conditions(results_df[results_df["metric"] == metric])
    if df.empty:
        return pd.DataFrame()

    present = set(df["model_label"].unique())
    if model_order is not None:
        models = [m for m in model_order if m in present]
    else:
        models = sorted(present)
    conditions = [c for c in _CONDITION_ORDER if c in df["perturbation_condition"].unique()]

    def _fmt(row) -> str:
        d = f"{row['observed_mean_delta']:+.3f}"
        lo = f"{row['ci_lower']:+.3f}"
        hi = f"{row['ci_upper']:+.3f}"
        star = "*" if row["reject"] else ""
        return f"{d} [{lo}, {hi}]{star}"

    data: dict[str, list[str]] = {cond: [] for cond in conditions}
    for model in models:
        for cond in conditions:
            subset = df[(df["model_label"] == model) & (df["perturbation_condition"] == cond)]
            data[cond].append(_fmt(subset.iloc[0]) if not subset.empty else "—")

    result = pd.DataFrame(data, index=models)
    result.index.name = "Model"
    return result


_CONDITION_ABBREV = {
    "Accent": "Accent",
    "Background noise": "Background noise",
    "Accent + Background noise": "Both",
}

_PAIRWISE_DISPLAY = {
    "accent_vs_background_noise": "Accent vs Background noise",
    "accent_vs_both": "Accent vs Both",
    "background_noise_vs_both": "Background noise vs Both",
}
_PAIRWISE_COLORS = {
    "Accent vs Background noise": "#882255",
    "Accent vs Both": "#CC6677",
    "Background noise vs Both": "#DDCC77",
}
_PAIRWISE_ORDER = [
    "Accent vs Background noise",
    "Accent vs Both",
    "Background noise vs Both",
]


def perturbation_pvalue_table(
    results_df: pd.DataFrame,
    metric: str,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """P-value table: model × (corrected p-value, significance) for each condition.

    Columns are a MultiIndex: (condition_abbrev, "p") and (condition_abbrev, "✓").
    p-values < 0.001 display as "< 0.001"; significance is "✓" or "".
    """
    df = _prettify_conditions(results_df[results_df["metric"] == metric])
    if df.empty:
        return pd.DataFrame()

    present = set(df["model_label"].unique())
    if model_order is not None:
        models = [m for m in model_order if m in present]
    else:
        models = sorted(present)
    conditions = [c for c in _CONDITION_ORDER if c in df["perturbation_condition"].unique()]

    tuples = []
    for cond in conditions:
        abbrev = _CONDITION_ABBREV.get(cond, cond)
        tuples.extend([(abbrev, "p"), (abbrev, "✓")])
    col_index = pd.MultiIndex.from_tuples(tuples)

    records = []
    for model in models:
        row_data = {}
        for cond in conditions:
            abbrev = _CONDITION_ABBREV.get(cond, cond)
            subset = df[(df["model_label"] == model) & (df["perturbation_condition"] == cond)]
            if subset.empty:
                row_data[(abbrev, "p")] = "—"
                row_data[(abbrev, "✓")] = ""
            else:
                r = subset.iloc[0]
                row_data[(abbrev, "p")] = _fmt_p(r["corrected_p"])
                row_data[(abbrev, "✓")] = "✓" if r["reject"] else ""
        records.append(row_data)

    result = pd.DataFrame(records, index=models, columns=col_index)
    result.index.name = "Model"
    return result


def perturbation_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Count of models with significant effects (reject=True) per metric × condition."""
    if results_df.empty:
        return pd.DataFrame()

    df = _prettify_conditions(results_df)
    metrics = sorted(df["metric"].unique())
    conditions = [c for c in _CONDITION_ORDER if c in df["perturbation_condition"].unique()]

    data: dict[str, list[int]] = {cond: [] for cond in conditions}
    for metric in metrics:
        mdf = df[df["metric"] == metric]
        for cond in conditions:
            data[cond].append(int(mdf[mdf["perturbation_condition"] == cond]["reject"].sum()))

    result = pd.DataFrame(data, index=metrics)
    result.index.name = "Metric"
    return result


def data_coverage_table(completeness_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table of data coverage per model, condition, and domain."""
    if completeness_df.empty:
        return pd.DataFrame()

    models = sorted(completeness_df["model_label"].unique())
    raw_conditions = sorted(
        completeness_df["condition_label"].unique(),
        key=lambda c: (
            _CONDITION_ORDER.index(_CONDITION_DISPLAY.get(c, c))
            if _CONDITION_DISPLAY.get(c, c) in _CONDITION_ORDER
            else 99
        ),
    )
    domains = sorted(completeness_df["domain"].unique())
    n_domains = len(domains)

    records = []
    for model in models:
        mdf = completeness_df[completeness_df["model_label"] == model]
        included = "Yes" if mdf["model_complete"].iloc[0] else "No"
        row: dict = {"Included": included}
        for cond in raw_conditions:
            display_name = _CONDITION_DISPLAY.get(cond, cond)
            cdf = mdf[mdf["condition_label"] == cond]
            n_complete = int(cdf["complete"].sum())
            row[display_name] = f"{n_complete}/{n_domains}"
        records.append(row)

    result = pd.DataFrame(records, index=models)
    result.index.name = "Model"
    return result


def perturbation_pairwise_pvalue_table(
    pairwise_df: pd.DataFrame,
    metric: str,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """P-value table for pairwise comparisons (secondary H-B family)."""
    df = pairwise_df[pairwise_df["metric"] == metric].copy()
    if df.empty:
        return pd.DataFrame()

    df["comparison_display"] = df["comparison"].map(lambda x: _PAIRWISE_DISPLAY.get(x, x))

    present = set(df["model_label"].unique())
    models = [m for m in model_order if m in present] if model_order else sorted(present)
    comparisons = [c for c in _PAIRWISE_ORDER if c in df["comparison_display"].unique()]

    tuples = []
    for comp in comparisons:
        tuples.extend([(comp, "p"), (comp, "✓")])
    col_index = pd.MultiIndex.from_tuples(tuples)

    records = {}
    for model in models:
        row_data = {}
        for comp in comparisons:
            cell = df[(df["model_label"] == model) & (df["comparison_display"] == comp)]
            if cell.empty:
                row_data[(comp, "p")] = ""
                row_data[(comp, "✓")] = ""
            else:
                r = cell.iloc[0]
                row_data[(comp, "p")] = _fmt_p(r["corrected_p"])
                row_data[(comp, "✓")] = "✓" if r["reject"] else ""
        records[model] = row_data

    result = pd.DataFrame.from_dict(records, orient="index")
    result.columns = col_index
    result.index.name = "Model"
    return result


def perturbation_pairwise_results_table(
    pairwise_df: pd.DataFrame,
    metric: str,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """Mean delta [95% CI] table for pairwise comparisons."""
    df = pairwise_df[pairwise_df["metric"] == metric].copy()
    if df.empty:
        return pd.DataFrame()

    df["comparison_display"] = df["comparison"].map(lambda x: _PAIRWISE_DISPLAY.get(x, x))

    present = set(df["model_label"].unique())
    models = [m for m in model_order if m in present] if model_order else sorted(present)
    comparisons = [c for c in _PAIRWISE_ORDER if c in df["comparison_display"].unique()]

    def _fmt_cell(row) -> str:
        d = f"{row['observed_mean_delta']:+.3f}"
        lo = f"{row['ci_lower']:+.3f}"
        hi = f"{row['ci_upper']:+.3f}"
        ast = _ast_tier(row["corrected_p"]) if row["reject"] else ""
        return f"{d} [{lo}, {hi}]{ast}"

    records: dict[str, dict] = {}
    for model in models:
        row_data = {}
        for comp in comparisons:
            cell = df[(df["model_label"] == model) & (df["comparison_display"] == comp)]
            row_data[comp] = _fmt_cell(cell.iloc[0]) if not cell.empty else ""
        records[model] = row_data

    result = pd.DataFrame.from_dict(records, orient="index")
    result.columns = comparisons
    result.index.name = "Model"
    return result


def perturbation_additivity_table(
    additivity_df: pd.DataFrame,
    metric: str,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """Summary table for the additivity residual test."""
    df = additivity_df[additivity_df["metric"] == metric]
    if df.empty:
        return pd.DataFrame()

    present = set(df["model_label"].unique())
    models = [m for m in model_order if m in present] if model_order else sorted(present)

    rows = {}
    for model in models:
        cell = df[df["model_label"] == model]
        if cell.empty:
            continue
        r = cell.iloc[0]
        d = f"{r['observed_mean_delta']:+.3f}"
        lo = f"{r['ci_lower']:+.3f}"
        hi = f"{r['ci_upper']:+.3f}"
        if r["reject"] and r["observed_mean_delta"] > 0:
            direction = "synergistic (+)"
        elif r["reject"] and r["observed_mean_delta"] < 0:
            direction = "sub-additive (−)"
        else:
            direction = "—"
        rows[model] = {
            "Mean residual [95% CI]": f"{d} [{lo}, {hi}]",
            "p (uncorrected)": _fmt_p(r["raw_p"]),
            "Direction": direction,
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Model"
    return result


def perturbation_placeholder_fig() -> go.Figure:
    return _empty_fig("Perturbations — run data_perturbations.py to populate")
