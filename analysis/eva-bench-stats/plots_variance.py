# analysis/eva-bench-stats/plots_variance.py
"""Plots for variance analysis.

Pure functions: take DataFrames, return go.Figure. No Streamlit, no file I/O.
"""

from itertools import combinations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plots_utils import empty_fig

_axis_style = {"title_font": {"color": "#222"}, "tickfont": {"color": "#222"}}
_disc_viridis = px.colors.sequential.Viridis
_COMPOSITE_GROUP_ORDER = {"EVA-A": 0, "EVA-X": 1, "EVA-overall": 2}
_COMPOSITE_STAT_ORDER = {"pass@1": 0, "pass@k": 1, "pass^k": 2, "mean": 3}


_VARIANCE_BUDGET_COLORS = {
    # Plotly Vivid qualitative palette — high saturation, spread across hue wheel
    "Domain": "#DAA51B",  # vivid amber-gold (variance budget tab only)
    "Judge": "#ED645A",  # vivid coral-red  (variance budget tab only)
    "Scenario": "#5D69B1",  # vivid blue-purple
    "Trial": "#E58606",  # vivid orange
    "Residual": "#CC61B0",  # vivid magenta (generic)
    "Judge + model × scenario interaction": "#CC61B0",  # vivid magenta (pooled LMM residual)
    "Judge stochasticity": "#52BCA3",  # vivid teal (per-model LMM residual)
}


def _stacked_variance_pct_bar(
    df: pd.DataFrame,
    pot_cols: list[tuple[str, str]],
    title: str,
) -> go.Figure:
    """Shared core for stacked % variance bar charts.

    Parameters
    ----------
    df : pd.DataFrame
        One row per metric, already filtered/sorted by the caller.
    pot_cols : list[tuple[str, str]]
        (display_label, pct_column) pairs in stack order.
    title : str
        Figure title.
    """
    import math

    metrics = df["metric"].tolist()
    fig = go.Figure()
    for label, col in pot_cols:
        if col not in df.columns:
            continue
        y_vals = (df[col] * 100).tolist()
        fig.add_trace(
            go.Bar(
                name=label,
                x=metrics,
                y=y_vals,
                marker_color=_VARIANCE_BUDGET_COLORS.get(label, "#888"),
                hovertemplate="<b>%{x}</b><br>" + label + ": %{y:.1f}%<extra></extra>",
            )
        )

    annotations = []
    if "sigma2_total" in df.columns:
        for metric, total in zip(df["metric"], df["sigma2_total"]):
            if pd.isna(total) or total <= 0:
                continue
            annotations.append(
                {
                    "x": metric,
                    "y": 102,
                    "text": f"σ={math.sqrt(float(total)):.3f}",
                    "showarrow": False,
                    "font": {"size": 11, "color": "#222"},
                    "yanchor": "bottom",
                }
            )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Metric",
        yaxis_title="% of total variance",
        yaxis={"range": [0, 115], "ticksuffix": "%"},
        legend_title="Source",
        height=520,
        margin={"r": 40, "t": 80},
        annotations=annotations,
    )
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def variance_budget_stacked_bar(budget_df: pd.DataFrame, model_id: str) -> go.Figure:
    """Stacked bar (% of total): one bar per metric, segmented into the variance pots.

    σ_total (the standard deviation, sqrt of σ²_total) is shown as an annotation
    on top of each bar so absolute magnitude and breakdown are both visible.
    """
    sub = budget_df[budget_df["model_id"] == model_id]
    if sub.empty:
        return empty_fig(f"No variance budget rows for {model_id!r}.")

    pot_cols = [
        ("Domain", "pct_domain"),
        ("Scenario", "pct_scenario"),
        ("Trial", "pct_trial"),
        ("Judge", "pct_judge"),
    ]
    return _stacked_variance_pct_bar(sub, pot_cols, f"Variance breakdown (% of total) — {model_id}")


def variance_budget_absolute_bar(budget_df: pd.DataFrame, model_id: str) -> go.Figure:
    """Stacked bar (absolute σ²): one bar per metric, height = total variance.

    Variances are additive, so stacking σ² values is mathematically correct.
    Bar height instantly conveys whether a metric has a large or small
    absolute variance — i.e., whether the % breakdown is worth attention.
    """
    sub = budget_df[budget_df["model_id"] == model_id]
    if sub.empty:
        return empty_fig(f"No variance budget rows for {model_id!r}.")

    metrics = sub["metric"].tolist()
    fig = go.Figure()
    pot_cols = [
        ("Domain", "sigma2_domain"),
        ("Scenario", "sigma2_scenario"),
        ("Trial", "sigma2_trial"),
        ("Judge", "sigma2_judge"),
    ]
    for label, col in pot_cols:
        if col not in sub.columns:
            continue
        y_vals = sub[col].tolist()
        fig.add_trace(
            go.Bar(
                name=label,
                x=metrics,
                y=y_vals,
                marker_color=_VARIANCE_BUDGET_COLORS[label],
                hovertemplate="<b>%{x}</b><br>" + label + ": σ²=%{y:.5f}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title=f"Absolute variance (σ²) — {model_id}",
        xaxis_title="Metric",
        yaxis_title="σ² (variance)",
        legend_title="Source",
        height=480,
        margin={"r": 40},
    )
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def llm_name(run_label: str) -> str:
    """Extract the LLM name from a run label 'stt / llm / tts'."""
    parts = run_label.split(" / ")
    return parts[1].strip() if len(parts) >= 3 else run_label


def clean_composite_label(col: str) -> str:
    """Convert a composite column name to a short display label."""
    for suffix, display in [("_pass_at_1", " pass@1"), ("_pass_at_k", " pass@k"), ("_pass_power_k", " pass^k")]:
        if col.endswith(suffix):
            return col[: -len(suffix)].replace("_pass", "") + display
    if col.endswith("_mean"):
        return col[:-5] + " mean"
    return col


def composite_sort_key(label: str) -> tuple:
    for g, go_idx in _COMPOSITE_GROUP_ORDER.items():
        if label.startswith(g):
            stat = label[len(g) + 1 :]
            return (go_idx, _COMPOSITE_STAT_ORDER.get(stat, 99))
    return (99, 99)


def compact_letters(models: list[str], sig_pairs: set) -> dict[str, str]:
    """Assign compact letter display (CLD) for pairwise comparisons.

    Returns a dict mapping model → letter string.
    Models that share a letter are NOT significantly different from each other.
    """
    if not sig_pairs:
        return {}
    letter_chars = "abcdefghij"
    assigned: dict[str, set] = {m: {0} for m in models}
    next_idx = 1
    for m1, m2 in combinations(models, 2):
        if frozenset([m1, m2]) in sig_pairs:
            common = assigned[m1] & assigned[m2]
            for c in common:
                assigned[m2].discard(c)
            if not assigned[m2]:
                assigned[m2].add(next_idx)
                for other in models:
                    if other != m2 and frozenset([other, m2]) not in sig_pairs:
                        assigned[other].add(next_idx)
                next_idx += 1
    return {m: "".join(letter_chars[i] for i in sorted(assigned[m])) for m in models}


def variance_histogram(
    df: pd.DataFrame,
    x_label: str,
    title: str,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> go.Figure:
    """Faceted histogram with one row per model and one column per metric."""
    if df.empty:
        return empty_fig("No data available.")
    n_rows = df["run_label"].nunique()
    # Plotly requires vertical_spacing <= 1/(rows-1). Cap to 0.5/(rows-1) so each
    # facet row keeps a usable plotting band, with a minimum floor for small N.
    row_spacing = min(0.15, 0.5 / max(n_rows - 1, 1)) if n_rows > 1 else 0.15
    height = max(650, 80 * n_rows + 200)
    fig = px.histogram(
        df,
        x="std",
        facet_row="run_label",
        facet_col="metric",
        color="run_label",
        opacity=0.85,
        labels={"std": x_label, "run_label": "Model(s)"},
        title=title,
        facet_row_spacing=row_spacing,
        height=height,
        color_discrete_map=color_map or {},
        category_orders={"run_label": label_order} if label_order else {},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], textangle=0))
    fig.update_layout(showlegend=False, margin={"r": 220})
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def variance_bar_fig(
    summary_df: pd.DataFrame,
    kw_df: pd.DataFrame,
    pw_df: pd.DataFrame,
    y_label: str,
    y_max: float | None = None,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> go.Figure:
    """Grouped bar chart of mean variance per metric, with K-W asterisks and CLD letters."""
    if summary_df.empty:
        return empty_fig("No summary data available.")
    if y_max is None:
        y_max = (summary_df["mean_std"] + summary_df["std_of_std"].fillna(0)).max() * 1.35

    metric_order = sorted(summary_df["metric"].unique())
    _cat_orders: dict = {"metric": metric_order}
    if label_order:
        _cat_orders["run_label"] = label_order
    fig = px.bar(
        summary_df,
        x="metric",
        y="mean_std",
        color="run_label",
        barmode="group",
        error_y="std_of_std",
        labels={"mean_std": y_label, "metric": "Metric", "run_label": "Model(s)"},
        category_orders=_cat_orders,
        color_discrete_map=color_map or {},
    )
    fig.update_layout(yaxis_range=[0, y_max], legend_title_text="Model(s)")
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)

    # Compact letter display
    letters_by_metric: dict[str, dict[str, str]] = {}
    if not kw_df.empty and not pw_df.empty:
        for _, kw_row in kw_df.iterrows():
            metric = kw_row["metric"]
            if not kw_row["significant"]:
                letters_by_metric[metric] = {}
                continue
            pw_metric = pw_df[pw_df["metric"] == metric]
            models_for_metric = summary_df[summary_df["metric"] == metric]["run_label"].tolist()
            sig_pairs: set = {
                frozenset([r["model_1"], r["model_2"]]) for _, r in pw_metric[pw_metric["significant"]].iterrows()
            }
            letters_by_metric[metric] = compact_letters(models_for_metric, sig_pairs)

    if any(letters_by_metric.values()):
        cat_index = {m: i for i, m in enumerate(metric_order)}
        n_traces = len(fig.data)
        bar_w = 0.8 / n_traces
        y_pad = y_max * 0.03

        for trace_idx, trace in enumerate(fig.data):
            model = trace.name
            x_offset = (trace_idx - (n_traces - 1) / 2) * bar_w
            model_data = summary_df[summary_df["run_label"] == model]

            for metric, cat_i in cat_index.items():
                letter = letters_by_metric.get(metric, {}).get(model, "")
                if not letter:
                    continue
                row = model_data[model_data["metric"] == metric]
                if row.empty:
                    continue
                bar_top = float(row["mean_std"].iloc[0])
                err = float(row["std_of_std"].fillna(0).iloc[0])
                y_pos = bar_top + err + y_pad
                fig.add_annotation(
                    x=cat_i + x_offset,
                    y=y_pos,
                    text=letter,
                    showarrow=False,
                    font={"size": 12, "color": "#222"},
                    xanchor="center",
                    yanchor="bottom",
                )

    # K-W significance asterisks on x-axis tick labels
    if not kw_df.empty:
        tick_texts = []
        for m in metric_order:
            kw_row = kw_df[kw_df["metric"] == m]
            if kw_row.empty:
                tick_texts.append(m)
                continue
            p = float(kw_row.iloc[0]["p"])
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            tick_texts.append(f"{m}<br><b>{stars}</b>" if stars else m)
        fig.update_xaxes(tickvals=metric_order, ticktext=tick_texts)

    return fig


def threshold_crossings_instability_fig(
    crossings_df: pd.DataFrame,
    n_runs: int,
    pass_thresholds: dict[str, float],
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Horizontal bar chart: scenario instability ranking by total pass/fail flip count."""
    if crossings_df.empty:
        return empty_fig("No borderline scenarios found.")
    ranked = (
        crossings_df.groupby("record_id")
        .size()
        .reset_index(name="total_flips")
        .sort_values("total_flips", ascending=True)
    )
    flip_ceiling = len(pass_thresholds) * n_runs * 3  # 3 trials per run
    fig = px.bar(
        ranked,
        x="total_flips",
        y="record_id",
        orientation="h",
        labels={
            "total_flips": "Total pass/fail flips (all metrics × models × trials)",
            "record_id": "Scenario",
        },
        color="total_flips",
        color_continuous_scale=_disc_viridis,
        range_color=[0, flip_ceiling],
    )
    fig.update_layout(
        yaxis={"autorange": "reversed"},
        xaxis_range=[0, flip_ceiling],
        coloraxis_showscale=False,
        height=max(400, len(ranked) * 18 + 100),
        margin={"l": 140, "r": 40, "b": 60},
    )
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def composite_stability_fig(
    stability_df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> go.Figure:
    """Line plot showing composite metric stability across iterations per run."""
    if stability_df.empty:
        return empty_fig("No composite stability data available.")

    composite_cols = [
        c
        for c in stability_df.columns
        if c not in ("run_id", "run_label", "iteration")
        and any(c.endswith(s) for s in ("_pass_at_1", "_pass_at_k", "_pass_power_k", "_mean"))
    ]
    if not composite_cols:
        return empty_fig("No composite columns found in stability data.")

    eva_ax_cols = sorted(
        [c for c in composite_cols if not clean_composite_label(c).startswith("EVA-overall")],
        key=lambda c: composite_sort_key(clean_composite_label(c)),
    )
    if not eva_ax_cols:
        eva_ax_cols = composite_cols

    melt_df = stability_df.melt(
        id_vars=["run_id", "run_label", "iteration"],
        value_vars=eva_ax_cols,
        var_name="composite_col",
        value_name="value",
    ).dropna(subset=["value"])
    melt_df["label"] = melt_df["composite_col"].map(clean_composite_label)
    melt_df["x_label"] = melt_df["label"].str.split(" ", n=1).str[-1]

    _color_map = color_map or {}

    eva_a_labels = sorted(
        [lb for lb in melt_df["label"].unique() if lb.startswith("EVA-A")],
        key=composite_sort_key,
    )
    eva_x_labels = sorted(
        [lb for lb in melt_df["label"].unique() if lb.startswith("EVA-X")],
        key=composite_sort_key,
    )
    eva_a_short = [lb.split(" ", 1)[-1] for lb in eva_a_labels]
    eva_x_short = [lb.split(" ", 1)[-1] for lb in eva_x_labels]

    summary_df = (
        melt_df.groupby(["run_label", "label", "x_label", "composite_col"])["value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["<b>EVA-A</b>", "<b>EVA-X</b>"],
        shared_yaxes=True,
    )

    for col_idx, (full_labels, short_labels) in enumerate(
        [(eva_a_labels, eva_a_short), (eva_x_labels, eva_x_short)], start=1
    ):
        sum_sub = summary_df[summary_df["label"].isin(full_labels)]
        tmp = px.scatter(
            sum_sub,
            x="x_label",
            y="mean",
            error_y="std",
            color="run_label",
            category_orders={"x_label": short_labels, "run_label": label_order or []},
            color_discrete_map=_color_map,
        )
        for trace in tmp.data:
            trace.showlegend = col_idx == 1
            trace.marker.size = 6
            fig.add_trace(trace, row=1, col=col_idx)
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=short_labels,
            tickangle=20,
            title_text="",
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title="Composite metric stability across iterations (mean ± std)",
        height=450,
        yaxis_range=[0, 1.05],
        yaxis2_range=[0, 1.05],
        legend={"title": "", "orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
        margin={"r": 220, "b": 80},
    )
    fig.update_annotations(font_size=16)
    fig.update_yaxes(title_text="Value (mean ± std across iterations)")
    return fig


def icc_bar_centered_fig(icc_pc_df: pd.DataFrame) -> go.Figure:
    """Bar chart of pooled-centered ICC per metric with 95% CI error bars."""
    if icc_pc_df.empty:
        return empty_fig("No pooled-centered ICC data available.")
    pc_plot = icc_pc_df.copy()
    pc_plot["error_lower"] = pc_plot["icc"] - pc_plot["ci_lower"]
    pc_plot["error_upper"] = pc_plot["ci_upper"] - pc_plot["icc"]
    fig = px.bar(
        pc_plot,
        x="metric",
        y="icc",
        error_y="error_upper",
        error_y_minus="error_lower",
        labels={"icc": "ICC (scenario)", "metric": "Metric"},
        category_orders={"metric": sorted(pc_plot["metric"].unique())},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(yaxis_range=[0, 1], height=350)
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def icc_bar_twoway_fig(icc_tw_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of two-way ICC (scenario + model components) with CI error bars."""
    if icc_tw_df.empty:
        return empty_fig("No two-way ICC data available.")
    tw_melt = icc_tw_df.melt(
        id_vars="metric",
        value_vars=["icc_scenario", "icc_model"],
        var_name="component",
        value_name="icc",
    )
    tw_melt["component"] = tw_melt["component"].map({"icc_scenario": "ICC scenario", "icc_model": "ICC model"})
    err_lo = icc_tw_df.set_index("metric")[["ci_lower_scenario", "ci_lower_model"]]
    err_hi = icc_tw_df.set_index("metric")[["ci_upper_scenario", "ci_upper_model"]]
    tw_melt["ci_lower"] = tw_melt.apply(
        lambda row: (
            err_lo.loc[row["metric"], "ci_lower_scenario"]
            if row["component"] == "ICC scenario"
            else err_lo.loc[row["metric"], "ci_lower_model"]
        ),
        axis=1,
    )
    tw_melt["ci_upper"] = tw_melt.apply(
        lambda row: (
            err_hi.loc[row["metric"], "ci_upper_scenario"]
            if row["component"] == "ICC scenario"
            else err_hi.loc[row["metric"], "ci_upper_model"]
        ),
        axis=1,
    )
    tw_melt["error_lower"] = tw_melt["icc"] - tw_melt["ci_lower"]
    tw_melt["error_upper"] = tw_melt["ci_upper"] - tw_melt["icc"]
    fig = px.bar(
        tw_melt,
        x="metric",
        y="icc",
        color="component",
        barmode="group",
        error_y="error_upper",
        error_y_minus="error_lower",
        labels={"icc": "ICC", "metric": "Metric", "component": ""},
        category_orders={"metric": sorted(tw_melt["metric"].unique())},
        color_discrete_map={"ICC scenario": "#636EFA", "ICC model": "#EF553B"},
    )
    fig.update_layout(yaxis_range=[0, 1], height=350)
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def icc_heatmap_fig(
    icc_pm_df: pd.DataFrame,
    label_order: list[str] | None = None,
) -> go.Figure:
    """Heatmap of per-model ICC values (metrics × models)."""
    if icc_pm_df.empty:
        return empty_fig("No per-model ICC data available.")
    pm_pivot = icc_pm_df.pivot(index="metric", columns="run_label", values="icc")
    if label_order:
        hm_cols = [c for c in label_order if c in pm_pivot.columns]
        if hm_cols:
            pm_pivot = pm_pivot[hm_cols]
    fig = px.imshow(
        pm_pivot,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        labels={"color": "ICC", "x": "Model", "y": "Metric"},
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_layout(
        height=max(250, len(pm_pivot) * 60 + 80),
        coloraxis_colorbar={"title": "ICC"},
        margin={"l": 200, "r": 40, "t": 30, "b": 80},
    )
    fig.update_xaxes(tickangle=30)
    return fig


def icc_bar_per_model_fig(
    icc_pm_df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> go.Figure:
    """Grouped bar chart of per-model ICC per metric with CI error bars."""
    if icc_pm_df.empty:
        return empty_fig("No per-model ICC data available.")
    pm_plot = icc_pm_df.copy()
    pm_plot["error_lower"] = pm_plot["icc"] - pm_plot["ci_lower"]
    pm_plot["error_upper"] = pm_plot["ci_upper"] - pm_plot["icc"]
    cat_orders: dict = {"metric": sorted(pm_plot["metric"].unique())}
    if label_order:
        cat_orders["run_label"] = label_order
    fig = px.bar(
        pm_plot,
        x="metric",
        y="icc",
        color="run_label",
        barmode="group",
        error_y="error_upper",
        error_y_minus="error_lower",
        labels={"icc": "ICC (scenario)", "metric": "Metric", "run_label": "Model"},
        color_discrete_map=color_map or {},
        category_orders=cat_orders,
    )
    fig.update_layout(
        yaxis_range=[0, 1],
        height=400,
        legend_title_text="Model",
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
        margin={"r": 220},
    )
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def deep_dive_scatter_fig(
    judge_var_df: pd.DataFrame,
    trial_var_df: pd.DataFrame,
    metric: str,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> go.Figure:
    """Scatter plot: judge std dev vs. trial std dev per (record, trial) for one metric."""
    if judge_var_df.empty or trial_var_df.empty:
        return empty_fig(f"No data for metric: {metric}")
    jv = judge_var_df[judge_var_df["metric"] == metric][["run_id", "run_label", "record_id", "trial", "std"]].rename(
        columns={"std": "judge_std"}
    )
    tv = trial_var_df[trial_var_df["metric"] == metric][["run_id", "run_label", "record_id", "std"]].rename(
        columns={"std": "trial_std"}
    )
    merged = pd.merge(jv, tv, on=["run_id", "run_label", "record_id"])
    if merged.empty:
        return empty_fig(f"No data for metric: {metric}")
    cat_orders: dict = {}
    if label_order:
        cat_orders["run_label"] = label_order
    fig = px.scatter(
        merged,
        x="trial_std",
        y="judge_std",
        color="run_label",
        hover_data=["record_id", "trial"],
        labels={"trial_std": "Trial std dev", "judge_std": "Judge std dev"},
        title=f"{metric}: judge vs. trial variance per (record, trial)",
        color_discrete_map=color_map or {},
        category_orders=cat_orders,
    )
    fig.add_hline(y=merged["judge_std"].mean(), line_dash="dash", line_color="gray", annotation_text="Mean judge std")
    fig.add_vline(x=merged["trial_std"].mean(), line_dash="dash", line_color="gray", annotation_text="Mean trial std")
    return fig


# ── LMM variance decomposition plots ──────────────────────────────────────────

_LMM_COMPONENT_ORDER = ["scenario", "trial", "judge", "residual"]
_LMM_COMPONENT_LABELS = {
    "scenario": "Scenario",
    "trial": "Trial",
    "judge": "Judge",
    "residual": "Residual",
}


def lmm_variance_stacked_bar(
    vc_df: pd.DataFrame,
    metric: str,
    title: str,
) -> go.Figure:
    """Stacked % bar for a single metric's LMM variance components.

    Components: Scenario, Trial, Judge (if present), Residual.
    Delegates to _stacked_variance_pct_bar after pivoting long→wide.

    Parameters
    ----------
    vc_df : pd.DataFrame
        lmm_variance_components DataFrame, long format (one row per component).
    metric : str
        Metric name to plot.
    title : str
        Figure title.
    """
    if vc_df.empty or "metric" not in vc_df.columns:
        return empty_fig(f"No LMM variance components for {metric!r}.")
    sub = vc_df[vc_df["metric"] == metric]
    if sub.empty:
        return empty_fig(f"No LMM variance components for {metric!r}.")

    wide_row: dict = {"metric": metric}
    for comp in _LMM_COMPONENT_ORDER:
        comp_rows = sub[sub["component"] == comp]
        if comp_rows.empty:
            continue
        wide_row[f"pct_{comp}"] = float(comp_rows["proportion"].iloc[0])
        wide_row[f"sigma2_{comp}"] = float(comp_rows["sigma2"].iloc[0])

    total = sub["total_variance"].iloc[0] if "total_variance" in sub.columns else float("nan")
    wide_row["sigma2_total"] = total
    wide_df = pd.DataFrame([wide_row])

    pot_cols = [(_LMM_COMPONENT_LABELS[c], f"pct_{c}") for c in _LMM_COMPONENT_ORDER if f"pct_{c}" in wide_df.columns]
    return _stacked_variance_pct_bar(wide_df, pot_cols, title)


def lmm_variance_stacked_bar_all(
    vc_df: pd.DataFrame,
    title: str,
    residual_label: str = "Residual",
    min_proportion: float = 0.001,
    metric_order: list[str] | None = None,
    judge_count: int = 0,
) -> go.Figure:
    """Stacked % bar for all metrics' LMM variance components on one chart.

    One bar per metric; segments are Scenario, Trial, Judge (if non-trivial), Residual.
    Components whose max proportion across all metrics is below min_proportion are omitted.
    If metric_order is given, metrics are displayed in that order. If judge_count > 0 and
    < len(metrics), a light gray dashed divider is drawn between the two groups.
    """
    if vc_df.empty or "metric" not in vc_df.columns:
        return empty_fig("No LMM variance components available.")
    all_metrics = sorted(vc_df["metric"].unique())
    if metric_order:
        all_metrics = [m for m in metric_order if m in set(all_metrics)]
    wide_rows = []
    for metric in all_metrics:
        sub = vc_df[vc_df["metric"] == metric]
        if sub.empty:
            continue
        row: dict = {"metric": metric}
        for comp in _LMM_COMPONENT_ORDER:
            comp_rows = sub[sub["component"] == comp]
            if comp_rows.empty:
                continue
            row[f"pct_{comp}"] = float(comp_rows["proportion"].iloc[0])
        total = sub["total_variance"].iloc[0] if "total_variance" in sub.columns else float("nan")
        row["sigma2_total"] = total
        wide_rows.append(row)
    if not wide_rows:
        return empty_fig("No LMM variance components available.")
    wide_df = pd.DataFrame(wide_rows)
    pot_cols = []
    for c in _LMM_COMPONENT_ORDER:
        col = f"pct_{c}"
        if col not in wide_df.columns or wide_df[col].max() < min_proportion:
            continue
        label = residual_label if c == "residual" else _LMM_COMPONENT_LABELS[c]
        pot_cols.append((label, col))
    fig = _stacked_variance_pct_bar(wide_df, pot_cols, title)
    fig.update_layout(legend_traceorder="reversed")
    n_metrics = len(wide_df)
    if 0 < judge_count < n_metrics:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=judge_count - 0.5,
            x1=judge_count - 0.5,
            y0=0,
            y1=1,
            line={"color": "lightgray", "width": 1.5, "dash": "dash"},
        )
        for x_pos, text in [(-0.45, "Judge-graded"), (judge_count - 0.45, "Deterministic")]:
            fig.add_annotation(
                x=x_pos,
                y=106,
                xref="x",
                yref="y",
                text=f"<i>{text}</i>",
                showarrow=False,
                font={"size": 10, "color": "gray"},
                xanchor="left",
                yanchor="bottom",
            )
    return fig


def lmm_variance_stacked_bar_grid(
    per_model_df: pd.DataFrame,
    metric: str,
    model_ids: list[str],
) -> go.Figure:
    """2×2 subplot grid of stacked % variance bars, one panel per model.

    Parameters
    ----------
    per_model_df : pd.DataFrame
        lmm_per_model_variance_components DataFrame, long format.
    metric : str
        Metric name to plot.
    model_ids : list[str]
        Ordered list of model IDs (determines subplot layout).
    """
    from plotly.subplots import make_subplots

    sub = per_model_df[per_model_df["metric"] == metric]
    if sub.empty:
        return empty_fig(f"No per-model LMM variance data for {metric!r}.")

    n = len(model_ids)
    n_cols = 2
    n_rows = (n + 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=model_ids,
        shared_yaxes=True,
    )

    legend_added: set[str] = set()
    for idx, model_id in enumerate(model_ids):
        row, col = divmod(idx, n_cols)
        model_sub = sub[sub["model_id"] == model_id]
        for comp in _LMM_COMPONENT_ORDER:
            comp_row = model_sub[model_sub["component"] == comp]
            if comp_row.empty:
                continue
            pct = float(comp_row["proportion"].iloc[0]) * 100
            label = _LMM_COMPONENT_LABELS[comp]
            show_legend = label not in legend_added
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=[metric],
                    y=[pct],
                    marker_color=_VARIANCE_BUDGET_COLORS.get(label, "#888"),
                    showlegend=show_legend,
                    legendgroup=label,
                    hovertemplate=f"<b>{model_id}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                ),
                row=row + 1,
                col=col + 1,
            )
            legend_added.add(label)

    fig.update_layout(
        barmode="stack",
        title_text=f"Variance decomposition per model — {metric}",
        height=420 * n_rows,
        legend_title="Source",
    )
    fig.update_yaxes(range=[0, 115], ticksuffix="%")
    return fig


def lmm_per_model_stacked_bar_rows(
    per_model_df: pd.DataFrame,
    model_ids: list[str],
    title: str = "Per-model variance decomposition — all metrics",
    residual_label: str = "Residual",
    min_proportion: float = 0.001,
    metric_order: list[str] | None = None,
    judge_count: int = 0,
) -> go.Figure:
    """Stacked % variance bars: one subplot row per model, all metrics on x-axis.

    Parameters
    ----------
    per_model_df : pd.DataFrame
        lmm_per_model_variance_components DataFrame, long format.
    model_ids : list[str]
        Ordered list of model IDs (one subplot row each).
    title : str
        Figure title.
    residual_label : str
        Display label for the residual component.
    min_proportion : float
        Components whose max proportion across all cells is below this threshold are omitted.
    metric_order : list[str] or None
        If given, display metrics in this order.
    judge_count : int
        If > 0 and < n_metrics, draw a dashed divider after this many metrics.
    """
    from plotly.subplots import make_subplots

    if per_model_df.empty or "model_id" not in per_model_df.columns:
        return empty_fig("No per-model LMM variance components available.")

    all_metrics = sorted(per_model_df["metric"].unique().tolist())
    if metric_order:
        all_metrics = [m for m in metric_order if m in set(all_metrics)]

    # Determine which components have any non-trivial proportion
    active_comps = []
    for comp in _LMM_COMPONENT_ORDER:
        max_pct = 0.0
        for model_id in model_ids:
            ms = per_model_df[per_model_df["model_id"] == model_id]
            for metric in all_metrics:
                cell = ms[(ms["metric"] == metric) & (ms["component"] == comp)]
                if not cell.empty:
                    max_pct = max(max_pct, float(cell["proportion"].iloc[0]))
        if max_pct >= min_proportion:
            active_comps.append(comp)

    n_rows = len(model_ids)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=model_ids,
        shared_xaxes=True,
        vertical_spacing=0.08 if n_rows <= 4 else 0.05,
    )

    seen: set[str] = set()
    for row_i, model_id in enumerate(model_ids, start=1):
        model_sub = per_model_df[per_model_df["model_id"] == model_id]
        for comp in active_comps:
            label = residual_label if comp == "residual" else _LMM_COMPONENT_LABELS[comp]
            color = _VARIANCE_BUDGET_COLORS.get(label, "#888")
            y_vals = []
            for metric in all_metrics:
                cell = model_sub[(model_sub["metric"] == metric) & (model_sub["component"] == comp)]
                y_vals.append(float(cell["proportion"].iloc[0]) * 100 if not cell.empty else 0.0)
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=all_metrics,
                    y=y_vals,
                    marker_color=color,
                    showlegend=label not in seen,
                    legendgroup=label,
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                ),
                row=row_i,
                col=1,
            )
            seen.add(label)

    fig.update_layout(
        barmode="stack",
        title=title,
        height=300 * n_rows + 80,
        legend_title="Source",
        legend_traceorder="reversed",
        margin={"r": 40, "t": 80},
    )
    fig.update_yaxes(range=[0, 115], ticksuffix="%")
    fig.update_xaxes(row=n_rows, col=1, **_axis_style)
    n_metrics = len(all_metrics)
    if 0 < judge_count < n_metrics:
        for row_i in range(1, n_rows + 1):
            xref = "x" if row_i == 1 else f"x{row_i}"
            fig.add_shape(
                type="line",
                xref=xref,
                yref="paper",
                x0=judge_count - 0.5,
                x1=judge_count - 0.5,
                y0=0,
                y1=1,
                line={"color": "lightgray", "width": 1.5, "dash": "dash"},
            )
        # Group labels in top subplot only (x-axis is shared across rows)
        for x_pos, text in [(-0.45, "Judge-graded"), (judge_count - 0.45, "Deterministic")]:
            fig.add_annotation(
                x=x_pos,
                y=106,
                xref="x",
                yref="y",
                text=f"<i>{text}</i>",
                showarrow=False,
                font={"size": 10, "color": "gray"},
                xanchor="left",
                yanchor="bottom",
            )
    return fig


def _lmm_absolute_bar_core(
    wide_df: pd.DataFrame,
    pot_cols: list[tuple[str, str]],
    title: str,
) -> go.Figure:
    """Core for absolute σ² stacked bar charts (LMM tab).

    Parameters
    ----------
    wide_df : pd.DataFrame
        One row per metric with sigma2_<comp> columns.
    pot_cols : list[tuple[str, str]]
        (display_label, column_name) pairs in stack order.
    title : str
        Figure title.
    """
    metrics = wide_df["metric"].tolist()
    fig = go.Figure()
    for label, col in pot_cols:
        if col not in wide_df.columns:
            continue
        y_vals = wide_df[col].tolist()
        fig.add_trace(
            go.Bar(
                name=label,
                x=metrics,
                y=y_vals,
                marker_color=_VARIANCE_BUDGET_COLORS.get(label, "#888"),
                hovertemplate="<b>%{x}</b><br>" + label + ": σ²=%{y:.5f}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Metric",
        yaxis_title="σ² (variance)",
        legend_title="Source",
        legend_traceorder="reversed",
        height=480,
        margin={"r": 40, "t": 80},
    )
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def lmm_variance_absolute_bar_all(
    vc_df: pd.DataFrame,
    title: str,
    residual_label: str = "Residual",
    min_proportion: float = 0.001,
    metric_order: list[str] | None = None,
    judge_count: int = 0,
    y_max: float | None = None,
) -> go.Figure:
    """Absolute σ² stacked bar for all metrics' LMM variance components on one chart."""
    if vc_df.empty or "metric" not in vc_df.columns:
        return empty_fig("No LMM variance components available.")
    all_metrics = sorted(vc_df["metric"].unique())
    if metric_order:
        all_metrics = [m for m in metric_order if m in set(all_metrics)]
    wide_rows = []
    for metric in all_metrics:
        sub = vc_df[vc_df["metric"] == metric]
        if sub.empty:
            continue
        row: dict = {"metric": metric}
        for comp in _LMM_COMPONENT_ORDER:
            comp_rows = sub[sub["component"] == comp]
            if comp_rows.empty:
                continue
            row[f"sigma2_{comp}"] = float(comp_rows["sigma2"].iloc[0])
        wide_rows.append(row)
    if not wide_rows:
        return empty_fig("No LMM variance components available.")
    wide_df = pd.DataFrame(wide_rows)
    pot_cols = []
    for c in _LMM_COMPONENT_ORDER:
        col = f"sigma2_{c}"
        if col not in wide_df.columns:
            continue
        comp_rows = vc_df[vc_df["component"] == c]
        if comp_rows.empty or comp_rows["proportion"].max() < min_proportion:
            continue
        label = residual_label if c == "residual" else _LMM_COMPONENT_LABELS[c]
        pot_cols.append((label, col))
    fig = _lmm_absolute_bar_core(wide_df, pot_cols, title)
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])
    n_metrics = len(wide_df)
    if 0 < judge_count < n_metrics:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=judge_count - 0.5,
            x1=judge_count - 0.5,
            y0=0,
            y1=1,
            line={"color": "lightgray", "width": 1.5, "dash": "dash"},
        )
        for x_pos, text in [(-0.45, "Judge-graded"), (judge_count - 0.45, "Deterministic")]:
            fig.add_annotation(
                x=x_pos,
                y=0.97,
                xref="x",
                yref="paper",
                text=f"<i>{text}</i>",
                showarrow=False,
                font={"size": 10, "color": "gray"},
                xanchor="left",
                yanchor="top",
            )
    return fig


def lmm_per_model_absolute_bar_rows(
    per_model_df: pd.DataFrame,
    model_ids: list[str],
    title: str = "Per-model absolute variance (σ²) — all metrics",
    residual_label: str = "Residual",
    min_proportion: float = 0.001,
    metric_order: list[str] | None = None,
    judge_count: int = 0,
    y_max: float | None = None,
) -> go.Figure:
    """Absolute σ² stacked bars: one subplot row per model, all metrics on x-axis."""
    from plotly.subplots import make_subplots

    if per_model_df.empty or "model_id" not in per_model_df.columns:
        return empty_fig("No per-model LMM variance components available.")

    all_metrics = sorted(per_model_df["metric"].unique().tolist())
    if metric_order:
        all_metrics = [m for m in metric_order if m in set(all_metrics)]

    active_comps = [
        comp
        for comp in _LMM_COMPONENT_ORDER
        if not per_model_df[per_model_df["component"] == comp].empty
        and per_model_df[per_model_df["component"] == comp]["proportion"].max() >= min_proportion
    ]

    n_rows = len(model_ids)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=model_ids,
        shared_xaxes=True,
        vertical_spacing=0.08 if n_rows <= 4 else 0.05,
    )

    seen: set[str] = set()
    for row_i, model_id in enumerate(model_ids, start=1):
        model_sub = per_model_df[per_model_df["model_id"] == model_id]
        for comp in active_comps:
            label = residual_label if comp == "residual" else _LMM_COMPONENT_LABELS[comp]
            color = _VARIANCE_BUDGET_COLORS.get(label, "#888")
            y_vals = []
            for metric in all_metrics:
                cell = model_sub[(model_sub["metric"] == metric) & (model_sub["component"] == comp)]
                y_vals.append(float(cell["sigma2"].iloc[0]) if not cell.empty else 0.0)
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=all_metrics,
                    y=y_vals,
                    marker_color=color,
                    showlegend=label not in seen,
                    legendgroup=label,
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: σ²=%{{y:.5f}}<extra></extra>",
                ),
                row=row_i,
                col=1,
            )
            seen.add(label)

    # Shared y-axis range
    if y_max is not None:
        y_range_max = y_max
    else:
        totals = per_model_df["total_variance"].dropna()
        y_range_max = float(totals.max()) * 1.15 if not totals.empty else 1.0

    fig.update_layout(
        barmode="stack",
        title=title,
        height=300 * n_rows + 80,
        legend_title="Source",
        legend_traceorder="reversed",
        margin={"r": 40, "t": 80},
    )
    fig.update_yaxes(tickformat=".4f", title_text="σ²", range=[0, y_range_max])
    fig.update_xaxes(row=n_rows, col=1, **_axis_style)

    n_metrics = len(all_metrics)
    if 0 < judge_count < n_metrics:
        for row_i in range(1, n_rows + 1):
            xref = "x" if row_i == 1 else f"x{row_i}"
            fig.add_shape(
                type="line",
                xref=xref,
                yref="paper",
                x0=judge_count - 0.5,
                x1=judge_count - 0.5,
                y0=0,
                y1=1,
                line={"color": "lightgray", "width": 1.5, "dash": "dash"},
            )
        for x_pos, text in [(-0.45, "Judge-graded"), (judge_count - 0.45, "Deterministic")]:
            fig.add_annotation(
                x=x_pos,
                y=0.97,
                xref="x",
                yref="paper",
                text=f"<i>{text}</i>",
                showarrow=False,
                font={"size": 10, "color": "gray"},
                xanchor="left",
                yanchor="top",
            )
    return fig


def lmm_forest_plot(
    fe_df: pd.DataFrame,
    metric: str,
    effect_type: str,
) -> go.Figure:
    """Forest plot of fixed effect coefficients for one metric.

    One point per level with horizontal CI bars and a vertical reference line at 0.
    Positive coefficient = above grand mean; negative = below grand mean
    (sum-to-zero coding).

    Parameters
    ----------
    fe_df : pd.DataFrame
        lmm_fixed_effects or lmm_per_model_fixed_effects DataFrame.
    metric : str
        Metric name to plot.
    effect_type : str
        ``"domain"`` or ``"model_id"`` — filters terms to this effect.
    """
    if fe_df.empty or "metric" not in fe_df.columns:
        return empty_fig(f"No fixed effects for {metric!r}.")
    sub = fe_df[fe_df["metric"] == metric]
    if sub.empty:
        return empty_fig(f"No fixed effects for {metric!r}.")

    # Keep only rows for the requested effect (patsy name pattern: C(<effect>, Sum)[S.<level>])
    pattern = f"C({effect_type}, Sum)[S."
    effect_rows = sub[sub["term"].str.startswith(pattern)].copy()
    if effect_rows.empty:
        return empty_fig(f"No {effect_type!r} fixed effect terms for {metric!r}.")

    # Extract level label from patsy term string
    effect_rows["level"] = effect_rows["term"].str.extract(r"\[S\.(.+)\]$")
    effect_rows = effect_rows.sort_values("coef")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=effect_rows["coef"],
            y=effect_rows["level"],
            mode="markers",
            marker={"size": 10, "color": "#1f77b4"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (effect_rows["ci_upper"] - effect_rows["coef"]).tolist(),
                "arrayminus": (effect_rows["coef"] - effect_rows["ci_lower"]).tolist(),
                "color": "#1f77b4",
                "thickness": 2,
            },
            hovertemplate="<b>%{y}</b><br>coef=%{x:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line={"color": "#888", "dash": "dash", "width": 1})

    effect_label = {"domain": "Domain", "model_id": "Model"}.get(effect_type, effect_type)
    fig.update_layout(
        title=f"{effect_label} fixed effects — {metric}",
        xaxis_title="Coefficient (deviation from grand mean)",
        yaxis_title=effect_label,
        height=max(300, 60 + len(effect_rows) * 40),
        margin={"l": 140, "r": 40, "t": 60},
    )
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    return fig


def lmm_component_boxplot(
    per_model_df: pd.DataFrame,
    metric_order: list[str] | None = None,
    residual_label: str = "Judge stochasticity",
) -> go.Figure:
    """Box + strip plot of % variance by component across models.

    x-axis = metric, one grouped box per component, individual points = model estimates.
    Deterministic metrics have no residual box (trial IS the residual there).

    Parameters
    ----------
    per_model_df : pd.DataFrame
        lmm_per_model_variance_components DataFrame, long format.
    metric_order : list[str] or None
        If given, display metrics in this order.
    residual_label : str
        Display label for the residual component.
    """
    df = per_model_df[per_model_df["component"] != "judge"].copy()
    df["proportion_pct"] = df["proportion"] * 100

    _label_map = {"scenario": "Scenario", "trial": "Trial", "residual": residual_label}
    df["comp_label"] = df["component"].map(_label_map)

    comp_order = ["Scenario", "Trial", residual_label]

    if metric_order:
        metrics = [m for m in metric_order if m in df["metric"].unique()]
    else:
        metrics = sorted(df["metric"].unique())

    _metric_display = {m: m.replace("_", " ") for m in metrics}

    fig = go.Figure()

    for comp in comp_order:
        cdf = df[df["comp_label"] == comp]
        if cdf.empty:
            continue
        color = _VARIANCE_BUDGET_COLORS.get(comp, "#888")

        x_vals, y_vals, hover_vals = [], [], []
        for m in metrics:
            mdf = cdf[cdf["metric"] == m]
            for _, row in mdf.iterrows():
                x_vals.append(_metric_display[m])
                y_vals.append(row["proportion_pct"])
                hover_vals.append(row.get("model_id", ""))

        fig.add_trace(
            go.Box(
                x=x_vals,
                y=y_vals,
                name=comp,
                marker_color=color,
                line_color=color,
                fillcolor=color,
                opacity=0.7,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                marker={"size": 7, "opacity": 1.0, "color": color, "line": {"width": 1, "color": "white"}},
                text=hover_vals,
                hovertemplate="<b>%{text}</b><br>%{y:.1f}%<extra>" + comp + "</extra>",
                legendgroup=comp,
            )
        )

    fig.update_layout(
        title="Variance components by metric — per-model estimates",
        boxmode="group",
        yaxis=dict(title="% of total variance", range=[0, 108], **_axis_style),
        xaxis=dict(tickangle=-30, **_axis_style),
        legend={"title": "Component", "orientation": "v"},
        height=480,
        margin={"b": 100, "t": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
