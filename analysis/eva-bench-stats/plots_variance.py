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


def _composite_sort_key(label: str) -> tuple:
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
    fig = px.histogram(
        df,
        x="std",
        facet_row="run_label",
        facet_col="metric",
        color="run_label",
        opacity=0.85,
        labels={"std": x_label, "run_label": "Model(s)"},
        title=title,
        facet_row_spacing=0.15,
        height=650,
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
        key=lambda c: _composite_sort_key(clean_composite_label(c)),
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

    _color_map = color_map or {}

    eva_a_labels = sorted(
        [lb for lb in melt_df["label"].unique() if lb.startswith("EVA-A")],
        key=_composite_sort_key,
    )
    eva_x_labels = sorted(
        [lb for lb in melt_df["label"].unique() if lb.startswith("EVA-X")],
        key=_composite_sort_key,
    )
    eva_a_short = [lb.split(" ", 1)[-1] for lb in eva_a_labels]
    eva_x_short = [lb.split(" ", 1)[-1] for lb in eva_x_labels]

    summary_df = (
        melt_df.groupby(["run_label", "label", "composite_col"])["value"].agg(mean="mean", std="std").reset_index()
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
            x="label",
            y="mean",
            error_y="std",
            color="run_label",
            category_orders={"label": full_labels, "run_label": label_order or []},
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
