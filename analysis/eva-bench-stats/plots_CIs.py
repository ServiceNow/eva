"""Plots and display tables for CI analysis."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PRETTY_METRIC_LABELS: dict[str, str] = {
    "task_completion": "Task Completion",
    "faithfulness": "Faithfulness",
    "agent_speech_fidelity": "Agent Speech Fidelity",
    "conversation_progression": "Conversation Progression",
    "turn_taking": "Turn Taking",
    "conciseness": "Conciseness",
    "EVA-A_mean": "EVA-A (mean)",
    "EVA-X_mean": "EVA-X (mean)",
    "EVA-A_pass": "EVA-A (pass@1)",
    "EVA-X_pass": "EVA-X (pass@1)",
}

DEFAULT_METRIC_ORDER: list[str] = [
    "EVA-A_pass",
    "EVA-X_pass",
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "turn_taking",
    "conciseness",
    "EVA-A_mean",
    "EVA-X_mean",
]


def metric_label(metric: str) -> str:
    return PRETTY_METRIC_LABELS.get(metric, metric)


def order_metrics(metrics: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(DEFAULT_METRIC_ORDER)}
    return sorted(metrics, key=lambda m: rank.get(m, 1_000))


def forest_plot(
    per_domain_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    metric: str,
    domains: list[str],
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Forest plot for one metric: 4 columns (3 domains + pooled), rows = models."""
    facets = list(domains) + ["pooled"]
    fig = make_subplots(rows=1, cols=len(facets), shared_yaxes=True, subplot_titles=facets)

    combined = pd.concat([
        per_domain_df[per_domain_df["metric"] == metric],
        pooled_df[pooled_df["metric"] == metric],
    ], ignore_index=True)

    if combined.empty:
        fig.update_layout(title=f"{metric_label(metric)} (no data)")
        return fig

    models = list(dict.fromkeys(combined["model_label"]))
    color_map = color_map or {}

    for col, facet in enumerate(facets, start=1):
        sub = combined[combined["domain"] == facet]
        for model in models:
            row = sub[sub["model_label"] == model]
            if row.empty:
                continue
            point = float(row["point_estimate"].iloc[0])
            lo = float(row["ci_lower"].iloc[0])
            hi = float(row["ci_upper"].iloc[0])
            color = color_map.get(model, "#3B82F6")
            fig.add_trace(
                go.Scatter(
                    x=[point],
                    y=[model],
                    mode="markers",
                    marker={"color": color, "size": 9},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [hi - point],
                        "arrayminus": [point - lo],
                        "color": color,
                        "thickness": 1.5,
                        "width": 4,
                    },
                    name=model,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=f"{metric_label(metric)} — 95% bootstrap CI",
        height=max(280, 28 * max(len(models), 4) + 80),
        margin={"l": 200, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "v"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def paper_summary_table(
    per_domain_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("EVA-A_mean", "EVA-X_mean"),
    domains: tuple[str, ...] = ("airline", "itsm", "medical_hr"),
) -> pd.DataFrame:
    """Wide table: one row per (model, metric); columns = each domain + pooled.

    Each cell is "{point:.3f} [{lo:.3f}, {hi:.3f}]".
    """
    parts = []
    rows = pd.concat([per_domain_df, pooled_df], ignore_index=True)
    for metric in metrics:
        for model, mdf in rows[rows["metric"] == metric].groupby("model_label"):
            row: dict[str, str] = {"model_label": model, "metric": metric_label(metric)}
            for facet in list(domains) + ["pooled"]:
                cell = mdf[mdf["domain"] == facet]
                if cell.empty:
                    row[facet] = ""
                    continue
                p = float(cell["point_estimate"].iloc[0])
                lo = float(cell["ci_lower"].iloc[0])
                hi = float(cell["ci_upper"].iloc[0])
                row[facet] = f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
            parts.append(row)
    return pd.DataFrame(parts)
