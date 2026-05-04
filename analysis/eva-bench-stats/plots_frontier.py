"""Plots and display tables for frontier analysis.

All functions take DataFrames and return Plotly figures or display-ready DataFrames.
No Streamlit calls, no file I/O.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plots_utils import empty_fig as _empty_fig
from plots_utils import fmt_p as _fmt_p

# Okabe-Ito colorblind-safe palette
_TYPE_COLORS = {
    "cascade": "#0072B2",
    "s2s": "#E69F00",
}
_TYPE_DISPLAY = {
    "cascade": "Cascade",
    "s2s": "S2S",
}
_QR_DIRECTION_COLORS = {
    "EVA-A": "#009E73",
    "EVA-X": "#CC79A7",
}
_QR_DASH = {0.75: "dash", 0.90: "dot"}


def frontier_scatter_plot(
    scores_df: pd.DataFrame,
    qr_results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> go.Figure:
    """EVA-A vs EVA-X scatter with quantile regression frontier lines.

    EVA-X ~ EVA-A lines are direct: y = intercept + slope*x.
    EVA-A ~ EVA-X lines are inverted onto the same axes: y = (x - intercept) / slope,
    skipped if |slope| < 1e-6. Non-significant lines (p >= alpha) get 40% opacity
    and "(n.s.)" in their legend label.
    """
    if scores_df.empty:
        return _empty_fig("No model scores available -- run data_frontier.py first.")

    fig = go.Figure()

    for model_type in ["cascade", "s2s"]:
        group = scores_df[scores_df["model_type"] == model_type]
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["eva_a"],
            y=group["eva_x"],
            mode="markers+text",
            name=_TYPE_DISPLAY.get(model_type, model_type),
            marker={
                "color": _TYPE_COLORS.get(model_type, "#888888"),
                "size": 10,
                "line": {"width": 1, "color": "white"},
            },
            text=group["model_label"],
            textposition="top right",
            textfont={"size": 9, "color": "#333"},
        ))

    x_min = float(scores_df["eva_a"].min())
    x_max = float(scores_df["eva_a"].max())
    pad = (x_max - x_min) * 0.05
    x_range = np.linspace(x_min - pad, x_max + pad, 200)

    for _, row in qr_results_df.iterrows():
        q = float(row["quantile"])
        x_name = str(row["x_name"])
        slope = float(row["slope"])
        intercept = float(row["intercept"])
        p_val = float(row["p_value"])

        significant = not math.isnan(p_val) and p_val < alpha
        opacity = 1.0 if significant else 0.4
        sig_suffix = "" if significant else " (n.s.)"
        dash = _QR_DASH.get(q, "solid")

        if x_name == "EVA-A":
            y_line = intercept + slope * x_range
            color = _QR_DIRECTION_COLORS["EVA-A"]
            label = f"QR q={q:.2f} (EVA-X ~ EVA-A){sig_suffix}"
        else:
            if abs(slope) < 1e-6:
                print(f"  Skipping inverted QR line EVA-A ~ EVA-X at q={q}: slope ~ 0")
                continue
            y_line = (x_range - intercept) / slope
            color = _QR_DIRECTION_COLORS["EVA-X"]
            label = f"QR q={q:.2f} (EVA-A ~ EVA-X, inv.){sig_suffix}"

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_line,
            mode="lines",
            name=label,
            line={"dash": dash, "color": color, "width": 1.5},
            opacity=opacity,
        ))

    fig.update_layout(
        title={
            "text": "EVA-A vs EVA-X -- Frontier Analysis (Quantile Regression)",
            "font": {"size": 16},
            "x": 0,
            "xanchor": "left",
        },
        xaxis_title="EVA-A pass@1",
        yaxis_title="EVA-X pass@1",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"gridcolor": "#ececec", "zerolinecolor": "#bbb"},
        yaxis={"gridcolor": "#ececec", "zerolinecolor": "#bbb"},
        legend={"title": ""},
        margin={"t": 60, "b": 60, "l": 65, "r": 10},
        height=520,
    )
    return fig


def qr_results_table(qr_results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Format QR results as a display table for st.dataframe."""
    if qr_results_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in qr_results_df.iterrows():
        slope = float(r["slope"])
        ci_lo = float(r["slope_ci_lower"])
        ci_hi = float(r["slope_ci_upper"])
        p = float(r["p_value"])
        significant = not math.isnan(p) and p < alpha

        slope_str = (
            f"{slope:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]"
            if not math.isnan(slope)
            else "n/a"
        )

        rows.append({
            "Quantile": f"q={float(r['quantile']):.2f}",
            "Direction": f"{r['y_name']} ~ {r['x_name']}",
            "Slope [95% CI]": slope_str,
            "p-value": _fmt_p(p),
            "Significant": "✓" if significant else "",
        })

    return pd.DataFrame(rows)


def frontier_placeholder_fig() -> go.Figure:
    return _empty_fig("Frontier -- run data_frontier.py and stats_frontier.py to populate")
