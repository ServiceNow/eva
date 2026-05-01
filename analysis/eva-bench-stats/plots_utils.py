# analysis/eva-bench-stats/plots_utils.py
"""Shared plot utilities used by all plots_*.py modules."""

import math

import plotly.graph_objects as go


def fmt_p(p: float) -> str:
    """Format a p-value for display: '< 0.001', 'n/a', or '{p:.3f}'."""
    if math.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def empty_fig(message: str) -> go.Figure:
    """Blank figure with a centered text annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#888"},
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300)
    return fig
