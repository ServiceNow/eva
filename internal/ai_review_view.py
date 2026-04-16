"""AI Review Streamlit view — renders LLM-powered analysis of pipeline benchmarks.

Extracted from analysis_app.py for modularity. All rendering functions for the
AI Review tab live here; pipeline logic lives in ai_review_pipeline.py.
"""

import asyncio
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import dotenv_values
from plotly.subplots import make_subplots

from eva.utils import router
from internal.ai_review_html_report import generate_report_zip
from internal.ai_review_pipeline import (
    ALL_ANALYSIS_METRICS,
    DEFAULT_JUDGE_MODEL,
    collect_combo_data,
    display_name,
    extract_pipeline_config,
    format_metric_name,
    get_available_models,
    get_combo_label,
    get_display_combo_label,
    get_record_directories,
    get_valid_trial_dirs,
    load_run_config,
    normalize_entity_type,
    run_full_pipeline,
)

# --- Constants ---

SEVERITY_COLORS = {
    "high_impact": "#f44336",
    "cascading": "#ff9800",
    "common": "#2196f3",
}

IDENTIFIER_ERROR_CATEGORIES = {
    "wrong_tool_params",
    "fabricating_tool_parameters",
    "fabricated_params",
    "number_mangling",
    "name_mangling",
    "disambiguation_failure",
    "question_quality",
    "redundant_statements",
    "information_loss",
}

NOISE_CATEGORIES = {"unknown", "llm_error", "parse_error", "error"}

# (y_name, x_name, held_name, y_idx, x_idx, held_idx) — indices into (label, llm, stt, tts) tuple
COMPONENT_PAIRS = [
    ("LLM", "STT", "TTS", 1, 2, 3),
    ("LLM", "TTS", "STT", 1, 3, 2),
    ("STT", "TTS", "LLM", 2, 3, 1),
]

COMPONENT_AXES = [
    ("LLM", 1, lambda llm, stt, tts: f"STT={stt}, TTS={tts}"),
    ("STT", 2, lambda llm, stt, tts: f"LLM={llm}, TTS={tts}"),
    ("TTS", 3, lambda llm, stt, tts: f"LLM={llm}, STT={stt}"),
]


# --- Report ZIP caching ---


def _pipeline_result_hash(pipeline_result: dict) -> str:
    """Return a stable hash string for the pipeline result to use as a cache key."""
    raw = json.dumps(pipeline_result, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


@st.cache_data(show_spinner="Generating report…")
def _cached_report_zip(_hash: str, pipeline_result: dict, recipient: str) -> bytes:
    """Cache ZIP generation keyed on pipeline result hash + recipient."""
    return generate_report_zip(pipeline_result, recipient=recipient)


# --- Data helpers ---


def _normalize_severity(severity: Any) -> list[str]:
    """Normalize severity to a list of tags (handles both old string and new list format)."""
    if isinstance(severity, list):
        return severity
    if isinstance(severity, str):
        return [severity]
    return ["common"]


def _severity_badges(severity: Any) -> str:
    """Return colored markdown badges for one or more severity tags."""
    tags = _normalize_severity(severity)
    parts = []
    for tag in tags:
        color = SEVERITY_COLORS.get(tag, "#999")
        parts.append(
            f'<span style="background-color:{color}; color:white; padding:2px 8px; border-radius:4px; font-size:0.85em;">{tag}</span>'
        )
    return " ".join(parts)


def _get_assignment_categories(assignment: dict) -> list[str]:
    """Extract category list from a Pass 2 assignment dict."""
    cats = assignment.get("categories", ["unknown"])
    if isinstance(cats, str):
        cats = [cats]
    return cats


def _count_pass2_categories(
    pass2: dict,
    metric_names: list[str] | None = None,
    include_only: set[str] | None = None,
) -> Counter:
    """Count error categories across pass2 assignments.

    Args:
        pass2: The pass2 dict from a combo report (metric_name -> list of assignments).
        metric_names: If given, only count these metrics. Defaults to all.
        include_only: If given, only count categories in this set.
    """
    counter: Counter = Counter()
    for metric_name, assignments in pass2.items():
        if metric_names and metric_name not in metric_names:
            continue
        for assignment in assignments:
            for cat in _get_assignment_categories(assignment):
                if include_only and cat not in include_only:
                    continue
                if cat not in NOISE_CATEGORIES:
                    counter[cat] += 1
    return counter


def _dl(label: str) -> str:
    """Shorthand for display combo label."""
    return get_display_combo_label(label)


def _replace_raw_labels(text: str, combos: dict) -> str:
    """Replace raw combo labels in free text with display names."""
    for raw_label in combos:
        display_label = get_display_combo_label(raw_label)
        if display_label != raw_label:
            text = text.replace(raw_label, display_label)
    return text


def _build_combo_metric_df(combos: dict, metrics: list[str]) -> pd.DataFrame:
    """Build a DataFrame with one row per combo, columns = Pipeline + metric scores."""
    rows = []
    for label, report in combos.items():
        ms = report.get("metric_scores", {})
        row = {"Pipeline": _dl(label)}
        for m in metrics:
            row[format_metric_name(m)] = ms.get(m)
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_pipeline_configs(combos: dict) -> list[tuple[str, str, str, str]]:
    """Parse combos into (label, llm, stt, tts) tuples."""
    parsed: list[tuple[str, str, str, str]] = []
    for label, report in combos.items():
        pc = report.get("pipeline_config", {})
        parsed.append(
            (
                label,
                pc.get("llm_model", "?"),
                pc.get("stt_model", "?"),
                pc.get("tts_model", "?"),
            )
        )
    return parsed


# --- Chart helpers ---


def _subplot_grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    """Return (n_rows, n_cols) for a subplot grid."""
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols


def _build_heatmap_matrix(
    y_vals: list[str],
    x_vals: list[str],
    coord_to_label: dict[tuple[str, str], str],
    value_fn: Callable[[str], float | None],
    fmt: str = ".3f",
) -> tuple[list[list[Any]], list[list[str]]]:
    """Build z-values and text matrices for a heatmap."""
    z: list[list[Any]] = []
    text: list[list[str]] = []
    for y_val in y_vals:
        z_row: list[Any] = []
        text_row: list[str] = []
        for x_val in x_vals:
            label = coord_to_label.get((y_val, x_val))
            if label:
                val = value_fn(label)
                z_row.append(val)
                text_row.append(f"{val:{fmt}}" if val is not None else "—")
            else:
                z_row.append(None)
                text_row.append("—")
        z.append(z_row)
        text.append(text_row)
    return z, text


def _render_cross_component_grid(
    items: list[tuple[str, Callable[[str], float | None]]],
    y_name: str,
    x_name: str,
    held_name: str,
    held_val: str,
    coord_to_label: dict[tuple[str, str], str],
    y_vals: list[str],
    x_vals: list[str],
    colorscale: str = "RdYlGn",
    title_prefix: str = "Metric scores",
    zmin: float = 0,
    zmax: float = 1,
    fmt: str = ".3f",
) -> None:
    """Build and display a subplot figure with one heatmap per item.

    Used by both metric-score and EVA-composite cross-component heatmaps.
    """
    n_items = len(items)
    n_rows, n_cols = _subplot_grid(n_items)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[name for name, _ in items],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (item_name, getter) in enumerate(items):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        z, text = _build_heatmap_matrix(y_vals, x_vals, coord_to_label, getter, fmt=fmt)

        is_pass = "pass" in item_name.lower()
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_vals,
                y=y_vals,
                text=text,
                texttemplate="%{text}",
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=(idx == 0),
                hovertemplate=(
                    f"{y_name}: %{{y}}<br>{x_name}: %{{x}}<br>"
                    f"{'Pass rate' if is_pass else 'Score'}: %{{z:.3f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=x_name if row == n_rows else "", row=row, col=col, tickangle=-35)
        fig.update_yaxes(title_text=y_name if col == 1 else "", row=row, col=col)

    fig.update_layout(
        height=max(400, 250 * n_rows),
        title_text=f"{title_prefix}: {y_name} × {x_name} (held: {held_name}={held_val})",
    )
    st.plotly_chart(fig, use_container_width=True)


def _combo_metric_scatter(
    combos: dict,
    x: str,
    y: str,
    size: str | None = None,
    color: str = "Pipeline",
    title: str = "",
    text_labels: bool = False,
    height: int = 450,
    reference_lines: dict | None = None,
) -> None:
    """Build a scatter plot of per-pipeline mean metric scores.

    Args:
        combos: Pipeline combo data dict.
        x: Metric key for x-axis.
        y: Metric key for y-axis.
        size: Optional metric key for bubble size.
        color: Column name for color encoding.
        title: Plot title.
        text_labels: Whether to show pipeline labels as text.
        height: Plot height in pixels.
        reference_lines: Dict with optional keys "hline", "vline", "diagonal" for reference lines.
    """
    rows = []
    for label, report in combos.items():
        ms = report.get("metric_scores", {})
        row: dict[str, Any] = {
            "Pipeline": _dl(label),
            format_metric_name(x): ms.get(x),
            format_metric_name(y): ms.get(y),
        }
        if size:
            row[format_metric_name(size)] = ms.get(size)
        rows.append(row)

    if not rows:
        return

    scatter_kwargs: dict[str, Any] = {
        "x": format_metric_name(x),
        "y": format_metric_name(y),
        "color": color,
        "title": title,
    }
    if size:
        scatter_kwargs["size"] = format_metric_name(size)
    if text_labels:
        scatter_kwargs["text"] = "Pipeline"

    fig = px.scatter(pd.DataFrame(rows), **scatter_kwargs)
    fig.update_traces(marker={"sizemin": 8})
    if text_labels:
        fig.update_traces(textposition="top center")
    fig.update_layout(height=height)

    if reference_lines:
        if "hline" in reference_lines:
            fig.add_hline(y=reference_lines["hline"], line_dash="dash", line_color="gray", opacity=0.5)
        if "vline" in reference_lines:
            fig.add_vline(x=reference_lines["vline"], line_dash="dash", line_color="gray", opacity=0.5)
        if reference_lines.get("diagonal"):
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"color": "gray", "dash": "dash"})

    st.plotly_chart(fig, use_container_width=True)


def _ensure_router_initialized() -> None:
    """Initialize the LiteLLM router if not already running."""
    try:
        router.get()
    except RuntimeError:
        env_vals = {**dotenv_values(".env"), **os.environ}
        model_list_json = env_vals.get("EVA_MODEL_LIST", "[]")
        router.init(json.loads(model_list_json))


# --- Tab 1: Pipeline Overview ---


def _load_metrics_summary_for_combo(report: dict) -> dict:
    """Load metrics_summary.json for a combo report, trying report cache then disk."""
    ms = report.get("metrics_summary")
    if ms and ms.get("overall_scores"):
        return ms
    for rd_str in report.get("run_dirs", []):
        ms_path = Path(rd_str) / "metrics_summary.json"
        if ms_path.exists():
            try:
                with open(ms_path) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _render_eva_composites(combos: dict):
    """Show EVA-A, EVA-X, EVA-overall composites as grouped bar chart, plus pass@k."""
    composite_names = ["EVA-A_mean", "EVA-X_mean", "EVA-overall_mean", "EVA-A_pass", "EVA-X_pass", "EVA-overall_pass"]
    display_names = ["EVA-A mean", "EVA-X mean", "EVA-overall mean", "EVA-A pass", "EVA-X pass", "EVA-overall pass"]
    bar_rows = []
    has_data = False
    summaries: dict[str, dict] = {}
    for label, report in combos.items():
        ms = _load_metrics_summary_for_combo(report)
        summaries[label] = ms
        overall = ms.get("overall_scores", {})
        for comp, display in zip(composite_names, display_names):
            val = overall.get(comp, {}).get("mean")
            if val is not None:
                has_data = True
                bar_rows.append({"Configuration": label, "Composite": display, "Score": val})

    if has_data:
        fig = px.bar(
            pd.DataFrame(bar_rows),
            x="Composite",
            y="Score",
            color="Configuration",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=400, yaxis_range=[0, 1.05], yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        _render_pass_at_k_table_from_summaries(summaries)
    else:
        st.info("No official EVA composite scores found in metrics_summary.json files.")


def _render_pass_at_k_table_from_summaries(summaries: dict[str, dict]):
    """Show pass@1, pass@k, pass^k table if multi-trial data exists."""
    rows = []
    has_data = False
    for label, ms in summaries.items():
        pass_k = ms.get("overall_scores", {}).get("pass_k", {})
        if not pass_k:
            continue
        has_data = True
        row: dict = {"Configuration": label}
        for comp_name, pk in pass_k.items():
            k = pk.get("k", "?")
            row[f"{comp_name} pass@1"] = pk.get("pass_at_1")
            row[f"{comp_name} pass@{k}"] = pk.get("pass_at_k")
            row[f"{comp_name} pass^{k}"] = pk.get("pass_power_k_observed")
        rows.append(row)

    if has_data:
        st.markdown("**pass@k / pass^k** (multi-trial reliability)")
        df = pd.DataFrame(rows).set_index("Configuration")
        st.dataframe(df.style.format("{:.3f}", na_rep="—"), use_container_width=True)
        st.caption(
            "pass@1 = probability a single draw passes. "
            "pass@k = probability at least 1 of k draws passes (optimistic). "
            "pass^k = probability all k draws pass (strict consistency)."
        )


def _render_per_metric_means(combos: dict):
    """Show per-metric mean scores as grouped bar chart."""
    bar_rows = []
    for label, report in combos.items():
        ms = report.get("metric_scores", {})
        for m in ALL_ANALYSIS_METRICS:
            score = ms.get(m)
            if score is not None:
                bar_rows.append({"Configuration": label, "Metric": format_metric_name(m), "Score": score})

    if bar_rows:
        fig = px.bar(
            pd.DataFrame(bar_rows),
            x="Metric",
            y="Score",
            color="Configuration",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=450, yaxis_range=[0, 1.05], yaxis_title="Mean Normalized Score", xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Issue Analysis ---


def _render_issue_analysis_tab(pipeline_result: dict):
    """Render per-combo issue analysis with cross-metric synthesis."""
    combos = pipeline_result.get("combos", {})
    if not combos:
        st.info("No combo data available.")
        return

    combo_labels = list(combos.keys())

    def _format_combo_option(label: str) -> str:
        return get_display_combo_label(label)

    if len(combo_labels) > 1:
        selected_label = st.selectbox(
            "Select Pipeline Configuration",
            combo_labels,
            format_func=_format_combo_option,
        )
    else:
        selected_label = combo_labels[0]

    report = combos[selected_label]
    pass1 = report.get("pass1", {})
    pass3 = report.get("pass3", {})

    pc = report.get("pipeline_config", {})
    if pc:
        stt = pc.get("stt_model")
        tts = pc.get("tts_model")
        if stt == "n/a" and tts == "n/a":
            # Speech-to-speech: single end-to-end model
            st.markdown(f"**S2S:** {display_name(pc.get('llm_model', '?'), 'llm')}")
        elif stt == "n/a":
            # Audio-Native: audio LLM + TTS, no separate STT
            st.markdown(
                f"**Audio LLM:** {display_name(pc.get('llm_model', '?'), 'llm')} · "
                f"**TTS:** {display_name(tts or '?', 'tts')}"
            )
        else:
            st.markdown(
                f"**STT:** {display_name(stt or '?', 'stt')} · "
                f"**LLM:** {display_name(pc.get('llm_model', '?'), 'llm')} · "
                f"**TTS:** {display_name(tts or '?', 'tts')}"
            )
    else:
        st.markdown(f"## {_dl(selected_label)}")

    cross_combo = pipeline_result.get("cross_combo", {})
    dominant = cross_combo.get("per_combo_dominant_failures", {}).get(selected_label, [])
    if dominant:
        st.markdown("**Dominant failures:**")
        st.markdown("".join(f"- {f}\n" for f in dominant))

    _render_combo_summary_stats(pass1)
    _render_cross_metric_synthesis(pass3)
    _render_per_metric_issues(pass1, report.get("failures", {}))


def _render_combo_summary_stats(pass1: dict):
    """Show top-level failure/issue counts and per-metric stats."""
    all_sample_ids: set[str] = set()
    for metric_name, p in pass1.items():
        if metric_name == "turn_taking":
            # Turn-taking tracks affected samples in stats, not issues
            for sid in p.get("turn_taking_stats", {}).get("affected_sample_ids", []):
                all_sample_ids.add(str(sid))
        else:
            for issue in p.get("issues", []):
                for sid in issue.get("affected_samples", []):
                    all_sample_ids.add(str(sid))
    total_issues = sum(len(p.get("issues", [])) for m, p in pass1.items() if m != "turn_taking")
    metric_stats = []
    for m in ALL_ANALYSIS_METRICS:
        p = pass1.get(m, {})
        if m == "turn_taking":
            tt_stats = p.get("turn_taking_stats", {})
            pct = tt_stats.get("overall_pct", {})
            metric_stats.append(
                {
                    "Metric": format_metric_name(m),
                    "Samples with issues": p.get("num_failures", 0),
                    "Issues found": (f"{pct.get('Late', 0):.0f}% late, {pct.get('Early / Interrupting', 0):.0f}% early")
                    if pct
                    else "—",
                }
            )
        else:
            metric_stats.append(
                {
                    "Metric": format_metric_name(m),
                    "Samples with issues": p.get("num_failures", 0),
                    "Issues found": len(p.get("issues", [])),
                }
            )
    cols = st.columns(3)
    cols[0].metric("Samples with issues", len(all_sample_ids))
    cols[1].metric("Issues identified", total_issues)
    cols[2].metric("Metrics analyzed", len(ALL_ANALYSIS_METRICS))
    st.dataframe(pd.DataFrame(metric_stats).set_index("Metric"), use_container_width=True)


def _render_cross_metric_synthesis(pass3: dict):
    """Render critical issues from cross-metric synthesis."""
    critical = pass3.get("critical_issues", [])

    with st.expander(
        f"Critical issues — {len(critical)} identified",
        expanded=True,
    ):
        if not critical:
            st.info("No critical issues identified")
            return

        rows = [
            {
                "Issue": issue.get("title", "Untitled"),
                "Severity": ", ".join(_normalize_severity(issue.get("severity", "common"))),
                "Samples": issue.get("sample_count", 0),
                "Source Metrics": ", ".join(issue.get("source_metrics", [])),
            }
            for issue in critical
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")

        for idx, issue in enumerate(critical, 1):
            sev_badge = _severity_badges(issue.get("severity", "common"))
            sample_count = issue.get("sample_count", 0)

            st.markdown(
                f"**{idx}. {issue.get('title', 'Untitled')}** &nbsp;&nbsp; "
                f"{sev_badge} &nbsp;&nbsp; `{sample_count} samples`",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Description:** {issue.get('description', '')}")
            source = issue.get("source_metrics", [])
            merged = issue.get("merged_from", [])
            if source:
                st.caption(f"Source metrics: {', '.join(source)}")
            if merged:
                st.caption(f"Merged from: {', '.join(merged)}")
            st.markdown("")


def _render_turn_taking_stats(p1: dict):
    """Render turn-taking as a deterministic stats table instead of LLM issues."""
    stats = p1.get("turn_taking_stats", {})
    if not stats:
        st.info("No turn-taking data available")
        return

    n_samples = stats.get("total_samples", 0)
    n_issues = stats.get("samples_with_issues", 0)
    pct = stats.get("overall_pct", {})

    # Overall distribution
    st.markdown("#### Turn-Level Distribution")
    dist_rows = []
    for label in ("On-Time", "Late", "Early / Interrupting", "Other / Indeterminate"):
        count = stats.get("overall_counts", {}).get(label, 0)
        p = pct.get(label, 0)
        if count > 0:
            dist_rows.append({"Category": label, "Turns": count, "% of Turns": f"{p:.1f}%"})
    if dist_rows:
        st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

    # Conversation-level averages
    st.markdown("#### Conversation-Level Averages")
    cols = st.columns(2)
    cols[0].metric("Conversations", n_samples)
    cols[1].metric("With issues", n_issues)

    avg_cols = st.columns(3)
    avg_cols[0].metric("Avg % on-time", f"{stats.get('avg_pct_on_time', 0):.1f}%")
    avg_cols[1].metric("Avg % late", f"{stats.get('avg_pct_late', 0):.1f}%")
    avg_cols[2].metric("Avg % early", f"{stats.get('avg_pct_early', 0):.1f}%")

    # Per-sample breakdown (sortable table)
    per_sample = stats.get("per_sample", [])
    if per_sample:
        with st.expander(f"Per-conversation breakdown ({len(per_sample)} samples)", expanded=False):
            sample_df = pd.DataFrame(per_sample)
            display_cols = ["sample_id", "num_turns", "On-Time", "Late", "Early / Interrupting", "pct_on_time"]
            display_cols = [c for c in display_cols if c in sample_df.columns]
            sample_df = sample_df[display_cols].rename(
                columns={
                    "sample_id": "Sample",
                    "num_turns": "Turns",
                    "pct_on_time": "% On-Time",
                }
            )
            st.dataframe(sample_df.sort_values("% On-Time"), use_container_width=True, hide_index=True)


def _render_per_metric_issues(pass1: dict, failures_data: dict):
    """Render issue cards for each metric."""
    total_failures = sum(p.get("num_failures", 0) for p in pass1.values())
    st.markdown(f"**Per-Metric Issues** — {total_failures} failures across {len(ALL_ANALYSIS_METRICS)} metrics")

    for metric_name in ALL_ANALYSIS_METRICS:
        p1 = pass1.get(metric_name, {})
        issues = p1.get("issues", [])
        n_failures = p1.get("num_failures", 0)

        # Turn-taking uses deterministic stats, not LLM issues
        if metric_name == "turn_taking":
            with st.expander(
                f"{format_metric_name(metric_name)} — {n_failures} conversations with timing issues",
                expanded=False,
            ):
                _render_turn_taking_stats(p1)
            continue

        with st.expander(
            f"{format_metric_name(metric_name)} — {n_failures} failures, {len(issues)} issues", expanded=False
        ):
            if not issues:
                st.info("No issues identified")
                continue

            table_rows = []
            for issue in issues:
                row = {
                    "Issue": issue.get("title", "Untitled"),
                    "Sample Count": issue.get("sample_count", 0),
                    "Affected Sample IDs": ", ".join(str(s) for s in issue.get("affected_samples", [])),
                }
                hallucinated = issue.get("hallucinated_sample_ids", [])
                if hallucinated:
                    row["Issue"] += f" ⚠️ ({len(hallucinated)} hallucinated IDs removed)"
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            st.markdown("---")

            metric_failures = failures_data.get(metric_name, {})
            for idx, issue in enumerate(issues, 1):
                _render_issue_card(idx, issue, metric_failures)


def _render_issue_card(idx: int, issue: dict, metric_failures: dict):
    """Render a single issue card as an expander with examples inside."""
    title = issue.get("title", "Untitled")
    sample_count = issue.get("sample_count", 0)
    samples = issue.get("affected_samples", [])

    with st.expander(f"Issue {idx}: {title} — {sample_count} samples", expanded=False):
        st.markdown(f"**Description:** {issue.get('description', '')}")

        if issue.get("causal_chain"):
            st.info(f"**Causal chain:** {issue['causal_chain']}")

        if samples:
            st.caption(f"Samples: {', '.join(str(s) for s in samples)}")

        hallucinated = issue.get("hallucinated_sample_ids", [])
        if hallucinated:
            st.warning(
                f"**Judge hallucinated {len(hallucinated)} sample ID(s)** that don't exist in the data "
                f"and were removed: {', '.join(str(s) for s in hallucinated)}"
            )

        if samples and metric_failures:
            example_ids = [s for s in samples[:3] if str(s) in metric_failures]
            if example_ids:
                st.markdown("**Examples:**")
                for sid in example_ids:
                    failure = metric_failures[str(sid)]
                    score = failure.get("score", "?")
                    explanation = failure.get("explanation", "No details available")
                    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                    st.markdown(f"**Sample {sid}** (score: {score_str})")
                    st.markdown(
                        f'<div style="max-height:300px;overflow-y:auto;background:#f6f6f6;'
                        f"padding:10px;border-radius:5px;font-family:monospace;font-size:13px;"
                        f'white-space:pre-wrap;margin-bottom:10px;">{explanation}</div>',
                        unsafe_allow_html=True,
                    )


# --- Tab 3: Model Comparison ---


def _render_executive_summary(pipeline_result: dict, cross_combo: dict):
    """Render the high-level summary: key findings and best pipeline."""
    combos = pipeline_result.get("combos", {})
    pattern_summary = pipeline_result.get("executive_summary", {})
    key_findings = pattern_summary.get("key_findings", [])
    best = cross_combo.get("best_overall", "")
    reasoning = cross_combo.get("best_overall_reasoning", "")

    if not key_findings and not best:
        return

    st.markdown("## High-Level Summary")
    st.caption("Synthesized from all pipeline profiles, cross-pipeline comparison, and component-isolated analysis.")

    if best:
        best_display = _replace_raw_labels(best, combos)
        reasoning_display = _replace_raw_labels(reasoning, combos) if reasoning else ""
        st.success(
            f"**Best overall pipeline:** {best_display}" + (f" — {reasoning_display}" if reasoning_display else "")
        )

    if key_findings:
        for i, finding in enumerate(key_findings, 1):
            finding = _replace_raw_labels(finding, combos)
            if " | " in finding:
                title, description = finding.split(" | ", 1)
                st.markdown(f"**{i}. {title}**")
                st.markdown(description.strip())
            else:
                st.markdown(f"**{i}.** {finding}")


def _render_model_comparison_tab(pipeline_result: dict):
    """Render model comparison: summary → scores → failures → components → length → STT."""
    combos = pipeline_result.get("combos", {})
    cross_combo = pipeline_result.get("cross_combo", {})

    if not combos:
        st.info("No combo data available.")
        return

    per_sample_rows = pipeline_result.get("per_sample_data", [])
    df = pd.DataFrame(per_sample_rows) if per_sample_rows else pd.DataFrame()

    # ── 1. Executive Summary ──
    _render_executive_summary(pipeline_result, cross_combo)

    # ── 2. Pipeline Scores ──
    st.markdown("## Pipeline Scores")
    st.caption("Quantitative comparison of all pipeline configurations across EVA composites and individual metrics.")
    _render_metric_profiles_combined(combos)
    with st.expander("EVA composite scores", expanded=False):
        _render_eva_composites(combos)
    with st.expander("Per-metric mean scores", expanded=False):
        _render_per_metric_means(combos)
    with st.expander("Full score comparison heatmap", expanded=False):
        _render_score_comparison_chart(combos)

    # ── 3. What Goes Wrong ──
    st.markdown("## Failure Analysis")
    st.caption("Where and how pipelines fail — from LLM-identified patterns to data-driven co-failure detection.")
    with st.expander("Cross-pipeline comparison", expanded=True):
        st.caption(
            "LLM synthesis comparing all pipelines: shared failures, per-pipeline weaknesses, and structural insights."
        )
        _render_cross_combo_comparison(cross_combo, combos)

    with st.expander("Error category breakdown", expanded=False):
        _render_failure_mode_breakdown(combos)

    with st.expander("Faithfulness vs task completion", expanded=False):
        _plot_faithfulness_vs_task_completion_inline(combos)

    with st.expander("Turn-taking vs task completion", expanded=False):
        _plot_turn_taking_vs_task_completion_inline(combos)

    # ── 4. Component Analysis ──
    st.markdown("## Component Analysis")
    st.caption(
        "Isolates the effect of each pipeline component (LLM, STT, TTS) by comparing "
        "configurations that differ in only one component."
    )
    with st.expander("Component-isolated heatmaps & LLM analysis", expanded=False):
        _render_component_heatmaps(combos)
        _render_component_comparison_results(pipeline_result)

    with st.expander("Cross-component interaction — per-metric scores", expanded=False):
        _render_cross_component_heatmaps(combos)

    with st.expander("Cross-component interaction — EVA composites & pass rates", expanded=False):
        _render_cross_component_eva_heatmaps(combos)

    # ── 5. Conversation Length Effects ──
    st.markdown("## Conversation Length Effects")
    st.caption("How metric scores change as conversations get longer.")
    if not df.empty and "num_turns" in df.columns:
        _plot_per_pipeline_degradation(df, combos)
    else:
        st.info("No per-sample turn count data available.")

    # ── 6. STT & Entity Error Analysis ──
    st.markdown("## STT & Entity Error Analysis")
    st.caption(
        "How speech-to-text errors — especially on critical entities like names, codes, and amounts — "
        "cascade into downstream failures."
    )
    if not df.empty:
        _plot_entity_accuracy_worse_than_wer(df)
    _plot_entity_type_failure_heatmap(combos)
    with st.expander("Task completion error categories by pipeline", expanded=False):
        _plot_error_category_heatmap_inline(combos)
    with st.expander("Identifier/code capture errors across metrics", expanded=False):
        _plot_identifier_error_heatmap_inline(combos)
    _render_error_propagation(combos)


def _render_score_comparison_chart(combos: dict):
    """Render a grouped bar chart of all scores across pipelines."""
    if len(combos) < 2:
        return

    rows = []
    for label, report in combos.items():
        ms = report.get("metric_scores", {})
        row: dict[str, Any] = {"Pipeline": _dl(label)}
        row["EVA-A Mean"] = report.get("eva_a_mean")
        row["EVA-A Pass@1"] = report.get("eva_a_pass1")
        row["EVA-X Mean"] = report.get("eva_x_mean")
        row["EVA-X Pass@1"] = report.get("eva_x_pass1")
        for m in ALL_ANALYSIS_METRICS:
            row[format_metric_name(m)] = ms.get(m)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Pipeline")
    metrics = list(df.columns)
    fig = go.Figure()
    for pipeline in df.index:
        values = [df.loc[pipeline, m] for m in metrics]
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name=pipeline,
                text=[f"{v:.3f}" if v is not None else "" for v in values],
                textposition="outside",
                textfont={"size": 10},
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.12], "title": "Score", "dtick": 0.25},
        xaxis={"tickangle": -40},
        height=450,
        margin={"l": 50, "r": 30, "t": 10, "b": 120},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.35, "xanchor": "center", "x": 0.5},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cross_combo_comparison(cross_combo: dict, combos: dict):
    """Render structured cross-combo comparison with all LLM-generated outputs."""
    if not cross_combo:
        return

    # Key findings from cross-combo analysis
    key_findings = cross_combo.get("key_findings", [])
    if key_findings:
        st.markdown("**Key findings:**")
        for i, f in enumerate(key_findings, 1):
            st.markdown(f"**{i}.** {_replace_raw_labels(f, combos)}")

    # Shared failures
    shared = cross_combo.get("shared_failures", [])
    if shared:
        st.markdown("**Shared across all pipelines:**")
        for f in shared:
            st.markdown(f"- {f}")

    # Dominant failures per config
    per_combo = cross_combo.get("per_combo_dominant_failures", {})
    if per_combo:
        st.markdown("**Dominant failures per pipeline:**")
        rows = []
        for label, failures in per_combo.items():
            if isinstance(failures, list):
                for f in failures:
                    rows.append({"Pipeline": _dl(label), "Dominant failure": f})
            else:
                rows.append({"Pipeline": _dl(label), "Dominant failure": str(failures)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Structural insights
    structural = cross_combo.get("structural_insights", [])
    if structural:
        st.markdown("**Structural & behavioral insights:**")
        for insight in structural:
            affected = insight.get("affected_configs", [])
            affected_display = [_replace_raw_labels(a, combos) for a in affected]
            scope = "all configurations" if "all" in affected else ", ".join(affected_display)
            finding = _replace_raw_labels(insight.get("finding", ""), combos)
            with st.container(border=True):
                st.markdown(f"**{insight.get('pattern', '')}** — {finding}")
                st.caption(f"Affects: {scope}")


def _render_failure_mode_breakdown(combos: dict):
    """Render a pivot table: error categories x pipelines, with counts. Data-driven from Pass 2."""
    if len(combos) < 2:
        return

    all_cats: Counter = Counter()
    combo_cat_counts: dict[str, Counter] = {}
    for label, report in combos.items():
        c = _count_pass2_categories(report.get("pass2", {}))
        combo_cat_counts[label] = c
        all_cats.update(c)

    if not all_cats:
        return

    st.caption("Error category counts from Pass 2 across all pipelines")

    top_cats = [cat for cat, _ in all_cats.most_common(20)]
    bar_rows = []
    for cat in top_cats:
        for label in combos:
            count = combo_cat_counts.get(label, Counter()).get(cat, 0)
            if count > 0:
                bar_rows.append({"Error Category": cat, "Pipeline": _dl(label), "Count": count})

    if bar_rows:
        fig = px.bar(
            pd.DataFrame(bar_rows),
            x="Error Category",
            y="Count",
            color="Pipeline",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=500, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    _render_ancillary_breakdown(combos)


def _render_ancillary_breakdown(combos: dict):
    """Render missed vs hallucinated ancillary comparison across pipelines.

    'Missed' = agent failed to execute or preserve state (state_mutation_error, partial_completion, missing_tool_call).
    'Hallucinated' = agent fabricated or misrepresented results (misrepresenting_tool_result, fabricated_params,
                     fabricating_tool_parameters, hallucination).
    """
    missed_cats = {"state_mutation_error", "partial_completion", "missing_tool_call"}
    hallucinated_cats = {
        "misrepresenting_tool_result",
        "fabricated_params",
        "fabricating_tool_parameters",
        "hallucination",
    }

    rows = []
    for label, report in combos.items():
        pass2 = report.get("pass2", {})
        missed = _count_pass2_categories(pass2, include_only=missed_cats)
        hallucinated = _count_pass2_categories(pass2, include_only=hallucinated_cats)
        missed_total = sum(missed.values())
        hallucinated_total = sum(hallucinated.values())
        if missed_total > 0 or hallucinated_total > 0:
            row: dict = {"Pipeline": _dl(label)}
            row["MISSED (total)"] = missed_total
            for cat in sorted(missed_cats):
                row[f"  {cat}"] = missed.get(cat, 0)
            row["HALLUCINATED (total)"] = hallucinated_total
            for cat in sorted(hallucinated_cats):
                row[f"  {cat}"] = hallucinated.get(cat, 0)
            rows.append(row)

    if not rows:
        return

    st.markdown("#### Missed vs hallucinated ancillaries")
    st.caption(
        "How often does the agent miss actions vs fabricate results? Totals are the sum of the indented columns to their right."
    )
    df = pd.DataFrame(rows).set_index("Pipeline").fillna(0).astype(int)
    st.dataframe(df, use_container_width=True)


def _render_component_heatmaps(combos: dict):
    """Render component-isolated heatmaps: vary one component, hold others constant.

    Produces two types of heatmap per group:
    1. Metric scores heatmap (varying component x metrics)
    2. Error categories heatmap (varying component x top error categories)
    """
    if len(combos) < 2:
        st.info("Need multiple pipeline configurations for component analysis.")
        return

    st.caption(
        "Isolates the effect of each pipeline component (LLM, STT, TTS) by grouping "
        "configurations that share the same other two components. Heatmaps show metric "
        "scores and error category counts for the varying component."
    )

    parsed = _parse_pipeline_configs(combos)

    any_rendered = False
    for comp_name, comp_idx, get_held_label in COMPONENT_AXES:
        comp_section = comp_name.lower()
        groups: dict[str, list[tuple[str, str]]] = {}
        for label, llm, stt, tts in parsed:
            components = (llm, stt, tts)
            held = get_held_label(
                display_name(llm, "llm"),
                display_name(stt, "stt"),
                display_name(tts, "tts"),
            )
            varying = components[comp_idx - 1]
            groups.setdefault(held, []).append((label, varying))

        for held_key, members in groups.items():
            if len(members) < 2:
                continue

            any_rendered = True
            st.markdown(f"#### {comp_name} effect ({held_key})")

            score_rows = []
            for label, varying in members:
                ms = combos[label].get("metric_scores", {})
                row: dict[str, Any] = {comp_name: display_name(varying, comp_section)}
                for m in ALL_ANALYSIS_METRICS:
                    row[format_metric_name(m)] = ms.get(m)
                score_rows.append(row)

            score_df = pd.DataFrame(score_rows).set_index(comp_name)
            fig_scores = px.imshow(
                score_df,
                text_auto=".3f",
                color_continuous_scale="RdYlGn",
                zmin=0,
                zmax=1,
                labels={"color": "Score"},
            )
            fig_scores.update_layout(
                height=max(300, 100 * len(members)),
                title="Metric scores",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_scores, use_container_width=True)

            cat_data: dict[str, Counter] = {}
            all_cats: Counter = Counter()
            for label, varying in members:
                c = _count_pass2_categories(combos[label].get("pass2", {}))
                dv = display_name(varying, comp_section)
                cat_data[dv] = c
                all_cats.update(c)

            if all_cats:
                top_cats = [cat for cat, _ in all_cats.most_common(12)]
                cat_rows = []
                for _label, varying in members:
                    dv = display_name(varying, comp_section)
                    row_cat: dict[str, Any] = {comp_name: dv}
                    for cat in top_cats:
                        row_cat[cat] = cat_data.get(dv, Counter()).get(cat, 0)
                    cat_rows.append(row_cat)

                cat_df = pd.DataFrame(cat_rows).set_index(comp_name)
                fig_cats = px.imshow(
                    cat_df,
                    text_auto=True,
                    color_continuous_scale="Reds",
                    labels={"color": "Count"},
                )
                fig_cats.update_layout(
                    height=max(350, 100 * len(members)),
                    title="Error categories",
                    xaxis_tickangle=-35,
                )
                st.plotly_chart(fig_cats, use_container_width=True)

            st.markdown("---")

    if not any_rendered:
        st.info("No component groups with 2+ configurations found for comparison.")


def _render_component_comparison_results(pipeline_result: dict):
    """Render LLM component comparison results: which component value is best and why."""
    component_isolation = pipeline_result.get("component_isolation", {})
    if not component_isolation:
        return

    st.markdown("---")
    st.markdown("#### LLM Component Analysis")
    st.caption(
        "For each pipeline component (LLM, STT, TTS), pipelines sharing the same "
        "other two components are compared to isolate the effect of changing just "
        "that one component. Input: metric scores and error category distributions."
    )

    component_labels = {"llm_model": "LLM", "stt_model": "STT", "tts_model": "TTS"}
    for component, report in component_isolation.items():
        label = component_labels.get(component, component)
        best = report.get("best_component")
        reasoning = report.get("reasoning", "")
        summary = report.get("summary", "")
        patterns = report.get("component_specific_patterns", [])

        if summary and "No comparable groups" in summary:
            continue

        with st.container(border=True):
            if best:
                st.markdown(f"**{label} — Best: {best}**")
            if reasoning:
                st.markdown(reasoning)
            if summary:
                st.caption(summary)

            if patterns:
                for p in patterns:
                    val = p.get("component_value", "?")
                    unique = p.get("unique_patterns", [])
                    if unique:
                        st.markdown(f"*{val}:* {'; '.join(unique)}")


def _render_cross_component_heatmaps(combos: dict):
    """Render heatmaps showing interaction between two components with the third held constant.

    For each pair (LLM x STT, LLM x TTS, STT x TTS), when the third component
    has a single value across combos, renders a grid of heatmaps -- one per metric --
    with the two varying components on axes and mean metric score as cell value.
    """
    if len(combos) < 3:
        st.info("Need at least 3 pipeline configurations for cross-component analysis.")
        return

    parsed = _parse_pipeline_configs(combos)

    any_rendered = False
    for y_name, x_name, held_name, y_idx, x_idx, held_idx in COMPONENT_PAIRS:
        held_groups: dict[str, list[tuple[str, str, str]]] = {}
        for entry in parsed:
            label = entry[0]
            held_groups.setdefault(entry[held_idx], []).append((label, entry[y_idx], entry[x_idx]))

        y_section = y_name.lower()
        x_section = x_name.lower()
        held_section = held_name.lower()
        for held_val, members in held_groups.items():
            y_vals = sorted({display_name(m[1], y_section) for m in members})
            x_vals = sorted({display_name(m[2], x_section) for m in members})
            if len(y_vals) < 2 or len(x_vals) < 2:
                continue

            any_rendered = True
            st.markdown(f"#### {y_name} × {x_name} ({held_name}={display_name(held_val, held_section)})")

            coord_to_label: dict[tuple[str, str], str] = {}
            for label, y_val, x_val in members:
                coord_to_label[(display_name(y_val, y_section), display_name(x_val, x_section))] = label

            # Metric score heatmaps via shared grid renderer
            metric_items: list[tuple[str, Callable[[str], float | None]]] = [
                (format_metric_name(m), lambda lbl, metric=m: combos[lbl].get("metric_scores", {}).get(metric))
                for m in ALL_ANALYSIS_METRICS
            ]
            _render_cross_component_grid(
                items=metric_items,
                y_name=y_name,
                x_name=x_name,
                held_name=held_name,
                held_val=held_val,
                coord_to_label=coord_to_label,
                y_vals=y_vals,
                x_vals=x_vals,
                colorscale="RdYlGn",
                title_prefix="Metric scores",
            )

            # Error category cross-component heatmap (unique to this function)
            all_cats: Counter = Counter()
            coord_to_cats: dict[tuple[str, str], Counter] = {}
            for label, y_val, x_val in members:
                c = _count_pass2_categories(combos[label].get("pass2", {}))
                coord_to_cats[(display_name(y_val, y_section), display_name(x_val, x_section))] = c
                all_cats.update(c)

            if all_cats:
                top_cats = [cat for cat, _ in all_cats.most_common(6)]
                n_cat_rows, n_cat_cols = _subplot_grid(len(top_cats))

                fig_cats = make_subplots(
                    rows=n_cat_rows,
                    cols=n_cat_cols,
                    subplot_titles=top_cats,
                    horizontal_spacing=0.08,
                    vertical_spacing=0.12,
                )

                max_count = 0
                for cat in top_cats:
                    for coord_cats in coord_to_cats.values():
                        max_count = max(max_count, coord_cats.get(cat, 0))

                for cat_idx, cat in enumerate(top_cats):
                    crow = cat_idx // n_cat_cols + 1
                    ccol = cat_idx % n_cat_cols + 1

                    z: list[list[Any]] = []
                    text: list[list[str]] = []
                    for y_val in y_vals:
                        z_row: list[Any] = []
                        text_row: list[str] = []
                        for x_val in x_vals:
                            count = coord_to_cats.get((y_val, x_val), Counter()).get(cat, 0)
                            z_row.append(count)
                            text_row.append(str(count))
                        z.append(z_row)
                        text.append(text_row)

                    fig_cats.add_trace(
                        go.Heatmap(
                            z=z,
                            x=x_vals,
                            y=y_vals,
                            text=text,
                            texttemplate="%{text}",
                            colorscale="Reds",
                            zmin=0,
                            zmax=max_count,
                            showscale=(cat_idx == 0),
                            hovertemplate=f"{y_name}: %{{y}}<br>{x_name}: %{{x}}<br>Count: %{{z}}<extra></extra>",
                        ),
                        row=crow,
                        col=ccol,
                    )

                    fig_cats.update_xaxes(
                        title_text=x_name if crow == n_cat_rows else "", row=crow, col=ccol, tickangle=-35
                    )
                    fig_cats.update_yaxes(title_text=y_name if ccol == 1 else "", row=crow, col=ccol)

                fig_cats.update_layout(
                    height=max(400, 250 * n_cat_rows),
                    title_text=f"Error categories: {y_name} × {x_name} (held: {held_name}={held_val})",
                )
                st.plotly_chart(fig_cats, use_container_width=True)

            st.markdown("---")

    if not any_rendered:
        st.info(
            "Cross-component heatmaps require at least 2 distinct values on both axes. "
            "Need more pipeline configurations with varied components."
        )


def _render_cross_component_eva_heatmaps(combos: dict):
    """Render cross-component heatmaps for EVA composite scores and pass rates.

    Shows EVA-A mean, EVA-X mean, EVA-overall mean, EVA-A pass, EVA-X pass,
    EVA-overall pass, and task_completion.
    """
    if len(combos) < 3:
        st.info("Need at least 3 pipeline configurations for cross-component analysis.")
        return

    summaries: dict[str, dict] = {}
    for label, report in combos.items():
        summaries[label] = _load_metrics_summary_for_combo(report)

    parsed = _parse_pipeline_configs(combos)

    eva_scores: list[tuple[str, Callable[[str], float | None]]] = [
        ("EVA-A mean", lambda label: combos[label].get("eva_a_mean", 0)),
        ("EVA-X mean", lambda label: combos[label].get("eva_x_mean", 0)),
        (
            "EVA-overall mean",
            lambda label: summaries[label].get("overall_scores", {}).get("EVA-overall_mean", {}).get("mean"),
        ),
        (
            "EVA-A pass",
            lambda label: summaries[label].get("overall_scores", {}).get("EVA-A_pass", {}).get("mean"),
        ),
        (
            "EVA-X pass",
            lambda label: summaries[label].get("overall_scores", {}).get("EVA-X_pass", {}).get("mean"),
        ),
        (
            "EVA-overall pass",
            lambda label: summaries[label].get("overall_scores", {}).get("EVA-overall_pass", {}).get("mean"),
        ),
        ("Task Completion", lambda label: combos[label].get("metric_scores", {}).get("task_completion")),
    ]

    any_rendered = False
    for y_name, x_name, held_name, y_idx, x_idx, held_idx in COMPONENT_PAIRS:
        held_groups: dict[str, list[tuple[str, str, str]]] = {}
        for entry in parsed:
            label = entry[0]
            held_groups.setdefault(entry[held_idx], []).append((label, entry[y_idx], entry[x_idx]))

        y_section = y_name.lower()
        x_section = x_name.lower()
        held_section = held_name.lower()
        for held_val, members in held_groups.items():
            y_vals = sorted({display_name(m[1], y_section) for m in members})
            x_vals = sorted({display_name(m[2], x_section) for m in members})
            if len(y_vals) < 2 or len(x_vals) < 2:
                continue

            any_rendered = True
            st.markdown(f"#### {y_name} × {x_name} ({held_name}={display_name(held_val, held_section)})")

            coord_to_label: dict[tuple[str, str], str] = {}
            for label, y_val, x_val in members:
                coord_to_label[(display_name(y_val, y_section), display_name(x_val, x_section))] = label

            _render_cross_component_grid(
                items=eva_scores,
                y_name=y_name,
                x_name=x_name,
                held_name=held_name,
                held_val=held_val,
                coord_to_label=coord_to_label,
                y_vals=y_vals,
                x_vals=x_vals,
                colorscale="RdYlGn",
                title_prefix="EVA scores",
            )
            st.markdown("---")

    if not any_rendered:
        st.info(
            "Cross-component heatmaps require at least 2 distinct values on both axes. "
            "Need more pipeline configurations with varied components."
        )


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        m = float(np.mean(values))
        return m, m
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def _plot_per_pipeline_degradation(df: pd.DataFrame, combos: dict):
    """Small-multiples line chart with quantile-based bins, bootstrap CIs, and sample size annotations."""
    st.caption(
        "Quantile-based bins on pooled turn distribution. Shaded bands = 95% bootstrap CI. "
        "Bins with n < 15 are annotated."
    )
    degrade_metrics = ALL_ANALYSIS_METRICS
    degrade_df = df.dropna(subset=["num_turns"]).copy()
    if degrade_df.empty:
        return

    min_n = 15
    n_quantile_bins = 5

    try:
        degrade_df["Turn Bin"], bin_edges = pd.qcut(
            degrade_df["num_turns"], q=n_quantile_bins, retbins=True, duplicates="drop"
        )
    except ValueError:
        return
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)]
    degrade_df["Turn Range"] = pd.cut(degrade_df["num_turns"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    pipeline_list = sorted(degrade_df["combo_label"].unique())
    n_pipelines = len(pipeline_list)
    if n_pipelines == 0:
        return

    sub_rows, sub_cols = _subplot_grid(n_pipelines)
    fig = make_subplots(
        rows=sub_rows,
        cols=sub_cols,
        subplot_titles=[_dl(p).split(" / ")[0] for p in pipeline_list],
        shared_yaxes=True,
    )
    colors = px.colors.qualitative.Set2

    sample_count_rows = []

    for idx, pipeline in enumerate(pipeline_list):
        p_df = degrade_df[degrade_df["combo_label"] == pipeline]
        r, c = idx // sub_cols + 1, idx % sub_cols + 1

        for lbl in bin_labels:
            bin_data = p_df[p_df["Turn Range"] == lbl]
            sample_count_rows.append({"Pipeline": _dl(pipeline), "Bin": lbl, "n": len(bin_data)})

        for m_idx, metric in enumerate(degrade_metrics):
            m_df = p_df.dropna(subset=[metric])
            if m_df.empty:
                continue

            means = []
            ci_lo = []
            ci_hi = []
            valid_labels = []
            for lbl in bin_labels:
                bin_data = m_df[m_df["Turn Range"] == lbl]
                if bin_data.empty:
                    continue
                vals = bin_data[metric].values
                valid_labels.append(lbl)
                means.append(float(np.mean(vals)))
                lo, hi = _bootstrap_ci(vals)
                ci_lo.append(lo)
                ci_hi.append(hi)

            if not valid_labels:
                continue

            color = colors[m_idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=valid_labels + valid_labels[::-1],
                    y=ci_hi + ci_lo[::-1],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r,
                col=c,
            )

            text_vals = []
            for lbl in valid_labels:
                n_in_bin = len(m_df[m_df["Turn Range"] == lbl])
                text_vals.append(f"n={n_in_bin}" if n_in_bin < min_n else "")

            fig.add_trace(
                go.Scatter(
                    x=valid_labels,
                    y=means,
                    mode="lines+markers+text",
                    text=text_vals,
                    textposition="top center",
                    textfont={"size": 9, "color": "red"},
                    name=format_metric_name(metric),
                    line={"color": color},
                    legendgroup=metric,
                    showlegend=(idx == 0),
                ),
                row=r,
                col=c,
            )

    fig.update_xaxes(type="category", categoryorder="array", categoryarray=bin_labels)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(
        height=350 * sub_rows, title_text="Per-pipeline: how each metric degrades as conversations get longer"
    )
    st.plotly_chart(fig, use_container_width=True)

    if sample_count_rows:
        count_df = pd.DataFrame(sample_count_rows).pivot_table(
            index="Pipeline", columns="Bin", values="n", aggfunc="sum", fill_value=0
        )
        count_df = count_df.reindex(columns=bin_labels)
        st.caption("Sample count per bin per pipeline")
        st.dataframe(count_df, use_container_width=True)


def _plot_error_category_heatmap_inline(combos: dict):
    """Heatmap of task_completion error categories by pipeline."""
    rows = []
    for label, report in combos.items():
        counter = _count_pass2_categories(report.get("pass2", {}), metric_names=["task_completion"])
        for cat, count in counter.items():
            rows.append({"Pipeline": _dl(label), "Error Category": cat, "Count": count})
    if rows:
        pivot = pd.DataFrame(rows).pivot_table(index="Pipeline", columns="Error Category", values="Count", fill_value=0)
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale="Reds", labels={"color": "Count"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _plot_identifier_error_heatmap_inline(combos: dict):
    """Heatmap of identifier-related error categories across all metrics."""
    st.caption("Categories linked to identifier corruption, verification loops, and fabricated parameters")
    rows = []
    for label, report in combos.items():
        pass2 = report.get("pass2", {})
        for metric_name, assignments in pass2.items():
            for assignment in assignments:
                for cat in _get_assignment_categories(assignment):
                    if cat in IDENTIFIER_ERROR_CATEGORIES:
                        rows.append(
                            {
                                "Pipeline": _dl(label),
                                "Metric": format_metric_name(metric_name),
                                "Category": cat,
                                "Count": 1,
                            }
                        )
    if rows:
        pivot = pd.DataFrame(rows).pivot_table(
            index="Pipeline", columns="Category", values="Count", aggfunc="sum", fill_value=0
        )
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale="Oranges", labels={"color": "Count"})
        fig.update_layout(height=max(350, 40 * len(pivot)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No identifier-related error categories found in Pass 2 data.")


def _plot_faithfulness_vs_task_completion_inline(combos: dict):
    """Scatter: faithfulness vs task completion per pipeline (mean scores)."""
    _combo_metric_scatter(
        combos,
        x="faithfulness",
        y="task_completion",
        size="conversation_progression",
        title="Low faithfulness co-occurs with low task completion",
    )


def _plot_turn_taking_vs_task_completion_inline(combos: dict):
    """Scatter with colored quadrant overlays describing each region."""
    rows = []
    for label, report in combos.items():
        ms = report.get("metric_scores", {})
        tt = ms.get("turn_taking")
        tc = ms.get("task_completion")
        if tt is not None and tc is not None:
            rows.append({"Pipeline": _dl(label), "Turn Taking": tt, "Task Completion": tc})
    if not rows:
        return

    fig = go.Figure()

    quadrants = [
        (0, 0.5, 0, 0.5, "rgba(239,83,80,0.10)", "Poor timing, incorrect"),
        (0.5, 1, 0, 0.5, "rgba(255,152,0,0.10)", "Good timing, incorrect"),
        (0, 0.5, 0.5, 1, "rgba(255,152,0,0.10)", "Poor timing, correct"),
        (0.5, 1, 0.5, 1, "rgba(76,175,80,0.10)", "Good timing, correct"),
    ]
    for x0, x1, y0, y1, color, _label in quadrants:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line={"width": 0}, layer="below")

    annotations = [
        (0.25, 0.25, "Poor timing\nIncorrect", "red"),
        (0.75, 0.25, "Good timing\nIncorrect", "darkorange"),
        (0.25, 0.75, "Poor timing\nCorrect", "darkorange"),
        (0.75, 0.75, "Good timing\nCorrect", "green"),
    ]
    for x, y, text, color in annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font={"size": 11, "color": color},
            opacity=0.5,
        )

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.4)

    scatter_df = pd.DataFrame(rows)
    colors_palette = px.colors.qualitative.Plotly
    for idx, (_, row) in enumerate(scatter_df.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[row["Turn Taking"]],
                y=[row["Task Completion"]],
                mode="markers+text",
                text=[row["Pipeline"]],
                textposition="top center",
                marker={"size": 14, "color": colors_palette[idx % len(colors_palette)]},
                name=row["Pipeline"],
                showlegend=True,
            )
        )

    fig.update_layout(
        height=550,
        title="Turn-taking quality vs task correctness",
        xaxis={"title": "Turn Taking", "range": [-0.02, 1.02]},
        yaxis={"title": "Task Completion", "range": [-0.02, 1.02]},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metric_profiles_combined(combos: dict):
    """Combined radar overlay showing all pipelines on one chart."""
    pipeline_labels = list(combos.keys())
    if len(pipeline_labels) < 2:
        return

    colors = px.colors.qualitative.Plotly
    metric_names = [format_metric_name(m) for m in ALL_ANALYSIS_METRICS]
    polar_axis = {"range": [0, 1], "showticklabels": True, "tickvals": [0, 0.25, 0.5, 0.75, 1.0]}

    fig_combined = go.Figure()
    for idx, label in enumerate(pipeline_labels):
        ms = combos[label].get("metric_scores", {})
        values = [ms.get(m, 0) for m in ALL_ANALYSIS_METRICS]
        values_closed = values + [values[0]]
        theta_closed = metric_names + [metric_names[0]]
        color = colors[idx % len(colors)]
        fig_combined.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=theta_closed,
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line={"color": color, "width": 2.5},
                name=_dl(label),
            )
        )
    fig_combined.update_layout(
        polar={"radialaxis": polar_axis},
        height=550,
        title="All pipelines — metric profile overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        margin={"l": 80, "r": 80, "t": 60, "b": 80},
    )
    st.plotly_chart(fig_combined, use_container_width=True)


def _render_error_propagation(combos: dict):
    """Render error propagation analysis with per-pipeline dropdowns."""
    has_data = any(report.get("error_propagation", {}).get("stats", {}).get("available") for report in combos.values())
    if not has_data:
        return

    st.markdown("### Error Propagation")
    st.caption(
        "How STT entity transcription errors cascade into downstream metric failures. "
        "Compares samples with entity errors vs clean samples."
    )

    available_combos = [
        label
        for label, report in combos.items()
        if report.get("error_propagation", {}).get("stats", {}).get("available")
    ]
    if not available_combos:
        return

    selected_combo = st.selectbox(
        "Select pipeline",
        available_combos,
        format_func=_dl,
        key="error_prop_pipeline",
    )

    report = combos[selected_combo]
    prop = report.get("error_propagation", {})
    stats = prop.get("stats", {})
    analysis = prop.get("analysis", {})

    n_errors = stats.get("n_entity_errors", 0)
    n_clean = stats.get("n_clean", 0)
    st.markdown(f"**{n_errors}** records contain entity errors, **{n_clean}** clean records")

    error_means = stats.get("error_means", {})
    clean_means = stats.get("clean_means", {})
    cofailure_rates = stats.get("cofailure_rates", {})
    baseline_rates = stats.get("baseline_rates", {})

    if error_means and clean_means:
        table_rows = []
        for m in error_means:
            err = error_means.get(m, 0)
            clean = clean_means.get(m, 0)
            delta = clean - err
            cofail = cofailure_rates.get(m, 0)
            baseline = baseline_rates.get(m, 0)
            table_rows.append(
                {
                    "Metric": format_metric_name(m),
                    "Mean Score (entity errors)": f"{err:.3f}",
                    "Mean Score (clean)": f"{clean:.3f}",
                    "Delta": f"{delta:+.3f}",
                    "Failure Rate (entity errors)": f"{cofail:.0%}",
                    "Failure Rate (clean)": f"{baseline:.0%}",
                }
            )
        st.dataframe(pd.DataFrame(table_rows).set_index("Metric"), use_container_width=True)
        st.info(
            "**Mean Score** = average normalized score (0–1) for records in each group.  \n"
            "**Failure Rate** = % of records in each group scoring below the passing threshold.  \n"
            "A large gap indicates entity errors are a likely cause of downstream failures.",
            icon=None,
        )

    if cofailure_rates and baseline_rates:
        bar_rows = []
        for m in cofailure_rates:
            bar_rows.append(
                {
                    "Metric": format_metric_name(m),
                    "Rate": cofailure_rates[m],
                    "Group": "Records with entity errors",
                }
            )
            bar_rows.append(
                {
                    "Metric": format_metric_name(m),
                    "Rate": baseline_rates[m],
                    "Group": "Clean records",
                }
            )
        fig = px.bar(
            pd.DataFrame(bar_rows),
            x="Metric",
            y="Rate",
            color="Group",
            barmode="group",
            title="Failure rate: records with entity errors vs clean records",
            color_discrete_map={
                "Records with entity errors": "#EF553B",
                "Clean records": "#636EFA",
            },
        )
        fig.update_layout(height=400, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    entity_impact = stats.get("entity_type_impact", {})
    if entity_impact:
        st.markdown("**Impact by entity type:**")
        type_rows = []
        for etype, data in sorted(entity_impact.items(), key=lambda x: x[1].get("count", 0), reverse=True):
            row: dict[str, Any] = {"Entity Type": etype, "Count": data.get("count", 0)}
            dm = data.get("downstream_means", {})
            for m in ["task_completion", "faithfulness", "conversation_progression"]:
                row[format_metric_name(m)] = f"{dm.get(m, 0):.3f}"
            cr = data.get("cofailure_rates", {})
            row["Task Failure Rate"] = f"{cr.get('task_completion', 0):.0%}"
            type_rows.append(row)
        if type_rows:
            st.dataframe(
                pd.DataFrame(type_rows).set_index("Entity Type"),
                use_container_width=True,
            )

    chains = analysis.get("causal_chains", [])
    overall = analysis.get("overall_finding", "")

    if overall:
        st.info(overall)

    if chains:
        st.markdown("**Causal chains identified:**")
        for idx, chain in enumerate(chains, 1):
            pattern = chain.get("pattern", "Unknown")
            mechanism = chain.get("mechanism", "")
            entity_types = chain.get("entity_types_involved", [])
            downstream = chain.get("downstream_effects", [])
            preventability = chain.get("preventability", "?")

            with st.container(border=True):
                st.markdown(f"**{idx}. {pattern}** &nbsp;&nbsp; Preventability: **{preventability}**")
                st.markdown(mechanism)
                if entity_types:
                    st.caption(f"Entity types: {', '.join(entity_types)}")
                if downstream:
                    st.caption(f"Downstream: {', '.join(downstream)}")


def _plot_entity_accuracy_worse_than_wer(df: pd.DataFrame):
    """Show count of samples where entity accuracy is worse than WER (conversation-level)."""
    stt_cols = {"transcription_accuracy_key_entities", "stt_wer"}
    if not stt_cols.issubset(df.columns):
        return
    plot_df = df.dropna(subset=list(stt_cols))
    if plot_df.empty:
        return

    below_diag = plot_df[plot_df["transcription_accuracy_key_entities"] < plot_df["stt_wer"]]
    pct = len(below_diag) / len(plot_df) * 100 if len(plot_df) > 0 else 0

    st.info(
        f"**Entity accuracy vs WER:** {len(below_diag)}/{len(plot_df)} samples ({pct:.0f}%) "
        "have entity accuracy worse than overall WER at the **conversation level** "
        "(both metrics are per-conversation scores). "
        "These are cases where STT gets most words right but fails on the critical entities "
        "(codes, names, amounts)."
    )


def _count_entity_totals_and_failures(
    combos: dict,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, int]]]:
    """Count total and failed entities per type per pipeline from metric details on disk.

    Returns:
        entity_data: {pipeline_label: {entity_type: {"total": N, "failed": N}}}
        pipeline_stats: {pipeline_label: {"conversations": N, "total_turns": N}}
    """
    entity_data: dict[str, dict[str, dict[str, int]]] = {}
    pipeline_stats: dict[str, dict[str, int]] = {}
    for label, report in combos.items():
        type_counts: dict[str, dict[str, int]] = {}
        per_convo_turns: list[int] = []
        per_convo_entities: list[int] = []
        for rd_str in report.get("run_dirs", []):
            run_dir = Path(rd_str)
            for record_dir in get_record_directories(run_dir):
                for _trial_label, data_path in get_valid_trial_dirs(record_dir):
                    metrics_path = data_path / "metrics.json"
                    if not metrics_path.exists():
                        continue
                    try:
                        with open(metrics_path) as f:
                            data = json.load(f)
                        metrics_dict = data.get("metrics", data)
                        ms = metrics_dict.get("transcription_accuracy_key_entities", {})
                        details = ms.get("details", {})
                    except Exception:
                        continue
                    per_turn = details.get("per_turn_entity_details", {})
                    if not per_turn:
                        continue
                    per_convo_turns.append(len(per_turn))
                    convo_entity_count = 0
                    for _tid, turn_data in per_turn.items():
                        for entity in turn_data.get("entities", []):
                            if entity.get("skipped"):
                                continue
                            etype = normalize_entity_type(entity.get("type", "unknown"))
                            if etype not in type_counts:
                                type_counts[etype] = {"total": 0, "failed": 0}
                            type_counts[etype]["total"] += 1
                            convo_entity_count += 1
                            if not entity.get("correct"):
                                type_counts[etype]["failed"] += 1
                    per_convo_entities.append(convo_entity_count)
        entity_data[label] = type_counts
        pipeline_stats[label] = {
            "conversations": len(per_convo_turns),
            "total_turns": sum(per_convo_turns),
            "per_convo_turns": per_convo_turns,
            "per_convo_entities": per_convo_entities,
        }
    return entity_data, pipeline_stats


def _plot_entity_type_failure_heatmap(combos: dict):
    """Heatmap: entity type failure rate (%) by pipeline."""
    entity_data, pipeline_stats = _count_entity_totals_and_failures(combos)

    rows = []
    for label, type_counts in entity_data.items():
        for etype, counts in type_counts.items():
            if counts["total"] > 0:
                rate = counts["failed"] / counts["total"] * 100
                rows.append(
                    {
                        "Pipeline": _dl(label),
                        "Entity Type": etype,
                        "Failure Rate (%)": round(rate, 1),
                        "Failed": counts["failed"],
                        "Total": counts["total"],
                    }
                )
    if not rows:
        return

    st.markdown("### Entity Type Failure Patterns")
    st.caption(
        "Failure rate = failed entities / total entities of that type. "
        "Shows which entity types (names, codes, amounts, dates) are hardest for each pipeline's STT."
    )
    # Per-pipeline summary: conversations, total entities, total failed
    summary_rows = []
    for label, type_counts in entity_data.items():
        total = sum(c["total"] for c in type_counts.values())
        failed = sum(c["failed"] for c in type_counts.values())
        stats = pipeline_stats.get(label, {})
        n_convos = stats.get("conversations", 0)
        convo_turns = stats.get("per_convo_turns", [])
        convo_entities = stats.get("per_convo_entities", [])
        median_turns = float(np.median(convo_turns)) if convo_turns else 0
        median_entities = float(np.median(convo_entities)) if convo_entities else 0
        summary_rows.append(
            {
                "Pipeline": _dl(label),
                "Conversations": n_convos,
                "Median Turns/Convo": median_turns,
                "Total Entities": total,
                "Median Entities/Convo": median_entities,
                "Failed": failed,
                "Failure Rate": f"{failed / total * 100:.1f}%" if total > 0 else "—",
            }
        )
    if summary_rows:
        st.caption(
            "Entity totals vary between pipelines because each conversation unfolds differently "
            "— longer conversations surface more entities for STT to transcribe."
        )
        st.dataframe(
            pd.DataFrame(summary_rows).set_index("Pipeline"),
            use_container_width=True,
        )

    # Distribution histograms for turns and entities per conversation
    dist_rows = []
    for label in entity_data:
        stats = pipeline_stats.get(label, {})
        for t in stats.get("per_convo_turns", []):
            dist_rows.append({"Pipeline": _dl(label), "Turns per Conversation": t})
    entity_dist_rows = []
    for label in entity_data:
        stats = pipeline_stats.get(label, {})
        for e in stats.get("per_convo_entities", []):
            entity_dist_rows.append({"Pipeline": _dl(label), "Entities per Conversation": e})

    if dist_rows or entity_dist_rows:
        col1, col2 = st.columns(2)
        if dist_rows:
            with col1:
                dist_df = pd.DataFrame(dist_rows)
                fig_turns = px.histogram(
                    dist_df,
                    x="Turns per Conversation",
                    color="Pipeline",
                    barmode="overlay",
                    opacity=0.6,
                    nbins=20,
                    title="Turns per conversation",
                )
                fig_turns.update_layout(height=300, margin={"t": 40, "b": 40})
                st.plotly_chart(fig_turns, use_container_width=True)
        if entity_dist_rows:
            with col2:
                edist_df = pd.DataFrame(entity_dist_rows)
                fig_ents = px.histogram(
                    edist_df,
                    x="Entities per Conversation",
                    color="Pipeline",
                    barmode="overlay",
                    opacity=0.6,
                    nbins=20,
                    title="Entities per conversation",
                )
                fig_ents.update_layout(height=300, margin={"t": 40, "b": 40})
                st.plotly_chart(fig_ents, use_container_width=True)

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No entity type failures found.")
        return

    # Sort entity types by total count descending
    type_order = df.groupby("Entity Type")["Total"].sum().sort_values(ascending=False).index.tolist()

    # Grouped stacked bar: one bar per pipeline per entity type
    # Full bar height = total, failed portion on top in a darker shade
    import plotly.graph_objects as go

    pipelines = sorted(df["Pipeline"].unique().tolist())
    palette_base = ["#93c5fd", "#86efac", "#fde68a", "#c4b5fd", "#fbcfe8"]
    palette_fail = ["#2563eb", "#16a34a", "#ca8a04", "#7c3aed", "#db2777"]

    fig = go.Figure()
    for i, pipeline in enumerate(pipelines):
        pdf = df[df["Pipeline"] == pipeline].set_index("Entity Type").reindex(type_order).fillna(0)
        passed = pdf["Total"] - pdf["Failed"]
        failed = pdf["Failed"]

        fig.add_trace(
            go.Bar(
                name=f"{pipeline}",
                x=type_order,
                y=passed,
                marker_color=palette_base[i % len(palette_base)],
                text=passed.astype(int),
                textposition="inside",
                legendgroup=pipeline,
                offsetgroup=i,
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{pipeline} (failed)",
                x=type_order,
                y=failed,
                marker_color=palette_fail[i % len(palette_fail)],
                text=failed.astype(int),
                textposition="inside",
                legendgroup=pipeline,
                offsetgroup=i,
                base=passed,
            )
        )

    fig.update_layout(
        barmode="group",
        height=480,
        yaxis_title="Entity Count",
        xaxis_title="Entity Type",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap view
    with st.expander("Failure rate heatmap", expanded=False):
        pivot = df.pivot_table(
            index="Pipeline", columns="Entity Type", values="Failure Rate (%)", aggfunc="mean", fill_value=0
        )
        if not pivot.empty:
            pivot = pivot[type_order]
            heatmap_fig = px.imshow(
                pivot,
                text_auto=".1f",
                color_continuous_scale="Reds",
                zmin=0,
                zmax=100,
                labels={"color": "Failure %"},
            )
            heatmap_fig.update_layout(height=max(300, 50 * len(pivot)))
            st.plotly_chart(heatmap_fig, use_container_width=True)


# --- Tab 5: Sample Viewer ---


def _render_sample_viewer(run_dirs: list[Path]):
    """Drill into a specific sample's metrics, transcript, and judge predictions."""
    st.markdown("### Sample Viewer")
    st.caption("Select a sample to inspect its judge predictions, transcript, and raw data.")

    sample_options = _build_sample_options(run_dirs)
    if not sample_options:
        st.info("No samples found in selected runs.")
        return

    selected_label = st.selectbox("Select Sample", list(sample_options.keys()))
    if not selected_label:
        return
    sample_dir = sample_options[selected_label]

    metrics_path = sample_dir / "metrics.json"
    if not metrics_path.exists():
        st.warning(f"No metrics.json found at {sample_dir}")
        return

    try:
        with open(metrics_path) as f:
            raw_metrics = json.load(f)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return

    metric_data = (
        raw_metrics.get("metrics", raw_metrics) if isinstance(raw_metrics.get("metrics"), dict) else raw_metrics
    )

    audio_path = sample_dir / "audio_mixed.wav"
    if audio_path.exists():
        st.audio(str(audio_path))

    _render_sample_scores(metric_data)
    _render_sample_judge_details(metric_data)
    _render_sample_transcript(sample_dir)


def _build_sample_options(run_dirs: list[Path]) -> dict[str, Path]:
    """Build a flat dict of sample_label -> data_path for all records in run_dirs."""
    options: dict[str, Path] = {}
    for run_dir in run_dirs:
        records_dir = run_dir / "records"
        if not records_dir.exists():
            continue
        for record_dir in sorted(records_dir.iterdir()):
            if not record_dir.is_dir():
                continue
            trial_dirs = sorted([d for d in record_dir.iterdir() if d.is_dir() and "failed_attempt" not in d.name])
            if trial_dirs:
                for trial_dir in trial_dirs:
                    label = f"{record_dir.name}/{trial_dir.name}"
                    if len(run_dirs) > 1:
                        label = f"{run_dir.name} / {label}"
                    options[label] = trial_dir
            else:
                label = record_dir.name
                if len(run_dirs) > 1:
                    label = f"{run_dir.name} / {label}"
                options[label] = record_dir
    return options


def _render_sample_scores(metric_data: dict):
    """Show metric scores table for a single sample."""
    st.markdown("#### Metric Scores")
    rows = []
    for name, val in sorted(metric_data.items()):
        if not isinstance(val, dict):
            continue
        rows.append(
            {
                "Metric": format_metric_name(name),
                "Score": val.get("score"),
                "Normalized": val.get("normalized_score"),
                "Error": val.get("error") or "",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)


def _render_sample_judge_details(metric_data: dict):
    """Show per-metric judge explanations and details."""
    st.markdown("#### Judge Predictions")
    for metric_name in ALL_ANALYSIS_METRICS:
        metric_val = metric_data.get(metric_name)
        if not metric_val or not isinstance(metric_val, dict):
            continue
        details = metric_val.get("details", {})
        if not details:
            continue
        score = metric_val.get("normalized_score")
        score_str = f"{score:.3f}" if score is not None else "N/A"
        with st.expander(f"{format_metric_name(metric_name)} — score: {score_str}", expanded=False):
            explanation = details.get("explanation", {})
            if isinstance(explanation, dict):
                dimensions = explanation.get("dimensions", explanation)
                for dim_name, dim_data in dimensions.items():
                    if isinstance(dim_data, dict):
                        rating = dim_data.get("rating", "N/A")
                        flagged = dim_data.get("flagged", False)
                        evidence = dim_data.get("evidence", "")
                        flag_icon = " **[FLAGGED]**" if flagged else ""
                        st.markdown(f"**{dim_name}** (rating={rating}){flag_icon}")
                        st.markdown(f"> {evidence}")
                    else:
                        st.markdown(f"**{dim_name}:** {dim_data}")
            elif explanation:
                st.markdown(str(explanation))

            per_turn = details.get("per_turn_explanations", {})
            if per_turn:
                st.markdown("**Per-turn explanations:**")
                for tid, exp in per_turn.items():
                    rating = details.get("per_turn_ratings", {}).get(str(tid), "")
                    modes = details.get("per_turn_failure_modes", {}).get(str(tid), [])
                    mode_str = f" [{', '.join(modes)}]" if modes else ""
                    st.markdown(f"- Turn {tid} (rating={rating}){mode_str}: {exp}")

            if metric_name == "task_completion" and details.get("diff"):
                st.markdown("**Database diff:**")
                st.json(details["diff"])

            with st.expander("Raw details JSON", expanded=False):
                st.json(details)


def _render_sample_transcript(sample_dir: Path):
    """Render the conversation transcript for a sample."""
    st.markdown("#### Transcript")
    transcript_path = sample_dir / "transcript.jsonl"
    if not transcript_path.exists():
        st.info("No transcript file found")
        return

    try:
        turns = []
        with open(transcript_path) as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))
        if turns:
            for turn in turns:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                if role in ("assistant", "user"):
                    st.chat_message(role).markdown(content)
                else:
                    st.markdown(f"**{role}:** {content}")
        else:
            st.info("Empty transcript")
    except Exception as e:
        st.error(f"Failed to load transcript: {e}")


# --- Entry Point ---


def render_ai_review(run_dirs: list[Path]):
    """Main entry point for the AI Review view."""
    st.markdown("### AI Review")
    st.caption("LLM-powered analysis of voice agent benchmark results across pipeline configurations.")

    config = _render_sidebar_config()

    selected_run_dirs = _render_run_selection(run_dirs)
    if not selected_run_dirs:
        st.info("Select at least one run directory from the sidebar.")
        return

    combo_data = collect_combo_data(selected_run_dirs)
    if combo_data:
        with st.expander(f"**Detected {len(combo_data)} pipeline configuration(s)**"):
            for label, data in combo_data.items():
                st.markdown(f"- {_dl(label)} ({len(data['records'])} records)")
    else:
        st.warning("No records with metrics found in the selected runs.")
        return

    pipeline_result = _run_or_load_pipeline(selected_run_dirs, config)
    if pipeline_result is None:
        st.info("Click **Run AI Review Pipeline** in the sidebar to start analysis.")
        return

    # Generate shareable ZIP (HTML report + audio samples)
    combo_labels = list(pipeline_result.get("combos", {}).keys())
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        recipient = st.text_input(
            "Report recipient",
            value="",
            placeholder="e.g. ElevenLabs",
            label_visibility="collapsed",
            help="Optional — prepended to the report title",
        )
    filename_parts = ["eva_report"]
    if recipient.strip():
        filename_parts.insert(0, recipient.strip().replace(" ", "_"))
    filename_parts.extend(combo_labels)
    filename = f"{'_'.join(filename_parts)}.zip".replace(" ", "_").replace("/", "-")
    report_zip = _cached_report_zip(_pipeline_result_hash(pipeline_result), pipeline_result, recipient.strip())
    with col_btn:
        st.download_button(
            label="Download Report ZIP",
            data=report_zip,
            file_name=filename,
            mime="application/zip",
            help="HTML report + audio samples from critical issues",
        )

    tab1, tab2, tab3 = st.tabs(["Issue Analysis", "Model Comparison", "Sample Viewer"])
    with tab1:
        _render_issue_analysis_tab(pipeline_result)
    with tab2:
        _render_model_comparison_tab(pipeline_result)
    with tab3:
        _render_sample_viewer(selected_run_dirs)


def _render_sidebar_config() -> dict:
    """Render AI Review sidebar controls and return config."""
    st.sidebar.header("AI Review Config")
    available_models = get_available_models()
    qp_judge_model = st.query_params.get("judge_model", DEFAULT_JUDGE_MODEL)
    default_judge = qp_judge_model if qp_judge_model in available_models else DEFAULT_JUDGE_MODEL
    default_idx = available_models.index(default_judge) if default_judge in available_models else 0
    judge_model = st.sidebar.selectbox("Judge Model", available_models, index=default_idx)
    st.query_params["judge_model"] = judge_model
    debug_mode = st.sidebar.checkbox("Debug mode (first 10 samples)")

    component_focus = st.sidebar.selectbox(
        "Report focus",
        ["all", "stt_tts", "llm"],
        format_func={"all": "All components", "stt_tts": "STT/TTS only", "llm": "LLM only"}.get,
        help="Filter key findings to only include issues relevant to the selected component(s).",
    )

    st.sidebar.markdown("**Force Rerun**")
    force_pass1 = st.sidebar.checkbox("Pass 1 (Summarization)")
    force_pass2 = st.sidebar.checkbox("Pass 2 (Categorization)")
    force_pass3 = st.sidebar.checkbox("Pass 3 (Synthesis)")
    force_reports = st.sidebar.checkbox("Model Reports")

    force_rerun: set[str] = set()
    if force_pass1:
        force_rerun.update({"pass1", "pass2", "pass3", "reports"})
    if force_pass2:
        force_rerun.update({"pass2", "pass3", "reports"})
    if force_pass3:
        force_rerun.update({"pass3", "reports"})
    if force_reports:
        force_rerun.add("reports")

    return {
        "judge_model": judge_model,
        "force_rerun": force_rerun,
        "max_samples": 10 if debug_mode else None,
        "component_focus": component_focus,
    }


def _has_any_valid_cache(run_dirs: list[Path], judge_model: str) -> bool:
    """Return True if any combo in .ai_review_cache/ has a valid metadata for these runs + model."""
    if not run_dirs:
        return False
    cache_root = run_dirs[0].parent / ".ai_review_cache"
    if not cache_root.exists():
        return False
    selected_run_strs = {str(d) for d in run_dirs}
    for subdir in cache_root.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        meta_path = subdir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("judge_model") != judge_model:
                continue
            cached_runs = set(meta.get("run_dirs", []))
            if cached_runs & selected_run_strs:
                return True
        except Exception:
            continue
    return False


def _render_run_selection(run_dirs: list[Path]) -> list[Path]:
    """Render run selection multiselect in sidebar."""
    st.sidebar.header("Run Selection")
    run_labels: dict[str, Path] = {}
    for d in run_dirs:
        try:
            rc = load_run_config(d)
            pc = extract_pipeline_config(rc)
            label = f"{d.name}  ({get_display_combo_label(get_combo_label(pc))})"
        except Exception:
            label = d.name
        run_labels[label] = d
    label_list = list(run_labels.keys())

    # Read selected run dir names from query params (comma-separated)
    qp_runs_raw = st.query_params.get("ai_runs", "")
    qp_run_names = {r.strip() for r in qp_runs_raw.split(",") if r.strip()} if qp_runs_raw else set()
    default_labels = [lbl for lbl in label_list if run_labels[lbl].name in qp_run_names] if qp_run_names else []
    if not default_labels:
        default_labels = label_list[:1]

    selected = st.sidebar.multiselect("Select Runs to Analyze", label_list, default=default_labels)

    # Save selected run dir names to query params
    st.query_params["ai_runs"] = ",".join(run_labels[lbl].name for lbl in selected)

    return [run_labels[lbl] for lbl in selected]


def _run_or_load_pipeline(run_dirs: list[Path], config: dict) -> dict | None:
    """Run pipeline if button clicked; auto-load from disk cache when available."""
    run_pipeline = st.sidebar.button("Run AI Review Pipeline", type="primary")

    # Detect run-selection changes and clear stale session state so the new
    # selection auto-loads its own cache (or prompts a fresh run).
    current_runs_key = tuple(sorted(str(d) for d in run_dirs))
    if st.session_state.get("ai_review_last_runs") != current_runs_key:
        st.session_state.pop("ai_review_result", None)
        st.session_state["ai_review_last_runs"] = current_runs_key

    pipeline_result = st.session_state.get("ai_review_result")

    should_auto_load = (
        pipeline_result is None
        and not run_pipeline
        and _has_any_valid_cache(run_dirs, config["judge_model"])
    )

    if run_pipeline or should_auto_load:
        _ensure_router_initialized()

        progress_bar = None
        status_text = None
        if run_pipeline:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

        def progress_callback(pct: float, msg: str) -> None:
            if progress_bar is not None:
                progress_bar.progress(min(pct, 1.0))
            if status_text is not None:
                status_text.text(msg)

        def result_callback(partial_result: dict) -> None:
            st.session_state["ai_review_result"] = partial_result

        spinner_msg = "Running AI Review Pipeline..." if run_pipeline else "Loading cached report..."
        with st.spinner(spinner_msg):
            result = asyncio.run(
                run_full_pipeline(
                    run_dirs=run_dirs,
                    judge_model=config["judge_model"],
                    force_rerun=config["force_rerun"],
                    max_samples=config["max_samples"],
                    progress_callback=progress_callback if run_pipeline else None,
                    result_callback=result_callback,
                    component_focus=config["component_focus"],
                )
            )
            st.session_state["ai_review_result"] = result
            pipeline_result = result

        if run_pipeline and progress_bar is not None and status_text is not None:
            progress_bar.progress(1.0)
            status_text.text("Pipeline complete!")

    return pipeline_result
