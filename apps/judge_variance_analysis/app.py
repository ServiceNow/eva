# apps/judge_variance_analysis/app.py
"""Streamlit app for interactive judge variance analysis.

Run from project root:
    uv run streamlit run apps/judge_variance_analysis/app.py --server.port 8502 --theme.primaryColor "#6b7280"
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from judge_variance_analysis.analysis import (
    JUDGE_METRICS,
    compute_composite_stability,
    compute_icc,
    compute_judge_variance,
    compute_judge_variance_summary,
    compute_q0_tests,
    compute_statistical_tests,
    compute_trajectory_variance,
    compute_trajectory_variance_summary,
    compute_within_type_tests,
)
from plotly.subplots import make_subplots

try:
    from judge_variance_analysis.generate_report import build_report as _build_report

    _REPORT_AVAILABLE = True
except ImportError:
    _build_report = None
    _REPORT_AVAILABLE = False

from judge_variance_analysis.load_data import (
    ARCHIVE_DIR,
    available_sources,
    get_run_metadata,
    latest_csv_timestamp,
    load_aggregate_scores,
    load_aggregate_scores_from_csv,
    load_scores,
    load_scores_from_csv,
)

st.set_page_config(page_title="EVA Judge Variance Analysis", layout="wide")

# Pass thresholds for EVA composite metrics (normalized_score scale)
PASS_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.5,  # EVA-A: faithfulness >= 0.5
    "agent_speech_fidelity": 0.95,  # EVA-A: agent_speech_fidelity >= 0.95
    "conversation_progression": 0.5,  # EVA-X: >= 0.5
    "turn_taking": 0.5,  # EVA-X: >= 0.5
    "conciseness": 0.5,  # EVA-X: >= 0.5
    # transcription_accuracy_key_entities: not part of any composite pass condition
}


@st.cache_data
def get_data(source: str = "csv"):
    if source == "csv":
        scores = load_scores_from_csv()
        agg = load_aggregate_scores_from_csv()
    else:
        scores = load_scores(ARCHIVE_DIR)
        agg = load_aggregate_scores(ARCHIVE_DIR)
    return scores, agg


def download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv(index=False).encode(), filename, "text/csv")


def _variance_histogram(
    df: pd.DataFrame,
    x_label: str,
    title: str,
    color_discrete_map: dict[str, str] | None = None,
    run_label_order: list[str] | None = None,
) -> go.Figure:
    """Faceted histogram with one row per model and one column per metric."""
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
        color_discrete_map=color_discrete_map or {},
        category_orders={"run_label": run_label_order} if run_label_order else {},
    )
    # Strip the "col=" / "row=" prefix from all facet labels and make them horizontal.
    # Plotly marks facet_row annotations with textangle=-90 (rotated); making them
    # textangle=0 keeps the model name readable on the right side of each row.
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], textangle=0))
    fig.update_layout(showlegend=False, margin={"r": 220})
    fig.update_xaxes(title_font={"color": "#222"}, tickfont={"color": "#222"})
    fig.update_yaxes(title_font={"color": "#222"}, tickfont={"color": "#222"})
    return fig


def _clean_composite_label(col: str) -> str:
    """Convert a composite column name to a short display label."""
    for suffix, display in [("_pass_at_1", " pass@1"), ("_pass_at_k", " pass@k"), ("_pass_power_k", " pass^k")]:
        if col.endswith(suffix):
            return col[: -len(suffix)].replace("_pass", "") + display
    if col.endswith("_mean"):
        return col[:-5] + " mean"
    return col


def _compact_letters(models: list[str], sig_pairs: set) -> dict[str, str]:
    """Assign compact letter display (CLD) for pairwise comparisons.

    Returns a dict mapping model → letter string.
    Models that share a letter are NOT significantly different from each other.
    """
    if not sig_pairs:
        return {}
    letter_chars = "abcdefghij"
    # assigned[m] = set of letter indices currently assigned to model m
    assigned: dict[str, set] = {m: {0} for m in models}
    next_idx = 1
    for m1, m2 in combinations(models, 2):
        if frozenset([m1, m2]) in sig_pairs:
            common = assigned[m1] & assigned[m2]
            for c in common:
                assigned[m2].discard(c)
            if not assigned[m2]:
                assigned[m2].add(next_idx)
                # Give the new letter to every model not significantly different from m2
                for other in models:
                    if other != m2 and frozenset([other, m2]) not in sig_pairs:
                        assigned[other].add(next_idx)
                next_idx += 1
    return {m: "".join(letter_chars[i] for i in sorted(assigned[m])) for m in models}


def _variance_bar_fig(
    summary_df: pd.DataFrame,
    kw_df: pd.DataFrame,
    pw_df: pd.DataFrame,
    y_label: str,
    y_max: float | None = None,
    color_discrete_map: dict[str, str] | None = None,
    run_label_order: list[str] | None = None,
) -> go.Figure:
    """Grouped bar chart of mean variance per metric, with K-W asterisks and CLD letters.

    Asterisks on x-axis tick labels indicate overall K-W significance.
    Letters above error bar tips are compact letter display for pairwise Bonferroni-corrected results.
    """
    if y_max is None:
        y_max = (summary_df["mean_std"] + summary_df["std_of_std"].fillna(0)).max() * 1.35

    # Ensure consistent metric order (alphabetical, matching groupby output)
    metric_order = sorted(summary_df["metric"].unique())
    _cat_orders: dict = {"metric": metric_order}
    if run_label_order:
        _cat_orders["run_label"] = run_label_order
    fig = px.bar(
        summary_df,
        x="metric",
        y="mean_std",
        color="run_label",
        barmode="group",
        error_y="std_of_std",
        labels={"mean_std": y_label, "metric": "Metric", "run_label": "Model(s)"},
        category_orders=_cat_orders,
        color_discrete_map=color_discrete_map or {},
    )
    fig.update_layout(yaxis_range=[0, y_max], legend_title_text="Model(s)")
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)

    # ── Compact letter display — placed above error bar tips via add_annotation ──
    # We use numeric x positions on the categorical axis: category index + bar offset.
    # With plotly's default bargap=0.2, each bar group occupies 0.8 units; for N models
    # that's 0.8/N per bar, centered symmetrically around the category integer index.
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
            letters_by_metric[metric] = _compact_letters(models_for_metric, sig_pairs)

    if any(letters_by_metric.values()):
        cat_index = {m: i for i, m in enumerate(metric_order)}
        n_traces = len(fig.data)
        bar_w = 0.8 / n_traces  # 0.8 = 1 - default bargap
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

    # ── K-W significance asterisks on x-axis tick labels ─────────────────────
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


def _threshold_crossings(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where judge stochasticity flipped a score across an EVA pass/fail threshold.

    For each (run_id, metric, record_id, trial): if the min normalized_score across
    iterations is below the threshold AND the max is at or above it (or vice versa),
    that trial had a pass/fail flip due to judge noise.

    Returns a DataFrame with one row per crossing, including min_score, max_score, threshold.
    Only metrics with defined PASS_THRESHOLDS are included.
    """
    rows = []
    for metric, threshold in PASS_THRESHOLDS.items():
        sub = scores_df[scores_df["metric"] == metric]
        for (run_id, run_label, record_id, trial), grp in sub.groupby(["run_id", "run_label", "record_id", "trial"]):
            s = grp["normalized_score"].values
            if len(s) < 2:
                continue
            lo, hi = float(s.min()), float(s.max())
            if lo < threshold <= hi:
                rows.append(
                    {
                        "run_id": run_id,
                        "run_label": run_label,
                        "metric": metric,
                        "record_id": record_id,
                        "trial": int(trial),
                        "min_score": lo,
                        "max_score": hi,
                        "threshold": threshold,
                    }
                )
    return pd.DataFrame(rows)


def _llm_name(run_label: str) -> str:
    """Extract the LLM name from a run label 'stt / llm / tts'."""
    parts = run_label.split(" / ")
    return parts[1].strip() if len(parts) >= 3 else run_label


def _fmt_p(p: float) -> str:
    """Format a p-value for display."""
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _model_list(rows: list) -> str:
    return " and ".join(
        f"{_llm_name(r['model'])} (p={_fmt_p(r['p_bonferroni'])}, delta={r['median_delta']:+.4f})" for r in rows
    )


_COMPOSITE_GROUP_ORDER = {"EVA-A": 0, "EVA-X": 1, "EVA-overall": 2}
_COMPOSITE_STAT_ORDER = {"pass@1": 0, "pass@k": 1, "pass^k": 2, "mean": 3}


def _composite_sort_key(label: str) -> tuple:
    for g, go in _COMPOSITE_GROUP_ORDER.items():  # noqa: F402
        if label.startswith(g):
            stat = label[len(g) + 1 :]
            return (go, _COMPOSITE_STAT_ORDER.get(stat, 99))
    return (99, 99)


_sources = available_sources()
_csv_ts = latest_csv_timestamp()
_csv_label = f"Pre-baked CSV ({_csv_ts})" if _csv_ts else "Pre-baked CSV"
if len(_sources) >= 2:
    _data_source = st.sidebar.radio(
        "Data source",
        _sources,
        format_func=lambda s: _csv_label if s == "csv" else "Raw JSON",
        horizontal=True,
    )
    st.sidebar.divider()
elif _sources:
    _data_source = _sources[0]
    _src_label = _csv_label if _data_source == "csv" else "Raw JSON"
    st.sidebar.caption(f"Data source: {_src_label}")
else:
    _data_source = "json"

scores_df, agg_df = get_data(_data_source)

if scores_df.empty:
    st.error(f"No data found in {ARCHIVE_DIR}. Run `run_iterations.py` first.")
    st.stop()

available_runs = sorted(scores_df["run_id"].unique())
available_metrics = [m for m in JUDGE_METRICS if m in scores_df["metric"].unique()]
metadata = get_run_metadata()

# ── Global run color/symbol/order maps ───────────────────────────────────────
# Assigned from all available runs so colors stay consistent regardless of
# sidebar selection.  Cascade: vivid cool family.  S2S: vivid warm family.
_CASCADE_COLORS = ["#3B82F6", "#99C945", "#A855F7"]  # electric blue, light green, vivid purple
_S2S_COLORS = ["#CC61B0", "#F97316", "#EAB308"]  # orchid pink, vivid orange, vivid amber

_run_label_lookup: dict[str, str] = (
    scores_df[["run_id", "run_label"]].drop_duplicates().set_index("run_id")["run_label"].to_dict()
)
_cascade_available = [r for r in available_runs if metadata.get(r, {}).get("type") == "cascade"]
_s2s_available = [r for r in available_runs if metadata.get(r, {}).get("type") != "cascade"]

RUN_COLOR_MAP: dict[str, str] = {}
RUN_SYMBOL_MAP: dict[str, str] = {}
RUN_GROUP_MAP: dict[str, str] = {}
for _i, _rid in enumerate(_cascade_available):
    _rl = _run_label_lookup.get(_rid, _rid)
    RUN_COLOR_MAP[_rl] = _CASCADE_COLORS[_i % len(_CASCADE_COLORS)]
    RUN_SYMBOL_MAP[_rl] = "circle"
    RUN_GROUP_MAP[_rl] = "cascade"
for _i, _rid in enumerate(_s2s_available):
    _rl = _run_label_lookup.get(_rid, _rid)
    RUN_COLOR_MAP[_rl] = _S2S_COLORS[_i % len(_S2S_COLORS)]
    RUN_SYMBOL_MAP[_rl] = "star"
    RUN_GROUP_MAP[_rl] = "s2s"

# Canonical display order: cascade runs first, then S2S — used in category_orders
# for every chart so legend and bar/box groups always appear in this sequence.
RUN_LABEL_ORDER: list[str] = [_run_label_lookup.get(rid, rid) for rid in _cascade_available + _s2s_available]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()

# Restore selections from URL params if present
_params = st.query_params
_default_runs = [r for r in _params.get("runs", "").split(",") if r in available_runs]
_default_metrics = [m for m in _params.get("metrics", "").split(",") if m in available_metrics]

st.sidebar.subheader("Models")
selected_runs = []
_PILL = (
    "display:inline-block; padding:1px 6px; border-radius:99px; "
    "font-size:10px; white-space:nowrap; margin-right:3px; margin-bottom:2px"
)

cascade_runs = [r for r in available_runs if metadata.get(r, {}).get("type") == "cascade"]
s2s_runs = [r for r in available_runs if metadata.get(r, {}).get("type") != "cascade"]

# Initialize individual run states on first render (before any widgets reference these keys)
for r in available_runs:
    if f"run_{r}" not in st.session_state:
        st.session_state[f"run_{r}"] = r in (_default_runs or available_runs)


def _on_cascade_all():
    val = st.session_state["_cascade_all"]
    for r in cascade_runs:
        st.session_state[f"run_{r}"] = val


def _on_s2s_all():
    val = st.session_state["_s2s_all"]
    for r in s2s_runs:
        st.session_state[f"run_{r}"] = val


if cascade_runs:
    # Sync master checkbox to reflect current individual states
    st.session_state["_cascade_all"] = all(st.session_state[f"run_{r}"] for r in cascade_runs)
    _hdr_col, _all_col = st.sidebar.columns([4, 1])
    with _hdr_col:
        st.markdown("##### Cascade")
    with _all_col:
        st.checkbox("all", key="_cascade_all", on_change=_on_cascade_all)
for r in cascade_runs:
    m = metadata.get(r, {})
    _col_cb, _col_pills = st.sidebar.columns([1, 10], gap="small")
    with _col_cb:
        _checked = st.checkbox(label=r, key=f"run_{r}", label_visibility="collapsed")
    with _col_pills:
        _rl_s = _run_label_lookup.get(r, r)
        _dot_color = RUN_COLOR_MAP.get(_rl_s, "#ccc")
        _dot = (
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{_dot_color};margin-right:4px;vertical-align:middle"></span>'
        )
        st.markdown(
            f'<div style="margin-top:3px; line-height:1.8">'
            f"{_dot}"
            f'<span style="{_PILL}; background:#dbeafe; color:#1d4ed8">{m.get("stt", "")}</span>'
            f'<span style="{_PILL}; background:#d1fae5; color:#065f46">{m.get("llm", "")}</span>'
            f'<span style="{_PILL}; background:#fef3c7; color:#78350f">{m.get("tts", "")}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    if _checked:
        selected_runs.append(r)

if s2s_runs:
    # Sync master checkbox to reflect current individual states
    st.session_state["_s2s_all"] = all(st.session_state[f"run_{r}"] for r in s2s_runs)
    _hdr_col, _all_col = st.sidebar.columns([4, 1])
    with _hdr_col:
        st.markdown("##### S2S and audio native")
    with _all_col:
        st.checkbox("all", key="_s2s_all", on_change=_on_s2s_all)
for r in s2s_runs:
    m = metadata.get(r, {})
    _col_cb, _col_pills = st.sidebar.columns([1, 10], gap="small")
    with _col_cb:
        _checked = st.checkbox(label=r, key=f"run_{r}", label_visibility="collapsed")
    with _col_pills:
        _rl_s = _run_label_lookup.get(r, r)
        _dot_color = RUN_COLOR_MAP.get(_rl_s, "#ccc")
        _dot = (
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{_dot_color};margin-right:4px;vertical-align:middle"></span>'
        )
        st.markdown(
            f'<div style="margin-top:3px; line-height:1.8">'
            f"{_dot}"
            f'<span style="{_PILL}; background:#ede9fe; color:#4c1d95">{m.get("s2s", "")}</span>'
            f'<span style="{_PILL}; background:#f1f5f9; color:#475569">{m.get("voice", "")}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    if _checked:
        selected_runs.append(r)

st.sidebar.subheader("Metrics")
selected_metrics = [
    m
    for m in available_metrics
    if st.sidebar.checkbox(
        m,
        value=(m in (_default_metrics or available_metrics)),
        key=f"metric_{m}",
    )
]

# Persist selections to URL
st.query_params["runs"] = ",".join(selected_runs)
st.query_params["metrics"] = ",".join(selected_metrics)

scores = scores_df[scores_df["run_id"].isin(selected_runs)]
agg = agg_df[agg_df["run_id"].isin(selected_runs)]

# ── Sidebar: Generate report ──────────────────────────────────────────────────
st.sidebar.divider()
with st.sidebar.expander("Generate report"):
    if not _REPORT_AVAILABLE:
        st.info("Report generation is not yet available in the shared version.")
    else:
        _SECTION_DESCS = {
            "A": "A · Judge Variance",
            "B": "B · Trajectory Variance",
            "C": "C · Judge vs. Trajectory Comparison",
            "D": "D · Composite Metric Stability",
            "E": "E · Borderline Scenarios",
            "F": "F · Per-Metric Deep Dive (untested)",
        }
        _include = {
            letter: st.checkbox(desc, value=(letter != "F"), key=f"rpt_sec_{letter}")
            for letter, desc in _SECTION_DESCS.items()
        }
        _exclude_for_report = {letter for letter, inc in _include.items() if not inc}

        if st.button("Generate", type="primary", disabled=not selected_runs):
            with st.spinner("Building report…"):
                st.session_state["_report_html"] = _build_report(
                    exclude_sections=_exclude_for_report,
                    scores_override=scores,
                    agg_override=agg,
                )

        if st.session_state.get("_report_html"):
            st.download_button(
                "Download report.html",
                data=st.session_state["_report_html"],
                file_name="report.html",
                mime="text/html",
            )

judge_var = compute_judge_variance(scores, selected_metrics)
traj_var = compute_trajectory_variance(scores, selected_metrics)
judge_summary = compute_judge_variance_summary(judge_var)
traj_summary = compute_trajectory_variance_summary(traj_var)
stat_results = compute_statistical_tests(judge_var, traj_var, selected_metrics)
q0_results = compute_q0_tests(judge_var, traj_var, selected_metrics)

# Within-type tests: run Q2/Q3 separately for cascade and S2S when both types are present.
_selected_run_type_map = {r: metadata.get(r, {}).get("type", "cascade") for r in selected_runs}
_selected_types = set(_selected_run_type_map.values())
within_type_results = (
    compute_within_type_tests(judge_var, traj_var, _selected_run_type_map, selected_metrics)
    if len(_selected_types) >= 2
    else {}
)
icc_results = compute_icc(scores, selected_metrics)

if judge_var.empty:
    st.warning("No data matches the current filters. Select at least one run and one metric.")
    st.stop()

# ── Global y-axis range for all variance plots ────────────────────────────────
# Shared across bar, box, and histogram plots so all charts are directly comparable.
_raw_var_max = max(
    judge_var["std"].max() if not judge_var.empty else 0.0,
    traj_var["std"].max() if not traj_var.empty else 0.0,
)
global_var_ymax = _raw_var_max * 1.25  # headroom for bar labels above error bars

# ── Tabs ─────────────────────────────────────────────────────────────────────
_axis_style = {"title_font": {"color": "#222"}, "tickfont": {"color": "#222"}}
_TYPE_LABELS = {"cascade": "Cascade", "s2s": "S2S / audio-native"}

tabs = st.tabs(
    [
        "Overview",
        "Variance overview",
        "Judge vs. trial variance",
        "Judge variance",
        "Trial variance",
        "EVA score stability",
        "Borderline scenarios",
        "Intraclass correlation",
        "Per-metric deep dive (untested)",
        "Statistical tests",
    ]
)

# ── Overview ─────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Overview")
    st.write("""
    This study measures two sources of variance in EVA metric scores:
    - **Judge variance** (stochasticity): the LLM judge producing different outputs when
      re-evaluating the *same* conversation (same audio, transcript, tool calls).
    - **Trial variance**: genuine differences in how conversations unfold across trials
      (different simulations of the same scenario).
    """)

    meta_rows = []
    for run_id in selected_runs:
        m = metadata.get(run_id, {})
        meta_rows.append({"run_id": run_id, **m})
    st.subheader("Runs in this analysis")
    st.dataframe(pd.DataFrame(meta_rows), width="stretch")

    st.subheader("Metric reference")
    _metric_rows = [
        (
            "faithfulness",
            "1, 2, or 3 per conversation",
            "5 faithfulness dimensions: fabricating tool parameters, misrepresenting tool results, "
            "violating policies, failing to disambiguate, hallucination",
            "1→0.0, 2→0.5, 3→1.0",
            "Mean of per-conversation normalized scores across all record × trial pairs",
        ),
        (
            "conversation_progression",
            "1, 2, or 3 per conversation",
            "4 progression dimensions: unnecessary tool calls, information loss, "
            "redundant statements, question quality",
            "1→0.0, 2→0.5, 3→1.0",
            "Mean of per-conversation normalized scores across all record × trial pairs",
        ),
        (
            "conciseness",
            "1, 2, or 3 per turn, then averaged",
            "Whether assistant responses were voice-appropriately brief and easy to digest",
            "mean of (1→0.0, 2→0.5, 3→1.0) across all turns",
            "Mean of per-conversation normalized scores across all record × trial pairs",
        ),
        (
            "turn_taking",
            "−1 (early/interrupting), 0 (on-time), +1 (late), null (excluded)",
            "Whether agent took the floor at the right time after each user utterance",
            "mean of per-turn scores (range −1 to +1), then mapped to 0–1",
            "Mean of per-conversation normalized scores across all record × trial pairs",
        ),
        (
            "agent_speech_fidelity",
            "0 or 1 per turn",
            "Whether TTS output faithfully reproduced intended text, especially key entities (codes, amounts, names)",
            "fraction of turns scored 1 (0 to 1)",
            "Mean of per-conversation normalized scores across all record × trial pairs",
        ),
        (
            "transcription_accuracy_key_entities",
            "fraction of entities correct",
            "Whether STT correctly transcribed key entities in user speech — cascade runs only",
            "fraction correct (0 to 1)",
            "Mean of per-conversation normalized scores across all record × trial pairs",
        ),
    ]
    _th = "text-align:left;padding:7px 14px;border-bottom:2px solid rgba(128,128,128,0.35);white-space:nowrap;font-weight:600"
    _td_base = "padding:7px 14px;vertical-align:top;border-bottom:1px solid rgba(128,128,128,0.15)"
    _td_nowrap = f"{_td_base};white-space:nowrap"
    _td_wrap = f"{_td_base};min-width:220px"
    _rows_html = "".join(
        f"<tr>"
        f"<td style='{_td_nowrap}'>{r[0]}</td>"
        f"<td style='{_td_nowrap}'>{r[1]}</td>"
        f"<td style='{_td_wrap}'>{r[2]}</td>"
        f"<td style='{_td_nowrap}'>{r[3]}</td>"
        f"<td style='{_td_wrap}'>{r[4]}</td>"
        f"</tr>"
        for r in _metric_rows
    )
    st.html(f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;font-family:inherit">
      <thead><tr>
        <th style="{_th}">metric</th>
        <th style="{_th}">raw score values</th>
        <th style="{_th}">what it measures</th>
        <th style="{_th}">normalization</th>
        <th style="{_th}">model-level aggregation</th>
      </tr></thead>
      <tbody>{_rows_html}</tbody>
    </table>
    """)

    st.subheader("Sample sizes")
    st.caption(
        "Judge variance n = number of (record, trial) pairs per model (each pair contributes "
        "one std dev across the 3 judge iterations). "
        "Trial variance n = number of records per model (each record contributes "
        "one std dev across the 3 simulation trials)."
    )
    if not judge_summary.empty and not traj_summary.empty:
        _n_judge = (
            judge_summary.groupby("run_label")["n"]
            .max()
            .reset_index()
            .rename(columns={"run_label": "model", "n": "judge variance n (record×trial pairs)"})
        )
        _n_traj = (
            traj_summary.groupby("run_label")["n"]
            .max()
            .reset_index()
            .rename(columns={"run_label": "model", "n": "trial variance n (records)"})
        )
        st.dataframe(_n_judge.merge(_n_traj, on="model"), width="stretch")

# ── Variance overview ─────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Variance overview")
    st.write("""
    High-level view of how much variance exists in each metric, and whether it is
    statistically distinguishable from zero. Judge variance (stochasticity) and trial
    variance (conversation-to-conversation) are shown on a shared scale for direct comparison.
    """)

    # ── Cross-model judge vs. trial bar chart ─────────────────────────────────
    st.subheader("Judge vs. trial variance per metric (averaged across models)")
    st.caption(
        "Each bar is the mean std dev for that variance type and metric, averaged across "
        "all selected models. Error bars show the spread (std dev) across models."
    )

    _j_agg = (
        judge_summary.groupby("metric")["mean_std"]
        .agg(mean="mean", std="std")
        .reset_index()
        .assign(variance_type="Judge (stochasticity)")
    )
    _t_agg = (
        traj_summary.groupby("metric")["mean_std"]
        .agg(mean="mean", std="std")
        .reset_index()
        .assign(variance_type="Trial (conversation-to-conversation)")
    )
    _vov_df = pd.concat([_j_agg, _t_agg], ignore_index=True)
    _metric_order_vov = sorted(_vov_df["metric"].unique())

    _vov_fig = px.bar(
        _vov_df,
        x="metric",
        y="mean",
        color="variance_type",
        barmode="group",
        error_y="std",
        category_orders={"metric": _metric_order_vov},
        color_discrete_map={
            "Judge (stochasticity)": "#636EFA",
            "Trial (conversation-to-conversation)": "#EF553B",
        },
        labels={"mean": "Mean std dev", "metric": "Metric", "variance_type": "Variance type"},
    )
    _vov_fig.update_layout(
        yaxis_range=[0, global_var_ymax],
        legend_title_text="Variance type",
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
        margin={"r": 260},
    )
    _vov_fig.update_xaxes(**_axis_style)
    _vov_fig.update_yaxes(**_axis_style)
    st.plotly_chart(_vov_fig, width="stretch")

    # ── Summary table: judge vs. trial mean std, ratio ───────────────────────
    _jt_table = pd.merge(
        judge_summary.groupby("metric")["mean_std"].mean().rename("mean judge std"),
        traj_summary.groupby("metric")["mean_std"].mean().rename("mean trial std"),
        left_index=True,
        right_index=True,
    ).reset_index()
    _jt_table["trial / judge"] = (_jt_table["mean trial std"] / _jt_table["mean judge std"]).round(1)
    _jt_table = _jt_table.sort_values("trial / judge", ascending=False).reset_index(drop=True)
    st.caption("Mean std dev averaged across selected models; ratio = trial ÷ judge.")
    st.dataframe(_jt_table.round({"mean judge std": 3, "mean trial std": 3}), hide_index=True, width="stretch")

    # ── Q0 significance summary ───────────────────────────────────────────────
    st.subheader("Is variance significantly greater than zero?")
    st.write(
        "One-sample Wilcoxon signed-rank test against 0 (one-sided, H₁: median > 0), "
        "pooled across all selected models per metric. "
        "A significant result means the variance is reliably non-zero — not just noise."
    )

    q0_j = q0_results.get("q0_judge_pooled", pd.DataFrame())
    q0_t = q0_results.get("q0_trial_pooled", pd.DataFrame())

    if not q0_j.empty and not q0_t.empty:
        _sig_map = {True: "✓ yes", False: "✗ no"}
        _q0_summary = pd.merge(
            q0_j[["metric", "median_std", "p", "significant"]].rename(
                columns={
                    "median_std": "judge median std",
                    "p": "judge p",
                    "significant": "judge sig?",
                }
            ),
            q0_t[["metric", "median_std", "p", "significant"]].rename(
                columns={
                    "median_std": "trial median std",
                    "p": "trial p",
                    "significant": "trial sig?",
                }
            ),
            on="metric",
        ).sort_values("metric")
        _q0_summary["judge p"] = _q0_summary["judge p"].map(_fmt_p)
        _q0_summary["trial p"] = _q0_summary["trial p"].map(_fmt_p)
        _q0_summary["judge sig?"] = _q0_summary["judge sig?"].map(_sig_map)
        _q0_summary["trial sig?"] = _q0_summary["trial sig?"].map(_sig_map)
        st.dataframe(
            _q0_summary.round({"judge median std": 4, "trial median std": 4}),
            hide_index=True,
            width="stretch",
        )

        with st.expander("Methodology"):
            st.markdown("""
**Why one-sample Wilcoxon against 0?**
Std devs are bounded at zero and right-skewed, so a t-test against 0 is inappropriate.
The Wilcoxon signed-rank test is a non-parametric alternative that ranks the absolute
values of the non-zero observations and tests whether they tend to be positive.

**Calculation steps:**
1. For each metric, collect all per-(record,trial) judge std devs (or per-record trial std devs)
   across all selected models.
2. Remove exact zeros (zero_method equivalent: only non-zero values are ranked).
3. Run `scipy.stats.wilcoxon(vals, alternative="greater")` — tests H₁: median > 0.
4. A significant result (p < 0.05) means variance is reliably above zero for that metric.

**Note:** This is not corrected for multiple comparisons across metrics, as each metric is
treated as a separate analysis question.
""")

# ── Judge vs. trial variance ──────────────────────────────────────────────────
with tabs[2]:
    st.header("Judge vs. trial variance")
    st.write("""
    **What this measures:** Side-by-side comparison of judge variance (stochasticity from
    re-running the same judge) vs. trial variance (differences across simulation trials).

    **Why it matters:** Answers the core question — which source of variance dominates?
    If judge variance is much larger than trial variance, the benchmark results are noisy;
    if trial variance dominates, the noise comes from the simulation itself.
    """)

    combined = pd.merge(
        judge_summary[["run_id", "run_label", "metric", "mean_std", "std_of_std"]].rename(
            columns={"mean_std": "judge_std", "std_of_std": "judge_std_err"}
        ),
        traj_summary[["run_id", "run_label", "metric", "mean_std", "std_of_std"]].rename(
            columns={"mean_std": "trajectory_std", "std_of_std": "traj_std_err"}
        ),
        on=["run_id", "run_label", "metric"],
    )

    active_runs = [r for r in selected_runs if not combined[combined["run_id"] == r].empty]
    run_labels_cmp = [combined[combined["run_id"] == r]["run_label"].iloc[0] for r in active_runs]

    COLORS = {
        "Judge (stochasticity)": "#636EFA",
        "Trial (across scenarios)": "#EF553B",
    }
    y_max = global_var_ymax

    st.subheader("Judge vs. trial variance across models")
    st.caption(
        "Error bars = std dev of per-(record,trial) std devs across the group. "
        "Asterisks indicate a significant difference between judge and trial variance "
        "for that model × metric (paired Wilcoxon signed-rank, Bonferroni-corrected): "
        "\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001."
    )

    fig = make_subplots(rows=len(active_runs), cols=1, subplot_titles=run_labels_cmp, shared_xaxes=True)
    for row_idx, run_id in enumerate(active_runs, start=1):
        run_data = combined[combined["run_id"] == run_id]
        for source, color in COLORS.items():
            is_judge = source == "Judge (stochasticity)"
            fig.add_trace(
                go.Bar(
                    name=source,
                    x=run_data["metric"],
                    y=run_data["judge_std"] if is_judge else run_data["trajectory_std"],
                    error_y={
                        "type": "data",
                        "array": (run_data["judge_std_err"] if is_judge else run_data["traj_std_err"]).fillna(0),
                    },
                    marker_color=color,
                    legendgroup=source,
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(range=[0, y_max], row=row_idx, col=1)

    # Significance asterisks: one per (model × metric) pair where Q1a is significant
    q1a_sig = stat_results["q1a"]
    q1a_sig = q1a_sig[q1a_sig["significant"]] if not q1a_sig.empty else pd.DataFrame()
    y_pad = y_max * 0.04
    for row_idx, run_id in enumerate(active_runs, start=1):
        run_data = combined[combined["run_id"] == run_id]
        run_label = run_data["run_label"].iloc[0]
        xref_str = "x" if row_idx == 1 else f"x{row_idx}"
        yref_str = "y" if row_idx == 1 else f"y{row_idx}"
        model_sig = q1a_sig[q1a_sig["model"] == run_label] if not q1a_sig.empty else pd.DataFrame()
        for _, qrow in model_sig.iterrows():
            metric = qrow["metric"]
            mdata = run_data[run_data["metric"] == metric]
            if mdata.empty:
                continue
            y_top = max(
                mdata["judge_std"].iloc[0] + mdata["judge_std_err"].fillna(0).iloc[0],
                mdata["trajectory_std"].iloc[0] + mdata["traj_std_err"].fillna(0).iloc[0],
            )
            p_bonf = qrow["p_bonferroni"]
            stars = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*"
            fig.add_annotation(
                x=metric,
                y=y_top + y_pad,
                text=f"<b>{stars}</b>",
                showarrow=False,
                xref=xref_str,
                yref=yref_str,
                font={"size": 14, "color": "#444"},
            )

    fig.update_layout(
        barmode="group",
        height=200 * len(active_runs),
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
        margin={"t": 40, "r": 160},
    )
    # Resize only subplot-title annotations (those lack a yref data reference)
    for ann in fig.layout.annotations:
        if ann.yref == "paper":
            ann.font.size = 12
    st.plotly_chart(fig, width="stretch")

    st.subheader("Comparison table")
    display_combined = combined.rename(columns={"trajectory_std": "trial_std", "traj_std_err": "trial_std_err"})
    st.dataframe(display_combined.round(4), width="stretch")
    download_button(display_combined, "variance_comparison.csv")

    judge_dom = combined[combined["judge_std"] > combined["trajectory_std"]]
    if not judge_dom.empty:
        st.warning(
            f"**Key finding:** Judge variance exceeds trial variance for "
            f"{len(judge_dom)} metric/run combinations: "
            f"{', '.join(judge_dom['metric'].unique())}."
        )
    else:
        st.info("**Key finding:** Trial variance exceeds judge variance for all metric/run combinations.")

    # ── Q1 statistical results ────────────────────────────────────────────────
    st.subheader("Statistical test: Is judge variance significantly different from trial variance?")
    q1a = stat_results["q1a"]
    q1b = stat_results["q1b"]

    if q1a.empty:
        st.info("Not enough data to run statistical tests (need ≥ 5 paired records per model × metric).")
    else:
        st.markdown("**Q1a — Paired Wilcoxon signed-rank test (per model × metric, Bonferroni-corrected)**")
        q1a_disp = q1a[
            [
                "metric",
                "model",
                "n_records",
                "median_judge_std",
                "median_traj_std",
                "median_delta",
                "p_bonferroni",
                "significant",
                "direction",
            ]
        ].copy()
        q1a_disp["model"] = q1a_disp["model"].apply(_llm_name)
        q1a_disp["p_bonferroni"] = q1a_disp["p_bonferroni"].map(_fmt_p)
        st.dataframe(q1a_disp.round({"median_judge_std": 4, "median_traj_std": 4, "median_delta": 4}), width="stretch")

        if not q1b.empty:
            st.markdown("**Q1b — Does the gap vary by model? (Kruskal-Wallis on per-record deltas)**")
            q1b_disp = q1b[["metric", "H", "p", "significant"]].copy()
            q1b_disp["H"] = q1b_disp["H"].round(2)
            q1b_disp["p"] = q1b_disp["p"].map(_fmt_p)
            st.dataframe(q1b_disp, width="stretch")

        st.markdown("**Plain-English interpretation:**")
        for metric in sorted(selected_metrics):
            q1a_sub = q1a[q1a["metric"] == metric] if not q1a.empty else pd.DataFrame()
            q1b_row = (
                q1b[q1b["metric"] == metric].iloc[0] if (not q1b.empty and (q1b["metric"] == metric).any()) else None
            )

            if q1a_sub.empty:
                st.markdown(f"- **{metric}**: insufficient data for test.")
                continue

            sig_less = [r for _, r in q1a_sub.iterrows() if r["significant"] and r["direction"] == "judge < trial"]
            sig_more = [r for _, r in q1a_sub.iterrows() if r["significant"] and r["direction"] == "judge > trial"]
            not_sig = [r for _, r in q1a_sub.iterrows() if not r["significant"]]

            sentences = []
            if sig_less:
                sentences.append(
                    f"Judge variance is significantly less than trial variance for {_model_list(sig_less)}"
                )
            if sig_more:
                sentences.append(
                    f"Judge variance is significantly greater than trial variance for {_model_list(sig_more)}"
                )
            if not_sig:
                sentences.append(
                    f"Judge variance is not significantly different from trial variance for {_model_list(not_sig)}"
                )

            q1b_txt = ""
            if q1b_row is not None:
                if q1b_row["significant"]:
                    q1b_txt = f" The size of the gap varies significantly across models (K-W p={_fmt_p(q1b_row['p'])})."
                else:
                    q1b_txt = f" The gap is consistent across models (K-W p={_fmt_p(q1b_row['p'])})."

            st.markdown(f"- **{metric}**: " + ". ".join(sentences) + "." + q1b_txt)

        with st.expander("Methodology"):
            st.markdown("""
**Why paired Wilcoxon signed-rank (Q1a)?**
We're comparing two variance estimates for the same set of records. Using the same records in
both groups removes the scenario-difficulty confound — a hard scenario would inflate both judge
and trial variance. Wilcoxon signed-rank is the non-parametric equivalent of a paired t-test;
it doesn't assume normality of the differences.

**Calculation steps (Q1a):**
1. For each (record, trial, metric, model): compute judge std dev across the 3 judge iterations.
2. Average those judge std devs over trials → one judge-variance estimate per (record, model, metric).
3. Trial variance per record is from `compute_trajectory_variance()` — std dev of the mean score
   across 3 trials (after averaging over iterations to remove judge noise).
4. Pair the two estimates by `record_id` and run a Wilcoxon signed-rank test
   (scipy.stats.wilcoxon, two-sided, zero_method="wilcox").
5. Bonferroni correction: multiply each p-value by the number of models being tested.

**Why Kruskal-Wallis on deltas (Q1b)?**
Q1b asks whether the *gap* between judge and trial variance is consistent across models or
model-dependent. We compute delta = mean_judge_std − traj_std for each record in each model,
then run Kruskal-Wallis across models. If significant, the judge-vs-trial relationship differs
by model — i.e., some models have noisier judges relative to their trial variance than others.
""")

    # ── Q0 per model ─────────────────────────────────────────────────────────
    st.subheader("Is variance significantly greater than zero? (per model)")
    st.write(
        "One-sample Wilcoxon signed-rank against 0 (H₁: median > 0) for judge and trial "
        "variance separately, per model × metric. "
        "**n/a** means all std devs for that model × metric were exactly 0 — the judge was "
        "perfectly deterministic, so the test cannot be run. This is itself informative: "
        "no variance to measure for that combination."
    )
    _q0_jm = q0_results.get("q0_judge_per_model", pd.DataFrame())
    _q0_tm = q0_results.get("q0_trial_per_model", pd.DataFrame())
    _sig_map_jt = {True: "✓ yes", False: "✗ no"}

    if _q0_jm.empty or _q0_tm.empty:
        st.info("Not enough data to run Q0 tests.")
    else:
        for run_id in active_runs:
            run_label = combined[combined["run_id"] == run_id]["run_label"].iloc[0]
            st.markdown(f"**{_llm_name(run_label)}** — {run_label}")
            _jm_sub = _q0_jm[_q0_jm["model"] == run_label]
            _tm_sub = _q0_tm[_q0_tm["model"] == run_label]
            if _jm_sub.empty or _tm_sub.empty:
                st.info("No data for this model.")
                continue
            _merged = pd.merge(
                _jm_sub[["metric", "median_std", "p", "significant"]].rename(
                    columns={
                        "median_std": "judge median std",
                        "p": "judge p",
                        "significant": "judge sig?",
                    }
                ),
                _tm_sub[["metric", "median_std", "p", "significant"]].rename(
                    columns={
                        "median_std": "trial median std",
                        "p": "trial p",
                        "significant": "trial sig?",
                    }
                ),
                on="metric",
            ).sort_values("metric")
            _merged["judge p"] = _merged["judge p"].map(_fmt_p)
            _merged["trial p"] = _merged["trial p"].map(_fmt_p)
            _merged["judge sig?"] = _merged["judge sig?"].map(_sig_map_jt)
            _merged["trial sig?"] = _merged["trial sig?"].map(_sig_map_jt)
            st.dataframe(
                _merged.round({"judge median std": 4, "trial median std": 4}),
                hide_index=True,
                width="stretch",
            )

# ── Judge variance ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Judge variance (stochasticity)")
    st.write("""
    **What this measures:** For each (record, trial) pair, how much does the metric score vary
    across the 3 judge re-runs on identical conversation data?

    **Why it matters:** If std dev here is high, the judge is unreliable — two runs of the
    same benchmark on the same data could yield meaningfully different scores.
    """)

    # Judge std dev per metric
    st.subheader("Judge std dev per metric")
    st.caption(
        "Each bar = mean std dev of normalized scores across all (record, trial) pairs "
        "for that run × metric; error bars = std dev of those std devs. "
        "Tick label markers show overall significance of judge variance differences across "
        "models (Kruskal-Wallis: * p\u202f<\u202f0.05, ** p\u202f<\u202f0.01, "
        "*** p\u202f<\u202f0.001). "
        "Bar labels are compact letter display (CLD): bars with different letters differ "
        "significantly from each other (pairwise Mann-Whitney U, Bonferroni-corrected)."
    )
    st.plotly_chart(
        _variance_bar_fig(
            judge_summary,
            stat_results.get("q2_kw", pd.DataFrame()),
            stat_results.get("q2_pairwise", pd.DataFrame()),
            "Mean std dev (judge)",
            y_max=global_var_ymax,
            color_discrete_map=RUN_COLOR_MAP,
            run_label_order=RUN_LABEL_ORDER,
        ),
        width="stretch",
    )

    _judge_var_filtered = judge_var[judge_var["metric"].isin(selected_metrics)]

    fig = px.box(
        _judge_var_filtered,
        x="metric",
        y="std",
        color="run_label",
        points="all",
        hover_data=["record_id", "trial"],
        labels={"std": "Std dev (judge)", "metric": "Metric", "run_label": "Model(s)"},
        title="Distribution (median, IQR, all points)",
        color_discrete_map=RUN_COLOR_MAP,
        category_orders={"run_label": RUN_LABEL_ORDER},
    )
    fig.update_traces(jitter=0.4, pointpos=0)
    fig.update_layout(yaxis_range=[0, global_var_ymax], legend_title_text="Model(s)")
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    st.plotly_chart(fig, width="stretch")

    download_button(judge_summary, "judge_variance_overview.csv")

    # Detailed distribution plots
    st.plotly_chart(
        _variance_histogram(
            judge_var,
            x_label="Std dev (judge)",
            title="Distribution of per-(record,trial) judge std dev",
            color_discrete_map=RUN_COLOR_MAP,
            run_label_order=RUN_LABEL_ORDER,
        ),
        width="stretch",
    )

    fig = px.box(
        scores[scores["metric"].isin(selected_metrics)],
        x="iteration",
        y="normalized_score",
        color="run_label",
        facet_col="metric",
        facet_col_wrap=3,
        labels={"normalized_score": "Score", "iteration": "Iteration", "run_label": "Model(s)"},
        title="Score distributions across iterations",
        facet_row_spacing=0.15,
        height=500,
        color_discrete_map=RUN_COLOR_MAP,
        category_orders={"run_label": RUN_LABEL_ORDER},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title_font={"color": "#222"}, tickfont={"color": "#222"})
    fig.update_yaxes(title_font={"color": "#222"}, tickfont={"color": "#222"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Summary statistics")

    # Mean std dev per metric, averaged across all selected models
    _cross_model_mean = (
        judge_summary.groupby("metric")["mean_std"]
        .mean()
        .reset_index()
        .rename(columns={"mean_std": "mean_std (across models)"})
        .set_index("metric")
        .T.reindex(columns=sorted(judge_summary["metric"].unique()))
    )
    _cross_model_mean.index = ["mean judge std dev (all models)"]
    _cross_model_mean.index.name = None
    st.dataframe(_cross_model_mean.round(4), width="stretch")

    st.caption(
        "**Column guide for per-model table:** `mean_std` = mean std dev across (record, trial) pairs; "
        "`std_of_std` = spread of those std devs; "
        "`std_min` / `std_max` = observed min and max std dev; "
        "`mean_range` = mean of (max score − min score) across pairs — a simpler spread "
        "measure than std dev; "
        "`pct_changed` = fraction of pairs where the score differed across iterations; "
        "`n` = number of (record, trial) pairs."
    )
    st.dataframe(judge_summary.round(4), width="stretch")
    download_button(judge_summary, "judge_variance_summary.csv")

    st.subheader("Std dev range per model (max − min)")
    st.caption(
        "How much do per-(record, trial) std devs vary within each model × metric? "
        "High delta = some pairs are very consistent, others very noisy."
    )
    _jdelta = judge_summary.assign(std_delta=judge_summary["std_max"] - judge_summary["std_min"]).pivot(
        index="run_label", columns="metric", values="std_delta"
    )
    _jdelta.columns.name = None
    _jdelta.index.name = "model"
    st.dataframe(_jdelta.round(4), width="stretch")

    worst = judge_summary.loc[judge_summary["mean_std"].idxmax()]
    st.info(
        f"**Key finding:** Highest judge std dev is {worst['mean_std']:.4f} for "
        f"**{worst['metric']}** (run: {worst['run_label']}). "
        f"{worst['pct_changed']:.1%} of (record, trial) pairs changed score across iterations."
    )

    # ── Q2 statistical results ────────────────────────────────────────────────
    st.subheader("Statistical test: Does judge variance differ across models?")
    q2_kw = stat_results["q2_kw"]
    q2_pw = stat_results["q2_pairwise"]

    if q2_kw.empty:
        st.info("Not enough data to run statistical tests (need ≥ 2 models with ≥ 2 observations).")
    else:
        # Build significant-pairs lookup
        sig_pairs_by_metric: dict[str, str] = {}
        if not q2_pw.empty:
            for _metric, _grp in q2_pw[q2_pw["significant"]].groupby("metric"):
                sig_pairs_by_metric[_metric] = "; ".join(
                    f"{_llm_name(r['model_1'])} vs {_llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                    for _, r in _grp.iterrows()
                )

        q2_disp = q2_kw[["metric", "H", "p", "significant", "n_models"]].copy()
        q2_disp["H"] = q2_disp["H"].round(2)
        q2_disp["p"] = q2_disp["p"].map(_fmt_p)
        q2_disp["significant pairs"] = q2_disp["metric"].map(lambda m: sig_pairs_by_metric.get(m, "—"))
        st.dataframe(q2_disp, width="stretch")

        st.markdown("**Plain-English interpretation:**")
        for _, row in q2_kw.iterrows():
            metric = row["metric"]
            pw_rows = q2_pw[q2_pw["metric"] == metric] if not q2_pw.empty else pd.DataFrame()
            sig_pw = pw_rows[pw_rows["significant"]] if not pw_rows.empty else pd.DataFrame()
            hstr = f"H={row['H']:.2f}, p={_fmt_p(row['p'])}"
            if row["significant"] and not sig_pw.empty:
                pairs_txt = ", ".join(
                    f"{_llm_name(r['model_1'])} vs {_llm_name(r['model_2'])}" for _, r in sig_pw.iterrows()
                )
                st.markdown(
                    f"- **{metric}**: Judge variance differs significantly across models "
                    f"({hstr}). Significant pairs (Bonferroni-corrected): {pairs_txt}."
                )
            elif row["significant"]:
                st.markdown(
                    f"- **{metric}**: Judge variance differs significantly across models "
                    f"({hstr}), but no individual pair survives Bonferroni correction."
                )
            else:
                st.markdown(f"- **{metric}**: No significant difference in judge variance across models ({hstr}).")

        if len(_selected_types) > 1:
            st.markdown(
                "> **Note:** The table above pools cascade and S2S models together. "
                "A significant result may simply reflect the type difference rather than "
                "within-type variation. Results within each type are shown below."
            )

        for _rtype in ("cascade", "s2s"):
            _wt = within_type_results.get(_rtype, {})
            _wt_kw = _wt.get("q2_kw", pd.DataFrame())
            _wt_pw = _wt.get("q2_pairwise", pd.DataFrame())
            if _wt_kw.empty:
                continue
            st.markdown(f"**Within {_TYPE_LABELS.get(_rtype, _rtype)} models**")
            _wt_sig: dict[str, str] = {}
            if not _wt_pw.empty:
                for _m, _g in _wt_pw[_wt_pw["significant"]].groupby("metric"):
                    _wt_sig[_m] = "; ".join(
                        f"{_llm_name(r['model_1'])} vs {_llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                        for _, r in _g.iterrows()
                    )
            _wt_disp = _wt_kw[["metric", "H", "p", "significant", "n_models"]].copy()
            _wt_disp["H"] = _wt_disp["H"].round(2)
            _wt_disp["p"] = _wt_disp["p"].map(_fmt_p)
            _wt_disp["significant pairs"] = _wt_disp["metric"].map(lambda m: _wt_sig.get(m, "—"))  # noqa: B023
            st.dataframe(_wt_disp, width="stretch")

        with st.expander("Methodology"):
            st.markdown("""
**Why Kruskal-Wallis?**
Per-(record,trial) judge std devs are not normally distributed — they are right-skewed with
many zeros. Kruskal-Wallis is a non-parametric one-way ANOVA on ranks; it tests whether the
distributions of judge std devs differ across models without assuming normality.

**Calculation steps:**
1. For each metric, collect all per-(record,trial) judge std devs grouped by model.
   (These come from `compute_judge_variance()`: std dev of the 3 iteration scores for each pair.)
2. Run Kruskal-Wallis H test (scipy.stats.kruskal) across the groups.
3. If the overall test is significant (p < 0.05): run pairwise Mann-Whitney U tests
   for all pairs of models (scipy.stats.mannwhitneyu, two-sided).
4. Apply Bonferroni correction: multiply each p-value by the number of pairs
   (e.g., 3 pairs for 3 models).
5. Report rank-biserial correlation as effect size: r = 1 − 2U/(n₁·n₂).
   Values near ±1 = large effect; near 0 = negligible. Positive r means model 1 has
   higher std devs than model 2.
6. When both cascade and S2S models are selected, tests are also run within each type
   separately so that cross-type differences don't dominate the signal.
""")

# ── Trial variance ────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Trial variance (conversation-to-conversation)")
    st.write("""
    **What this measures:** For each scenario (record), how much does the metric score vary
    across the 3 simulation trials (different conversations of the same scenario)?
    Judge noise is removed by averaging scores across iterations before computing std dev.

    **Why it matters:** High trial variance means the same scenario plays out very
    differently across trials — the system is sensitive to conversation randomness.
    """)

    # Trial std dev per metric
    st.subheader("Trial std dev per metric")
    st.caption(
        "Each bar = mean std dev of normalized scores across records for that run × metric; "
        "error bars = std dev of those std devs. "
        "Tick label markers show overall significance of trial variance differences across "
        "models (Kruskal-Wallis: * p\u202f<\u202f0.05, ** p\u202f<\u202f0.01, "
        "*** p\u202f<\u202f0.001). "
        "Bar labels are compact letter display (CLD): bars with different letters differ "
        "significantly from each other (pairwise Mann-Whitney U, Bonferroni-corrected)."
    )
    st.plotly_chart(
        _variance_bar_fig(
            traj_summary,
            stat_results.get("q3_kw", pd.DataFrame()),
            stat_results.get("q3_pairwise", pd.DataFrame()),
            "Mean std dev (trial)",
            y_max=global_var_ymax,
            color_discrete_map=RUN_COLOR_MAP,
            run_label_order=RUN_LABEL_ORDER,
        ),
        width="stretch",
    )

    _traj_var_filtered = traj_var[traj_var["metric"].isin(selected_metrics)]

    fig = px.box(
        _traj_var_filtered,
        x="metric",
        y="std",
        color="run_label",
        points="all",
        hover_data=["record_id"],
        labels={"std": "Std dev (trial)", "metric": "Metric", "run_label": "Model(s)"},
        title="Distribution (median, IQR, all points)",
        color_discrete_map=RUN_COLOR_MAP,
        category_orders={"run_label": RUN_LABEL_ORDER},
    )
    fig.update_traces(jitter=0.4, pointpos=0)
    fig.update_layout(yaxis_range=[0, global_var_ymax], legend_title_text="Model(s)")
    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)
    st.plotly_chart(fig, width="stretch")

    download_button(traj_summary, "trial_variance_overview.csv")

    st.plotly_chart(
        _variance_histogram(
            traj_var,
            x_label="Std dev (trial)",
            title="Distribution of per-scenario trial std dev",
            color_discrete_map=RUN_COLOR_MAP,
            run_label_order=RUN_LABEL_ORDER,
        ),
        width="stretch",
    )

    st.subheader("Summary statistics")
    st.caption(
        "**Column guide:** `mean_std` = mean std dev across records; "
        "`std_of_std` = spread of those std devs; "
        "`std_min` / `std_max` = observed min and max std dev; "
        "`mean_range` = mean of (max score − min score) across records — a simpler spread "
        "measure than std dev; "
        "`n` = number of records."
    )
    st.dataframe(traj_summary.round(4), width="stretch")
    download_button(traj_summary, "trial_variance_summary.csv")

    st.subheader("Std dev range per model (max − min)")
    st.caption("How much do per-record std devs vary within each model × metric?")
    _tdelta = traj_summary.assign(std_delta=traj_summary["std_max"] - traj_summary["std_min"]).pivot(
        index="run_label", columns="metric", values="std_delta"
    )
    _tdelta.columns.name = None
    _tdelta.index.name = "model"
    st.dataframe(_tdelta.round(4), width="stretch")

    worst = traj_summary.loc[traj_summary["mean_std"].idxmax()]
    st.info(
        f"**Key finding:** Highest trial std dev is {worst['mean_std']:.4f} for "
        f"**{worst['metric']}** (run: {worst['run_label']})."
    )

    # ── Q3 statistical results ────────────────────────────────────────────────
    st.subheader("Statistical test: Does trial variance differ across models?")
    q3_kw = stat_results.get("q3_kw", pd.DataFrame())
    q3_pw = stat_results.get("q3_pairwise", pd.DataFrame())

    if q3_kw.empty:
        st.info("Not enough data to run statistical tests (need ≥ 2 models with ≥ 2 observations).")
    else:
        sig_pairs_q3: dict[str, str] = {}
        if not q3_pw.empty:
            for _metric, _grp in q3_pw[q3_pw["significant"]].groupby("metric"):
                sig_pairs_q3[_metric] = "; ".join(
                    f"{_llm_name(r['model_1'])} vs {_llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                    for _, r in _grp.iterrows()
                )

        q3_disp = q3_kw[["metric", "H", "p", "significant", "n_models"]].copy()
        q3_disp["H"] = q3_disp["H"].round(2)
        q3_disp["p"] = q3_disp["p"].map(_fmt_p)
        q3_disp["significant pairs"] = q3_disp["metric"].map(lambda m: sig_pairs_q3.get(m, "—"))
        st.dataframe(q3_disp, width="stretch")

        st.markdown("**Plain-English interpretation:**")
        for _, row in q3_kw.iterrows():
            metric = row["metric"]
            pw_rows = q3_pw[q3_pw["metric"] == metric] if not q3_pw.empty else pd.DataFrame()
            sig_pw = pw_rows[pw_rows["significant"]] if not pw_rows.empty else pd.DataFrame()
            hstr = f"H={row['H']:.2f}, p={_fmt_p(row['p'])}"
            if row["significant"] and not sig_pw.empty:
                pairs_txt = ", ".join(
                    f"{_llm_name(r['model_1'])} vs {_llm_name(r['model_2'])}" for _, r in sig_pw.iterrows()
                )
                st.markdown(
                    f"- **{metric}**: Trial variance differs significantly across models "
                    f"({hstr}). Significant pairs (Bonferroni-corrected): {pairs_txt}."
                )
            elif row["significant"]:
                st.markdown(
                    f"- **{metric}**: Trial variance differs significantly across models "
                    f"({hstr}), but no individual pair survives Bonferroni correction."
                )
            else:
                st.markdown(f"- **{metric}**: No significant difference in trial variance across models ({hstr}).")

        if len(_selected_types) > 1:
            st.markdown(
                "> **Note:** The table above pools cascade and S2S models together. "
                "A significant result may simply reflect the type difference rather than "
                "within-type variation. Results within each type are shown below."
            )

        for _rtype in ("cascade", "s2s"):
            _wt = within_type_results.get(_rtype, {})
            _wt_kw = _wt.get("q3_kw", pd.DataFrame())
            _wt_pw = _wt.get("q3_pairwise", pd.DataFrame())
            if _wt_kw.empty:
                continue
            st.markdown(f"**Within {_TYPE_LABELS.get(_rtype, _rtype)} models**")
            _wt_sig_q3: dict[str, str] = {}
            if not _wt_pw.empty:
                for _m, _g in _wt_pw[_wt_pw["significant"]].groupby("metric"):
                    _wt_sig_q3[_m] = "; ".join(
                        f"{_llm_name(r['model_1'])} vs {_llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                        for _, r in _g.iterrows()
                    )
            _wt_disp_q3 = _wt_kw[["metric", "H", "p", "significant", "n_models"]].copy()
            _wt_disp_q3["H"] = _wt_disp_q3["H"].round(2)
            _wt_disp_q3["p"] = _wt_disp_q3["p"].map(_fmt_p)
            _wt_disp_q3["significant pairs"] = _wt_disp_q3["metric"].map(lambda m: _wt_sig_q3.get(m, "—"))  # noqa: B023
            st.dataframe(_wt_disp_q3, width="stretch")

        with st.expander("Methodology"):
            st.markdown("""
**Same approach as Q2 (judge variance), applied to trial variance.**

Trial variance is measured as the std dev of a scenario's score across 3 simulation trials,
after averaging over judge iterations to remove judge noise. The resulting per-record std dev
is the unit of analysis here — one value per (record, model, metric).

**Why the same test (Kruskal-Wallis + Mann-Whitney U)?**
The question is structurally identical to Q2: does this variance measure differ across models?
The data have the same properties — bounded at zero, right-skewed — so the same non-parametric
approach applies. The one practical difference is that Q3 groups contain N records per model
(not N×K (record,trial) pairs as in Q2), so the groups are smaller and the test is slightly
less powerful for the same true effect size.

**Calculation steps:**
1. For each metric: group per-record trial std devs by model (from compute_trajectory_variance()).
2. Run Kruskal-Wallis H test (scipy.stats.kruskal) across the groups.
3. If significant: run pairwise Mann-Whitney U for all model pairs, Bonferroni-corrected.
4. Report rank-biserial correlation r = 1 − 2U/(n₁·n₂) as effect size.
5. When both cascade and S2S models are selected, tests are also run within each type
   separately so that cross-type differences don't dominate the signal.
""")

# ── EVA score stability ────────────────────────────────────────────────────────
with tabs[5]:
    st.header("EVA score stability")
    st.write("""
    **What this measures:** For each iteration, the headline EVA composite metrics are
    recomputed from scratch using that iteration's judge scores. If judge stochasticity
    is low, these numbers should be nearly identical across iterations.

    **Why it matters:** Even if individual metric std dev is small, it could systematically
    flip borderline scenarios, shifting the headline numbers that teams use to compare models.
    """)

    with st.expander("Composite metric definitions"):
        st.markdown("""
| Composite | Components | Pass condition |
|---|---|---|
| **EVA-A** | task_completion, faithfulness, agent_speech_fidelity | task_completion == 1.0 AND faithfulness >= 0.5 AND agent_speech_fidelity >= 0.95 |
| **EVA-X** | conversation_progression, turn_taking, conciseness | all >= 0.5 |
| **EVA-overall** | — | EVA-A pass AND EVA-X pass (derived) |
| **_mean variants** | same components as _pass | simple mean of normalized scores |

**Statistics computed per composite, per iteration:**
- **pass@1**: average fraction of trials that pass per scenario — expected pass rate for a single-trial benchmark
- **pass@k (k=3)**: fraction of scenarios where at least one trial passes — probability of seeing a passing result if you run once
- **pass^k (k=3)**: average (c/n)³ per scenario — theoretical probability all 3 draws pass (conservative lower bound)
- **mean**: mean of the composite value across all (record, trial) rows
        """)

    composite_df = compute_composite_stability(agg)

    if composite_df.empty:
        st.warning("No aggregate data available. Run the iterations first.")
    else:
        composite_cols = [c for c in composite_df.columns if c not in ("run_id", "run_label", "iteration")]
        # Exclude EVA-overall; sort so EVA-A metrics come before EVA-X within each group
        eva_ax_cols = sorted(
            [c for c in composite_cols if not _clean_composite_label(c).startswith("EVA-overall")],
            key=lambda c: _composite_sort_key(_clean_composite_label(c)),
        )

        melt_df = composite_df.melt(
            id_vars=["run_id", "run_label", "iteration"],
            value_vars=eva_ax_cols,
            var_name="composite_col",
            value_name="value",
        ).dropna(subset=["value"])
        melt_df["label"] = melt_df["composite_col"].map(_clean_composite_label)
        # Short label: strip "EVA-A " / "EVA-X " prefix for use as x-axis tick
        melt_df["x_label"] = melt_df["label"].str.split(" ", n=1).str[-1]

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

        # Use global color/symbol/group maps (defined at top level, consistent across all charts)
        _color_map = RUN_COLOR_MAP
        _symbol_map = RUN_SYMBOL_MAP
        _group_map = RUN_GROUP_MAP
        _GROUP_TITLE = {"cascade": "Cascade", "s2s": "S2S"}

        # ── Mean ± std scatter: EVA-A | EVA-X ─────────────────────────────────
        summary_df = (
            melt_df.groupby(["run_label", "label", "x_label", "composite_col"])["value"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        fig_scatter = make_subplots(
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
                category_orders={"x_label": short_labels, "run_label": RUN_LABEL_ORDER},
                color_discrete_map=_color_map,
            )
            for trace in tmp.data:
                rl = trace.name
                grp = _group_map.get(rl, "cascade")
                trace.showlegend = col_idx == 1
                trace.legendgroup = grp
                trace.legendgrouptitle = {"text": _GROUP_TITLE.get(grp, grp)}
                trace.marker.size = 6
                trace.marker.symbol = _symbol_map.get(rl, "circle")
                fig_scatter.add_trace(trace, row=1, col=col_idx)
            fig_scatter.update_xaxes(
                categoryorder="array",
                categoryarray=short_labels,
                tickangle=20,
                title_text="",
                row=1,
                col=col_idx,
            )
        fig_scatter.update_layout(
            title="Variance across judge iterations (mean ± std err)",
            height=450,
            yaxis_range=[0, 1.05],
            yaxis2_range=[0, 1.05],
            legend={
                "title": "",
                "orientation": "v",
                "yanchor": "middle",
                "y": 0.5,
                "xanchor": "left",
                "x": 1.02,
                "groupclick": "toggleitem",
            },
            margin={"r": 220, "b": 80},
        )
        fig_scatter.update_annotations(font_size=16)
        fig_scatter.update_yaxes(title_text="Value (mean ± std across iterations)")
        st.plotly_chart(fig_scatter, width="stretch")

        # ── Delta table (above values table) ──────────────────────────────────
        st.subheader("Delta (max − min) across iterations, per model")
        delta_df = composite_df.groupby("run_label")[eva_ax_cols].agg(lambda x: x.max() - x.min())
        delta_df.columns = [_clean_composite_label(c) for c in delta_df.columns]
        delta_df = delta_df.reset_index()
        _num_cols = delta_df.select_dtypes("number").columns
        delta_df[_num_cols] = delta_df[_num_cols].apply(lambda s: s.map(lambda v: float(f"{v:.2g}")))
        st.dataframe(delta_df, width="stretch")

        # ── Values per iteration (clean columns, EVA-A/X only) ────────────────
        st.subheader("Values per iteration")
        _vpi = composite_df[["run_id", "run_label", "iteration"] + eva_ax_cols].copy()
        _vpi = _vpi.rename(columns={c: _clean_composite_label(c) for c in eva_ax_cols})
        st.dataframe(_vpi.round(4), width="stretch")
        download_button(composite_df, "composite_stability.csv")

        ranges = composite_df.groupby("run_label")[eva_ax_cols].agg(lambda x: x.max() - x.min())
        max_range = ranges.max().max()
        max_metric = _clean_composite_label(ranges.max().idxmax())
        st.info(f"**Key finding:** Largest composite shift across iterations is {max_range:.4f} for **{max_metric}**.")

# ── Borderline scenarios ───────────────────────────────────────────────────────
with tabs[6]:
    st.header("Borderline scenarios")
    st.write("""
    **What this measures:** (Record, trial) pairs where judge stochasticity flipped the score
    across an EVA composite pass/fail threshold — meaning one judge iteration scored the metric
    as passing and another as failing on identical conversation data.

    **Why it matters:** These are the scenarios where judge noise directly affects benchmark
    conclusions. Only metrics that feed into EVA-A or EVA-X pass conditions are included;
    `transcription_accuracy_key_entities` is excluded (no composite pass threshold).
    """)

    crossings = _threshold_crossings(scores)

    # Fixed at 150 — the total number of (record, trial) pairs per model
    _y_max_pairs = 150

    # ── Bar chart: flip count per metric per model ────────────────────────────
    if crossings.empty:
        st.info("No pass/fail threshold crossings found in the selected data.")
    else:
        flip_counts_bar = crossings.groupby(["run_label", "metric"]).size().reset_index(name="flip_count")
        # Ensure all (run_label, metric) combinations appear even if count is 0
        _all_combos = pd.MultiIndex.from_product(
            [flip_counts_bar["run_label"].unique(), list(PASS_THRESHOLDS.keys())],
            names=["run_label", "metric"],
        )
        flip_counts_bar = (
            flip_counts_bar.set_index(["run_label", "metric"]).reindex(_all_combos, fill_value=0).reset_index()
        )
        fig = px.bar(
            flip_counts_bar,
            x="metric",
            y="flip_count",
            color="run_label",
            barmode="group",
            labels={"flip_count": "Pass/fail flips (record×trial pairs)", "metric": "Metric"},
            title="Judge-stochasticity pass/fail flips per metric",
            color_discrete_map=RUN_COLOR_MAP,
            category_orders={"run_label": RUN_LABEL_ORDER},
        )
        fig.update_layout(yaxis_range=[0, _y_max_pairs], legend_title_text="Model(s)")
        fig.update_xaxes(**_axis_style)
        fig.update_yaxes(**_axis_style)
        st.plotly_chart(fig, width="stretch")

        # ── Heatmap: one subplot per metric, x = model, y = record ──────────────
        st.subheader("Scenario instability heatmap")
        st.write(
            "Each cell shows how many trials (out of 3) had a pass/fail flip for that "
            "scenario and model. A cell of 3 means the score crossed the threshold on "
            "every trial — maximum judge instability for that scenario."
        )

        all_records = sorted(scores["record_id"].unique())
        threshold_metrics = list(PASS_THRESHOLDS.keys())
        active_run_ids = [r for r in selected_runs if not scores[scores["run_id"] == r].empty]
        # Shortened model labels for x-axis (LLM name only)
        short_labels = [_llm_name(scores[scores["run_id"] == r]["run_label"].iloc[0]) for r in active_run_ids]
        full_labels = [scores[scores["run_id"] == r]["run_label"].iloc[0] for r in active_run_ids]

        # Reversed viridis: yellow (light) = 0 flips, dark purple = 3 flips
        _viridis_4_rev = pc.sample_colorscale("Viridis", [1, 2 / 3, 1 / 3, 0])
        _disc_viridis = [
            [0.00, _viridis_4_rev[0]],
            [0.25, _viridis_4_rev[0]],
            [0.25, _viridis_4_rev[1]],
            [0.50, _viridis_4_rev[1]],
            [0.50, _viridis_4_rev[2]],
            [0.75, _viridis_4_rev[2]],
            [0.75, _viridis_4_rev[3]],
            [1.00, _viridis_4_rev[3]],
        ]

        n_metrics = len(threshold_metrics)
        if n_metrics > 0 and active_run_ids:
            fig_heat = make_subplots(
                rows=1,
                cols=n_metrics,
                subplot_titles=[m.replace("_", " ") for m in threshold_metrics],
                shared_yaxes=True,
                horizontal_spacing=0.04,
            )
            for col_idx, metric in enumerate(threshold_metrics, start=1):
                metric_crossings = crossings[crossings["metric"] == metric]
                # Count flips per (run_label, record_id)
                flip_per = metric_crossings.groupby(["run_id", "record_id"]).size().reset_index(name="flip_count")
                # Build pivot: y=record_id, x=model (short label)
                pivot_rows = []
                for run_id, short in zip(active_run_ids, short_labels):
                    sub = flip_per[flip_per["run_id"] == run_id].set_index("record_id")["flip_count"]
                    col_vals = [float(sub.get(rec, 0)) for rec in all_records]
                    pivot_rows.append((short, col_vals))
                # z shape: (n_records, n_models) — rows=records, cols=models
                z_vals = [[row[1][i] for row in pivot_rows] for i in range(len(all_records))]
                x_labels = [row[0] for row in pivot_rows]

                fig_heat.add_trace(
                    go.Heatmap(
                        z=z_vals,
                        x=x_labels,
                        y=all_records,
                        colorscale=_disc_viridis,
                        zmin=0,
                        zmax=3,
                        showscale=(col_idx == n_metrics),
                        colorbar={
                            "title": "# trials<br>flipped",
                            "tickvals": [0, 1, 2, 3],
                            "ticktext": ["0", "1", "2", "3"],
                            "len": 0.5,
                        },
                        hovertemplate=("record: %{y}<br>model: %{x}<br>trials flipped: %{z:.0f}<extra></extra>"),
                        customdata=[[full_labels[j] for j in range(len(active_run_ids))] for _ in all_records],
                    ),
                    row=1,
                    col=col_idx,
                )
                fig_heat.update_xaxes(tickangle=30, row=1, col=col_idx)
                fig_heat.update_yaxes(autorange="reversed", row=1, col=col_idx)

            _heat_height = max(600, len(all_records) * 22 + 160)
            fig_heat.update_layout(
                height=_heat_height,
                margin={"l": 140, "r": 120, "b": 140},
            )
            fig_heat.update_annotations(font_size=11)
            st.plotly_chart(fig_heat, width="stretch")

        download_button(crossings, "borderline_scenarios.csv")

        # ── Scenario instability ranking ───────────────────────────────────────
        st.subheader("Scenario instability ranking")
        st.write(
            "Total flip count per scenario across all metrics and models. "
            "Scenarios with flips across multiple models are universally sensitive; "
            "those concentrated in one model are model-specific."
        )
        _ranked = (
            crossings.groupby("record_id")
            .size()
            .reset_index(name="total_flips")
            .sort_values("total_flips", ascending=True)  # ascending so reversed axis puts highest at top
        )
        _max_flips = int(_ranked["total_flips"].max()) if not _ranked.empty else 1
        # Maximum possible: n_threshold_metrics × n_models × 3 trials
        _flip_ceiling = len(PASS_THRESHOLDS) * len(active_run_ids) * 3

        fig_rank = px.bar(
            _ranked,
            x="total_flips",
            y="record_id",
            orientation="h",
            labels={
                "total_flips": "Total pass/fail flips (all metrics × models × trials)",
                "record_id": "Scenario",
            },
            color="total_flips",
            color_continuous_scale=_disc_viridis,
            range_color=[0, _flip_ceiling],
        )
        fig_rank.update_layout(
            yaxis={"autorange": "reversed"},
            xaxis_range=[0, _flip_ceiling],
            coloraxis_showscale=False,
            height=max(400, len(_ranked) * 18 + 100),
            margin={"l": 140, "r": 40, "b": 60},
        )
        fig_rank.update_xaxes(**_axis_style)
        fig_rank.update_yaxes(**_axis_style)
        st.plotly_chart(fig_rank, width="stretch")

        # Breakdown: how many models contributed flips per scenario
        _model_flip_counts = (
            crossings.groupby(["record_id", "run_id"])
            .size()
            .reset_index(name="n")
            .groupby("record_id")["run_id"]
            .count()
            .reset_index(name="n_models_with_flips")
        )
        _ranked_detail = (
            _ranked.merge(_model_flip_counts, on="record_id", how="left")
            .sort_values("total_flips", ascending=False)
            .reset_index(drop=True)
        )
        _ranked_detail.index += 1
        _ranked_detail.index.name = "rank"
        st.dataframe(_ranked_detail, width="stretch")

# ── Intraclass correlation ────────────────────────────────────────────────────
with tabs[7]:
    st.header("Intraclass correlation (ICC)")
    st.write("""
    **What this measures:** ICC = σ²_scenario / σ²_total quantifies what fraction
    of score variance is attributable to *scenario identity* — i.e., how much of
    the spread in scores comes from some scenarios being consistently harder or
    easier, vs. noise from trial-to-trial conversation differences.

    **High ICC** → scores primarily reflect genuine scenario difficulty differences;
    the benchmark discriminates well across scenarios.
    **Low ICC** → scores are dominated by within-scenario noise; scenario identity
    explains little of the variance.

    Two pooled estimates and one per-model breakdown are shown.
    """)

    _icc_pm = icc_results.get("per_model", pd.DataFrame())
    _icc_pc = icc_results.get("pooled_centered", pd.DataFrame())
    _icc_tw = icc_results.get("pooled_twoway", pd.DataFrame())

    # ── Section A: Pooled ICC ─────────────────────────────────────────────────
    st.subheader("Pooled ICC")

    st.markdown("**Option A — Centered (within-model variance explained by scenario)**")
    st.caption(
        "Each model's mean score is subtracted before pooling. ICC answers: "
        "what fraction of within-model score variance is explained by scenario identity?"
    )
    if _icc_pc.empty:
        st.info("No pooled-centered ICC data available.")
    else:
        _pc_plot = _icc_pc.copy()
        _pc_plot["error_lower"] = _pc_plot["icc"] - _pc_plot["ci_lower"]
        _pc_plot["error_upper"] = _pc_plot["ci_upper"] - _pc_plot["icc"]
        _fig_pc = px.bar(
            _pc_plot,
            x="metric",
            y="icc",
            error_y="error_upper",
            error_y_minus="error_lower",
            labels={"icc": "ICC (scenario)", "metric": "Metric"},
            category_orders={"metric": sorted(_pc_plot["metric"].unique())},
            color_discrete_sequence=["#636EFA"],
        )
        _fig_pc.update_layout(yaxis_range=[0, 1], height=350)
        _fig_pc.update_xaxes(**_axis_style)
        _fig_pc.update_yaxes(**_axis_style)
        st.plotly_chart(_fig_pc, use_container_width=True)

    st.markdown("**Option B — Two-way random effects with interaction**")
    st.caption(
        "ICC_scenario = σ²_scenario / σ²_total. ICC_model = σ²_model / σ²_total. "
        "Both are fractions of total variance (scenario + model + interaction + residual). "
        "F-tests use MS_interaction as denominator (Cornfield-Tukey rule for random effects)."
    )
    if _icc_tw.empty:
        st.info("No two-way ICC data available.")
    else:
        _tw_melt = _icc_tw.melt(
            id_vars="metric",
            value_vars=["icc_scenario", "icc_model"],
            var_name="component",
            value_name="icc",
        )
        _tw_melt["component"] = _tw_melt["component"].map({"icc_scenario": "ICC scenario", "icc_model": "ICC model"})
        _err_lo = _icc_tw.set_index("metric")[["ci_lower_scenario", "ci_lower_model"]]
        _err_hi = _icc_tw.set_index("metric")[["ci_upper_scenario", "ci_upper_model"]]
        _tw_melt["ci_lower"] = _tw_melt.apply(
            lambda row: (
                _err_lo.loc[row["metric"], "ci_lower_scenario"]
                if row["component"] == "ICC scenario"
                else _err_lo.loc[row["metric"], "ci_lower_model"]
            ),
            axis=1,
        )
        _tw_melt["ci_upper"] = _tw_melt.apply(
            lambda row: (
                _err_hi.loc[row["metric"], "ci_upper_scenario"]
                if row["component"] == "ICC scenario"
                else _err_hi.loc[row["metric"], "ci_upper_model"]
            ),
            axis=1,
        )
        _tw_melt["error_lower"] = _tw_melt["icc"] - _tw_melt["ci_lower"]
        _tw_melt["error_upper"] = _tw_melt["ci_upper"] - _tw_melt["icc"]

        _fig_tw = px.bar(
            _tw_melt,
            x="metric",
            y="icc",
            color="component",
            barmode="group",
            error_y="error_upper",
            error_y_minus="error_lower",
            labels={"icc": "ICC", "metric": "Metric", "component": ""},
            category_orders={"metric": sorted(_tw_melt["metric"].unique())},
            color_discrete_map={
                "ICC scenario": "#636EFA",
                "ICC model": "#EF553B",
            },
        )
        _fig_tw.update_layout(yaxis_range=[0, 1], height=350)
        _fig_tw.update_xaxes(**_axis_style)
        _fig_tw.update_yaxes(**_axis_style)
        st.plotly_chart(_fig_tw, use_container_width=True)

        st.markdown("**Scenario × model interaction F-test**")
        st.caption(
            "A significant interaction means models do not rank scenarios consistently "
            "— some scenarios are disproportionately harder/easier for specific models."
        )
        _int_disp = _icc_tw[["metric", "f_interaction", "p_interaction", "sigma2_interaction", "sigma2_total"]].copy()
        _int_disp["f_interaction"] = _int_disp["f_interaction"].round(2)
        _int_disp["p_interaction"] = _int_disp["p_interaction"].map(_fmt_p)
        _int_disp["sigma2_interaction"] = _int_disp["sigma2_interaction"].round(4)
        _int_disp["% of total variance"] = (
            (_int_disp["sigma2_interaction"] / _int_disp["sigma2_total"].replace(0, float("nan"))) * 100
        ).round(1)
        _int_disp = _int_disp.drop(columns=["sigma2_total"])
        st.dataframe(_int_disp, hide_index=True, use_container_width=True)

    # ── Section B: Per-model ICC ──────────────────────────────────────────────
    st.subheader("Per-model ICC")
    st.caption(
        "One-way ANOVA per (model, metric): ICC = σ²_scenario / (σ²_scenario + σ²_residual). "
        "How much of this model's score variance is explained by which scenario it is?"
    )

    if _icc_pm.empty:
        st.info("No per-model ICC data available.")
    else:
        _pm_pivot = _icc_pm.pivot(index="metric", columns="run_label", values="icc")
        _hm_cols = [c for c in RUN_LABEL_ORDER if c in _pm_pivot.columns]
        _pm_pivot = _pm_pivot[_hm_cols]

        _fig_hm = px.imshow(
            _pm_pivot,
            color_continuous_scale="Blues",
            zmin=0,
            zmax=1,
            labels={"color": "ICC", "x": "Model", "y": "Metric"},
            aspect="auto",
            text_auto=".2f",
        )
        _fig_hm.update_layout(
            height=max(250, len(_pm_pivot) * 60 + 80),
            coloraxis_colorbar={"title": "ICC"},
            margin={"l": 200, "r": 40, "t": 30, "b": 80},
        )
        _fig_hm.update_xaxes(tickangle=30)
        st.plotly_chart(_fig_hm, use_container_width=True)

        _pm_plot = _icc_pm.copy()
        _pm_plot["error_lower"] = _pm_plot["icc"] - _pm_plot["ci_lower"]
        _pm_plot["error_upper"] = _pm_plot["ci_upper"] - _pm_plot["icc"]
        _fig_pm = px.bar(
            _pm_plot,
            x="metric",
            y="icc",
            color="run_label",
            barmode="group",
            error_y="error_upper",
            error_y_minus="error_lower",
            labels={"icc": "ICC (scenario)", "metric": "Metric", "run_label": "Model"},
            color_discrete_map=RUN_COLOR_MAP,
            category_orders={
                "metric": sorted(_pm_plot["metric"].unique()),
                "run_label": RUN_LABEL_ORDER,
            },
        )
        _fig_pm.update_layout(
            yaxis_range=[0, 1],
            height=400,
            legend_title_text="Model",
            legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
            margin={"r": 220},
        )
        _fig_pm.update_xaxes(**_axis_style)
        _fig_pm.update_yaxes(**_axis_style)
        st.plotly_chart(_fig_pm, use_container_width=True)

        with st.expander("Full per-model ICC table"):
            _pm_disp = _icc_pm[
                [
                    "run_label",
                    "metric",
                    "icc",
                    "ci_lower",
                    "ci_upper",
                    "sigma2_scenario",
                    "sigma2_residual",
                    "n_scenarios",
                    "n_trials",
                    "f_stat",
                    "p_value",
                ]
            ].copy()
            _pm_disp["p_value"] = _pm_disp["p_value"].map(_fmt_p)
            st.dataframe(_pm_disp.round(4), hide_index=True, use_container_width=True)
            download_button(_pm_disp, "icc_per_model.csv")

    # ── Key finding callout ───────────────────────────────────────────────────
    if not _icc_pc.empty:
        _icc_max = _icc_pc.loc[_icc_pc["icc"].idxmax()]
        _icc_min = _icc_pc.loc[_icc_pc["icc"].idxmin()]
        st.info(
            f"**Key finding:** Highest pooled ICC: **{_icc_max['metric']}** "
            f"({_icc_max['icc']:.2f}, 95% CI [{_icc_max['ci_lower']:.2f}–{_icc_max['ci_upper']:.2f}]). "
            f"Lowest: **{_icc_min['metric']}** "
            f"({_icc_min['icc']:.2f}, 95% CI [{_icc_min['ci_lower']:.2f}–{_icc_min['ci_upper']:.2f}])."
        )

# ── Per-metric deep dive ─────────────────────────────────────────────────────
with tabs[8]:
    st.header("Per-metric deep dive")
    st.write("""
    **What this measures:** For each metric, how does judge variance relate to trial
    variance at the individual scenario level? Scenarios in the upper-right are high-variance
    from *both* sources; upper-left are judge-driven; lower-right are trial-driven.

    **Why it matters:** Motivated by the original question about `transcription_accuracy_key_entities`
    varying 87–92% across runs — this scatter reveals whether that spread is judge noise or
    genuine differences in how the STT model transcribed different conversations.
    """)

    selected_metric = st.selectbox(
        "Metric",
        selected_metrics,
        index=selected_metrics.index("transcription_accuracy_key_entities")
        if "transcription_accuracy_key_entities" in selected_metrics
        else 0,
    )

    jv = judge_var[judge_var["metric"] == selected_metric][["run_id", "run_label", "record_id", "trial", "std"]].rename(
        columns={"std": "judge_std"}
    )
    tv = traj_var[traj_var["metric"] == selected_metric][["run_id", "run_label", "record_id", "std"]].rename(
        columns={"std": "traj_std"}
    )
    merged = pd.merge(jv, tv, on=["run_id", "run_label", "record_id"])

    if merged.empty:
        st.warning("No data for this metric in the selected runs.")
    else:
        fig = px.scatter(
            merged,
            x="traj_std",
            y="judge_std",
            color="run_label",
            hover_data=["record_id", "trial"],
            labels={"traj_std": "Trial std dev", "judge_std": "Judge std dev"},
            title=f"{selected_metric}: judge vs. trial variance per (record, trial)",
            color_discrete_map=RUN_COLOR_MAP,
            category_orders={"run_label": RUN_LABEL_ORDER},
        )
        fig.add_hline(
            y=merged["judge_std"].mean(), line_dash="dash", line_color="gray", annotation_text="Mean judge std"
        )
        fig.add_vline(
            x=merged["traj_std"].mean(), line_dash="dash", line_color="gray", annotation_text="Mean trial std"
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(merged.round(4), width="stretch")
        download_button(merged, f"deep_dive_{selected_metric}.csv")

        mean_j = merged["judge_std"].mean()
        mean_t = merged["traj_std"].mean()
        dominant = "judge stochasticity" if mean_j > mean_t else "trial differences"
        st.info(
            f"**Key finding:** For **{selected_metric}**, mean judge std dev = {mean_j:.4f}, "
            f"mean trial std dev = {mean_t:.4f}. "
            f"Variance is primarily driven by **{dominant}**."
        )

# ── Statistical tests (full details) ─────────────────────────────────────────
with tabs[9]:
    st.header("Statistical tests")
    st.write("""
    Full results for all statistical tests. High-level summaries with plain-English
    interpretations appear inline on the **Judge vs. trial variance**, **Judge variance**,
    and **Trial variance** tabs.
    """)

    q1a = stat_results.get("q1a", pd.DataFrame())
    q1b = stat_results.get("q1b", pd.DataFrame())
    q2_kw = stat_results.get("q2_kw", pd.DataFrame())
    q2_pw = stat_results.get("q2_pairwise", pd.DataFrame())
    q3_kw = stat_results.get("q3_kw", pd.DataFrame())
    q3_pw = stat_results.get("q3_pairwise", pd.DataFrame())

    # ── Q0 ────────────────────────────────────────────────────────────────────
    with st.expander("Q0 — Is variance significantly greater than zero? (→ Variance overview tab)"):
        st.markdown("One-sample Wilcoxon signed-rank (H₁: median > 0). Non-zero std devs only.")

        st.markdown("**Judge variance — pooled across models**")
        _q0jp = q0_results.get("q0_judge_pooled", pd.DataFrame())
        if _q0jp.empty:
            st.info("No results.")
        else:
            _q0jp_disp = _q0jp.copy()
            _q0jp_disp["p"] = _q0jp_disp["p"].map(_fmt_p)
            _q0jp_disp["W"] = _q0jp_disp["W"].round(1)
            st.dataframe(_q0jp_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
            download_button(_q0jp, "stat_q0_judge_pooled.csv")

        st.markdown("**Judge variance — per model**")
        _q0jm = q0_results.get("q0_judge_per_model", pd.DataFrame())
        if _q0jm.empty:
            st.info("No results.")
        else:
            _q0jm_disp = _q0jm.copy()
            _q0jm_disp["p"] = _q0jm_disp["p"].map(_fmt_p)
            _q0jm_disp["W"] = _q0jm_disp["W"].round(1)
            _q0jm_disp["model"] = _q0jm_disp["model"].apply(_llm_name)
            st.dataframe(_q0jm_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
            download_button(_q0jm, "stat_q0_judge_per_model.csv")

        st.markdown("**Trial variance — pooled across models**")
        _q0tp = q0_results.get("q0_trial_pooled", pd.DataFrame())
        if _q0tp.empty:
            st.info("No results.")
        else:
            _q0tp_disp = _q0tp.copy()
            _q0tp_disp["p"] = _q0tp_disp["p"].map(_fmt_p)
            _q0tp_disp["W"] = _q0tp_disp["W"].round(1)
            st.dataframe(_q0tp_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
            download_button(_q0tp, "stat_q0_trial_pooled.csv")

        st.markdown("**Trial variance — per model**")
        _q0tm = q0_results.get("q0_trial_per_model", pd.DataFrame())
        if _q0tm.empty:
            st.info("No results.")
        else:
            _q0tm_disp = _q0tm.copy()
            _q0tm_disp["p"] = _q0tm_disp["p"].map(_fmt_p)
            _q0tm_disp["W"] = _q0tm_disp["W"].round(1)
            _q0tm_disp["model"] = _q0tm_disp["model"].apply(_llm_name)
            st.dataframe(_q0tm_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
            download_button(_q0tm, "stat_q0_trial_per_model.csv")

        with st.expander("Q0 full methodology"):
            st.markdown("""
**Test choice:** One-sample Wilcoxon signed-rank test (scipy.stats.wilcoxon, alternative="greater").

**Why non-parametric?** Std devs are bounded at zero and right-skewed; a one-sample t-test
against 0 would violate normality assumptions. The Wilcoxon test ranks the absolute values
of the non-zero observations and tests whether they are systematically positive.

**Calculation steps:**
1. For each metric (and optionally each model): collect the relevant std devs.
   - Judge: per-(record,trial) std devs across 3 judge iterations.
   - Trial: per-record std devs across 3 trials (judge-noise-removed).
2. Remove exact zeros before ranking (equivalent to zero_method="wilcox").
3. Run scipy.stats.wilcoxon(nonzero_vals, alternative="greater").
4. p < 0.05 → variance is significantly greater than zero for this metric/model.

**Pooled vs. per-model:** Pooled analysis combines all models' std devs, giving higher power
to detect small but consistent variance. Per-model analysis can reveal if a specific model's
judge is unusually deterministic (e.g., always returns the same score for a metric).
""")

        # ── Q1a ───────────────────────────────────────────────────────────────────
    with st.expander("Q1a — Paired Wilcoxon: judge vs. trial variance (→ Judge vs. trial variance tab)"):
        if q1a.empty:
            st.info("No results (need ≥ 5 paired records per model × metric).")
        else:
            _q1a_disp = q1a.copy()
            _q1a_disp["p_raw"] = _q1a_disp["p_raw"].map(_fmt_p)
            _q1a_disp["p_bonferroni"] = _q1a_disp["p_bonferroni"].map(_fmt_p)
            _q1a_disp["W"] = _q1a_disp["W"].round(1)
            _q1a_disp["model"] = _q1a_disp["model"].apply(_llm_name)
            st.dataframe(
                _q1a_disp.round({"median_judge_std": 4, "median_traj_std": 4, "median_delta": 4}), width="stretch"
            )
            download_button(q1a, "stat_q1a_wilcoxon.csv")

        with st.expander("Q1a full methodology"):
            st.markdown("""
**Test choice:** Wilcoxon signed-rank test (non-parametric paired t-test).

**Why paired?** Both judge std dev and trial std dev are computed from the same set of records.
Pairing by record_id removes the scenario-difficulty confound — a hard scenario would inflate
both variance measures, making an unpaired comparison misleading.

**Why non-parametric?** The differences between paired judge/trial std devs are not normally
distributed (bounded at zero, right-skewed). The Wilcoxon test ranks the *absolute values* of
the differences and tests whether positive and negative ranks are symmetric around zero.

**Calculation steps:**
1. For each (record, trial, metric, model): judge std dev = std dev of scores across 3 iterations.
2. Average judge std devs over trials per record →
   one judge-variance estimate per (record, model, metric).
   (This matches the granularity of trial variance, which is computed per record.)
3. Trial variance per record = std dev of the iteration-averaged scores across 3 trials
   (from compute_trajectory_variance()).
4. Merge on record_id. For each model × metric pair:
   run scipy.stats.wilcoxon(judge_stds, traj_stds, alternative="two-sided", zero_method="wilcox").
   zero_method="wilcox" excludes tied pairs (where judge_std == traj_std) — conservative choice.
5. Bonferroni correction: multiply p_raw by the number of models tested per metric.
6. Direction is determined by the sign of the median delta (judge_std − traj_std).
""")

        # ── Q1b ───────────────────────────────────────────────────────────────────
    with st.expander(
        "Q1b — Kruskal-Wallis: does the judge-vs-trial gap vary across models? (→ Judge vs. trial variance tab)"
    ):
        if q1b.empty:
            st.info("No results.")
        else:
            _q1b_disp = q1b.copy()
            _q1b_disp["H"] = _q1b_disp["H"].round(3)
            _q1b_disp["p"] = _q1b_disp["p"].map(_fmt_p)
            st.dataframe(_q1b_disp, width="stretch")
            download_button(q1b, "stat_q1b_delta_kruskal_wallis.csv")

        with st.expander("Q1b full methodology"):
            st.markdown("""
**What this tests:** Whether the *gap* between judge and trial variance differs across models.

**Why a separate test?** Q1a answers "is judge variance different from trial variance for each
model?" but doesn't say whether that relationship is model-dependent. Q1b tests the interaction:
do different models have different judge-vs-trial variance ratios?

**Calculation steps:**
1. For each (record, model, metric): compute delta = mean_judge_std − traj_std.
   (Using the same per-record aggregation as Q1a.)
2. Group the deltas by model.
3. Run Kruskal-Wallis across groups (same rationale as Q2: non-parametric, right-skewed data).
4. A significant result means the gap is not uniform across models — some models may have
   judge variance that dominates more (or less) than trial variance compared to others.

**No Bonferroni correction here:** Q1b is one test per metric (not a family of pairwise tests),
so no correction is needed within this test. The 0.05 threshold is applied per metric.
""")

        # ── Q2 ────────────────────────────────────────────────────────────────────
    with st.expander("Q2 — Does judge variance differ across models? (→ Judge variance tab)"):
        st.markdown("**Kruskal-Wallis (overall test, per metric)**")
        if q2_kw.empty:
            st.info("No results (need ≥ 2 models).")
        else:
            _q2_kw_disp = q2_kw.copy()
            _q2_kw_disp["H"] = _q2_kw_disp["H"].round(3)
            _q2_kw_disp["p"] = _q2_kw_disp["p"].map(_fmt_p)
            st.dataframe(_q2_kw_disp, width="stretch")
            download_button(q2_kw, "stat_q2_kruskal_wallis.csv")

        st.markdown("**Pairwise Mann-Whitney U (Bonferroni-corrected)**")
        if q2_pw.empty:
            st.info("No pairwise results.")
        else:
            _q2_pw_disp = q2_pw.copy()
            _q2_pw_disp["p_raw"] = _q2_pw_disp["p_raw"].map(_fmt_p)
            _q2_pw_disp["p_bonferroni"] = _q2_pw_disp["p_bonferroni"].map(_fmt_p)
            _q2_pw_disp["U"] = _q2_pw_disp["U"].round(1)
            _q2_pw_disp["effect_r"] = _q2_pw_disp["effect_r"].round(3)
            st.dataframe(_q2_pw_disp, width="stretch")
            download_button(q2_pw, "stat_q2_pairwise.csv")

        if len(_selected_types) > 1:
            st.markdown(
                "> **Note:** Tables above pool all models. Within-type results (cascade / S2S "
                "separately) are shown below to isolate within-group variation."
            )
        for _rtype in ("cascade", "s2s"):
            _wt = within_type_results.get(_rtype, {})
            _wt_kw2 = _wt.get("q2_kw", pd.DataFrame())
            _wt_pw2 = _wt.get("q2_pairwise", pd.DataFrame())
            if _wt_kw2.empty:
                continue
            st.markdown(f"**Within {_TYPE_LABELS.get(_rtype, _rtype)} — K-W**")
            _d = _wt_kw2.copy()
            _d["H"] = _d["H"].round(3)
            _d["p"] = _d["p"].map(_fmt_p)
            st.dataframe(_d, width="stretch")
            if not _wt_pw2.empty:
                st.markdown(f"**Within {_TYPE_LABELS.get(_rtype, _rtype)} — pairwise MWU**")
                _dp = _wt_pw2.copy()
                _dp["p_raw"] = _dp["p_raw"].map(_fmt_p)
                _dp["p_bonferroni"] = _dp["p_bonferroni"].map(_fmt_p)
                _dp["U"] = _dp["U"].round(1)
                _dp["effect_r"] = _dp["effect_r"].round(3)
                st.dataframe(_dp, width="stretch")

        with st.expander("Q2 full methodology"):
            st.markdown("""
**Test choice:** Kruskal-Wallis (non-parametric one-way ANOVA on ranks).

**Why non-parametric?** Per-(record,trial) judge std devs are right-skewed with many zeros —
they do not satisfy the normality assumption required for a one-way ANOVA.

**Why ranks?** Kruskal-Wallis converts all observations to ranks across all groups combined,
then tests whether the average rank differs across groups. This is robust to skew and outliers.

**Calculation steps:**
1. For each metric: group per-(record,trial) judge std devs by model.
2. Run scipy.stats.kruskal(*groups). The H statistic approximates a χ² distribution
   with df = n_models − 1 under the null that all groups have the same distribution.
3. If p < 0.05: run pairwise Mann-Whitney U (scipy.stats.mannwhitneyu, two-sided) for
   all pairs of models.
4. Bonferroni correction: multiply each p_raw by the number of pairs
   (3 pairs for 3 models, capped at 1.0).
5. Effect size: rank-biserial correlation r = 1 − 2U/(n₁·n₂).
   Ranges from −1 to +1. Positive means model_1 tends to have larger std devs.
   Convention: |r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.

**Family-wise error:** The 0.05 threshold is applied per metric independently.
No cross-metric correction is applied, consistent with treating each metric as a
separate analysis question.
""")

        # ── Q3 ────────────────────────────────────────────────────────────────────
    with st.expander("Q3 — Does trial variance differ across models? (→ Trial variance tab)"):
        st.markdown("**Kruskal-Wallis (overall test, per metric)**")
        if q3_kw.empty:
            st.info("No results (need ≥ 2 models).")
        else:
            _q3_kw_disp = q3_kw.copy()
            _q3_kw_disp["H"] = _q3_kw_disp["H"].round(3)
            _q3_kw_disp["p"] = _q3_kw_disp["p"].map(_fmt_p)
            st.dataframe(_q3_kw_disp, width="stretch")
            download_button(q3_kw, "stat_q3_kruskal_wallis.csv")

        st.markdown("**Pairwise Mann-Whitney U (Bonferroni-corrected)**")
        if q3_pw.empty:
            st.info("No pairwise results.")
        else:
            _q3_pw_disp = q3_pw.copy()
            _q3_pw_disp["p_raw"] = _q3_pw_disp["p_raw"].map(_fmt_p)
            _q3_pw_disp["p_bonferroni"] = _q3_pw_disp["p_bonferroni"].map(_fmt_p)
            _q3_pw_disp["U"] = _q3_pw_disp["U"].round(1)
            _q3_pw_disp["effect_r"] = _q3_pw_disp["effect_r"].round(3)
            st.dataframe(_q3_pw_disp, width="stretch")
            download_button(q3_pw, "stat_q3_pairwise.csv")

        for _rtype in ("cascade", "s2s"):
            _wt = within_type_results.get(_rtype, {})
            _wt_kw3 = _wt.get("q3_kw", pd.DataFrame())
            _wt_pw3 = _wt.get("q3_pairwise", pd.DataFrame())
            if _wt_kw3.empty:
                continue
            st.markdown(f"**Within {_TYPE_LABELS.get(_rtype, _rtype)} — K-W**")
            _d3 = _wt_kw3.copy()
            _d3["H"] = _d3["H"].round(3)
            _d3["p"] = _d3["p"].map(_fmt_p)
            st.dataframe(_d3, width="stretch")
            if not _wt_pw3.empty:
                st.markdown(f"**Within {_TYPE_LABELS.get(_rtype, _rtype)} — pairwise MWU**")
                _dp3 = _wt_pw3.copy()
                _dp3["p_raw"] = _dp3["p_raw"].map(_fmt_p)
                _dp3["p_bonferroni"] = _dp3["p_bonferroni"].map(_fmt_p)
                _dp3["U"] = _dp3["U"].round(1)
                _dp3["effect_r"] = _dp3["effect_r"].round(3)
                st.dataframe(_dp3, width="stretch")

        with st.expander("Q3 full methodology"):
            st.markdown("""
**Test choice:** Same as Q2 — Kruskal-Wallis + pairwise Mann-Whitney U with Bonferroni correction.

**Why the same test?** The question is structurally identical to Q2: does this variance measure
differ across models? Trial std devs have the same distributional properties as judge std devs
(bounded at zero, right-skewed), so the same non-parametric approach is appropriate.

**Key difference from Q2:** Trial variance is at the per-record level (one std dev per record,
computed across 3 trials after averaging over judge iterations). Q2's groups contained N×K
(record,trial) observations; Q3's groups contain only N records per model. This means Q3 has
smaller groups and is slightly less powerful for the same true effect size.

**Calculation steps:**
1. For each metric: group per-record trial std devs by model (from compute_trajectory_variance()).
2. Run scipy.stats.kruskal(*groups).
3. If significant: pairwise Mann-Whitney U, Bonferroni-corrected.
4. Effect size: rank-biserial correlation r = 1 − 2U/(n₁·n₂).
""")

    # ── Q4 ────────────────────────────────────────────────────────────────────
    with st.expander("Q4 — Intraclass correlation (→ Intraclass correlation tab)"):
        st.markdown("**Per-model ICC (one-way ANOVA)**")
        if _icc_pm.empty:
            st.info("No results.")
        else:
            _q4_pm = _icc_pm[
                ["run_label", "metric", "icc", "ci_lower", "ci_upper", "f_stat", "p_value", "n_scenarios", "n_trials"]
            ].copy()
            _q4_pm["p_value"] = _q4_pm["p_value"].map(_fmt_p)
            st.dataframe(_q4_pm.round(4), hide_index=True, use_container_width=True)
            download_button(_icc_pm, "stat_q4_icc_per_model.csv")

        st.markdown("**Pooled ICC — centered (Option A)**")
        if _icc_pc.empty:
            st.info("No results.")
        else:
            st.dataframe(_icc_pc.round(4), hide_index=True, use_container_width=True)
            download_button(_icc_pc, "stat_q4_icc_pooled_centered.csv")

        st.markdown("**Pooled ICC — two-way random effects (Option B)**")
        if _icc_tw.empty:
            st.info("No results.")
        else:
            _q4_tw = _icc_tw.copy()
            for col in ["p_scenario", "p_model", "p_interaction"]:
                _q4_tw[col] = _q4_tw[col].map(_fmt_p)
            st.dataframe(_q4_tw.round(4), hide_index=True, use_container_width=True)
            download_button(_icc_tw, "stat_q4_icc_pooled_twoway.csv")

        with st.expander("Q4 full methodology"):
            st.markdown("""
**What ICC measures here**

ICC = σ²_scenario / σ²_total. Scores are first averaged over judge iterations (removing
judge noise), giving one score per (model, scenario, trial). ICC then asks: of all the
variance in those scores, what fraction is attributable to which scenario it is?

**Per-model: one-way ANOVA ICC(1,1)**

For each (model, metric): one-way ANOVA with n = 50 scenario groups, k = 3 trials per group.

Variance components:
- σ²_scenario = max(0, (MS_between − MS_within) / k)
- σ²_residual = MS_within

ICC = σ²_scenario / (σ²_scenario + σ²_residual)

Confidence intervals: Shrout & Fleiss (1979) exact CI using the F distribution:
- ci_lower = max(0, (F_obs/F_upper − 1) / (F_obs/F_upper + k − 1))
- ci_upper = min(1, (F_obs/F_lower − 1) / (F_obs/F_lower + k − 1))

where F_lower, F_upper are the α/2 and 1−α/2 critical values of F(df_between, df_within).

**Pooled centered (Option A)**

Each model's mean score is subtracted before pooling. The one-way ANOVA is then run with
k = n_models × n_trials observations per scenario group. This answers: what fraction of
within-model score variance is explained by scenario identity, pooled across models?
The centering removes model-level mean differences so they do not inflate σ²_residual.

**Pooled two-way (Option B)**

Model: Y_ijk = μ + α_i (scenario) + β_j (model) + (αβ)_ij (interaction) + ε_ijk (residual/trial)

All four terms are treated as random effects. Variance components from expected mean squares
(Cornfield-Tukey rule): the correct denominator for testing scenario and model main effects
is MS_interaction, not MS_residual, because the interaction term inflates the expected value
of MS_scenario and MS_model.

Variance components:
- σ²_residual    = MS_residual
- σ²_interaction = max(0, (MS_interaction − MS_residual) / n_trials)
- σ²_scenario    = max(0, (MS_scenario − MS_interaction) / (n_trials × n_models))
- σ²_model       = max(0, (MS_model − MS_interaction)    / (n_trials × n_scenarios))
- σ²_total       = σ²_scenario + σ²_model + σ²_interaction + σ²_residual

ICC_scenario = σ²_scenario / σ²_total
ICC_model    = σ²_model    / σ²_total

F-tests (random-effects Cornfield-Tukey denominators):
- F_scenario    = MS_scenario    / MS_interaction  (df = df_s, df_sm)
- F_model       = MS_model       / MS_interaction  (df = df_m, df_sm)
- F_interaction = MS_interaction / MS_residual     (df = df_sm, df_e)

Degrees of freedom (n_s=50, n_m=6, n_t=3 for most metrics):
- df_scenario = 49, df_model = 5, df_interaction = 245, df_residual = 600

Confidence intervals: Satterthwaite approximation for σ²_scenario:
- L = MS_scenario − MS_interaction (linear combination)
- Effective df: df_L = L² / (MS_scenario²/df_scenario + MS_interaction²/df_interaction)
- 95% CI on σ²_scenario: [df_L × σ²_scenario / χ²(df_L, 0.975),  df_L × σ²_scenario / χ²(df_L, 0.025)]
- CI on ICC_scenario: divide bounds by σ²_total (σ²_total treated as fixed — standard approximation)

**Interpreting negative variance components**

Variance component estimates can be slightly negative due to sampling variability when the
true value is near zero. These are clipped to 0. A clipped estimate means the factor explains
essentially none of the variance; no strong conclusion can be drawn about the sign.

**Interpreting the scenario × model interaction**

A significant interaction F-test (F_interaction, df_interaction = 245, df_residual = 600)
means models do not rank scenarios consistently — some scenarios are disproportionately
harder or easier for specific models. A non-significant interaction supports the additivity
assumption and means the benchmark discriminates scenarios consistently across models.
""")


# ── Tab persistence via URL param ─────────────────────────────────────────────
components.html(
    """
<script>
(function() {
    function run() {
        var tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        if (!tabs.length) { setTimeout(run, 100); return; }

        // On load: restore saved tab if not already active
        var params = new URLSearchParams(window.parent.location.search);
        var savedIdx = parseInt(params.get("tab") || "0");
        if (savedIdx > 0 && tabs[savedIdx] &&
                tabs[savedIdx].getAttribute("aria-selected") !== "true") {
            tabs[savedIdx].click();
        }

        // On click: update URL param (replaceState so back button isn't affected)
        tabs.forEach(function(tab, idx) {
            tab.addEventListener("click", function() {
                var url = new URL(window.parent.location);
                url.searchParams.set("tab", idx);
                window.parent.history.replaceState({}, "", url);
            });
        });
    }
    run();
})();
</script>
""",
    height=0,
)
