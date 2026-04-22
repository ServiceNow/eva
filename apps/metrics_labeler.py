#!/usr/bin/env python3
"""Streamlit labeling app for EVA test sets.

Supports three metrics:
  - conciseness           (per-turn ratings + failure modes + explanation)
  - conversation_progression (single whole-trace rating + explanation)
  - faithfulness          (single whole-trace rating + explanation)

Each record has 3 judges (judge_1, judge_2, judge_3) whose outputs are shown
side-by-side. Two reviews (different labeler names) are collected per record.

Usage:
    streamlit run apps/conciseness_labeler.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]

METRICS = {
    "conciseness": {
        "dataset": REPO / "eva_conciseness_test_set.jsonl",
        "labels": REPO / "conciseness_labels.json",
        "granularity": "per_turn",
    },
    "conversation_progression": {
        "dataset": REPO / "eva_conversation_progression_test_set.jsonl",
        "labels": REPO / "conversation_progression_labels.json",
        "granularity": "whole",
    },
    "faithfulness": {
        "dataset": REPO / "eva_faithfulness_test_set.jsonl",
        "labels": REPO / "faithfulness_labels.json",
        "granularity": "whole",
    },
}

JUDGE_KEYS = ["judge_1", "judge_2", "judge_3"]
RATING_CHOICES = [1, 2, 3]
REVIEWS_PER_RECORD = 2

FAILURE_MODES_BY_METRIC = {
    "conciseness": [
        "verbosity_or_filler",
        "excess_information_density",
        "over_enumeration_or_list_exhaustion",
        "contextually_disproportionate_detail",
    ],
    "faithfulness": [
        "fabricating_tool_parameters",
        "misrepresenting_tool_result",
        "violating_policies",
        "failing_to_disambiguate",
        "hallucination",
    ],
    "conversation_progression": [
        "unnecessary_tool_calls",
        "information_loss",
        "redundant_statements",
        "question_quality",
    ],
}
# Backwards-compat alias still used by normalize helper for conciseness.
FAILURE_MODE_CHOICES = FAILURE_MODES_BY_METRIC["conciseness"]


@st.cache_data
def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    migrated: dict = {}
    for rid, val in data.items():
        if isinstance(val, list):
            migrated[rid] = val
        elif isinstance(val, dict):
            migrated[rid] = [val] if (val.get("per_turn") or val.get("rating") is not None) else []
        else:
            migrated[rid] = []
    return migrated


def save_labels(path: Path, labels: dict) -> None:
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


def assistant_turns_by_id(conversation_trace: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for t in conversation_trace:
        if t.get("role") == "assistant" and t.get("content"):
            tid = t.get("turn_id")
            if tid is None:
                continue
            out.setdefault(int(tid), []).append(t)
    return out


def render_turn_block(trace: list[dict], turn_id: int) -> None:
    for t in trace:
        if t.get("turn_id") != turn_id:
            continue
        role = t.get("role")
        ttype = t.get("type")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(t.get("content") or "")
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(t.get("content") or "")
        elif ttype == "tool_call":
            with st.expander(f"🔧 tool_call: `{t.get('tool_name')}`", expanded=False):
                st.json(t.get("parameters") or {})
        elif ttype == "tool_response":
            with st.expander(f"↩️ tool_response: `{t.get('tool_name')}`", expanded=False):
                resp = t.get("tool_response")
                if isinstance(resp, (dict, list)):
                    st.json(resp)
                else:
                    st.code(str(resp))


def normalize_judge_failure_modes(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(s).strip() for s in raw]
    else:
        items = []
    return [i for i in items if i in FAILURE_MODE_CHOICES]


def _prettify(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _render_whole_trace_judge(idx: int, details: dict) -> None:
    """Parse judge details (rating + structured explanation dict) and display nicely."""
    rating = details.get("rating")
    explanation = details.get("explanation")

    st.markdown(f"### Judge {idx}")
    color = {1: "red", 2: "orange", 3: "green"}.get(rating, "gray")
    rating_display = rating if rating is not None else "—"
    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;color:{color};line-height:1.2;margin:0.25rem 0 0.5rem 0'>"
        f"Overall rating: {rating_display}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Most whole-trace judges return explanation as {"dimensions": {...}}.
    if isinstance(explanation, dict):
        dims = explanation.get("dimensions") or {}
        if dims:
            for dim_name, dim in dims.items():
                if not isinstance(dim, dict):
                    continue
                flagged = dim.get("flagged")
                dim_rating = dim.get("rating")
                evidence = dim.get("evidence", "")
                flag_icon = "🚩" if flagged else "✅"
                header = f"**{flag_icon} {_prettify(dim_name)}**"
                if dim_rating is not None:
                    header += f" — rating `{dim_rating}`"
                st.markdown(header)
                if evidence:
                    st.markdown(f"> {evidence}")
                st.markdown("")
        else:
            # Unknown dict shape — fall back to JSON rendering.
            st.json(explanation, expanded=True)
    elif isinstance(explanation, str) and explanation.strip():
        st.markdown(f"> {explanation}")
    else:
        st.caption("_No explanation._")


def render_judge_prompts(record: dict) -> None:
    """Single collapsible scrollable pane with the judge prompt (identical across judges)."""
    with st.expander("📜 Full judge prompt", expanded=True):
        prompt = ""
        for jkey in JUDGE_KEYS:
            prompt = (record.get(jkey, {}).get("details") or {}).get("judge_prompt", "")
            if prompt:
                break
        box = st.container(height=320, border=True)
        with box:
            if prompt:
                st.text(prompt)
            else:
                st.info("No judge_prompt available.")


def render_judges_per_turn(record: dict, turn_id: int) -> None:
    """For conciseness: stack each judge's per-turn rating/fm/explanation."""
    for i, jkey in enumerate(JUDGE_KEYS, start=1):
        judge = record.get(jkey, {}) or {}
        details = judge.get("details", {}) or {}
        rating = (details.get("per_turn_ratings") or {}).get(str(turn_id))
        fm_raw = (details.get("per_turn_failure_modes") or {}).get(str(turn_id))
        expl = (details.get("per_turn_explanations") or {}).get(str(turn_id), "")

        st.markdown(f"**Judge {i}**")
        cols = st.columns([1, 3])
        with cols[0]:
            st.markdown(f"Rating: **{rating if rating is not None else '—'}**")
        with cols[1]:
            if fm_raw:
                st.markdown(f"_Failure modes:_ `{fm_raw}`")
        if expl:
            st.markdown(f"> {expl}")
        st.markdown("")


def _metric_stats(metric: str) -> dict:
    cfg = METRICS[metric]
    records = load_dataset(str(cfg["dataset"]))
    labels = load_labels(cfg["labels"])
    total = len(records) * REVIEWS_PER_RECORD
    done = sum(min(len(labels.get(r["original_id"], [])), REVIEWS_PER_RECORD) for r in records)
    per_labeler: Counter = Counter()
    for reviews in labels.values():
        for r in reviews:
            name = (r.get("labeler") or "?").strip() or "?"
            per_labeler[name] += 1
    return {
        "num_records": len(records),
        "total": total,
        "done": done,
        "per_labeler": per_labeler,
    }


def _render_landing() -> None:
    st.title("EVA Labeling")
    st.caption("Pick a metric to start or continue labeling.")

    cols = st.columns(len(METRICS))
    for col, metric in zip(cols, METRICS.keys()):
        stats = _metric_stats(metric)
        with col:
            with st.container(border=True):
                st.markdown(f"### {metric}")
                pct = stats["done"] / stats["total"] if stats["total"] else 0.0
                st.progress(pct, text=f"{stats['done']} / {stats['total']} reviews")
                st.caption(f"{stats['num_records']} records × {REVIEWS_PER_RECORD} reviews")

                st.markdown("**By labeler**")
                if stats["per_labeler"]:
                    for name, count in stats["per_labeler"].most_common():
                        st.markdown(f"- **{name}** — {count}")
                else:
                    st.caption("_No reviews yet._")

                if st.button(
                    f"Open {metric} →",
                    key=f"open_{metric}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["selected_metric"] = metric
                    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="EVA Labeler",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not st.session_state.get("selected_metric"):
        _render_landing()
        return

    metric = st.session_state["selected_metric"]
    cfg = METRICS[metric]

    # --- Sidebar: back + labeler + record selection ---
    with st.sidebar:
        if st.button("← Back to metrics", use_container_width=True):
            st.session_state["selected_metric"] = None
            st.rerun()

        st.header("Labeler")
        labeler_name = st.text_input(
            "Your name",
            value=st.session_state.get("labeler_name", ""),
            key="labeler_name",
        )

        records = load_dataset(str(cfg["dataset"]))
        labels = load_labels(cfg["labels"])

        st.header("Record")
        idx = st.selectbox(
            "Select record",
            options=list(range(len(records))),
            format_func=lambda i: f"{i:02d}",
            key=f"record_idx_{metric}",
        )

        completed = sum(
            1 for r in records if len(labels.get(r["original_id"], [])) >= REVIEWS_PER_RECORD
        )
        st.caption(f"Metric: **{metric}**")
        st.caption(f"Dataset: `{cfg['dataset'].name}`")
        st.caption(f"Labels file: `{cfg['labels'].name}`")
        st.caption(f"Fully reviewed: {completed}/{len(records)}")

    record = records[idx]
    rid = record["original_id"]
    trace = record["conversation_trace"]

    st.title(f"EVA Labeler — {metric}")
    st.caption(f"Domain: **{record.get('domain', 'airline')}** · Record: **{rid}** (#{idx:02d})")

    reviews = labels.get(rid, [])
    name_norm = labeler_name.strip().lower()
    my_review = next(
        (r for r in reviews if name_norm and r.get("labeler", "").strip().lower() == name_norm),
        None,
    )
    other_reviews = [r for r in reviews if r is not my_review]

    if len(reviews) == 0:
        st.info(f"No reviews yet for **{rid}**. {REVIEWS_PER_RECORD} reviews needed.")
    elif my_review is not None:
        names = ", ".join(r.get("labeler", "?") for r in reviews)
        st.info(
            f"You already have a review for **{rid}** — editing it. "
            f"Total reviews: {len(reviews)}/{REVIEWS_PER_RECORD} ({names})."
        )
    elif len(reviews) >= REVIEWS_PER_RECORD:
        names = ", ".join(r.get("labeler", "?") for r in reviews)
        st.warning(
            f"⚠️ **{rid}** already has {len(reviews)}/{REVIEWS_PER_RECORD} reviews "
            f"(by {names}). Saving is disabled — pick another record."
        )
    else:
        names = ", ".join(r.get("labeler", "?") for r in reviews)
        st.info(
            f"**{rid}** has {len(reviews)}/{REVIEWS_PER_RECORD} reviews so far "
            f"(by {names}). You'll add the next one."
        )

    # Judge prompts
    render_judge_prompts(record)

    st.divider()

    # Scope for resetting form state when metric/record/labeler changes
    init_scope = f"{metric}::{name_norm}::{rid}"
    scope_changed = st.session_state.get("_last_scope") != init_scope
    st.session_state["_last_scope"] = init_scope

    can_save = my_review is not None or len(reviews) < REVIEWS_PER_RECORD

    if cfg["granularity"] == "per_turn":
        new_per_turn = _render_per_turn_mode(
            record, trace, rid, my_review, scope_changed
        )
        payload_builder = lambda: {"per_turn": new_per_turn}
    else:
        payload_builder = _render_whole_trace_mode(
            metric, record, trace, rid, my_review, scope_changed
        )

    # --- Save ---
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("💾 Save labels", type="primary", use_container_width=True, disabled=not can_save):
            name = labeler_name.strip()
            if not name:
                st.error("Please enter your name in the sidebar before saving.")
            elif my_review is None and any(
                r.get("labeler", "").strip().lower() == name.lower() for r in other_reviews
            ):
                st.error("A review with this name already exists for this record.")
            else:
                new_review = {
                    "labeler": name,
                    "target": record.get("target"),
                    **payload_builder(),
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
                updated = [r for r in reviews if r is not my_review] + [new_review]
                labels[rid] = updated
                save_labels(cfg["labels"], labels)
                st.success(f"Saved labels for {rid} (review {len(updated)}/{REVIEWS_PER_RECORD})")
    with col_b:
        for r in reviews:
            st.caption(f"• **{r.get('labeler','?')}** — saved {r.get('updated_at','?')}")


def _render_per_turn_mode(record, trace, rid, my_review, scope_changed) -> dict:
    """Render the per-turn UI (conciseness) and return the new per_turn dict."""
    ordered_turn_ids: list[int] = []
    for t in trace:
        tid = t.get("turn_id")
        if tid is None:
            continue
        tid = int(tid)
        if tid not in ordered_turn_ids:
            ordered_turn_ids.append(tid)

    asst_turns = assistant_turns_by_id(trace)
    source_per_turn = (my_review or {}).get("per_turn", {})

    new_per_turn: dict[str, dict] = {}

    st.subheader("Conversation & ratings")
    turns_box = st.container(height=700, border=True)
    with turns_box:
        for turn_id in ordered_turn_ids:
            st.subheader(f"Turn {turn_id}")
            left, right = st.columns([3, 2])

            with left:
                render_turn_block(trace, turn_id)

            with right:
                if turn_id not in asst_turns:
                    st.caption("_No assistant response in this turn._")
                    continue

                render_judges_per_turn(record, turn_id)

                st.markdown("---")
                st.markdown("**Your label**")

                rating_key = f"rating_{rid}_{turn_id}"
                fm_key = f"fm_{rid}_{turn_id}"
                expl_key = f"expl_{rid}_{turn_id}"

                existing = source_per_turn.get(str(turn_id), {})

                if scope_changed:
                    st.session_state[rating_key] = existing.get("rating")
                    st.session_state[fm_key] = list(existing.get("failure_modes", []))
                    st.session_state[expl_key] = existing.get("explanation", "")

                current = st.session_state.get(rating_key)
                rating_index = RATING_CHOICES.index(current) if current in RATING_CHOICES else None
                rating_value = st.radio(
                    "Your rating",
                    RATING_CHOICES,
                    index=rating_index,
                    horizontal=True,
                    key=rating_key,
                )
                fm_value = st.multiselect(
                    "Failure modes",
                    options=FAILURE_MODES_BY_METRIC["conciseness"],
                    key=fm_key,
                )
                expl_value = st.text_area(
                    "Quick explanation",
                    key=expl_key,
                    height=90,
                    placeholder="Briefly justify your rating…",
                )

                new_per_turn[str(turn_id)] = {
                    "rating": rating_value,
                    "failure_modes": fm_value,
                    "explanation": expl_value,
                }

            st.divider()

    return new_per_turn


def _render_whole_trace_mode(metric, record, trace, rid, my_review, scope_changed):
    """Render the whole-trace UI (conv_progression / faithfulness).
    Conversation is on the left, each judge gets its own column on the right —
    all scroll independently.
    Returns a zero-arg callable that produces the payload dict at save time."""
    st.subheader("Conversation & judges")

    pane_height = 700
    conv_col, judges_col = st.columns([3, 3])

    with conv_col:
        st.markdown("**Conversation**")
        conv_box = st.container(height=pane_height, border=True)
        with conv_box:
            ordered_turn_ids: list[int] = []
            for t in trace:
                tid = t.get("turn_id")
                if tid is None:
                    continue
                tid = int(tid)
                if tid not in ordered_turn_ids:
                    ordered_turn_ids.append(tid)
            for turn_id in ordered_turn_ids:
                st.markdown(f"**Turn {turn_id}**")
                render_turn_block(trace, turn_id)
                st.markdown("")

    with judges_col:
        st.markdown("**Judges**")
        judges_box = st.container(height=pane_height, border=True)
        with judges_box:
            for i, jkey in enumerate(JUDGE_KEYS, start=1):
                judge = record.get(jkey, {}) or {}
                details = judge.get("details", {}) or {}
                _render_whole_trace_judge(i, details)
                st.markdown("---")

    st.divider()

    # Your label (single rating + failure modes + explanation for whole trace)
    st.subheader("Your label")
    rating_key = f"whole_rating_{rid}"
    fm_key = f"whole_fm_{rid}"
    expl_key = f"whole_expl_{rid}"

    existing = my_review or {}

    if scope_changed:
        st.session_state[rating_key] = existing.get("rating")
        st.session_state[fm_key] = list(existing.get("failure_modes", []))
        st.session_state[expl_key] = existing.get("explanation", "")

    current = st.session_state.get(rating_key)
    rating_index = RATING_CHOICES.index(current) if current in RATING_CHOICES else None
    rating_value = st.radio(
        "Your rating",
        RATING_CHOICES,
        index=rating_index,
        horizontal=True,
        key=rating_key,
    )
    fm_options = FAILURE_MODES_BY_METRIC.get(metric, [])
    fm_value = st.multiselect(
        "Failure modes",
        options=fm_options,
        key=fm_key,
    )
    expl_value = st.text_area(
        "Quick explanation",
        key=expl_key,
        height=120,
        placeholder="Briefly justify your rating for the whole trace…",
    )

    def build_payload():
        return {
            "rating": rating_value,
            "failure_modes": fm_value,
            "explanation": expl_value,
        }

    return build_payload


if __name__ == "__main__":
    main()
