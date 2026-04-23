"""Side-by-side comparison of a clean vs perturbed benchmark run."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

CLEAN_RUN = "/Users/tara.bogavelli/third_eva/EVA-Bench3/output/2026-04-23_07-58-27.130648_scribe-v2_gpt-5.4-low_sonic-3"
PERTURBED_RUN = "/Users/tara.bogavelli/EVA-Bench/output/2026-04-23_08-01-35.295168_scribe-v2_gpt-5.4-low_sonic-3"

AUDIO_FILES = {
    "mixed": "audio_mixed.wav",
    "user": "audio_user.wav",
    "assistant": "audio_assistant.wav",
}

# Categories mirror src/eva/metrics/aggregation.py:EVA_COMPOSITES.
ACCURACY_METRICS = ["task_completion", "faithfulness", "agent_speech_fidelity"]
EXPERIENCE_METRICS = ["conversation_progression", "turn_taking", "conciseness"]
DIAGNOSTIC_METRICS = [
    "authentication_success",
    "conversation_correctly_finished",
    "pronunciation",
    "response_speed",
    "speakability",
    "stt_wer",
    "tool_call_validity",
    "transcription_accuracy_key_entities",
    "user_behavioral_fidelity",
    "user_speech_fidelity",
]

CATEGORIES = [
    ("Accuracy", ACCURACY_METRICS),
    ("Experience", EXPERIENCE_METRICS),
    ("Diagnostic", DIAGNOSTIC_METRICS),
]

JUDGE_EXPLANATION_METRICS = [
    "faithfulness",
    "conversation_progression",
    "conciseness",
    "agent_speech_fidelity",
    "pronunciation",
    "transcription_accuracy_key_entities",
]


@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float):
    with open(path) as f:
        return json.load(f)


def load_json(path: Path):
    if not path.exists():
        return None
    return _load_json(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _load_jsonl(path: str, mtime: float):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return _load_jsonl(str(path), path.stat().st_mtime)


def list_sample_ids(run_root: Path) -> set[str]:
    records = run_root / "records"
    if not records.is_dir():
        return set()
    return {p.name for p in records.iterdir() if p.is_dir()}


def list_trials(run_root: Path, sample_id: str) -> set[str]:
    sample = run_root / "records" / sample_id
    if not sample.is_dir():
        return set()
    return {p.name for p in sample.iterdir() if p.is_dir()}


def render_transcript(turns: list[dict]):
    if not turns:
        st.info("Empty transcript.")
        return
    for turn in turns:
        role = turn.get("type", "user")
        role_mapped = "assistant" if role == "assistant" else "user"
        with st.chat_message(role_mapped):
            st.markdown(turn.get("content", ""))


def metric_entry(blob: dict, name: str) -> dict:
    m = (blob.get("metrics") or {}).get(name) or {}
    return m if isinstance(m, dict) else {}


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _color_score(v):
    """Green → red gradient on [0, 1]. Returns CSS `background-color` string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "background-color: #f2f2f2; color: #666"
    try:
        x = max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return ""
    # red → yellow → green interpolation
    if x < 0.5:
        t = x / 0.5
        r, g, b = 230, int(100 + 155 * t), 100
    else:
        t = (x - 0.5) / 0.5
        r, g, b = int(230 - 130 * t), 200, int(100 + 50 * t)
    return f"background-color: rgba({r},{g},{b},0.55)"


def _color_delta(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x > 0:
        alpha = min(0.7, 0.2 + abs(x))
        return f"background-color: rgba(100, 200, 120, {alpha})"
    if x < 0:
        alpha = min(0.7, 0.2 + abs(x))
        return f"background-color: rgba(230, 110, 110, {alpha})"
    return "background-color: #f2f2f2"


def build_category_df(clean_blob: dict, pert_blob: dict, metric_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in metric_names:
        c = metric_entry(clean_blob, name)
        p = metric_entry(pert_blob, name)
        cs = c.get("normalized_score")
        ps = p.get("normalized_score")
        try:
            delta = float(ps) - float(cs) if cs is not None and ps is not None else None
        except (TypeError, ValueError):
            delta = None
        rows.append({
            "metric": name,
            "clean": cs,
            "perturbed": ps,
            "delta": delta,
            "skipped_clean": bool(c.get("skipped", False)) if c else None,
            "skipped_perturbed": bool(p.get("skipped", False)) if p else None,
        })
    return pd.DataFrame(rows)


def render_category(title: str, df: pd.DataFrame, clean_blob: dict, pert_blob: dict):
    st.markdown(f"#### {title}")
    if df.empty:
        st.info("No metrics in this category.")
        return
    styled = (
        df.style
        .format({"clean": _fmt, "perturbed": _fmt, "delta": _fmt})
        .map(_color_score, subset=["clean", "perturbed"])
        .map(_color_delta, subset=["delta"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _parse_maybe_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
    return value


def _score_badge(value) -> str:
    return f"`{_fmt(value)}`"


def _header_summary(entry: dict):
    if not entry:
        st.write("—")
        return
    parts = [f"score: {_score_badge(entry.get('score'))}", f"normalized: {_score_badge(entry.get('normalized_score'))}"]
    st.markdown(" • ".join(parts))
    if entry.get("error"):
        st.error(entry["error"])
    if entry.get("skipped"):
        st.caption("(skipped)")


def _render_dimensioned(entry: dict):
    """For faithfulness / conversation_progression: explanation is JSON with per-dimension evidence."""
    _header_summary(entry)
    if not entry:
        return
    details = entry.get("details") or {}
    rating = details.get("rating")
    if rating is not None:
        st.markdown(f"**Overall rating:** {rating}")
    parsed = _parse_maybe_json(details.get("explanation"))
    if isinstance(parsed, dict) and "dimensions" in parsed:
        for dim, body in parsed["dimensions"].items():
            if isinstance(body, dict):
                flagged = body.get("flagged")
                r = body.get("rating")
                icon = "🚩" if flagged else "✅"
                bits = [icon, f"**{dim}**"]
                if r is not None:
                    bits.append(f"(rating: {r})")
                st.markdown(" ".join(bits))
                ev = body.get("evidence")
                if ev:
                    st.markdown(f"> {ev}")
            else:
                st.markdown(f"**{dim}:** {body}")
    elif isinstance(parsed, str):
        st.markdown(parsed)
    elif parsed is not None:
        st.json(parsed, expanded=False)


def _iter_turns(details: dict):
    """Yield (turn_key_sort, turn_key_str, rating, explanation, extras)."""
    expl = details.get("per_turn_explanations") or {}
    ratings = details.get("per_turn_ratings") or {}
    if isinstance(expl, dict):
        keys = list(expl.keys())
    else:
        keys = list(range(len(expl)))
    def sort_key(k):
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))
    for k in sorted(keys, key=sort_key):
        e = expl[k] if isinstance(expl, dict) else expl[k]
        r = ratings.get(k) if isinstance(ratings, dict) else (ratings[k] if isinstance(ratings, list) and isinstance(k, int) and k < len(ratings) else None)
        yield k, r, e


def _render_per_turn_simple(entry: dict, show_rating: bool = True):
    """agent_speech_fidelity — rating + short explanation per turn."""
    _header_summary(entry)
    if not entry:
        return
    details = entry.get("details") or {}
    for k, r, e in _iter_turns(details):
        icon = "✅" if (isinstance(r, (int, float)) and r >= 1) else ("⚠️" if r is not None else "•")
        rating_txt = f" (rating: {r})" if show_rating and r is not None else ""
        st.markdown(f"{icon} **Turn {k}**{rating_txt}")
        if e:
            st.markdown(f"> {e}")


def _render_conciseness(entry: dict):
    _header_summary(entry)
    if not entry:
        return
    details = entry.get("details") or {}
    failure_modes = details.get("per_turn_failure_modes") or {}
    normalized = details.get("per_turn_normalized") or {}
    for k, r, e in _iter_turns(details):
        norm = normalized.get(k) if isinstance(normalized, dict) else None
        fms = failure_modes.get(k) if isinstance(failure_modes, dict) else None
        icon = "✅" if (isinstance(norm, (int, float)) and norm >= 1.0) else ("⚠️" if norm is not None else "•")
        head = f"{icon} **Turn {k}**"
        if r is not None:
            head += f" — rating: {r}"
        if isinstance(norm, (int, float)):
            head += f" • normalized: {norm:.2f}"
        st.markdown(head)
        if fms:
            st.markdown(f"failure modes: `{', '.join(fms)}`")
        if e:
            st.markdown(f"> {e}")


def _render_pronunciation(entry: dict):
    _header_summary(entry)
    if not entry:
        return
    details = entry.get("details") or {}
    transcripts = details.get("per_turn_transcripts") or {}
    errors = details.get("per_turn_errors") or {}
    for k, r, e in _iter_turns(details):
        icon = "✅" if (isinstance(r, (int, float)) and r >= 1) else ("⚠️" if r is not None else "•")
        head = f"{icon} **Turn {k}**" + (f" — rating: {r}" if r is not None else "")
        st.markdown(head)
        tr = transcripts.get(k) if isinstance(transcripts, dict) else None
        if tr:
            st.caption(f"transcript: {tr}")
        if e:
            st.markdown(f"> {e}")
        err = errors.get(k) if isinstance(errors, dict) else None
        if isinstance(err, dict):
            flagged_dims = [d for d, body in err.items() if isinstance(body, dict) and body.get("flagged")]
            if flagged_dims:
                for d in flagged_dims:
                    ev = err[d].get("evidence", "")
                    st.markdown(f"🚩 **{d}** — {ev}")


def _render_transcription_accuracy(entry: dict):
    _header_summary(entry)
    if not entry:
        return
    details = entry.get("details") or {}
    entity_details = details.get("per_turn_entity_details") or {}
    normalized = details.get("per_turn_normalized") or {}
    for k, r, e in _iter_turns(details):
        norm = normalized.get(k) if isinstance(normalized, dict) else None
        # -1 means N/A for this metric
        if isinstance(norm, (int, float)) and norm < 0:
            icon = "⚪"
        elif isinstance(norm, (int, float)) and norm >= 1.0:
            icon = "✅"
        elif norm is not None:
            icon = "⚠️"
        else:
            icon = "•"
        head = f"{icon} **Turn {k}**"
        if r is not None:
            head += f" — rating: {r}"
        st.markdown(head)
        if e:
            st.markdown(f"> {e}")
        ed = entity_details.get(k) if isinstance(entity_details, dict) else None
        if isinstance(ed, dict):
            entities = ed.get("entities") or []
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                correct = ent.get("correct")
                mark = "✅" if correct else ("🚩" if correct is False else "•")
                t = ent.get("type", "?")
                val = ent.get("value")
                tx = ent.get("transcribed_value")
                analysis = ent.get("analysis", "")
                st.markdown(f"{mark} `{t}` — expected: `{val}` | transcribed: `{tx}`")
                if analysis:
                    st.caption(analysis)


RENDERERS = {
    "faithfulness": _render_dimensioned,
    "conversation_progression": _render_dimensioned,
    "agent_speech_fidelity": _render_per_turn_simple,
    "conciseness": _render_conciseness,
    "pronunciation": _render_pronunciation,
    "transcription_accuracy_key_entities": _render_transcription_accuracy,
}


def render_judge_explanations(clean_blob: dict, pert_blob: dict):
    st.markdown("### Judge explanations")
    for name in JUDGE_EXPLANATION_METRICS:
        c = metric_entry(clean_blob, name)
        p = metric_entry(pert_blob, name)
        cs = _fmt(c.get("normalized_score")) if c else "—"
        ps = _fmt(p.get("normalized_score")) if p else "—"
        render = RENDERERS.get(name)
        with st.expander(f"{name}   —   clean: {cs}  |  perturbed: {ps}", expanded=False):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Clean**")
                if render:
                    render(c)
                else:
                    st.json(c, expanded=False)
            with col_r:
                st.markdown("**Perturbed**")
                if render:
                    render(p)
                else:
                    st.json(p, expanded=False)


def render_side_top(label: str, run_root: Path, sample_id: str, trial: str, track: str):
    st.subheader(label)
    trial_dir = run_root / "records" / sample_id / trial
    if not trial_dir.is_dir():
        st.warning(f"No such trial at `{trial_dir}`.")
        return None, None
    audio_path = trial_dir / AUDIO_FILES[track]
    if audio_path.exists():
        st.audio(str(audio_path))
    else:
        st.info(f"Missing `{AUDIO_FILES[track]}`.")
    return load_json(trial_dir / "metrics.json") or {}, trial_dir


def render_side_conversation(run_root: Path, sample_id: str, trial: str):
    trial_dir = run_root / "records" / sample_id / trial
    render_transcript(load_jsonl(trial_dir / "transcript.jsonl"))


def main():
    st.set_page_config(page_title="Run Compare", layout="wide")
    st.title("Clean vs Perturbed Run")

    with st.sidebar:
        st.header("Runs")
        clean_path = st.text_input("Clean run path", CLEAN_RUN)
        perturbed_path = st.text_input("Perturbed run path", PERTURBED_RUN)
        clean_root = Path(clean_path)
        perturbed_root = Path(perturbed_path)

        shared = sorted(list_sample_ids(clean_root) & list_sample_ids(perturbed_root))
        if not shared:
            st.error("No shared sample IDs between the two runs.")
            st.stop()

        sample_id = st.selectbox("Sample", shared)
        shared_trials = sorted(
            list_trials(clean_root, sample_id) & list_trials(perturbed_root, sample_id)
        )
        if not shared_trials:
            st.error(f"No shared trials for sample `{sample_id}`.")
            st.stop()
        trial = st.selectbox("Trial", shared_trials)
        track = st.radio("Audio track", list(AUDIO_FILES.keys()), index=0)

    cfg = load_json(perturbed_root / "config.json") or {}
    pert = (cfg.get("perturbation") or {})
    pert_desc = ", ".join(f"{k}={v}" for k, v in pert.items() if v not in (None, False)) or "none"
    st.markdown(f"**Sample** `{sample_id}` / `{trial}` — perturbations: `{pert_desc}`")

    col_clean, col_pert = st.columns(2)
    with col_clean:
        clean_blob, _ = render_side_top("Clean", clean_root, sample_id, trial, track)
    with col_pert:
        pert_blob, _ = render_side_top("Perturbed", perturbed_root, sample_id, trial, track)

    if clean_blob is None or pert_blob is None:
        return

    st.divider()
    for title, metric_names in CATEGORIES:
        df = build_category_df(clean_blob, pert_blob, metric_names)
        render_category(title, df, clean_blob, pert_blob)

    st.divider()
    render_judge_explanations(clean_blob, pert_blob)

    st.divider()
    st.markdown("### Conversation trace")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Clean**")
        render_side_conversation(clean_root, sample_id, trial)
    with col_r:
        st.markdown("**Perturbed**")
        render_side_conversation(perturbed_root, sample_id, trial)


if __name__ == "__main__":
    main()
