#!/usr/bin/env python3
"""Streamlit Analysis App for EVA benchmark results.

Visualizes benchmark outputs including transcripts, metrics, conversation traces,
and audio. Supports cross-run comparison, run-level overviews, and per-record
detail exploration.

Usage:
    streamlit run src/eva/app/analysis.py
"""

import html
import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from diff_viewer import diff_viewer

import eva.metrics  # noqa: F401
from apps.audio_plots import preload_audio_data, render_audio_analysis_tab
from eva.metrics.registry import get_global_registry
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, RecordMetrics

# ============================================================================
# Configuration
# ============================================================================

_DEFAULT_OUTPUT_DIR = os.environ.get("EVA_OUTPUT_DIR", "output")


def _build_metric_group_map() -> dict[str, str]:
    """Build metric name -> display category mapping from the metric registry."""
    registry = get_global_registry()
    return {
        # Normalize category to title case, take first segment if compound
        name: getattr(metric_class, "category", "other").split("/")[0].replace("_", " ").strip().title()
        for name, metric_class in registry.get_all().items()
    }


_METRIC_GROUP: dict[str, str] = _build_metric_group_map()

# Ordered categories for display; anything not listed sorts to the end
_CATEGORY_ORDER = ["Accuracy", "Experience", "Conversation Quality", "Diagnostic", "Validation"]

# Per-turn metrics keyed by the role they apply to
_KNOWN_PER_TURN_METRICS: dict[str, str] = {
    "transcription_accuracy_key_entities": "user",
    "stt_wer": "user",
    "agent_speech_fidelity": "assistant",
    "user_speech_fidelity": "user",
    "turn_taking": "assistant",
    "speech_fluency": "assistant",
    "llm_to_tts_accuracy": "assistant",
    "conciseness": "assistant",
    "speakability": "assistant",
    "conversation_progression": "assistant",
}

_ACRONYMS = {"tts", "stt", "wer", "llm", "db"}

# Categories to include in bar charts (exclude Diagnostic/Validation)
_BAR_CHART_CATEGORIES = {"Accuracy", "Experience"}

_CATEGORY_COLORS = {
    "Accuracy": "#3366CC",
    "Experience": "#FF9900",
    "Conversation Quality": "#22AA99",
    "Diagnostic": "#994499",
    "Validation": "#999999",
    "Other": "#AAAAAA",
}

_NON_NORMALIZED_METRICS = {"response_speed"}

# EVA composite scores to show in the bar chart
_EVA_BAR_COMPOSITES = ["EVA-A_pass", "EVA-X_pass", "EVA-A_mean", "EVA-X_mean"]

_EVA_COMPOSITE_DISPLAY = {
    "EVA-A_pass": "EVA-A pass@1",
    "EVA-X_pass": "EVA-X pass@1",
    "EVA-A_mean": "EVA-A Mean",
    "EVA-X_mean": "EVA-X Mean",
}


# ============================================================================
# Data Loading
# ============================================================================


def get_run_directories(output_dir: Path) -> list[Path]:
    """Get all run directories in output_dir, sorted newest first."""
    if not output_dir.exists():
        return []
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and (d / "records").exists()]
    return sorted(run_dirs, key=lambda d: d.name, reverse=True)


def _system_name_from_run(run_dir: Path) -> str:
    """Extract the system name from a run folder name (<timestamp>_<system_name>)."""
    m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+_(.+)$", run_dir.name)
    return m.group(1) if m else run_dir.name


def filter_latest_runs(run_dirs: list[Path]) -> list[Path]:
    """Keep only the most recent run per system name.

    Assumes run_dirs is already sorted newest-first (as returned by
    get_run_directories), so the first occurrence of each system name wins.
    """
    seen: set[str] = set()
    result = []
    for d in run_dirs:
        system = _system_name_from_run(d)
        if system not in seen:
            seen.add(system)
            result.append(d)
    return result


def get_record_directories(run_dir: Path) -> list[Path]:
    """Get all record directories in a run, sorted by record ID."""
    records_dir = run_dir / "records"
    if not records_dir.exists():
        return []
    return sorted([d for d in records_dir.iterdir() if d.is_dir()], key=lambda d: d.name)


def load_record_result(record_dir: Path) -> ConversationResult | None:
    """Load ConversationResult from result.json."""
    result_path = record_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        with open(result_path) as f:
            return ConversationResult(**json.load(f))
    except Exception as e:
        st.error(f"Failed to load result.json: {e}")
        return None


def load_record_metrics(record_dir: Path) -> RecordMetrics | None:
    """Load RecordMetrics from metrics.json."""
    metrics_path = record_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return RecordMetrics(**json.load(f))
    except Exception as e:
        st.error(f"Failed to load metrics.json: {e}")
        return None


def load_evaluation_record(run_dir: Path, record_id: str) -> EvaluationRecord | None:
    """Load EvaluationRecord from dataset referenced in config."""
    config_path = run_dir / "config.json"
    dataset_path = None
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                dataset_path = Path(config.get("dataset_path", ""))
        except Exception:
            pass

    if not dataset_path or not dataset_path.exists():
        return None

    try:
        records = EvaluationRecord.load_dataset(dataset_path)
        for record in records:
            if record.id == record_id:
                return record
        return None
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


def _load_run_config(run_dir: Path) -> dict:
    """Load config.json for a run."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_metrics_summary(run_dir: Path) -> dict:
    """Load metrics_summary.json for a run."""
    path = run_dir / "metrics_summary.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def format_transcript(transcript_path: Path) -> pd.DataFrame:
    """Load and format transcript.jsonl as a DataFrame."""
    if not transcript_path.exists():
        return pd.DataFrame()
    try:
        with open(transcript_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if not entries:
            return pd.DataFrame()
        df = pd.DataFrame(entries)
        cols = [c for c in ["timestamp", "role", "content"] if c in df.columns]
        cols.extend(c for c in df.columns if c not in cols)
        return df[cols] if cols else df
    except Exception as e:
        st.error(f"Failed to load transcript: {e}")
        return pd.DataFrame()


# ============================================================================
# Helpers
# ============================================================================


def _is_sub_metric(name: str) -> bool:
    """Return True if the metric name is a sub-metric (contains __ separator)."""
    return "__" in name


def _sort_metrics_by_category(metric_names: list[str]) -> list[str]:
    """Sort metric names grouped by category."""

    def _sort_key(m: str) -> tuple[int, str]:
        cat = _METRIC_GROUP.get(m, "Other")
        idx = _CATEGORY_ORDER.index(cat) if cat in _CATEGORY_ORDER else len(_CATEGORY_ORDER)
        return (idx, m)

    return sorted(metric_names, key=_sort_key)


def _make_category_header_rename(metric_names: list[str]) -> dict[str, str]:
    """Prefix metric names with their category for display."""
    return {m: f"[{_METRIC_GROUP.get(m, 'Other')}] {m}" for m in metric_names}


def _format_metric_name(name: str) -> str:
    """Format a metric name for display, preserving acronyms."""
    parts = name.split("_")
    return " ".join(p.upper() if p.lower() in _ACRONYMS else p.capitalize() for p in parts)


def _score_color(score: float | None) -> str:
    """Return a hex color for a normalized score."""
    if score is None:
        return "#888"
    if score >= 0.8:
        return "#4caf50"
    if score >= 0.5:
        return "#ff9800"
    return "#f44336"


def _classify_metrics(
    metrics: RecordMetrics,
) -> tuple[dict[str, dict], dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """Split metrics into conversation-level, user per-turn, and assistant per-turn."""
    conversation_metrics: dict[str, dict] = {}
    user_per_turn: dict[str, dict[str, dict]] = {}
    assistant_per_turn: dict[str, dict[str, dict]] = {}

    for name, metric_score in metrics.metrics.items():
        details = metric_score.details or {}
        per_turn_keys = [k for k in details if k.startswith("per_turn")]
        known_role = _KNOWN_PER_TURN_METRICS.get(name)

        if per_turn_keys or (known_role and not details):
            entry = {
                "score": metric_score.score,
                "normalized_score": metric_score.normalized_score,
                "details": details,
            }
            role = known_role or "user"
            if role == "assistant":
                assistant_per_turn[name] = entry
            else:
                user_per_turn[name] = entry
        else:
            conversation_metrics[name] = {
                "score": metric_score.score,
                "normalized_score": metric_score.normalized_score,
                "details": details,
                "error": metric_score.error,
            }

    return conversation_metrics, user_per_turn, assistant_per_turn


def _get_turn_metric_info(turn_id: int, per_turn_metrics: dict[str, dict]) -> list[dict]:
    """Get per-turn metric info (rating, explanation) for a given turn_id."""
    results = []
    tid = str(turn_id)
    for metric_name, metric_data in per_turn_metrics.items():
        details = metric_data["details"]
        rating = None
        explanation = ""

        if metric_name == "turn_taking":
            label = (details.get("per_turn_judge_timing_ratings") or {}).get(tid)
            if label is None:
                continue
            label_lower = label.strip().lower()
            if label_lower == "on-time":
                rating = 1.0
            elif label_lower == "late":
                rating = 0.3
            else:
                rating = 0.0
            latency = (details.get("per_turn_latency") or {}).get(tid)
            latency_str = f" ({latency:.3f}s)" if latency is not None else ""
            explanation = (details.get("per_turn_judge_timing_explanations") or {}).get(tid, "")
            results.append(
                {
                    "metric_name": metric_name,
                    "short_name": f"{label}{latency_str}",
                    "rating": rating,
                    "explanation": explanation,
                }
            )
            continue

        if "per_turn_ratings" in details and tid in details["per_turn_ratings"]:
            rating = details["per_turn_ratings"][tid]
        elif "per_turn_normalized" in details and tid in details["per_turn_normalized"]:
            rating = details["per_turn_normalized"][tid]
        elif "per_turn_wer" in details and tid in details["per_turn_wer"]:
            wer = details["per_turn_wer"][tid]
            rating = round(1.0 - wer, 3) if wer is not None else None

        if "per_turn_explanations" in details and tid in details["per_turn_explanations"]:
            explanation = details["per_turn_explanations"][tid]

        if rating is None:
            continue

        short_name = metric_name.replace("transcription_accuracy_", "").replace("_", " ")
        results.append(
            {
                "metric_name": metric_name,
                "short_name": short_name,
                "rating": rating,
                "explanation": explanation,
            }
        )
    return results


def _render_turn_metric_badges(turn_metrics: list[dict]) -> str:
    """Build HTML badge strings from turn metric info."""
    badges = []
    for info in turn_metrics:
        color = _score_color(info["rating"])
        tooltip = html.escape(info["explanation"], quote=True).replace("$", "&#36;") if info["explanation"] else ""
        badges.append(
            f'<span title="{tooltip}" style="display:inline-block; background:{color}22; '
            f"color:{color}; border:1px solid {color}44; border-radius:4px; "
            f'padding:1px 6px; margin:2px 4px 2px 0; font-size:0.75em; cursor:help;">'
            f"{info['short_name']}: {info['rating']:.2f}</span>"
        )
    return "".join(badges)


def _get_record_data_dirs(record_dir: Path) -> list[tuple[str, Path]]:
    """Get data directories for a record, handling trial/non-trial layouts."""
    trial_dirs = []
    if record_dir.exists():
        trial_dirs = sorted(
            [
                d
                for d in record_dir.iterdir()
                if d.is_dir() and any(f for f in d.iterdir() if f.suffix in (".json", ".wav", ".jsonl"))
            ],
            key=lambda d: d.name,
        )
    if trial_dirs:
        return [(d.name, d) for d in trial_dirs]
    return [("", record_dir)]


def _extract_model_details(run_config: dict) -> dict[str, str]:
    """Extract model details from a run config into a flat dict for display."""
    model_cfg = run_config.get("pipeline") or run_config.get("model") or {}
    details: dict[str, str] = {}

    # Speech-to-speech
    s2s = model_cfg.get("s2s") or model_cfg.get("realtime_model") or ""
    if s2s:
        s2s_params = model_cfg.get("s2s_params") or {}
        label = s2s_params.get("alias") or s2s_params.get("model") or s2s
        details["S2S"] = label
    else:
        # Audio LLM
        audio_llm = model_cfg.get("audio_llm") or ""
        if audio_llm:
            audio_llm_params = model_cfg.get("audio_llm_params") or {}
            details["Audio LLM"] = audio_llm_params.get("alias") or audio_llm_params.get("model") or audio_llm
        else:
            # Cascade: LLM
            llm = model_cfg.get("llm") or model_cfg.get("llm_model") or ""
            if llm:
                details["LLM"] = llm

        # STT (cascade only, not S2S/AudioLLM)
        if not audio_llm:
            stt = model_cfg.get("stt") or model_cfg.get("stt_model") or ""
            if stt:
                stt_params = model_cfg.get("stt_params") or {}
                label = stt_params.get("alias") or stt_params.get("model") or stt
                details["STT"] = f"{stt} ({label})" if label != stt else stt

        # TTS (cascade and AudioLLM)
        tts = model_cfg.get("tts") or model_cfg.get("tts_model") or ""
        if tts:
            tts_params = model_cfg.get("tts_params") or {}
            label = tts_params.get("alias") or tts_params.get("model") or tts
            details["TTS"] = f"{tts} ({label})" if label != tts else tts

    # Turn strategy
    turn_strategy = model_cfg.get("turn_strategy")
    if turn_strategy:
        details["Turn Strategy"] = turn_strategy

    return details


def _model_suffix_from_config(run_config: dict) -> str:
    """Build a short model suffix from config.json, mirroring folder naming on newer runs."""
    model_cfg = run_config.get("pipeline") or run_config.get("model") or {}
    if not model_cfg:
        return ""

    # Speech-to-speech
    s2s = model_cfg.get("s2s") or model_cfg.get("realtime_model") or ""
    if s2s:
        s2s_params = model_cfg.get("s2s_params") or {}
        parts = [s2s_params.get("alias") or s2s_params.get("model") or s2s]
        return "_".join(p for p in parts if p)

    # Audio LLM (2-part)
    audio_llm = model_cfg.get("audio_llm") or ""
    if audio_llm:
        audio_llm_params = model_cfg.get("audio_llm_params") or {}
        tts_params = model_cfg.get("tts_params") or {}
        parts = [
            audio_llm_params.get("alias") or audio_llm_params.get("model") or audio_llm,
            tts_params.get("alias") or tts_params.get("model") or model_cfg.get("tts") or "",
        ]
        return "_".join(p for p in parts if p)

    # Cascade (stt + llm + tts)
    stt_params = model_cfg.get("stt_params") or {}
    tts_params = model_cfg.get("tts_params") or {}
    parts = [
        stt_params.get("alias") or stt_params.get("model") or model_cfg.get("stt") or model_cfg.get("stt_model") or "",
        model_cfg.get("llm") or model_cfg.get("llm_model") or "",
        tts_params.get("alias") or tts_params.get("model") or model_cfg.get("tts") or model_cfg.get("tts_model") or "",
    ]
    return "_".join(p for p in parts if p)


_TIMESTAMP_RUN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)_(.+)$")
_TIMESTAMP_ONLY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)$")


def _get_system_and_timestamp(run_name: str, run_config: dict) -> tuple[str, str]:
    """Return (system_name, timestamp) for a run."""
    m = _TIMESTAMP_RUN_RE.match(run_name)
    if m:
        return m.group(2), m.group(1)
    # Timestamp-only directory (no system name suffix) — still extract the timestamp
    m = _TIMESTAMP_ONLY_RE.match(run_name)
    if m:
        suffix = _model_suffix_from_config(run_config)
        return suffix or "", m.group(1)
    suffix = _model_suffix_from_config(run_config)
    if suffix and suffix not in run_name:
        return f"{suffix} ({run_name})", ""
    return run_name, ""


def _get_run_label(run_name: str, run_config: dict) -> str:
    """Build a display label for a run (used for chart legends)."""
    system, timestamp = _get_system_and_timestamp(run_name, run_config)
    return f"{system} ({timestamp})" if timestamp else system


def _color_cell(val):
    """Color-code a metric cell value. Uses semi-transparent backgrounds for dark/light mode compat."""
    if pd.isna(val):
        return "background-color: rgba(128, 128, 128, 0.15)"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if val >= 0.8:
            return "background-color: rgba(76, 175, 80, 0.2)"
        elif val >= 0.5:
            return "background-color: rgba(255, 152, 0, 0.2)"
        else:
            return "background-color: rgba(244, 67, 54, 0.2)"
    return ""


def _collect_run_metrics(run_dir: Path) -> tuple[list[dict], list[str]]:
    """Collect all metrics rows for a run. Returns (rows, metric_names).

    Rows for failed attempts (directories named *_failed_attempt_*) are marked
    with ``_is_failed_attempt=True`` so the caller can filter them.
    """
    record_dirs = get_record_directories(run_dir)
    rows: list[dict] = []
    all_metric_names: set[str] = set()

    for record_dir in record_dirs:
        record_id = record_dir.name
        data_dirs = _get_record_data_dirs(record_dir)

        for trial_label, data_path in data_dirs:
            metrics = load_record_metrics(data_path)
            if not metrics:
                continue

            is_failed_attempt = "_failed_attempt_" in trial_label
            row: dict = {"record": record_id, "_is_failed_attempt": is_failed_attempt}
            if trial_label:
                row["trial"] = trial_label

            for metric_name, metric_score in metrics.metrics.items():
                all_metric_names.add(metric_name)
                row[metric_name] = (
                    None
                    if metric_score.error
                    else metric_score.normalized_score
                    if metric_score.normalized_score is not None
                    else metric_score.score
                )

                sub_metrics = getattr(metric_score, "sub_metrics", None)
                if sub_metrics:
                    for sub_key, sub_ms in sub_metrics.items():
                        col = f"{metric_name}__{sub_key}"
                        row[col] = (
                            None
                            if sub_ms.error
                            else sub_ms.normalized_score
                            if sub_ms.normalized_score is not None
                            else sub_ms.score
                        )
                        all_metric_names.add(col)
                        if col not in _METRIC_GROUP:
                            _METRIC_GROUP[col] = _METRIC_GROUP.get(metric_name, "Other")
                        if metric_name in _NON_NORMALIZED_METRICS:
                            _NON_NORMALIZED_METRICS.add(col)

            for comp_name, comp_value in metrics.aggregate_metrics.items():
                row[comp_name] = comp_value

            rows.append(row)

    return rows, sorted(all_metric_names)


def _extract_eva_scatter_point(
    run_dir: Path,
    run_config: dict,
    metrics_summary: dict,
) -> dict | None:
    """Extract EVA-A/EVA-X scores from a run's metrics_summary for all view modes.

    Returns a dict with keys: label, run, model_details, and per-view scores
    (pass_at_1, pass_at_k, pass_power_k, mean) each containing eva_a and eva_x.
    Returns None if no overall_scores available.
    """
    overall = metrics_summary.get("overall_scores", {})
    if not overall:
        return None

    model_suffix = _model_suffix_from_config(run_config)
    result: dict = {
        "run": run_dir.name,
        "label": _get_run_label(run_dir.name, run_config),
        "short_label": model_suffix or run_dir.name,
        "model_details": _extract_model_details(run_config),
        "pipeline_type": _classify_pipeline_type(run_config),
    }

    # pass@1: mean of EVA-A_pass and EVA-X_pass
    eva_a_pass = (overall.get("EVA-A_pass") or {}).get("mean")
    eva_x_pass = (overall.get("EVA-X_pass") or {}).get("mean")
    if eva_a_pass is not None and eva_x_pass is not None:
        result["pass_at_1"] = {"eva_a": eva_a_pass, "eva_x": eva_x_pass}

    # mean: mean of EVA-A_mean and EVA-X_mean
    eva_a_mean = (overall.get("EVA-A_mean") or {}).get("mean")
    eva_x_mean = (overall.get("EVA-X_mean") or {}).get("mean")
    if eva_a_mean is not None and eva_x_mean is not None:
        result["mean"] = {"eva_a": eva_a_mean, "eva_x": eva_x_mean}

    # pass@k and pass^k from pass_k section
    pass_k = overall.get("pass_k", {})
    eva_a_pk = pass_k.get("EVA-A_pass", {})
    eva_x_pk = pass_k.get("EVA-X_pass", {})
    if eva_a_pk.get("pass_at_k") is not None and eva_x_pk.get("pass_at_k") is not None:
        result["pass_at_k"] = {
            "eva_a": eva_a_pk["pass_at_k"],
            "eva_x": eva_x_pk["pass_at_k"],
        }
    if eva_a_pk.get("pass_power_k_observed") is not None and eva_x_pk.get("pass_power_k_observed") is not None:
        result["pass_power_k"] = {
            "eva_a": eva_a_pk["pass_power_k_observed"],
            "eva_x": eva_x_pk["pass_power_k_observed"],
        }

    return result


def _compute_pareto_frontier(points: list[dict]) -> list[dict]:
    """Compute the Pareto frontier (non-dominated points) from a list of {x, y} dicts."""
    frontier = []
    for p in points:
        dominated = any(q["x"] >= p["x"] and q["y"] >= p["y"] and (q["x"] > p["x"] or q["y"] > p["y"]) for q in points)
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p["x"])


_SCATTER_VIEW_MODES = {
    "pass@1": {
        "key": "pass_at_1",
        "description": (
            "Average of per-sample scores, where each sample scores 1 if all metrics "
            "in the category surpass metric-specific thresholds, else 0."
        ),
    },
    "pass@k (k=3)": {
        "key": "pass_at_k",
        "description": (
            "Percent of scenarios where at least 1 of k=3 trials surpasses "
            "metric-specific thresholds in all metrics in the category."
        ),
    },
    "pass^k (k=3)": {
        "key": "pass_power_k",
        "description": ("Per-scenario probability of all k=3 trials succeeding, averaged across scenarios."),
    },
    "Mean": {
        "key": "mean",
        "description": (
            "Average of per-sample scores, where each sample's score is the mean of the sub-metrics in that category."
        ),
    },
}

_MODEL_COLORS = px.colors.qualitative.Prism + px.colors.qualitative.Antique + px.colors.qualitative.Pastel

# Pipeline type colors (matching website ScatterPlot.tsx)
_PIPELINE_TYPE_COLORS = {
    "Cascade": "#A78BFA",  # purple
    "Audio-Native": "#34D399",  # emerald
    "Speech-to-Speech": "#F59E0B",  # amber
    "Unknown": "#9CA3AF",  # gray
}

# Pipeline type labels
_PIPELINE_CASCADE = "Cascade"
_PIPELINE_S2S = "Speech-to-Speech"
_PIPELINE_AUDIO_NATIVE = "Audio-Native"
_PIPELINE_UNKNOWN = "Unknown"


def _classify_pipeline_type(run_config: dict) -> str:
    """Classify a run's pipeline type from its config."""
    model_cfg = run_config.get("pipeline") or run_config.get("model") or {}
    if model_cfg.get("realtime_model") == "ultravox":
        return _PIPELINE_AUDIO_NATIVE
    if model_cfg.get("s2s") or model_cfg.get("realtime_model"):
        return _PIPELINE_S2S
    if model_cfg.get("audio_llm"):
        return _PIPELINE_AUDIO_NATIVE
    if model_cfg.get("llm") or model_cfg.get("llm_model"):
        return _PIPELINE_CASCADE
    return _PIPELINE_UNKNOWN


def _extract_llm_model_name(run_config: dict) -> str:
    """Extract the primary model name from config."""
    details = _extract_model_details(run_config)
    return next(
        (details[k] for k in ("S2S", "Audio LLM", "LLM") if k in details),
        "unknown",
    )


def _extract_all_models(run_config: dict) -> set[str]:
    """Extract all model names with their role (LLM/STT/TTS) from config."""
    details = _extract_model_details(run_config)
    models: set[str] = set()
    for role in ("S2S", "Audio LLM", "LLM"):
        if role in details:
            models.add(f"{details[role]} (LLM)")
            break
    if "STT" in details:
        models.add(f"{details['STT']} (STT)")
    if "TTS" in details:
        models.add(f"{details['TTS']} (TTS)")
    return models or {"unknown"}


def _extract_providers(run_config: dict) -> set[str]:
    """Extract all providers (LLM, STT, TTS) from config."""
    model_list = run_config.get("model_list") or []
    model_cfg = run_config.get("pipeline") or run_config.get("model") or {}
    providers: set[str] = set()

    # LLM provider from model_list
    primary = _extract_llm_model_name(run_config)
    for entry in model_list:
        if entry.get("model_name") == primary:
            litellm_model = (entry.get("litellm_params") or {}).get("model", "")
            if "qwen" in litellm_model:
                providers.add("qwen")
            elif "/" in litellm_model:
                providers.add(litellm_model.split("/")[0])
            break

    # Audio Native provider — prefer model name (e.g. ultravox) over serving platform (e.g. vllm)
    audio_llm_params = model_cfg.get("audio_llm_params") or {}
    audio_llm_provider = (
        audio_llm_params.get("alias") or audio_llm_params.get("model") or model_cfg.get("audio_llm") or ""
    )
    if audio_llm_provider:
        providers.add(audio_llm_provider)
    # S2S provider
    s2s = model_cfg.get("s2s")
    if s2s:
        providers.add(s2s)
    # Realtime provider
    realtime = model_cfg.get("realtime_model")
    if realtime:
        providers.add(realtime)
    # STT provider — normalize to base provider name
    stt = model_cfg.get("stt_model") or model_cfg.get("stt") or ""
    if stt:
        stt_model = (model_cfg.get("stt_params") or {}).get("model") or ""
        if "voxtral" in stt_model:
            providers.add("mistral")
        else:
            providers.add(stt.split("-")[0] if "-" in stt else stt)

    # TTS provider — normalize to base provider name
    tts = model_cfg.get("tts_model") or model_cfg.get("tts") or ""
    if tts:
        providers.add(tts.split("-")[0] if "-" in tts else tts)

    return providers or {"unknown"}


def _render_eva_scatter_plot(scatter_data: list[dict]):
    """Render the EVA-A vs EVA-X scatter plot with view mode selector."""
    # Filter to views that have data in at least one run
    available_views = {
        label: cfg for label, cfg in _SCATTER_VIEW_MODES.items() if any(cfg["key"] in d for d in scatter_data)
    }

    if not available_views:
        st.info(
            "No EVA composite scores available for scatter plot. "
            "Ensure runs have the required metrics (task_completion, faithfulness, "
            "agent_speech_fidelity, conversation_progression, turn_taking, conciseness)."
        )
        return

    view_labels = list(available_views.keys())
    selected_view = st.segmented_control(
        "View mode",
        view_labels,
        default=view_labels[0],
        key="scatter_view_mode",
        label_visibility="collapsed",
    )
    show_pareto = st.toggle("Pareto frontier", value=True, key="scatter_show_pareto")
    label_mode = st.segmented_control(
        "Labels",
        ["Pareto", "All", "None"],
        default="Pareto",
        key="scatter_label_mode",
    )
    if label_mode is None:
        label_mode = "Pareto"
    if selected_view is None:
        selected_view = view_labels[0]
    view_cfg = available_views[selected_view]
    view_key = view_cfg["key"]

    # Build plot data
    plot_points = []
    for d in scatter_data:
        scores = d.get(view_key)
        if scores and "eva_a" in scores and "eva_x" in scores:
            details = d.get("model_details", {})
            plot_points.append(
                {
                    "label": d["label"],
                    "short_label": d.get("short_label", d["label"]),
                    "x": scores["eva_a"],
                    "y": scores["eva_x"],
                    "llm": details.get("LLM", ""),
                    "stt": details.get("STT", ""),
                    "tts": details.get("TTS", ""),
                    "pipeline_type": d.get("pipeline_type", _PIPELINE_UNKNOWN),
                }
            )

    if not plot_points:
        st.info(f"No runs have data for the '{selected_view}' view.")
        return

    # Determine subscript label
    subscript_map = {
        "pass_at_1": "pass@1",
        "pass_at_k": "pass@k",
        "pass_power_k": "pass^k",
        "mean": "mean",
    }
    subscript = subscript_map.get(view_key, view_key)

    st.html(
        f"<div style='text-align:center'>"
        f"<h4 style='margin-bottom:0.2em'>EVA-A vs EVA-X ({subscript})</h4>"
        f"<p style='color:gray;font-size:0.85em;margin-top:0'>{view_cfg['description']}</p>"
        f"</div>"
    )

    fig = go.Figure()

    # Pareto frontier
    frontier = _compute_pareto_frontier([{"x": p["x"], "y": p["y"]} for p in plot_points])
    frontier_set = {(p["x"], p["y"]) for p in frontier}
    if show_pareto and len(frontier) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[p["x"] for p in frontier],
                y=[p["y"] for p in frontier],
                mode="lines",
                line={"color": "#06B6D4", "width": 2, "dash": "dash"},
                name="Pareto Frontier",
                hoverinfo="skip",
            )
        )

    # Data points — color by pipeline type
    # Track which types we've added to legend to avoid duplicates
    legend_types_shown: set[str] = set()
    for p in plot_points:
        ptype = p["pipeline_type"]
        color = _PIPELINE_TYPE_COLORS.get(ptype, _PIPELINE_TYPE_COLORS[_PIPELINE_UNKNOWN])
        hover_parts = [
            f"<b>{p['label']}</b>",
            f"EVA-A<sub>{subscript}</sub>: {p['x']:.3f}",
            f"EVA-X<sub>{subscript}</sub>: {p['y']:.3f}",
            f"Type: {ptype}",
            *(f"{model.upper()}: {p[model]}" for model in ("llm", "stt", "tts") if p[model]),
        ]

        on_frontier = (p["x"], p["y"]) in frontier_set
        show_text = label_mode == "All" or (label_mode == "Pareto" and on_frontier)
        show_legend = ptype not in legend_types_shown
        legend_types_shown.add(ptype)
        fig.add_trace(
            go.Scatter(
                x=[p["x"]],
                y=[p["y"]],
                mode="markers+text" if show_text else "markers",
                marker={"color": color, "size": 14},
                text=[p["short_label"]] if show_text else None,
                textposition="middle right",
                textfont={"size": 10},
                name=ptype,
                legendgroup=ptype,
                showlegend=show_legend,
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis={
            "title": f"Accuracy (EVA-A<sub>{subscript}</sub>)",
            "range": [-0.05, 1.05],
            "dtick": 0.2,
            "gridcolor": "rgba(128,128,128,0.2)",
            "constrain": "domain",
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "title": f"Experience (EVA-X<sub>{subscript}</sub>)",
            "range": [-0.05, 1.05],
            "dtick": 0.2,
            "gridcolor": "rgba(128,128,128,0.2)",
            "constrain": "domain",
        },
        height=700,
        showlegend=True,
        legend={"yanchor": "middle", "y": 0.5},
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 60, "r": 10, "t": 10, "b": 60},
    )

    st.plotly_chart(fig)


# ============================================================================
# Views
# ============================================================================


def render_cross_run_comparison(run_dirs: list[Path]):
    """Render a comparison view across multiple runs."""
    st.markdown("### Cross-Run Comparison")
    st.caption("Compare aggregate metrics across all runs that have metrics data.")

    # Pre-load configs for filtering
    run_configs = {d.name: _load_run_config(d) for d in run_dirs}

    # Collect filter options
    all_models: set[str] = set()
    all_providers: set[str] = set()
    all_types: set[str] = set()
    for cfg in run_configs.values():
        all_models.update(_extract_all_models(cfg))
        all_providers.update(_extract_providers(cfg))
        all_types.add(_classify_pipeline_type(cfg))

    # Render filters
    sel_types = st.multiselect("Pipeline Type", sorted(all_types), default=all_types, key="type", bind="query-params")
    sel_providers = st.multiselect(
        "Provider", sorted(all_providers), default=all_providers, key="provider", bind="query-params"
    )
    sel_models = st.multiselect("Model", sorted(all_models), default=all_models, key="model", bind="query-params")

    # Apply filters
    filtered_dirs = [
        d
        for d in run_dirs
        if _extract_all_models(run_configs[d.name]) & set(sel_models)
        and _extract_providers(run_configs[d.name]) & set(sel_providers)
        and _classify_pipeline_type(run_configs[d.name]) in sel_types
    ]

    if not filtered_dirs:
        st.warning("No runs match the selected filters.")
        return

    run_summaries: list[dict] = []
    all_metric_names: set[str] = set()
    scatter_data: list[dict] = []

    for run_dir in filtered_dirs:
        run_name = run_dir.name
        run_config = run_configs[run_name]
        metrics_summary = _load_metrics_summary(run_dir)

        # Try metrics_summary.json first, fall back to per-record loading
        if metrics_summary and "per_metric" in metrics_summary:
            per_metric = metrics_summary["per_metric"]
            metric_names = list(per_metric.keys())
            all_metric_names.update(metric_names)
            model_details = _extract_model_details(run_config)
            system_name, run_timestamp = _get_system_and_timestamp(run_name, run_config)
            summary: dict = {
                "run": run_name,
                "run_output_dir": str(run_dir.parent),
                "label": _get_run_label(run_name, run_config),
                "system_name": system_name,
                "run_timestamp": run_timestamp,
                "records": metrics_summary.get("total_records", 0),
                "pipeline_type": _classify_pipeline_type(run_config),
                **model_details,
            }
            for m, stats in per_metric.items():
                if stats.get("mean") is not None:
                    summary[m] = stats["mean"]
                for sub_key, sub_stats in (stats.get("sub_metrics") or {}).items():
                    if sub_stats.get("mean") is not None:
                        col = f"{m}__{sub_key}"
                        summary[col] = sub_stats["mean"]
                        all_metric_names.add(col)
                        if col not in _METRIC_GROUP:
                            _METRIC_GROUP[col] = _METRIC_GROUP.get(m, "Other")
                        if m in _NON_NORMALIZED_METRICS:
                            _NON_NORMALIZED_METRICS.add(col)
            # Add EVA composite scores from overall_scores
            overall = metrics_summary.get("overall_scores", {})
            for composite in _EVA_BAR_COMPOSITES:
                val = (overall.get(composite) or {}).get("mean")
                if val is not None:
                    summary[composite] = val
            run_summaries.append(summary)
        else:
            rows, metric_names = _collect_run_metrics(run_dir)
            if not rows:
                continue
            all_metric_names.update(metric_names)
            df = pd.DataFrame(rows)
            model_details = _extract_model_details(run_config)
            system_name, run_timestamp = _get_system_and_timestamp(run_name, run_config)
            summary = {
                "run": run_name,
                "run_output_dir": str(run_dir.parent),
                "label": _get_run_label(run_name, run_config),
                "system_name": system_name,
                "run_timestamp": run_timestamp,
                "records": len(df),
                "pipeline_type": _classify_pipeline_type(run_config),
                **model_details,
            }
            for m in metric_names:
                if m in df.columns:
                    vals = df[m].dropna()
                    if len(vals) > 0:
                        summary[m] = vals.mean()
            run_summaries.append(summary)

        # Scatter plot data from metrics_summary
        point = _extract_eva_scatter_point(run_dir, run_config, metrics_summary)
        if point:
            scatter_data.append(point)

    if not run_summaries:
        st.warning("No runs with metrics data found")
        return

    metric_names = _sort_metrics_by_category(sorted(all_metric_names))
    if not st.session_state.get("show_sub_metrics"):
        metric_names = [m for m in metric_names if not _is_sub_metric(m)]

    col_rename = _make_category_header_rename(metric_names)
    summary_df = pd.DataFrame(run_summaries)

    ordered_metrics = [m for m in metric_names if m in summary_df.columns]

    # EVA-A vs EVA-X scatter plot
    if scatter_data:
        _render_eva_scatter_plot(scatter_data)

    st.markdown("#### Mean Metrics per Run")

    # Grouped bar chart: Accuracy/Experience metrics + EVA composites
    bar_metrics = [m for m in ordered_metrics if _METRIC_GROUP.get(m) in _BAR_CHART_CATEGORIES]
    bar_composites = [c for c in _EVA_BAR_COMPOSITES if c in summary_df.columns]
    bar_keys = bar_composites + bar_metrics
    bar_labels = [_EVA_COMPOSITE_DISPLAY[c] for c in bar_composites] + [_format_metric_name(m) for m in bar_metrics]
    if bar_keys and len(run_summaries) > 1:
        bar_fig = go.Figure()
        for i, (_, row) in enumerate(summary_df.iterrows()):
            bar_fig.add_trace(
                go.Bar(
                    x=bar_labels,
                    y=[row.get(m) for m in bar_keys],
                    name=row["label"],
                    marker_color=_MODEL_COLORS[i % len(_MODEL_COLORS)],
                )
            )
        bar_fig.update_layout(
            barmode="group",
            yaxis={"title": "Score", "range": [0, 1.05]},
            xaxis={"tickangle": -45},
            legend={"orientation": "h", "yanchor": "top", "y": -0.55, "xanchor": "center", "x": 0.5},
            height=450,
            margin={"l": 50, "r": 10, "t": 10, "b": 120},
        )
        st.plotly_chart(bar_fig)

    # Split metrics into three groups by category
    eva_a_composites = [c for c in ["EVA-A_pass", "EVA-A_mean"] if c in summary_df.columns]
    eva_x_composites = [c for c in ["EVA-X_pass", "EVA-X_mean"] if c in summary_df.columns]
    accuracy_metrics = [m for m in ordered_metrics if _METRIC_GROUP.get(m) == "Accuracy"]
    experience_metrics = [m for m in ordered_metrics if _METRIC_GROUP.get(m) == "Experience"]
    other_metrics = [m for m in ordered_metrics if _METRIC_GROUP.get(m) not in {"Accuracy", "Experience"}]

    multiple_output_dirs = summary_df["run_output_dir"].nunique() > 1
    id_cols = ["system_name", "run_timestamp"] + (["run_output_dir"] if multiple_output_dirs else []) + ["records"]
    id_rename = {
        "system_name": "System",
        "run_timestamp": "Timestamp",
        "run_output_dir": "Output Dir",
        "records": "# Records",
    }
    link_series = "/run_overview?output_dir=" + summary_df["run_output_dir"] + "&run=" + summary_df["run"]

    def _show_subtable(heading: str, composites: list, metrics: list) -> None:
        if not composites and not metrics:
            return
        st.markdown(f"#### {heading}")
        composite_rename = {c: f"[EVA] {_EVA_COMPOSITE_DISPLAY[c]}" for c in composites}
        cols = id_cols + composites + metrics
        sub_df = summary_df[cols].copy()
        sub_df.insert(0, "link", link_series)
        sub_df = sub_df.rename(columns={**id_rename, **composite_rename, **col_rename})
        score_cols = [composite_rename[c] for c in composites] + [col_rename[m] for m in metrics]
        styled = sub_df.style.map(_color_cell, subset=score_cols)
        styled = styled.format(dict.fromkeys(score_cols, "{:.3f}"), na_rep="—")
        st.dataframe(
            styled,
            hide_index=True,
            column_config={"link": st.column_config.LinkColumn(" ", display_text="🔍", width=40)},
        )

    _show_subtable("Accuracy Metrics (EVA-A)", eva_a_composites, accuracy_metrics)
    _show_subtable("Experience Metrics (EVA-X)", eva_x_composites, experience_metrics)
    _show_subtable("Diagnostic & Other Metrics", [], other_metrics)

    csv = summary_df.drop(columns=["label"]).to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="cross_run_comparison.csv", mime="text/csv")


def render_run_overview(run_dir: Path):
    """Render run-level overview with aggregate metrics and per-record table."""
    # Model details at the top
    run_config = _load_run_config(run_dir)
    model_details = _extract_model_details(run_config)
    if model_details:
        cols = st.columns(len(model_details))
        for col, (label, value) in zip(cols, model_details.items()):
            col.metric(label, value)
        st.divider()

    rows, metric_names = _collect_run_metrics(run_dir)

    if not rows:
        st.warning("No metrics data found for any record")
        return

    has_trials = "trial" in rows[0]
    df = pd.DataFrame(rows)
    metric_names = _sort_metrics_by_category(metric_names)
    if not st.session_state.get("show_sub_metrics"):
        metric_names = [m for m in metric_names if not _is_sub_metric(m)]

    col_rename = _make_category_header_rename(metric_names)

    # --- Aggregate summary ---
    st.markdown("### Aggregate Metrics")

    summary_data = {}
    for m in metric_names:
        if m in df.columns:
            vals = df[m].dropna()
            if len(vals) > 0:
                summary_data[m] = {
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "min": vals.min(),
                    "max": vals.max(),
                    "count": int(len(vals)),
                }

    if summary_data:
        is_light = st.get_option("theme.base") == "light"
        error_bar_color = "#333333" if is_light else "#E0E0E0"

        # Separate non-normalized metrics (e.g. response_speed in seconds)
        normalized_data = {m: s for m, s in summary_data.items() if m not in _NON_NORMALIZED_METRICS}
        standalone_data = {m: s for m, s in summary_data.items() if m in _NON_NORMALIZED_METRICS}

        # Build a dataframe for the bar chart (normalized 0-1 metrics only)
        chart_rows = [
            {
                "metric": _format_metric_name(m),
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "count": stats["count"],
                "category": _METRIC_GROUP.get(m, "Other"),
            }
            for m, stats in normalized_data.items()
        ]

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows)

            # Sort by category order then metric name within category
            cat_order = {c: i for i, c in enumerate(_CATEGORY_ORDER + ["Other"])}
            chart_df["_cat_sort"] = chart_df["category"].map(lambda c: cat_order.get(c, len(cat_order)))
            chart_df = chart_df.sort_values(["_cat_sort", "metric"], ascending=[True, True]).drop(columns=["_cat_sort"])

            fig = go.Figure()
            for cat in _CATEGORY_ORDER + ["Other"]:
                cat_data = chart_df[chart_df["category"] == cat]
                if cat_data.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        y=cat_data["metric"],
                        x=cat_data["mean"],
                        orientation="h",
                        name=cat,
                        marker_color=_CATEGORY_COLORS.get(cat, "#AAAAAA"),
                        error_x={
                            "type": "data",
                            "array": cat_data["std"].tolist(),
                            "visible": True,
                            "color": error_bar_color,
                            "thickness": 2,
                        },
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Mean: %{x:.3f}<br>"
                            "Std: %{customdata[0]:.3f}<br>"
                            "Min: %{customdata[1]:.3f}<br>"
                            "Max: %{customdata[2]:.3f}<br>"
                            "n=%{customdata[3]}"
                            "<extra></extra>"
                        ),
                        customdata=cat_data[["std", "min", "max", "count"]].values,
                    )
                )

            fig.update_layout(
                xaxis_title="Score",
                xaxis_range=[0, 1.05],
                yaxis={"categoryorder": "array", "categoryarray": list(reversed(chart_df["metric"].tolist()))},
                legend_title="Category",
                height=max(350, len(chart_df) * 35 + 100),
                margin={"l": 10, "r": 10, "t": 10, "b": 40},
            )
            st.plotly_chart(fig)

        # Render standalone metrics (non-normalized) as separate bar charts
        for m, stats in standalone_data.items():
            display_name = _format_metric_name(m)
            cat = _METRIC_GROUP.get(m, "Other")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=[display_name],
                    x=[stats["mean"]],
                    orientation="h",
                    marker_color=_CATEGORY_COLORS.get(cat, "#AAAAAA"),
                    error_x={
                        "type": "data",
                        "array": [stats["std"]],
                        "visible": True,
                        "color": error_bar_color,
                        "thickness": 2,
                    },
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Mean: %{x:.3f}s<br>"
                        f"Std: {stats['std']:.3f}s<br>"
                        f"Min: {stats['min']:.3f}s<br>"
                        f"Max: {stats['max']:.3f}s<br>"
                        f"n={stats['count']}"
                        "<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                xaxis_title="Seconds",
                yaxis={"categoryorder": "array", "categoryarray": [display_name]},
                height=150,
                margin={"l": 10, "r": 10, "t": 10, "b": 40},
                showlegend=False,
            )
            st.plotly_chart(fig)

    st.divider()

    # --- Per-record table ---
    st.markdown("### Per-Record Metrics")

    has_failed_attempts = df["_is_failed_attempt"].any()
    show_failed = False
    if has_failed_attempts:
        show_failed = st.toggle("Show failed attempts", value=False)

    table_df = df if show_failed else df[~df["_is_failed_attempt"]]

    run_name = run_dir.name
    leading_cols = ["record"]
    if has_trials:
        leading_cols.append("trial")
    composite_cols = [c for c in _EVA_BAR_COMPOSITES if c in table_df.columns]
    ordered_metrics = [m for m in metric_names if m in table_df.columns]
    table_df = table_df[leading_cols + composite_cols + ordered_metrics]

    # Add link column to navigate to Record Detail
    def _record_link(row):
        params = f"/record_detail?output_dir={run_dir.parent}&run={run_name}&record={row['record']}"
        if "trial" in row and pd.notna(row.get("trial")):
            params += f"&trial={row['trial']}"
        return params

    table_df = table_df.copy()
    table_df.insert(0, "link", table_df.apply(_record_link, axis=1))
    composite_rename = {c: _EVA_COMPOSITE_DISPLAY[c] for c in composite_cols}
    table_df = table_df.rename(columns={**col_rename, **composite_rename})

    renamed_composites = [composite_rename[c] for c in composite_cols]
    renamed_metrics = [col_rename[m] for m in ordered_metrics]
    score_cols = renamed_composites + renamed_metrics
    styled = table_df.style.map(_color_cell, subset=score_cols)
    styled = styled.format(dict.fromkeys(score_cols, "{:.3f}"), na_rep="—")
    st.dataframe(
        styled,
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn(" ", display_text="🔍", width=40),
        },
    )

    csv = table_df.drop(columns=["link"]).to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=f"{run_dir.name}_metrics.csv", mime="text/csv")


def _render_eva_composite_cards(metrics: RecordMetrics) -> None:
    """Render EVA composite scores (EVA-A/EVA-X pass@1 and Mean) as st.metric cards."""
    composites = [(c, metrics.aggregate_metrics.get(c)) for c in _EVA_BAR_COMPOSITES]
    composites = [(c, v) for c, v in composites if v is not None]
    if not composites:
        return
    cols = st.columns(len(composites))
    for col, (name, value) in zip(cols, composites):
        col.metric(_EVA_COMPOSITE_DISPLAY[name], f"{value:.3f}")


def render_metrics_tab(metrics: RecordMetrics | None):
    """Render the metrics tab with judge ratings and scores."""
    if not metrics:
        st.warning("No metrics.json found for this record")
        return

    st.markdown("### Metrics")
    _render_eva_composite_cards(metrics)

    # Group metrics by category, preserving insertion order within each group
    grouped: dict[str, list[tuple[str, object]]] = {}
    for metric_name, metric_score in metrics.metrics.items():
        cat = _METRIC_GROUP.get(metric_name, "Other")
        grouped.setdefault(cat, []).append((metric_name, metric_score))

    for cat in _CATEGORY_ORDER + [c for c in grouped if c not in _CATEGORY_ORDER]:
        if cat not in grouped:
            continue
        st.markdown(f"#### {cat}")
        for metric_name, metric_score in grouped[cat]:
            with st.expander(
                f"**{metric_name}**: {metric_score.normalized_score:.3f}"
                if metric_score.normalized_score is not None
                else f"**{metric_name}**"
            ):
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.metric("Score", f"{metric_score.score:.3f}" if metric_score.score is not None else "N/A")
                    st.metric(
                        "Normalized",
                        f"{metric_score.normalized_score:.3f}" if metric_score.normalized_score is not None else "N/A",
                    )
                    if metric_score.error:
                        st.error(f"Error: {metric_score.error}")

                    # Dimension scores (e.g. faithfulness, conversation_progression)
                    explanation = metric_score.details.get("explanation") if metric_score.details else None
                    dimensions = explanation.get("dimensions") if isinstance(explanation, dict) else None
                    if dimensions:
                        st.markdown("**Dimensions**")
                        for dim_name, dim_data in dimensions.items():
                            if isinstance(dim_data, dict):
                                rating = dim_data.get("rating")
                                flagged = dim_data.get("flagged", False)
                                label = dim_name.replace("_", " ").title()
                                score_str = f"{rating}/3" if rating is not None else "N/A"
                                prefix = "⚠ " if flagged else ""
                                st.markdown(f"{prefix}**{label}:** {score_str}")

                with col2:
                    if metric_score.details:
                        st.markdown("**Details:**")
                        if "explanation" in metric_score.details:
                            st.write(metric_score.details["explanation"])

                        if "judge_prompt" in metric_score.details:
                            with st.expander("View Judge Prompt"):
                                prompt = metric_score.details["judge_prompt"]
                                if isinstance(prompt, str):
                                    st.text(prompt)
                                else:
                                    st.json(prompt)
                        elif "judge_prompts" in metric_score.details:
                            with st.expander("View Judge Prompts"):
                                prompts = metric_score.details["judge_prompts"]
                                if isinstance(prompts, list):
                                    for i, prompt in enumerate(prompts):
                                        st.markdown(f"**Turn {i + 1}:**")
                                        st.text(prompt)
                                        st.divider()
                                else:
                                    st.json(prompts)

                        details_to_show = {
                            k: v
                            for k, v in metric_score.details.items()
                            if k not in ["explanation", "judge_prompt", "judge_prompts"]
                        }
                        if details_to_show:
                            st.json(details_to_show)


def render_processed_data_tab(metrics: RecordMetrics | None):
    """Render the processed data tab with all context variables."""
    if not metrics or not metrics.context:
        st.warning("No processed data available (metrics context not found)")
        return

    context = metrics.context

    st.markdown("### Processed Variables from Metrics Context")
    st.info("These variables are processed by the MetricsContextProcessor and used for metric computation.")

    # Conversation trace
    with st.expander("Conversation Trace", expanded=True):
        if context.get("conversation_trace"):
            for turn in context["conversation_trace"]:
                role = turn.get("role", "")
                content = turn.get("content", turn.get("message", ""))
                turn_type = turn.get("type", "")

                if role == "user":
                    st.markdown(f"**User:** {content}")
                elif role == "assistant":
                    st.markdown(f"**Assistant:** {content}")
                elif turn_type == "tool_call":
                    tool_name = turn.get("tool_name", "unknown")
                    params = turn.get("parameters", {})
                    st.code(f"[Tool Call: {tool_name}({params})]", language=None)
                elif turn_type == "tool_response":
                    response = turn.get("tool_response", "")
                    st.code(f"[Tool Response: {response}]", language=None)
        else:
            st.info("No conversation trace data")

    # Tool data
    with st.expander("Tool Parameters"):
        if context.get("tool_params"):
            st.json(context["tool_params"])
        else:
            st.info("No tool parameters data")

    with st.expander("Tool Responses"):
        if context.get("tool_responses"):
            st.json(context["tool_responses"])
        else:
            st.info("No tool responses data")

    # Transcripts by speaker
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Assistant Transcript (by Turn)"):
            if context.get("transcribed_assistant_turns"):
                st.json(context["transcribed_assistant_turns"])
            else:
                st.info("No assistant transcript data")
    with col2:
        with st.expander("User Transcript (by Turn)"):
            if context.get("transcribed_user_turns"):
                st.json(context["transcribed_user_turns"])
            else:
                st.info("No user transcript data")

    # TTS text
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Assistant TTS Text (by Turn)"):
            if context.get("intended_assistant_turns"):
                st.json(context["intended_assistant_turns"])
            else:
                st.info("No assistant TTS text data")
    with col2:
        with st.expander("User TTS Text (by Turn)"):
            if context.get("intended_user_turns"):
                st.json(context["intended_user_turns"])
            else:
                st.info("No user TTS text data")

    # Stats
    with st.expander("Conversation Statistics"):
        stats = {}
        if "num_assistant_turns" in context:
            stats["Assistant Turns"] = context["num_assistant_turns"]
        if "num_user_turns" in context:
            stats["User Turns"] = context["num_user_turns"]
        if "conversation_finished" in context:
            stats["Conversation Finished"] = context["conversation_finished"]
        if stats:
            st.json(stats)
        else:
            st.info("No statistics data")

    with st.expander("Agent Instructions"):
        if context.get("agent_instructions"):
            st.text(context["agent_instructions"])
        else:
            st.info("No agent instructions data")


def render_conversation_trace_tab(metrics: RecordMetrics | None, record_dir: Path):
    """Render conversation trace with chat-like visualization and per-turn metrics."""
    if not metrics or not metrics.context:
        st.warning("No processed data available (metrics context not found)")
        return

    context = metrics.context
    trace = context.get("conversation_trace", [])
    if not trace:
        st.info("No conversation trace data")
        return

    # Classify metrics
    conversation_metrics, user_per_turn, assistant_per_turn = _classify_metrics(metrics)
    all_per_turn = {**user_per_turn, **assistant_per_turn}
    has_turn_metrics = bool(all_per_turn)

    # Build lookup dicts
    intended_user = context.get("intended_user_turns", {})
    transcribed_assistant = context.get("transcribed_assistant_turns", {})
    intended_user = {int(k): v for k, v in intended_user.items()} if intended_user else {}
    transcribed_assistant = {int(k): v for k, v in transcribed_assistant.items()} if transcribed_assistant else {}

    # WER details
    stt_wer_details = user_per_turn.get("stt_wer", {}).get("details", {})
    per_turn_wer = stt_wer_details.get("per_turn_wer", {})
    per_turn_errors = stt_wer_details.get("per_turn_errors", {})

    # --- Metrics overview at the top ---
    all_top_metrics = {**conversation_metrics}
    for name, m in all_per_turn.items():
        all_top_metrics[name] = m

    if all_top_metrics:
        selected = st.session_state.get("selected_metric")

        st.html(
            """<style>
            div[class*="st-key-metric_btn_"] button {
                height: 4.5rem;
            }
            </style>"""
        )
        st.markdown("### Metrics Overview")
        _render_eva_composite_cards(metrics)

        grouped: dict[str, list[str]] = {}
        for name in all_top_metrics:
            group = _METRIC_GROUP.get(name, "Other")
            grouped.setdefault(group, []).append(name)

        # Accuracy / Experience / Diagnostic / Validation side by side in 4 columns
        col_groups = ["Accuracy", "Experience", "Diagnostic", "Validation"]
        present_col_groups = [g for g in col_groups if grouped.get(g)]
        if present_col_groups:
            outer_cols = st.columns(len(present_col_groups))
            for outer_col, group in zip(outer_cols, present_col_groups):
                names_in_group = grouped[group]
                with outer_col:
                    st.caption(group)
                    for name in names_in_group:
                        m = all_top_metrics[name]
                        score = m["normalized_score"]
                        display_name = _format_metric_name(name)
                        score_str = f"{score:.3f}" if score is not None else "N/A"
                        icon = None if score is None else "🟢" if score >= 0.8 else "🟡" if score >= 0.4 else "🔴"
                        st.button(
                            f"{display_name}\n{score_str}",
                            key=f"metric_btn_{name}",
                            on_click=st.session_state.update,
                            kwargs={"selected_metric": None if selected == name else name},
                            type="primary" if selected == name else "secondary",
                            icon=icon,
                            width="stretch",
                        )

        # Any remaining groups (Other, Conversation Quality, etc.) rendered below
        for group in [g for g in grouped if g not in col_groups]:
            names_in_group = grouped[group]
            st.caption(group)
            cols = st.columns(min(len(names_in_group), 5))
            for i, name in enumerate(names_in_group):
                m = all_top_metrics[name]
                score = m["normalized_score"]
                with cols[i % len(cols)]:
                    display_name = _format_metric_name(name)
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    icon = None if score is None else "🟢" if score >= 0.8 else "🟡" if score >= 0.4 else "🔴"
                    st.button(
                        f"{display_name}\n{score_str}",
                        key=f"metric_btn_{name}",
                        on_click=st.session_state.update,
                        kwargs={"selected_metric": None if selected == name else name},
                        type="primary" if selected == name else "secondary",
                        icon=icon,
                        width="stretch",
                    )

        # Show details for selected metric
        if selected and selected in conversation_metrics:
            m = conversation_metrics[selected]
            details = m.get("details", {})
            details_to_show = {
                k: v for k, v in details.items() if k not in ("judge_prompt", "judge_prompts", "judge_raw_response")
            }
            st.markdown(f"**{_format_metric_name(selected)}**")
            explanation = details.get("explanation") or details.get("message", "")
            if explanation and isinstance(explanation, str):
                st.write(explanation)

            # Scenario DB diff for task_completion
            if selected == "task_completion":
                expected_db = context.get("expected_scenario_db")
                final_db = context.get("final_scenario_db")
                if expected_db and final_db:
                    expected_str = json.dumps(expected_db, indent=2, sort_keys=True, default=str)
                    actual_str = json.dumps(final_db, indent=2, sort_keys=True, default=str)
                    diff_viewer(expected_str, actual_str, lang="json", key="task_completion_diff")
            elif details_to_show:
                st.json(details_to_show)

        elif selected and selected in all_per_turn:
            m = all_per_turn[selected]
            details = m.get("details", {})
            st.markdown(f"**{_format_metric_name(selected)}** — per-turn breakdown")

            turn_ids: set[str] = set()
            for k, v in details.items():
                if k.startswith("per_turn") and isinstance(v, dict):
                    turn_ids.update(v.keys())
            turn_ids_sorted = sorted(turn_ids, key=lambda x: int(x) if x.isdigit() else x)

            if turn_ids_sorted:
                for tid_str in turn_ids_sorted:
                    if selected == "turn_taking":
                        label = (details.get("per_turn_judge_timing_ratings") or {}).get(tid_str, "")
                        latency = (details.get("per_turn_latency") or {}).get(tid_str)
                        explanation = (details.get("per_turn_judge_timing_explanations") or {}).get(tid_str, "")
                        latency_str = f"{latency:.3f}s" if latency is not None else "N/A"
                        if label.strip().lower() == "on-time":
                            color = "#4caf50"
                        elif label.strip().lower() == "late":
                            color = "#ff9800"
                        else:
                            color = "#f44336"
                        st.html(
                            f'<div style="margin-bottom:8px; padding:6px 10px; background:rgba(128,128,128,0.1); border-radius:6px; border-left:3px solid {color};">'
                            f"<strong>Turn {tid_str}</strong> — "
                            f'<span style="color:{color}; font-weight:600;">{label}</span> '
                            f'<span style="opacity:0.6;">(latency: {latency_str})</span>'
                            f"{'<br><span style=font-size:0.88em;opacity:0.75;>' + html.escape(str(explanation)) + '</span>' if explanation else ''}"
                            f"</div>"
                        )
                    else:
                        rating = None
                        explanation = ""
                        if "per_turn_ratings" in details:
                            rating = details["per_turn_ratings"].get(tid_str)
                        elif "per_turn_normalized" in details:
                            rating = details["per_turn_normalized"].get(tid_str)
                        elif "per_turn_wer" in details:
                            wer = details["per_turn_wer"].get(tid_str)
                            rating = round(1.0 - wer, 3) if wer is not None else None
                        if "per_turn_explanations" in details:
                            explanation = details["per_turn_explanations"].get(tid_str, "")

                        color = _score_color(rating)
                        rating_str = f"{rating:.2f}" if isinstance(rating, (int, float)) else str(rating)
                        st.html(
                            f'<div style="margin-bottom:8px; padding:6px 10px; background:rgba(128,128,128,0.1); border-radius:6px; border-left:3px solid {color};">'
                            f"<strong>Turn {tid_str}</strong> — "
                            f'<span style="color:{color}; font-weight:600;">{rating_str}</span>'
                            f"{'<br><span style=font-size:0.88em;opacity:0.75;>' + html.escape(str(explanation)) + '</span>' if explanation else ''}"
                            f"</div>"
                        )
            else:
                details_to_show = {
                    k: v for k, v in details.items() if k not in ("judge_prompt", "judge_prompts", "judge_raw_response")
                }
                if details_to_show:
                    st.json(details_to_show)

            st.divider()

    # --- Conversation Trace ---
    st.markdown("### Conversation Trace")

    if all_per_turn:
        legend_parts = [f"`{mn.replace('transcription_accuracy_', '').replace('_', ' ')}`" for mn in all_per_turn]
        st.caption(f"Per-turn metrics shown inline: {', '.join(legend_parts)}")

    prev_role = None
    for entry in trace:
        role = entry.get("role", "")
        content = html.escape(entry.get("content", "")).replace("$", "&#36;")
        entry_type = entry.get("type", "")
        turn_id = entry.get("turn_id")
        tool_name = entry.get("tool_name", "")

        if role == "assistant":
            raw_reverse = transcribed_assistant.get(turn_id, "") if turn_id is not None else ""
            reverse_text = html.escape(raw_reverse).replace("$", "&#36;")
            reverse_html = ""
            if raw_reverse and reverse_text != content:
                reverse_html = (
                    f'<div style="font-size:0.78em; opacity:0.65; margin-top:6px; '
                    f'border-top:1px solid rgba(25,118,210,0.2); padding-top:4px;">'
                    f"<em>transcribed (STT): {reverse_text}</em></div>"
                )
            badges_html = ""
            turn_metrics = []
            if turn_id is not None and assistant_per_turn:
                turn_metrics = _get_turn_metric_info(turn_id, assistant_per_turn)
                if turn_metrics:
                    badges_html = (
                        f'<div style="margin-top:6px; border-top:1px solid rgba(25,118,210,0.2); '
                        f'padding-top:4px;">{_render_turn_metric_badges(turn_metrics)}</div>'
                    )

            if has_turn_metrics:
                col_left, col_right = st.columns([3, 1])
            else:
                col_left = st.container()
                col_right = None

            with col_left:
                st.html(
                    f'<div style="background-color:rgba(25, 118, 210, 0.15); padding:10px 14px; border-radius:8px; '
                    f'border-left:4px solid #1976d2; margin-bottom:8px; color:inherit;">'
                    f'<strong style="color:#42a5f5;">Assistant</strong> '
                    f'<span style="font-size:0.72em; opacity:0.5;">turn {turn_id}</span><br>{content}'
                    f"{reverse_html}{badges_html}</div>"
                )

            if turn_metrics and col_right is not None:
                explanations = [m for m in turn_metrics if m["explanation"]]
                if explanations:
                    with col_right:
                        if prev_role != "assistant":
                            st.html(
                                '<div style="border-top:2px solid #1976d2; margin-bottom:6px; padding-top:4px;">'
                                '<strong style="color:#42a5f5; font-size:0.85em;">Assistant Metrics</strong></div>'
                            )
                        for m in explanations:
                            color = _score_color(m["rating"])
                            st.html(
                                f'<div style="font-size:0.82em; margin-bottom:8px;">'
                                f"<strong>{m['short_name']}</strong> "
                                f'<span style="color:{color}; font-weight:600;">{m["rating"]:.2f}</span>'
                                f'<br><span style="font-size:0.85em; opacity:0.7;">{html.escape(str(m["explanation"]))}</span>'
                                f"</div>"
                            )
            prev_role = "assistant"

        elif role == "user":
            raw_reverse = intended_user.get(turn_id, "") if turn_id is not None else ""
            reverse_text = html.escape(raw_reverse).replace("$", "&#36;")
            reverse_html = ""
            if raw_reverse and reverse_text != content:
                reverse_html = (
                    f'<div style="font-size:0.78em; opacity:0.65; margin-top:6px; '
                    f'border-top:1px solid rgba(123,31,162,0.2); padding-top:4px;">'
                    f"<em>intended (TTS): {reverse_text}</em></div>"
                )
            badges_html = ""
            turn_metrics = []
            if turn_id is not None and user_per_turn:
                turn_metrics = _get_turn_metric_info(turn_id, user_per_turn)
                if turn_metrics:
                    badges_html = (
                        f'<div style="margin-top:6px; border-top:1px solid rgba(123,31,162,0.2); '
                        f'padding-top:4px;">{_render_turn_metric_badges(turn_metrics)}</div>'
                    )

            # WER error details
            wer_html = ""
            tid = str(turn_id) if turn_id is not None else ""
            if tid in per_turn_wer:
                wer_val = per_turn_wer[tid]
                errors = per_turn_errors.get(tid, {})
                error_parts = []
                for sub in errors.get("substitutions", []):
                    error_parts.append(f"{sub['expected']} -> {sub['actual']}")
                for d in errors.get("deletions", []):
                    error_parts.append(f"-{d}")
                for ins in errors.get("insertions", []):
                    error_parts.append(f"+{ins}")
                wer_color = _score_color(1.0 - wer_val if wer_val is not None else None)
                wer_pct = f"{wer_val:.0%}" if wer_val is not None else "?"
                errors_str = ", ".join(error_parts) if error_parts else "no errors"
                wer_html = (
                    f'<div style="font-size:0.78em; margin-top:4px; '
                    f'border-top:1px solid rgba(123,31,162,0.2); padding-top:4px; color:{wer_color};">'
                    f"WER {wer_pct}: {errors_str}</div>"
                )

            if has_turn_metrics:
                col_left, col_right = st.columns([3, 1])
            else:
                col_left = st.container()
                col_right = None

            with col_left:
                st.html(
                    f'<div style="background-color:rgba(123, 31, 162, 0.15); padding:10px 14px; border-radius:8px; '
                    f'border-left:4px solid #7b1fa2; margin-bottom:8px; color:inherit;">'
                    f'<strong style="color:#ce93d8;">User</strong> '
                    f'<span style="font-size:0.72em; opacity:0.5;">turn {turn_id}</span><br>{content}'
                    f"{reverse_html}{wer_html}{badges_html}</div>"
                )

            if turn_metrics and col_right is not None:
                explanations = [m for m in turn_metrics if m["explanation"]]
                if explanations:
                    with col_right:
                        if prev_role != "user":
                            st.html(
                                '<div style="border-top:2px solid #7b1fa2; margin-bottom:6px; padding-top:4px;">'
                                '<strong style="color:#ce93d8; font-size:0.85em;">User Metrics</strong></div>'
                            )
                        for m in explanations:
                            color = _score_color(m["rating"])
                            st.html(
                                f'<div style="font-size:0.82em; margin-bottom:8px;">'
                                f"<strong>{m['short_name']}</strong> "
                                f'<span style="color:{color}; font-weight:600;">{m["rating"]:.2f}</span>'
                                f'<br><span style="font-size:0.85em; opacity:0.7;">{html.escape(str(m["explanation"]))}</span>'
                                f"</div>"
                            )
            prev_role = "user"

        elif tool_name:
            if has_turn_metrics:
                col_left, _ = st.columns([3, 1])
            else:
                col_left = st.container()
            with col_left:
                if entry_type == "tool_call":
                    params_str = json.dumps(entry.get("parameters", {}), indent=2)
                    with st.expander(f"Tool Call — `{tool_name}`", expanded=False):
                        st.code(params_str, language="json")
                elif entry_type == "tool_response":
                    tool_response = entry.get("tool_response", "")
                    response_str = (
                        json.dumps(tool_response, indent=2) if isinstance(tool_response, dict) else str(tool_response)
                    )
                    with st.expander(f"Tool Response — `{tool_name}`", expanded=False):
                        st.code(response_str, language="json")


def _render_sidebar_run_metadata(run_name: str, run_config: dict):
    """Render run metadata in the sidebar."""
    metadata_parts = [f"**Run:** {run_name}"]
    for label, value in _extract_model_details(run_config).items():
        metadata_parts.append(f"**{label}:** {value}")
    if run_config.get("num_trials"):
        metadata_parts.append(f"**Trials:** {run_config['num_trials']}")
    provenance = run_config.get("provenance", {})
    if provenance.get("git_branch"):
        metadata_parts.append(f"**Branch:** {provenance['git_branch']}")
    st.sidebar.info("\n\n".join(metadata_parts))


def _get_run_dirs():
    """Get run directories, showing an error if none found."""
    output_dirs_input = st.sidebar.text_area(
        "Output directories (one per line)", value=_DEFAULT_OUTPUT_DIR, key="output_dir", bind="query-params"
    )
    output_dirs = [Path(stripped) for p in output_dirs_input.splitlines() if (stripped := p.strip())]

    run_dirs = [rd for od in output_dirs for rd in get_run_directories(od)]

    latest_only = st.sidebar.toggle("Latest run per system only", value=True)
    if latest_only:
        run_dirs = filter_latest_runs(run_dirs)

    st.sidebar.toggle("Show sub-metrics", value=False, key="show_sub_metrics")

    if not run_dirs:
        st.error(f"No run directories found in: {', '.join(str(d) for d in output_dirs)}")
        st.stop()

    return run_dirs


def _select_run(run_dirs: list[Path]):
    st.sidebar.header("Run Selection")
    multiple_output_dirs = len({d.parent for d in run_dirs}) > 1
    selected_run_dir = st.sidebar.selectbox(
        "Select Run",
        run_dirs,
        format_func=lambda d: str(d) if multiple_output_dirs else d.name,
        key="run",
        bind="query-params",
    )

    run_config = _load_run_config(selected_run_dir)
    if run_config:
        _render_sidebar_run_metadata(selected_run_dir.name, run_config)
    else:
        st.sidebar.info(f"**Run:** {selected_run_dir.name}")

    return selected_run_dir


def render_record_detail(selected_run_dir: Path):
    record_dirs = get_record_directories(selected_run_dir)

    if not record_dirs:
        st.error(f"No records found in {selected_run_dir / 'records'}")
        return

    # Sidebar: record selection
    st.sidebar.header("Record Selection")
    record_names = [d.name for d in record_dirs]
    selected_record_name = st.sidebar.selectbox("Select Record", record_names, key="record", bind="query-params")
    selected_record_dir = selected_run_dir / "records" / selected_record_name

    # Detect trial subdirectories
    trial_dirs = (
        sorted(
            [
                d
                for d in selected_record_dir.iterdir()
                if d.is_dir() and any(f for f in d.iterdir() if f.suffix in (".json", ".wav", ".jsonl"))
            ],
            key=lambda d: d.name,
        )
        if selected_record_dir.exists()
        else []
    )

    selected_trial = None
    if trial_dirs:
        trial_names = [d.name for d in trial_dirs]
        selected_trial = st.sidebar.selectbox("Select Trial", trial_names, key="trial", bind="query-params")
        selected_record_dir = selected_record_dir / selected_trial

    # Load data
    result = load_record_result(selected_record_dir)
    metrics = load_record_metrics(selected_record_dir)
    eval_record = load_evaluation_record(selected_run_dir, selected_record_name)

    # Header
    header = f"Record: {selected_record_name}"
    if selected_trial:
        header += f" / {selected_trial}"
    st.header(header)

    # Result summary
    if result:
        col1, col2, col3 = st.columns(3)
        with col1:
            if result.completed:
                st.success(f"Completed ({result.conversation_ended_reason or 'ok'})")
            else:
                st.error(f"Failed: {result.error or 'unknown'}")
        with col2:
            st.metric("Duration", f"{result.duration_seconds:.1f}s")
        with col3:
            st.metric("Turns", result.num_turns)

    # Audio player
    audio_path = selected_record_dir / "audio_mixed.wav"
    if audio_path.exists():
        st.markdown("### Audio Recording")
        st.audio(str(audio_path))

    # User goal & ground truth
    with st.expander("User Goal", expanded=False):
        if eval_record:
            try:
                user_goal_data = json.loads(eval_record.user_goal)
                st.json(user_goal_data)
            except (json.JSONDecodeError, TypeError):
                st.write(eval_record.user_goal)
        else:
            st.info("No user goal available")

    if eval_record and hasattr(eval_record, "ground_truth") and eval_record.ground_truth:
        with st.expander("Ground Truth (Expected Scenario DB)", expanded=False):
            st.json(eval_record.ground_truth.expected_scenario_db)

    st.divider()

    # Pre-load audio data before the tabs so the cache is warm when the user
    # opens the Audio Analysis tab (or switches trials).
    preload_audio_data(selected_record_dir)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Conversation Trace",
            "Transcript",
            "Metrics Detail",
            "Processed Data",
            "Turn Taking Analysis",
        ]
    )

    with tab1:
        render_conversation_trace_tab(metrics, selected_record_dir)

    with tab2:
        st.markdown("### Transcript")
        if metrics and metrics.context and "turns_transcript" in metrics.context:
            try:
                turns = metrics.context["turns_transcript"]
                if turns:
                    transcript_df = pd.DataFrame(turns)
                    column_config = {}
                    if "content" in transcript_df.columns:
                        column_config["content"] = st.column_config.TextColumn("content", width="large")
                    if "timestamp" in transcript_df.columns:
                        column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                    if "role" in transcript_df.columns:
                        column_config["role"] = st.column_config.TextColumn("role", width="small")
                    st.dataframe(transcript_df, hide_index=True, column_config=column_config)
                else:
                    st.info("No transcript data available")
            except Exception:
                transcript_df = format_transcript(selected_record_dir / "transcript.jsonl")
                if not transcript_df.empty:
                    st.dataframe(transcript_df, hide_index=True)
        else:
            transcript_df = format_transcript(selected_record_dir / "transcript.jsonl")
            if not transcript_df.empty:
                column_config = {}
                if "content" in transcript_df.columns:
                    column_config["content"] = st.column_config.TextColumn("content", width="large")
                if "timestamp" in transcript_df.columns:
                    column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                if "role" in transcript_df.columns:
                    column_config["role"] = st.column_config.TextColumn("role", width="small")
                st.dataframe(transcript_df, hide_index=True, column_config=column_config)
            else:
                st.info("No transcript data available")

    with tab3:
        render_metrics_tab(metrics)

    with tab4:
        render_processed_data_tab(metrics)

    with tab5:
        render_audio_analysis_tab(selected_record_dir)


# ============================================================================
# Main App
# ============================================================================


def cross_run_comparison():
    render_cross_run_comparison(_get_run_dirs())


def run_overview():
    render_run_overview(_select_run(_get_run_dirs()))


def record_detail():
    render_record_detail(_select_run(_get_run_dirs()))


def main():
    st.set_page_config(page_title="EVA Results Analysis", layout="wide", page_icon="website/public/favicon.svg")

    pages = (
        st.Page(cross_run_comparison, title="Cross-Run Comparison", icon=":material/compare_arrows:"),
        st.Page(run_overview, title="Run Overview", icon=":material/summarize:"),
        st.Page(record_detail, title="Record Detail", icon=":material/article:"),
    )
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
