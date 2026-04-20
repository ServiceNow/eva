#!/usr/bin/env python3
"""Streamlit Analysis App for Voice Agent Benchmark

Visualizes benchmark outputs including transcripts, logs, metrics, and audio.
Provides LLM-based analysis of conversation logs.

Usage:
    streamlit run scripts/analysis_app.py
"""

import asyncio
import html
import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from diff_viewer import diff_viewer
from pydub import AudioSegment

from eva.models.config import RunConfig
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, RecordMetrics
from eva.utils import router
from eva.utils.llm_client import LLMClient
from eva.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(os.environ.get("EVA_OUTPUT_DIR", "output"))
ANALYSIS_FILE = "analysis.json"


# ============================================================================
# Helper Functions
# ============================================================================


def get_run_directories(output_dir: Path | None = None) -> list[Path]:
    """Get all run directories in output/, sorted by timestamp (newest first)."""
    base = output_dir if output_dir is not None else OUTPUT_DIR
    if not base.exists():
        return []

    run_dirs = [d for d in base.iterdir() if d.is_dir() and (d / "records").exists()]
    # Sort by directory name (which is timestamp format YYYYMMDD_HHMMSS)
    return sorted(run_dirs, key=lambda d: d.name, reverse=True)


def get_record_directories(run_dir: Path) -> list[Path]:
    """Get all record directories in a run, sorted by record ID."""
    records_dir = run_dir / "records"
    if not records_dir.exists():
        return []

    record_dirs = [d for d in records_dir.iterdir() if d.is_dir()]
    return sorted(record_dirs, key=lambda d: d.name)


def load_record_result(record_dir: Path) -> ConversationResult | None:
    """Load ConversationResult from result.json."""
    result_path = record_dir / "result.json"
    if not result_path.exists():
        return None

    try:
        with open(result_path) as f:
            data = json.load(f)
        return ConversationResult(**data)
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
            data = json.load(f)
        return RecordMetrics(**data)
    except Exception as e:
        st.error(f"Failed to load metrics.json: {e}")
        return None


def load_evaluation_record(run_dir: Path, record_id: str, dataset_path: Path | None = None) -> EvaluationRecord | None:
    """Load EvaluationRecord from dataset."""
    # Try to find dataset path from config
    if not dataset_path:
        config_path = run_dir / "config.json"
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


def format_transcript(transcript_path: Path) -> pd.DataFrame:
    """Load and format transcript.jsonl as a DataFrame."""
    if not transcript_path.exists():
        return pd.DataFrame()

    entries = []
    try:
        with open(transcript_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if not entries:
            return pd.DataFrame()

        # Create DataFrame with key columns
        df = pd.DataFrame(entries)
        # Select and order columns if they exist
        cols = []
        if "timestamp" in df.columns:
            cols.append("timestamp")
        if "role" in df.columns:
            cols.append("role")
        if "content" in df.columns:
            cols.append("content")

        # Add any other columns
        for col in df.columns:
            if col not in cols:
                cols.append(col)

        return df[cols] if cols else df

    except Exception as e:
        st.error(f"Failed to load transcript: {e}")
        return pd.DataFrame()


def format_pipecat_logs(pipecat_path: Path, log_type_filter: str | None = None) -> pd.DataFrame:
    """Load and format pipecat logs as a DataFrame with optional filtering."""
    if not pipecat_path.exists():
        return pd.DataFrame()

    entries = []
    try:
        with open(pipecat_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    # Apply filter if specified
                    if log_type_filter and entry.get("type") != log_type_filter:
                        continue

                    entries.append(entry)

        if not entries:
            return pd.DataFrame()

        df = pd.DataFrame(entries)

        # Reorder columns for readability
        priority_cols = ["timestamp", "type", "source", "destination"]
        cols = [c for c in priority_cols if c in df.columns]
        cols.extend([c for c in df.columns if c not in cols])

        return df[cols]

    except Exception as e:
        st.error(f"Failed to load pipecat logs: {e}")
        return pd.DataFrame()


async def generate_llm_analysis(
    record_dir: Path, user_goal: str, agent_instructions: str, llm_client: LLMClient
) -> dict:
    """Generate LLM analysis of the conversation logs."""
    logs_path = record_dir / "logs.log"

    if not logs_path.exists():
        return {"error": "logs.log not found"}

    # Read logs
    logs_content = logs_path.read_text()

    # Truncate if too long (keep last 10000 chars to focus on recent errors)
    if len(logs_content) > 10000:
        logs_content = "...[truncated]...\n" + logs_content[-10000:]

    prompt = f"""You are an expert at analyzing voice agent conversation logs.

## User's Goal
{user_goal}

## Agent Instructions
{agent_instructions}

## Conversation Logs
{logs_content}

## Task
Analyze the conversation logs and provide:

1. **Brief Summary**: 2-3 sentence summary of what happened in the conversation
2. **Simulation Issues**: Were there any errors or failures in the simulation itself? (STT errors, TTS errors, connection issues, timeouts, etc.)
3. **Agent Performance Issues**:
   - Did the LLM make tool calling errors?
   - Did the agent go off policy or violate instructions?
   - Were there any hallucinations?
   - Did the agent fail to complete the user's goal?
   - Any other problematic behavior?

Respond in JSON format:
{{
    "summary": "<2-3 sentence summary>",
    "simulation_issues": [
        {{"issue": "<detailed description citing from logs>", "severity": "high|medium|low"}}
    ],
    "agent_issues": [
        {{"issue": "<detailed description citing from logs>", "severity": "high|medium|low"}}
    ],
    "overall_assessment": "success|partial_success|failure"
}}
"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response_text = await llm_client.generate_text(messages, response_format={"type": "json_object"})

        analysis = json.loads(response_text)
        analysis["generated_at"] = datetime.now().isoformat()
        return analysis

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {"error": str(e)}


def load_or_generate_analysis(
    record_dir: Path, user_goal: str, agent_instructions: str, force_regenerate: bool = False
) -> dict:
    """Load cached analysis or generate new one."""
    analysis_path = record_dir / ANALYSIS_FILE

    # Try to load cached analysis
    if not force_regenerate and analysis_path.exists():
        try:
            with open(analysis_path) as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load cached analysis: {e}")

    # Generate new analysis
    with st.spinner("Generating LLM analysis..."):
        config = RunConfig(_env_file=".env")
        router.init(config.model_list)
        llm_client = LLMClient(model="gpt-5.2", params={"max_tokens": 10000})
        analysis = asyncio.run(generate_llm_analysis(record_dir, user_goal, agent_instructions, llm_client))

    # Cache the result
    try:
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to cache analysis: {e}")

    return analysis


# ============================================================================
# UI Components
# ============================================================================


def render_llm_analysis(analysis: dict):
    """Render the LLM analysis section."""
    if "error" in analysis:
        st.error(f"Analysis Error: {analysis['error']}")
        return

    # Overall assessment with color
    assessment = analysis.get("overall_assessment", "unknown")
    if assessment == "success":
        st.success("**Overall Assessment:** ✅ Success")
    elif assessment == "partial_success":
        st.warning("**Overall Assessment:** ⚠️ Partial Success")
    else:
        st.error("**Overall Assessment:** ❌ Failure")

    # Summary
    st.markdown("### Summary")
    st.write(analysis.get("summary", "No summary available"))

    # Issues
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Simulation Issues")
        sim_issues = analysis.get("simulation_issues", [])
        if sim_issues:
            for issue in sim_issues:
                severity = issue.get("severity", "low")
                icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                st.markdown(f"{icon} **{severity.upper()}**: {issue.get('issue', '')}")
        else:
            st.info("No simulation issues detected")

    with col2:
        st.markdown("### Agent Performance Issues")
        agent_issues = analysis.get("agent_issues", [])
        if agent_issues:
            for issue in agent_issues:
                severity = issue.get("severity", "low")
                icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                st.markdown(f"{icon} **{severity.upper()}**: {issue.get('issue', '')}")
        else:
            st.info("No agent performance issues detected")

    # Timestamp
    generated_at = analysis.get("generated_at")
    if generated_at:
        st.caption(f"Analysis generated at: {generated_at}")


def check_authentication_success(record_dir: Path) -> bool | None:
    """Check if get_reservation was called and returned status: 'success'.

    Checks conversation_messages in audit_log.json for a tool call to get_reservation
    followed by a tool response with status 'success'.

    Returns True if authentication succeeded, False if get_reservation was called but
    failed, and None if get_reservation was never called.
    """
    audit_log_path = record_dir / "audit_log.json"
    if not audit_log_path.exists():
        return None

    try:
        with open(audit_log_path) as f:
            data = json.load(f)
    except Exception:
        return None

    get_reservation_called = False

    # Check transcript entries (tool_call/tool_response entries with value dicts)
    for entry in data.get("transcript", []):
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, dict):
            continue

        if entry.get("type") == "tool_call" and value.get("tool") == "get_reservation":
            get_reservation_called = True
        elif entry.get("type") == "tool_response" and value.get("tool") == "get_reservation":
            get_reservation_called = True
            response = value.get("response", {})
            if isinstance(response, dict) and response.get("status") == "success":
                return True

    # Fallback: check conversation_messages (OpenAI-style format)
    for msg in data.get("conversation_messages", []):
        if not isinstance(msg, dict):
            continue
        for tool_call in msg.get("tool_calls", []):
            func = tool_call.get("function", {})
            if func.get("name") == "get_reservation":
                get_reservation_called = True
        if msg.get("role") == "tool" and get_reservation_called:
            try:
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = json.loads(content)
                if isinstance(content, dict) and content.get("status") == "success":
                    return True
            except (json.JSONDecodeError, TypeError):
                pass

    return False if get_reservation_called else None


def render_metrics_tab(metrics: RecordMetrics | None):
    """Render the metrics tab with judge ratings and scores."""
    if not metrics:
        st.warning("No metrics.json found for this record")
        return

    # Individual metrics
    st.markdown("### Metrics")

    for metric_name, metric_score in metrics.metrics.items():
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

            with col2:
                if metric_score.details:
                    st.markdown("**Details:**")

                    # Special handling for judge metrics
                    if "explanation" in metric_score.details:
                        st.write(metric_score.details["explanation"])

                    # Show judge prompt in expander if available
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

                    # Show other details (excluding explanation and judge prompts)
                    details_to_show = {
                        k: v
                        for k, v in metric_score.details.items()
                        if k not in ["explanation", "judge_prompt", "judge_prompts"]
                    }
                    if details_to_show:
                        st.json(details_to_show)


def render_processed_data_tab(metrics: RecordMetrics | None, record_dir: Path):
    """Render the processed data tab with all context variables."""
    if not metrics or not hasattr(metrics, "context") or not metrics.context:
        st.warning("No processed data available (metrics context not found)")
        return

    context = metrics.context

    st.markdown("### Processed Variables from Metrics Context")
    st.info("These variables are processed by the MetricsContextProcessor and used for metric computation.")

    # Conversation with Tool Calls
    with st.expander("🔧 conversation_trace", expanded=True):
        if "conversation_trace" in context and context["conversation_trace"]:
            turns = context["conversation_trace"]
            for turn in turns:
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
                    tool_name = turn.get("tool_name", "unknown")
                    response = turn.get("tool_response", "")
                    response_str = str(response)
                    st.code(f"[Tool Response: {response_str}]", language=None)
        else:
            st.info("No conversation with tools data")

    # Tool Parameters
    with st.expander("🔧 Tool Parameters"):
        if "tool_params" in context and context["tool_params"]:
            st.json(context["tool_params"])
        else:
            st.info("No tool parameters data")

    # Tool Responses
    with st.expander("📥 Tool Responses"):
        if "tool_responses" in context and context["tool_responses"]:
            st.json(context["tool_responses"])
        else:
            st.info("No tool responses data")

    # Transcript by Speaker
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🗣️ Assistant Transcript (by Turn)"):
            if "transcribed_assistant_turns" in context and context["transcribed_assistant_turns"]:
                st.json(context["transcribed_assistant_turns"])
            else:
                st.info("No assistant transcript data")

    with col2:
        with st.expander("👤 User Transcript (by Turn)"):
            if "transcribed_user_turns" in context and context["transcribed_user_turns"]:
                st.json(context["transcribed_user_turns"])
            else:
                st.info("No user transcript data")

    # TTS Text
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔊 Assistant TTS Text (by Turn)"):
            if "intended_assistant_turns" in context and context["intended_assistant_turns"]:
                st.json(context["intended_assistant_turns"])
            else:
                st.info("No assistant TTS text data")

    with col2:
        with st.expander("🔊 User TTS Text (by Turn)"):
            if "intended_user_turns" in context and context["intended_user_turns"]:
                st.json(context["intended_user_turns"])
            else:
                st.info("No user TTS text data")

    # Audio Timestamps
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("⏱️ Assistant Audio Timestamps"):
            if "audio_timestamps_assistant_turns" in context and context["audio_timestamps_assistant_turns"]:
                # Format timestamps nicely
                timestamps = context["audio_timestamps_assistant_turns"]
                if isinstance(timestamps, dict):
                    formatted = {
                        f"Turn {k}": ", ".join(f"{seg[0]:.3f}s - {seg[1]:.3f}s" for seg in v)
                        if isinstance(v, list) and v and isinstance(v[0], list)
                        else str(v)
                        for k, v in timestamps.items()
                    }
                    st.json(formatted)
                else:
                    st.json(timestamps)
            else:
                st.info("No assistant audio timestamps")

    with col2:
        with st.expander("⏱️ User Audio Timestamps"):
            if "audio_timestamps_user_turns" in context and context["audio_timestamps_user_turns"]:
                # Format timestamps nicely
                timestamps = context["audio_timestamps_user_turns"]
                if isinstance(timestamps, dict):
                    formatted = {
                        f"Turn {k}": ", ".join(f"{seg[0]:.3f}s - {seg[1]:.3f}s" for seg in v)
                        if isinstance(v, list) and v and isinstance(v[0], list)
                        else str(v)
                        for k, v in timestamps.items()
                    }
                    st.json(formatted)
                else:
                    st.json(timestamps)
            else:
                st.info("No user audio timestamps")

    # Basic Stats
    with st.expander("📊 Conversation Statistics"):
        stats = {}
        if "num_assistant_turns" in context:
            stats["Assistant Turns"] = context["num_assistant_turns"]
        if "num_user_turns" in context:
            stats["User Turns"] = context["num_user_turns"]
        if "conversation_finished" in context:
            stats["Conversation Finished"] = context["conversation_finished"]
        if "agent_instructions" in context and context["agent_instructions"]:
            stats["Has Agent Instructions"] = True

        if stats:
            st.json(stats)
        else:
            st.info("No statistics data")

    # Agent Instructions
    with st.expander("📋 Agent Instructions"):
        if "agent_instructions" in context and context["agent_instructions"]:
            st.text(context["agent_instructions"])
        else:
            st.info("No agent instructions data")


# Map metric names to display groups for the overview section
_METRIC_GROUP: dict[str, str] = {
    "task_completion": "Accuracy",
    "faithfulness": "Accuracy",
    "agent_speech_fidelity": "Accuracy",
    "stt_wer": "Accuracy",
    "transcription_accuracy_key_entities": "Accuracy",
    "llm_to_tts_accuracy": "Accuracy",
    "conciseness": "Experience",
    "conversation_progression": "Experience",
    "speakability": "Experience",
    "speech_fluency": "Experience",
    "turn_taking": "Experience",
    "response_speed": "Experience",
    "tool_call_correctness": "Accuracy",
    "authentication": "Accuracy",
    "cost": "Validation",
    "user_behavioral_fidelity": "Validation",
    "conversation_finished": "Validation",
    "user_speech_fidelity": "Validation",
}

_CATEGORY_ORDER = ["Accuracy", "Experience", "Validation"]


def _sort_metrics_by_category(metric_names: list[str]) -> list[str]:
    """Sort metric names grouped by category, maintaining alphabetical order within each group."""

    def _sort_key(m: str) -> tuple[int, str]:
        cat = _METRIC_GROUP.get(m, "Other")
        idx = _CATEGORY_ORDER.index(cat) if cat in _CATEGORY_ORDER else len(_CATEGORY_ORDER)
        return (idx, m)

    return sorted(metric_names, key=_sort_key)


def _make_category_header_rename(metric_names: list[str]) -> dict[str, str]:
    """Create a column rename mapping that prefixes metric names with their category."""
    rename = {}
    for m in metric_names:
        cat = _METRIC_GROUP.get(m, "Other")
        rename[m] = f"[{cat}] {m}"
    return rename


# Metrics known to be per-turn for a specific role, even if details are empty on failure
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
}


def _classify_metrics(
    metrics: RecordMetrics,
) -> tuple[dict[str, dict], dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """Split metrics into conversation-level, user per-turn, and assistant per-turn.

    Returns:
        (conversation_metrics, user_per_turn_metrics, assistant_per_turn_metrics)
    """
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
            # Determine role
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


_ACRONYMS = {"tts", "stt", "wer", "llm", "db"}


def _format_metric_name(name: str) -> str:
    """Format a metric name for display, preserving acronyms in upper case."""
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


def _get_turn_metric_info(turn_id: int, per_turn_metrics: dict[str, dict]) -> list[dict]:
    """Get per-turn metric info (rating, explanation) for a given turn_id."""
    results = []
    tid = str(turn_id)
    for metric_name, metric_data in per_turn_metrics.items():
        details = metric_data["details"]
        rating = None
        explanation = ""

        # turn_taking uses different detail keys (all keyed by string turn_id)
        if metric_name == "turn_taking":
            label = (details.get("per_turn_judge_timing_ratings") or {}).get(tid)
            if label is None:
                continue
            # Map string label to numeric for color coding
            label_lower = label.strip().lower()
            if label_lower == "on-time":
                rating = 1.0
            elif label_lower == "late":
                rating = 0.3
            else:  # Early / Interrupting
                rating = 0.0
            latency = (details.get("per_turn_latency") or {}).get(tid)
            latency_str = f" ({latency:.3f}s)" if latency is not None else ""
            explanation = (details.get("per_turn_judge_timing_explanations") or {}).get(tid, "")
            short_name = f"{label}{latency_str}"
            results.append(
                {
                    "metric_name": metric_name,
                    "short_name": short_name,
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
    """Build HTML badge strings from turn metric info list."""
    badges = []
    for info in turn_metrics:
        color = _score_color(info["rating"])
        tooltip = html.escape(info["explanation"], quote=True) if info["explanation"] else ""
        badges.append(
            f'<span title="{tooltip}" style="display:inline-block; background:{color}22; '
            f"color:{color}; border:1px solid {color}44; border-radius:4px; "
            f'padding:1px 6px; margin:2px 4px 2px 0; font-size:0.75em; cursor:help;">'
            f"{info['short_name']}: {info['rating']:.2f}</span>"
        )
    return "".join(badges)


@st.cache_data(show_spinner=False)
def _load_audio(audio_path: str) -> bytes | None:
    """Load a WAV file and return raw bytes (cached)."""
    p = Path(audio_path)
    if not p.exists():
        return None
    return p.read_bytes()


def _compute_clock_mapping(record_dir: Path, audio_bytes: bytes) -> dict:
    """Compute ElevenLabs-to-WAV clock mapping from elevenlabs events and WAV duration.

    Returns dict with session_started_sec, clock_scale, or empty if unavailable.
    """
    elevenlabs_path = record_dir / "elevenlabs_events.jsonl"
    if not elevenlabs_path.exists():
        return {}

    session_started_sec = None
    session_ended_sec = None
    try:
        with open(elevenlabs_path) as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") != "connection_state":
                    continue
                data = event.get("data", {})
                state = data.get("state")
                ts = event.get("timestamp")
                if ts is None:
                    continue
                ts_sec = ts / 1000.0
                if state == "session_started" and session_started_sec is None:
                    session_started_sec = ts_sec
                elif state == "session_ended":
                    session_ended_sec = ts_sec
    except Exception:
        return {}

    if session_started_sec is None or session_ended_sec is None:
        return {}

    session_duration = session_ended_sec - session_started_sec
    if session_duration <= 0:
        return {}

    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        wav_duration_sec = len(audio) / 1000.0
    except Exception:
        return {}

    return {
        "session_started_sec": session_started_sec,
        "clock_scale": wav_duration_sec / session_duration,
    }


def _el_ts_to_wav_offset(clock_map: dict, el_ts_sec: float) -> float:
    """Convert an ElevenLabs absolute timestamp to a WAV-file offset in seconds."""
    session_start = clock_map["session_started_sec"]
    scale = clock_map["clock_scale"]
    return (el_ts_sec - session_start) * scale


def _extract_audio_segment(audio_bytes: bytes, start_sec: float, end_sec: float, padding_sec: float = 0.5) -> bytes:
    """Extract a segment from a WAV file, return as WAV bytes."""
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    start_ms = max(0, int((start_sec - padding_sec) * 1000))
    end_ms = min(len(audio), int((end_sec + padding_sec) * 1000))
    segment = audio[start_ms:end_ms]
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return buf.getvalue()


def _get_turn_timestamps(turn_id: int, role: str, context: dict) -> list | None:
    """Get the raw [start, end] timestamp pair for a turn."""
    if role == "assistant":
        timestamps = context.get("audio_timestamps_assistant_turns", {})
    else:
        timestamps = context.get("audio_timestamps_user_turns", {})

    segs = timestamps.get(str(turn_id)) or timestamps.get(turn_id)
    if not segs:
        return None
    # Flatten segments to [earliest_start, latest_end]
    return [min(s[0] for s in segs), max(s[1] for s in segs)]


def render_conversation_trace_tab(metrics: RecordMetrics | None, record_dir: Path):
    """Render the conversation trace tab with a chat-like visualization.

    Shows conversation-level metrics at the top, then the trace with per-turn
    metrics inlined next to the corresponding user turns.
    """
    if not metrics or not hasattr(metrics, "context") or not metrics.context:
        st.warning("No processed data available (metrics context not found)")
        return

    context = metrics.context
    trace = context.get("conversation_trace", [])
    if not trace:
        st.info("No conversation trace data")
        return

    # Load audio and clock mapping for per-turn playback
    audio_mixed_path = record_dir / "audio_mixed.wav"
    audio_bytes = _load_audio(str(audio_mixed_path))
    clock_map = _compute_clock_mapping(record_dir, audio_bytes) if audio_bytes else {}

    # Classify metrics
    conversation_metrics, user_per_turn, assistant_per_turn = _classify_metrics(metrics)
    all_per_turn = {**user_per_turn, **assistant_per_turn}
    has_turn_metrics = bool(all_per_turn)

    # Build lookup dicts for the "reverse" of each turn
    intended_user = context.get("intended_user_turns", {})
    transcribed_assistant = context.get("transcribed_assistant_turns", {})
    intended_user = {int(k): v for k, v in intended_user.items()} if intended_user else {}
    transcribed_assistant = {int(k): v for k, v in transcribed_assistant.items()} if transcribed_assistant else {}

    # Build per-turn WER error details for inline display
    stt_wer_details = user_per_turn.get("stt_wer", {}).get("details", {})
    per_turn_wer = stt_wer_details.get("per_turn_wer", {})
    per_turn_errors = stt_wer_details.get("per_turn_errors", {})

    # --- All metrics at the top, grouped by category ---
    all_top_metrics = {**conversation_metrics}
    for name, m in all_per_turn.items():
        all_top_metrics[name] = m

    if all_top_metrics:
        # CSS to make metric buttons equal height
        st.markdown(
            """<style>
            div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
                height: 4.5rem;
                white-space: normal;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        st.markdown("### Metrics Overview")

        # Group metrics by category
        grouped: dict[str, list[str]] = {}
        for name in all_top_metrics:
            group = _METRIC_GROUP.get(name, "Other")
            grouped.setdefault(group, []).append(name)

        # Render groups in a fixed order
        group_order = ["Accuracy", "Experience", "Validation", "Other"]
        for group in group_order:
            names_in_group = grouped.get(group)
            if not names_in_group:
                continue
            st.caption(group)
            cols = st.columns(min(len(names_in_group), 5))
            for i, name in enumerate(names_in_group):
                m = all_top_metrics[name]
                col = cols[i % len(cols)]
                score = m["normalized_score"]
                with col:
                    display_name = _format_metric_name(name)
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    icon = None
                    if name != "cost" and score is not None:
                        if score >= 0.8:
                            icon = "🟢"
                        elif score >= 0.4:
                            icon = "🟡"
                        else:
                            icon = "🔴"
                    if st.button(
                        f"{display_name}\n{score_str}", key=f"metric_btn_{name}", use_container_width=True, icon=icon
                    ):
                        if st.session_state.get("selected_metric") == name:
                            st.session_state["selected_metric"] = None
                        else:
                            st.session_state["selected_metric"] = name
                        st.rerun()

        # Show details for selected metric
        selected = st.session_state.get("selected_metric")
        if selected and selected in conversation_metrics:
            m = conversation_metrics[selected]
            details = m.get("details", {})
            details_to_show = {
                k: v for k, v in details.items() if k not in ("judge_prompt", "judge_prompts", "judge_raw_response")
            }
            display_name = _format_metric_name(selected)
            st.markdown(f"**{display_name}**")
            explanation = details.get("explanation") or details.get("message", "")
            if explanation and isinstance(explanation, str):
                st.write(explanation)

            # Special handling: task_completion shows a scenario DB diff
            if selected == "task_completion":
                expected_db = context.get("expected_scenario_db")
                final_db = context.get("final_scenario_db")
                if expected_db and final_db:
                    expected_str = json.dumps(expected_db, indent=2, sort_keys=True, default=str)
                    actual_str = json.dumps(final_db, indent=2, sort_keys=True, default=str)
                    diff_viewer(
                        expected_str,
                        actual_str,
                        lang="json",
                        key="task_completion_diff",
                    )
            elif details_to_show:
                st.json(details_to_show)

        elif selected and selected in all_per_turn:
            m = all_per_turn[selected]
            details = m.get("details", {})
            display_name = _format_metric_name(selected)
            st.markdown(f"**{display_name}** — per-turn breakdown")

            # Collect all turn IDs from any per_turn_* key
            turn_ids: set[str] = set()
            for k, v in details.items():
                if k.startswith("per_turn") and isinstance(v, dict):
                    turn_ids.update(v.keys())
            turn_ids_sorted = sorted(turn_ids, key=lambda x: int(x) if x.isdigit() else x)

            if turn_ids_sorted:
                for tid_str in turn_ids_sorted:
                    # Gather rating
                    rating = None
                    explanation = ""
                    label = ""

                    if selected == "turn_taking":
                        label = (details.get("per_turn_judge_timing_ratings") or {}).get(tid_str, "")
                        latency = (details.get("per_turn_latency") or {}).get(tid_str)
                        explanation = (details.get("per_turn_judge_timing_explanations") or {}).get(tid_str, "")
                        latency_str = f"{latency:.3f}s" if latency is not None else "N/A"
                        # Color from label
                        if label.strip().lower() == "on-time":
                            color = "#4caf50"
                        elif label.strip().lower() == "late":
                            color = "#ff9800"
                        else:
                            color = "#f44336"
                        st.markdown(
                            f'<div style="margin-bottom:8px; padding:6px 10px; background:#fafafa; border-radius:6px; border-left:3px solid {color};">'
                            f"<strong>Turn {tid_str}</strong> — "
                            f'<span style="color:{color}; font-weight:600;">{label}</span> '
                            f'<span style="opacity:0.6;">(latency: {latency_str})</span>'
                            f"{'<br><span style=font-size:0.88em;opacity:0.75;>' + html.escape(str(explanation)) + '</span>' if explanation else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
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
                        rating_str = (
                            f"{rating:.2f}" if isinstance(rating, (int, float)) and rating is not None else str(rating)
                        )
                        st.markdown(
                            f'<div style="margin-bottom:8px; padding:6px 10px; background:#fafafa; border-radius:6px; border-left:3px solid {color};">'
                            f"<strong>Turn {tid_str}</strong> — "
                            f'<span style="color:{color}; font-weight:600;">{rating_str}</span>'
                            f"{'<br><span style=font-size:0.88em;opacity:0.75;>' + html.escape(str(explanation)) + '</span>' if explanation else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
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
        legend_parts = []
        for metric_name in all_per_turn:
            short_name = metric_name.replace("transcription_accuracy_", "").replace("_", " ")
            legend_parts.append(f"`{short_name}`")
        st.caption(f"Per-turn metrics shown inline: {', '.join(legend_parts)}")

    for entry in trace:
        role = entry.get("role", "")
        content = entry.get("content", "")
        entry_type = entry.get("type", "")
        turn_id = entry.get("turn_id")
        tool_name = entry.get("tool_name", "")

        if role == "assistant":
            reverse_text = transcribed_assistant.get(turn_id, "") if turn_id is not None else ""
            reverse_html = ""
            if reverse_text and reverse_text != content:
                reverse_html = (
                    f'<div style="font-size:0.78em; opacity:0.65; margin-top:6px; '
                    f'border-top:1px solid rgba(25,118,210,0.2); padding-top:4px;">'
                    f"<em>transcribed (STT): {reverse_text}</em></div>"
                )
            # Assistant per-turn metric badges
            turn_metrics = []
            badges_html = ""
            if turn_id is not None and assistant_per_turn:
                turn_metrics = _get_turn_metric_info(turn_id, assistant_per_turn)
                if turn_metrics:
                    badges_html = _render_turn_metric_badges(turn_metrics)
                    badges_html = (
                        f'<div style="margin-top:6px; border-top:1px solid rgba(25,118,210,0.2); '
                        f'padding-top:4px;">{badges_html}</div>'
                    )

            if has_turn_metrics:
                col_left, col_right = st.columns([3, 1])
            else:
                col_left = st.container()
                col_right = None

            with col_left:
                st.markdown(
                    f'<div style="background-color:#e8f4fd; padding:10px 14px; border-radius:8px; '
                    f'border-left:4px solid #1976d2; margin-bottom:8px;">'
                    f'<strong style="color:#1976d2;">Assistant</strong> <span style="font-size:0.72em; opacity:0.5;">turn {turn_id}</span><br>{content}'
                    f"{reverse_html}{badges_html}</div>",
                    unsafe_allow_html=True,
                )
                if audio_bytes and clock_map and turn_id is not None:
                    ts = _get_turn_timestamps(turn_id, "assistant", context)
                    if ts:
                        start_off = _el_ts_to_wav_offset(clock_map, ts[0])
                        end_off = _el_ts_to_wav_offset(clock_map, ts[1])
                        seg = _extract_audio_segment(audio_bytes, start_off, end_off, padding_sec=5.0)
                        st.audio(seg, format="audio/wav")

            if turn_metrics and col_right is not None:
                explanations = [m for m in turn_metrics if m["explanation"]]
                if explanations:
                    with col_right:
                        for m in explanations:
                            color = _score_color(m["rating"])
                            st.markdown(
                                f'<div style="font-size:0.82em; margin-bottom:8px;">'
                                f"<strong>{m['short_name']}</strong> "
                                f'<span style="color:{color}; font-weight:600;">{m["rating"]:.2f}</span>'
                                f'<br><span style="font-size:0.85em; opacity:0.7;">{html.escape(str(m["explanation"]))}</span>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        elif role == "user":
            reverse_text = intended_user.get(turn_id, "") if turn_id is not None else ""
            reverse_html = ""
            if reverse_text and reverse_text != content:
                reverse_html = (
                    f'<div style="font-size:0.78em; opacity:0.65; margin-top:6px; '
                    f'border-top:1px solid rgba(123,31,162,0.2); padding-top:4px;">'
                    f"<em>intended (TTS): {reverse_text}</em></div>"
                )
            # User per-turn metric badges
            turn_metrics = []
            badges_html = ""
            if turn_id is not None and user_per_turn:
                turn_metrics = _get_turn_metric_info(turn_id, user_per_turn)
                if turn_metrics:
                    badges_html = _render_turn_metric_badges(turn_metrics)
                    badges_html = (
                        f'<div style="margin-top:6px; border-top:1px solid rgba(123,31,162,0.2); '
                        f'padding-top:4px;">{badges_html}</div>'
                    )

            # STT WER error details inside the bubble
            wer_html = ""
            tid = str(turn_id) if turn_id is not None else ""
            if tid in per_turn_wer:
                wer_val = per_turn_wer[tid]
                errors = per_turn_errors.get(tid, {})
                error_parts = []
                for sub in errors.get("substitutions", []):
                    error_parts.append(f"{sub['expected']} → {sub['actual']}")
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
                st.markdown(
                    f'<div style="background-color:#f3e8fd; padding:10px 14px; border-radius:8px; '
                    f'border-left:4px solid #7b1fa2; margin-bottom:8px;">'
                    f'<strong style="color:#7b1fa2;">User</strong> <span style="font-size:0.72em; opacity:0.5;">turn {turn_id}</span><br>{content}'
                    f"{reverse_html}{wer_html}{badges_html}</div>",
                    unsafe_allow_html=True,
                )
                if audio_bytes and clock_map and turn_id is not None:
                    ts = _get_turn_timestamps(turn_id, "user", context)
                    if ts:
                        start_off = _el_ts_to_wav_offset(clock_map, ts[0])
                        end_off = _el_ts_to_wav_offset(clock_map, ts[1])
                        seg = _extract_audio_segment(audio_bytes, start_off, end_off, padding_sec=5.0)
                        st.audio(seg, format="audio/wav")

            # Per-turn metric explanations in the right column
            if turn_metrics and col_right is not None:
                explanations = [m for m in turn_metrics if m["explanation"]]
                if explanations:
                    with col_right:
                        for m in explanations:
                            color = _score_color(m["rating"])
                            st.markdown(
                                f'<div style="font-size:0.82em; margin-bottom:8px;">'
                                f"<strong>{m['short_name']}</strong> "
                                f'<span style="color:{color}; font-weight:600;">{m["rating"]:.2f}</span>'
                                f'<br><span style="font-size:0.85em; opacity:0.7;">{html.escape(str(m["explanation"]))}</span>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

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


def _get_record_data_dirs(record_dir: Path) -> list[tuple[str, Path]]:
    """Get the data directories for a record, handling both trial and non-trial layouts.

    Returns a list of (label, path) tuples. For non-trial records, returns a single
    entry with an empty label. For trial records, returns one entry per trial.
    """
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
    # No trials — the record dir itself holds the data
    return [("", record_dir)]


def render_run_overview(run_dir: Path):
    """Render a run-level overview with aggregate metrics and a per-record table."""
    rows, metric_names = _collect_run_metrics(run_dir)

    if not rows:
        st.warning("No metrics data found for any record")
        return

    has_trials = "trial" in rows[0]

    df = pd.DataFrame(rows)

    # Sort metrics by category
    metric_names = _sort_metrics_by_category(metric_names)
    col_rename = _make_category_header_rename(metric_names)

    # --- Aggregate summary ---
    st.markdown("### Aggregate Metrics")

    # Compute mean per metric (across all records/trials)
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
        # Show metric cards grouped by category
        grouped: dict[str, list[tuple[str, dict]]] = {}
        for m, stats in summary_data.items():
            cat = _METRIC_GROUP.get(m, "Other")
            grouped.setdefault(cat, []).append((m, stats))

        for cat in _CATEGORY_ORDER + ["Other"]:
            if cat not in grouped:
                continue
            st.markdown(f"**{cat}**")
            cols = st.columns(min(len(grouped[cat]), 4))
            for i, (m, stats) in enumerate(grouped[cat]):
                with cols[i % len(cols)]:
                    st.metric(
                        m,
                        f"{stats['mean']:.3f}",
                        help=f"std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}, n={stats['count']}",
                    )

        # Also show summary table
        summary_df = pd.DataFrame(summary_data).T
        summary_df.index.name = "metric"
        summary_df["category"] = [_METRIC_GROUP.get(m, "Other") for m in summary_df.index]
        summary_df = summary_df[["category", "mean", "std", "min", "max", "count"]].round(3)
        st.dataframe(summary_df, use_container_width=True)

    st.divider()

    # --- Per-record table ---
    st.markdown("### Per-Record Metrics")

    # Reorder columns: record, trial (if present), then metrics sorted by category
    leading_cols = ["record"]
    if has_trials:
        leading_cols.append("trial")
    ordered_metrics = [m for m in metric_names if m in df.columns]
    ordered_cols = leading_cols + ordered_metrics
    df = df[ordered_cols]
    df = df.rename(columns=col_rename)

    # Style: color-code metric values
    def _color_cell(val):
        if pd.isna(val):
            return "background-color: #f5f5f5; color: #999;"
        if isinstance(val, (int, float)):
            if val >= 0.8:
                return "background-color: #e8f5e9;"
            elif val >= 0.5:
                return "background-color: #fff3e0;"
            else:
                return "background-color: #ffebee;"
        return ""

    renamed_metrics = [col_rename[m] for m in ordered_metrics]
    styled = df.style.map(_color_cell, subset=renamed_metrics)
    styled = styled.format(dict.fromkeys(renamed_metrics, "{:.3f}"), na_rep="—")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # CSV download
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=f"{run_dir.name}_metrics.csv", mime="text/csv")


# AI Review view is in ai_review_view.py
from internal.ai_review_view import render_ai_review  # noqa: E402

# ============================================================================
# Main App
# ============================================================================


def _load_run_config(run_dir: Path) -> dict:
    """Load config.json for a run, returning empty dict on failure."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_run_label(run_name: str, run_config: dict) -> str:
    """Build a short label for a run from its config."""
    pipeline = run_config.get("pipeline") or run_config.get("model") or {}
    model = pipeline.get("realtime_model") or pipeline.get("llm_model") or ""
    if not model:
        return run_name
    parts = [model]
    stt = pipeline.get("stt_model")
    tts = pipeline.get("tts_model")
    if stt:
        parts.append(f"STT: {stt}")
    if tts:
        parts.append(f"TTS: {tts}")
    return f"{run_name} ({', '.join(parts)})"


def _render_sidebar_run_metadata(run_name: str, run_config: dict):
    """Render run metadata in the sidebar info box."""
    pipeline = run_config.get("pipeline") or run_config.get("model") or {}
    metadata_parts = [f"**Run:** {run_name}"]
    if pipeline.get("realtime_model"):
        metadata_parts.append(f"**Realtime Model:** {pipeline['realtime_model']}")
    elif pipeline.get("llm_model"):
        metadata_parts.append(f"**LLM Model:** {pipeline['llm_model']}")
    if pipeline.get("stt_model"):
        metadata_parts.append(f"**STT:** {pipeline['stt_model']}")
    if pipeline.get("tts_model"):
        metadata_parts.append(f"**TTS:** {pipeline['tts_model']}")
    if run_config.get("num_trials"):
        metadata_parts.append(f"**Trials:** {run_config['num_trials']}")
    provenance = run_config.get("provenance", {})
    if provenance.get("git_branch"):
        metadata_parts.append(f"**Branch:** {provenance['git_branch']}")
    st.sidebar.info("\n\n".join(metadata_parts))


def _collect_run_metrics(run_dir: Path) -> tuple[list[dict], list[str]]:
    """Collect all metrics rows for a run. Returns (rows, metric_names)."""
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

            row: dict = {"record": record_id}
            if trial_label:
                row["trial"] = trial_label

            for metric_name, metric_score in metrics.metrics.items():
                all_metric_names.add(metric_name)
                if metric_score.error:
                    row[metric_name] = None
                else:
                    row[metric_name] = (
                        metric_score.normalized_score
                        if metric_score.normalized_score is not None
                        else metric_score.score
                    )

            # Computed on-the-fly: authentication check
            auth = check_authentication_success(data_path)
            all_metric_names.add("authentication")
            if auth is True:
                row["authentication"] = 1.0
            elif auth is False:
                row["authentication"] = 0.0

            rows.append(row)

    return rows, sorted(all_metric_names)


def render_cross_run_comparison(run_dirs: list[Path]):
    """Render a comparison view across multiple runs."""
    st.markdown("### Cross-Run Comparison")
    st.caption("Compare aggregate metrics across all runs that have metrics data.")

    # Collect per-run aggregates
    run_summaries: list[dict] = []
    all_metric_names: set[str] = set()

    for run_dir in run_dirs:
        run_name = run_dir.name
        run_config = _load_run_config(run_dir)
        rows, metric_names = _collect_run_metrics(run_dir)
        if not rows:
            continue

        all_metric_names.update(metric_names)
        df = pd.DataFrame(rows)

        summary: dict = {
            "run": run_name,
            "label": _get_run_label(run_name, run_config),
            "records": len(df),
        }

        for m in metric_names:
            if m in df.columns:
                vals = df[m].dropna()
                if len(vals) > 0:
                    summary[m] = vals.mean()

        run_summaries.append(summary)

    if not run_summaries:
        st.warning("No runs with metrics data found")
        return

    metric_names = _sort_metrics_by_category(sorted(all_metric_names))
    col_rename = _make_category_header_rename(metric_names)
    summary_df = pd.DataFrame(run_summaries)

    # --- Summary table ---
    st.markdown("#### Mean Metrics per Run")

    ordered_metrics = [m for m in metric_names if m in summary_df.columns]
    display_cols = ["label", "records"] + ordered_metrics
    display_df = summary_df[display_cols].copy()
    display_df = display_df.rename(columns={"label": "Run", "records": "# Records", **col_rename})
    renamed_metrics = [col_rename[m] for m in ordered_metrics]

    def _color_cell(val):
        if pd.isna(val):
            return "background-color: #f5f5f5; color: #999;"
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if val >= 0.8:
                return "background-color: #e8f5e9;"
            elif val >= 0.5:
                return "background-color: #fff3e0;"
            elif val < 0.5:
                return "background-color: #ffebee;"
        return ""

    styled = display_df.style.map(_color_cell, subset=renamed_metrics)
    styled = styled.format(dict.fromkeys(renamed_metrics, "{:.3f}"), na_rep="—")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Bar chart comparison ---
    if len(run_summaries) > 1 and metric_names:
        st.markdown("#### Metric Comparison")
        chart_options = [f"{col_rename[m]}" for m in metric_names]
        selected_display = st.selectbox("Select metric to compare", chart_options)
        # Map back to original name
        reverse_rename = {v: k for k, v in col_rename.items()}
        selected_metric = reverse_rename.get(selected_display, "")
        if selected_metric in summary_df.columns:
            chart_df = summary_df[["label", selected_metric]].dropna(subset=[selected_metric])
            chart_df = chart_df.rename(columns={"label": "Run", selected_metric: "Score"})
            st.bar_chart(chart_df, x="Run", y="Score")

    # CSV download
    csv = summary_df.drop(columns=["label"]).to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        file_name="cross_run_comparison.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Voice Agent Benchmark Analysis", page_icon="🎙️", layout="wide")

    st.title("🎙️ Voice Agent Benchmark Analysis")

    # Read query params for deep linking
    query_params = st.query_params

    # Sidebar: Output directory (with query param support for shareable links)
    st.sidebar.header("Configuration")
    default_output_dir = query_params.get("output_dir", str(OUTPUT_DIR))
    output_dir_input = st.sidebar.text_input("Output Directory", value=default_output_dir)
    effective_output_dir = Path(output_dir_input) if output_dir_input else OUTPUT_DIR
    st.query_params["output_dir"] = str(effective_output_dir)

    # Get run directories
    run_dirs = get_run_directories(effective_output_dir)

    if not run_dirs:
        st.error(f"No run directories found in {effective_output_dir}")
        return

    # View mode toggle
    st.sidebar.header("View")
    default_view = query_params.get("view", "Record Detail")
    view_options = ["Cross-Run Comparison", "Run Overview", "Record Detail", "AI Review"]
    default_view_idx = view_options.index(default_view) if default_view in view_options else 2
    view_mode = st.sidebar.radio("View", view_options, index=default_view_idx, label_visibility="collapsed")

    if view_mode == "Cross-Run Comparison":
        st.query_params.update({"view": "Cross-Run Comparison"})
        render_cross_run_comparison(run_dirs)
        return

    if view_mode == "AI Review":
        st.query_params.update({"view": "AI Review"})
        render_ai_review(run_dirs)
        return

    # Sidebar: Run selection (for Run Overview and Record Detail)
    st.sidebar.header("Run Selection")

    run_dir_names = [d.name for d in run_dirs]
    default_run_idx = 0
    if "run" in query_params and query_params["run"] in run_dir_names:
        default_run_idx = run_dir_names.index(query_params["run"])
    selected_run_name = st.sidebar.selectbox(
        "Select Run",
        run_dir_names,
        index=default_run_idx,
    )

    selected_run_dir = effective_output_dir / selected_run_name

    # Display run metadata from config.json
    run_config = _load_run_config(selected_run_dir)
    if run_config:
        _render_sidebar_run_metadata(selected_run_name, run_config)
    else:
        st.sidebar.info(f"**Run:** {selected_run_name}")

    if view_mode == "Run Overview":
        st.query_params.update({"run": selected_run_name, "view": "Run Overview"})
        render_run_overview(selected_run_dir)
        return

    # Get records
    record_dirs = get_record_directories(selected_run_dir)

    if not record_dirs:
        st.error(f"No records found in {selected_run_dir / 'records'}")
        return

    # Sidebar: Record selection
    st.sidebar.header("Record Selection")
    record_names = [d.name for d in record_dirs]
    default_record_idx = 0
    if "record" in query_params and query_params["record"] in record_names:
        default_record_idx = record_names.index(query_params["record"])
    selected_record_name = st.sidebar.selectbox("Select Record", record_names, index=default_record_idx)

    selected_record_dir = selected_run_dir / "records" / selected_record_name

    # Detect trial subdirectories (e.g. trial_0, trial_1, ...)
    # A subdirectory is a trial if it's a directory and contains typical record files
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
        default_trial_idx = 0
        if "trial" in query_params and query_params["trial"] in trial_names:
            default_trial_idx = trial_names.index(query_params["trial"])
        selected_trial = st.sidebar.selectbox("Select Trial", trial_names, index=default_trial_idx)
        selected_record_dir = selected_record_dir / selected_trial

    # Update query params to reflect current selection (enables shareable URLs)
    new_params = {"run": selected_run_name, "record": selected_record_name, "view": "Record Detail"}
    if selected_trial:
        new_params["trial"] = selected_trial
    st.query_params.update(new_params)

    # Load record data
    load_record_result(selected_record_dir)
    metrics = load_record_metrics(selected_record_dir)
    eval_record = load_evaluation_record(selected_run_dir, selected_record_name)

    # Display record header
    header = f"Record: {selected_record_name}"
    if selected_trial:
        header += f" / {selected_trial}"
    st.header(header)

    # Computed on-the-fly metrics
    auth_result = check_authentication_success(selected_record_dir)
    if auth_result is True:
        st.success("Authentication: Passed (get_reservation returned success)")
    elif auth_result is False:
        st.error("Authentication: Failed (get_reservation called but did not return success)")
    else:
        st.warning("Authentication: N/A (get_reservation was not called)")

    # Audio player (prominent at top)
    audio_path = selected_record_dir / "audio_mixed.wav"
    if audio_path.exists():
        st.markdown("### 🔊 Audio Recording")
        st.audio(str(audio_path))
    else:
        st.warning("No audio file found")

    st.divider()

    # Get agent instructions from audit log
    audit_log_path = selected_record_dir / "audit_log.json"
    if audit_log_path.exists():
        try:
            with open(audit_log_path) as f:
                audit_log = json.load(f)
            llm_prompts = audit_log.get("llm_prompts", [])
            if llm_prompts and len(llm_prompts) > 0:
                first_prompt = llm_prompts[0].get("prompt", [])
                if first_prompt and len(first_prompt) > 0:
                    system_message = first_prompt[0]
                    if isinstance(system_message, dict) and system_message.get("role") == "system":
                        _ = system_message.get("content", "")
        except Exception:
            pass

    _ = eval_record.user_goal if eval_record else "Unknown"

    # with col2:
    # st.markdown("### ✅ Key Objectives")
    with st.expander("✅ Key Objectives", expanded=False):
        if eval_record and hasattr(eval_record, "ground_truth") and eval_record.ground_truth:
            objectives = getattr(eval_record.ground_truth, "key_objectives_required", None)
            if objectives:
                st.json(objectives)
                # for i, obj in enumerate(objectives, 1):
                #     st.markdown(f"{i}. {obj}")
            else:
                st.info("No key objectives defined")
        else:
            st.info("No key objectives defined")

    # User goal below in collapsed expander
    with st.expander("📋 User Goal", expanded=False):
        if eval_record:
            # Try to parse user_goal as JSON for better formatting
            try:
                user_goal_data = json.loads(eval_record.user_goal)
                st.json(user_goal_data)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, display as regular text
                st.write(eval_record.user_goal)
        else:
            st.info("No user goal available")

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "💬 Conversation Trace",
            "📝 Transcript",
            "📋 Audit Log",
            "🔍 Pipecat Logs",
            "🎤 ElevenLabs Logs",
            "📊 Metrics",
            "🔧 Processed Data",
        ]
    )

    with tab1:
        render_conversation_trace_tab(metrics, selected_record_dir)

    with tab2:
        st.markdown("### Transcript")

        # Try to load from metrics context first
        if metrics and hasattr(metrics, "context") and metrics.context and "turns_transcript" in metrics.context:
            try:
                turns = metrics.context["turns_transcript"]
                if turns:
                    transcript_df = pd.DataFrame(turns)

                    # Configure columns - make content column much wider
                    column_config = {}
                    if "content" in transcript_df.columns:
                        column_config["content"] = st.column_config.TextColumn("content", width="large")
                    if "timestamp" in transcript_df.columns:
                        column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                    if "role" in transcript_df.columns:
                        column_config["role"] = st.column_config.TextColumn("role", width="small")

                    st.dataframe(transcript_df, use_container_width=True, hide_index=True, column_config=column_config)
                else:
                    st.info("No transcript data available")
            except Exception as e:
                st.error(f"Failed to load transcript from context: {e}")
                # Fall back to file-based loading
                transcript_df = format_transcript(selected_record_dir / "transcript.jsonl")
                if not transcript_df.empty:
                    column_config = {}
                    if "content" in transcript_df.columns:
                        column_config["content"] = st.column_config.TextColumn("content", width="large")
                    if "timestamp" in transcript_df.columns:
                        column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                    if "role" in transcript_df.columns:
                        column_config["role"] = st.column_config.TextColumn("role", width="small")
                    st.dataframe(transcript_df, use_container_width=True, hide_index=True, column_config=column_config)
        else:
            # Fallback to file-based loading
            transcript_df = format_transcript(selected_record_dir / "transcript.jsonl")
            if not transcript_df.empty:
                # Configure columns - make content column much wider
                column_config = {}
                if "content" in transcript_df.columns:
                    column_config["content"] = st.column_config.TextColumn("content", width="large")
                if "timestamp" in transcript_df.columns:
                    column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                if "role" in transcript_df.columns:
                    column_config["role"] = st.column_config.TextColumn("role", width="small")

                st.dataframe(transcript_df, use_container_width=True, hide_index=True, column_config=column_config)
            else:
                st.info("No transcript data available")

    with tab3:
        st.markdown("### Audit Log")
        audit_log_path = selected_record_dir / "audit_log.json"
        if audit_log_path.exists():
            try:
                with open(audit_log_path) as f:
                    audit_log = json.load(f)
                st.json(audit_log)
            except Exception as e:
                st.error(f"Failed to load audit log: {e}")
        else:
            st.info("No audit log available")

    with tab4:
        st.markdown("### Pipecat Logs")

        # Get unique log types for filtering
        pipecat_path = selected_record_dir / "pipecat_logs.jsonl"
        if pipecat_path.exists():
            try:
                # Extract unique types
                log_types = set()
                with open(pipecat_path) as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            log_types.add(entry.get("type", "unknown"))

                log_types = sorted(log_types)

                # Filter dropdown
                selected_type = st.selectbox("Filter by log type", ["All"] + log_types)

                # Load and display
                filter_type = None if selected_type == "All" else selected_type
                pipecat_df = format_pipecat_logs(pipecat_path, filter_type)

                if not pipecat_df.empty:
                    # Configure columns - make data column much wider, others narrower
                    column_config = {}
                    if "data" in pipecat_df.columns:
                        column_config["data"] = st.column_config.TextColumn("data", width="large")
                    if "timestamp" in pipecat_df.columns:
                        column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                    if "type" in pipecat_df.columns:
                        column_config["type"] = st.column_config.TextColumn("type", width="small")
                    if "source" in pipecat_df.columns:
                        column_config["source"] = st.column_config.TextColumn("source", width="medium")
                    if "destination" in pipecat_df.columns:
                        column_config["destination"] = st.column_config.TextColumn("destination", width="medium")

                    st.dataframe(pipecat_df, use_container_width=True, hide_index=True, column_config=column_config)
                else:
                    st.info("No pipecat logs match the filter")

            except Exception as e:
                st.error(f"Failed to load pipecat logs: {e}")
        else:
            st.info("No pipecat logs available")

    with tab5:
        st.markdown("### ElevenLabs Logs")
        elevenlabs_path = selected_record_dir / "elevenlabs_events.jsonl"
        if elevenlabs_path.exists():
            try:
                entries = []
                with open(elevenlabs_path) as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))

                if entries:
                    df = pd.DataFrame(entries)

                    # Configure columns - make data column much wider, others narrower
                    column_config = {}
                    if "data" in df.columns:
                        column_config["data"] = st.column_config.TextColumn("data", width="large")
                    if "timestamp" in df.columns:
                        column_config["timestamp"] = st.column_config.TextColumn("timestamp", width="small")
                    if "type" in df.columns:
                        column_config["type"] = st.column_config.TextColumn("type", width="small")
                    if "event" in df.columns:
                        column_config["event"] = st.column_config.TextColumn("event", width="small")

                    st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config)
                else:
                    st.info("No ElevenLabs events")
            except Exception as e:
                st.error(f"Failed to load ElevenLabs logs: {e}")
        else:
            st.info("No ElevenLabs logs available")

    with tab6:
        render_metrics_tab(metrics)

    with tab7:
        render_processed_data_tab(metrics, selected_record_dir)


if __name__ == "__main__":
    main()
