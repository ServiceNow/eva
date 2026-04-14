"""AI Review Pipeline for EVA.

Multi-pass LLM analysis pipeline that examines voice agent benchmark results
to identify failure patterns, categorize errors, and produce cross-metric
and cross-model synthesis reports.

No Streamlit dependency — this module is purely data + LLM logic.
"""

import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eva.models.results import RecordMetrics
from eva.utils.json_utils import extract_and_load_json
from eva.utils.llm_client import LLMClient
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# ---
# Constants
# ---

EVA_A_METRICS = ["task_completion", "faithfulness", "agent_speech_fidelity"]
EVA_X_METRICS = ["turn_taking", "conversation_progression", "conciseness"]
STT_METRICS = ["transcription_accuracy_key_entities"]
DEBUG_METRICS = ["response_speed"]
ALL_ANALYSIS_METRICS = EVA_A_METRICS + EVA_X_METRICS + STT_METRICS

_ACRONYMS = {"tts", "stt", "wer", "llm", "db"}

DEFAULT_JUDGE_MODEL = "gpt-5.2"
DEFAULT_BATCH_SIZE = 25
DEFAULT_FAILURE_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Entity type taxonomy — canonical categories for key entity evaluation
# ---------------------------------------------------------------------------
# The LLM judge assigns free-form "type" labels to entities. We normalize
# these into a fixed, mutually-exclusive taxonomy so that cross-pipeline
# comparisons are apples-to-apples. Any raw type not in the map passes
# through unchanged (and should be added here when first observed).
#
# Taxonomy (12 categories):
#
#   identifier     Alphanumeric booking/confirmation codes, reference numbers,
#                  ticket numbers. High-entropy strings that must be captured
#                  character-perfectly for tool lookups to succeed.
#
#   name           Passenger first/last names, including spelled-out variants.
#                  Case-insensitive but exact-match required.
#
#   flight         Airline designator + flight number (e.g. "SK 703").
#                  Includes the carrier prefix — a single dropped letter
#                  (SK → S) makes the entire identifier invalid.
#
#   airport        Airport IATA codes (SEA, DCA) and airport names.
#                  Also covers route expressions (SEA → BOS) since the
#                  atomic unit is the airport code.
#
#   location       City names, places, origins, destinations — geographic
#                  references that are not IATA airport codes.
#
#   date           Calendar dates and years (e.g. "August 23", "2026").
#
#   time           Clock times, AM/PM, durations, time windows, timezone
#                  references (e.g. "2:00 PM Eastern", "within 24 hours").
#
#   amount         Monetary values, prices, fees, budget constraints
#                  (e.g. "$120", "under $100"). Includes currency references.
#
#   seat           Seat numbers (21C), seat preferences (aisle/window),
#                  cabin class, fare class — anything about where/how
#                  the passenger sits.
#
#   contact        Email addresses, phone numbers — passenger contact info.
#
#   voucher        Voucher codes, travel credit codes, compensation
#                  reference strings.
#
#   other          Anything that doesn't fit above: gate numbers, bag counts,
#                  loyalty tiers, selection numbers, organization names, etc.
# ---------------------------------------------------------------------------

ENTITY_TYPE_TAXONOMY: dict[str, str] = {
    # identifier
    "confirmation_code": "identifier",
    "confirmation_code_character": "identifier",
    "confirmation_code_partial": "identifier",
    "confirmation_code_part": "identifier",
    "confirmation_code_last_character": "identifier",
    "confirmation_code_nato": "identifier",
    "confirmation_code_nato_spelling": "identifier",
    "reference_number": "identifier",
    "booking_id": "identifier",
    "ticket_number": "identifier",
    "identifier": "identifier",
    # name
    "name": "name",
    "name_spelling": "name",
    "spelled_name": "name",
    "last_name": "name",
    "first_name": "name",
    "passenger_name": "name",
    "spelling": "name",
    "spelling_nato": "name",
    "spelling_letter": "name",
    # flight
    "flight_number": "flight",
    "airline": "flight",
    # airport
    "airport": "airport",
    "airport_code": "airport",
    "airport_name": "airport",
    "airport_route": "airport",
    "flight_route": "airport",
    "route": "airport",
    "location_code": "airport",
    # location
    "place": "location",
    "place_name": "location",
    "location": "location",
    "destination": "location",
    "origin": "location",
    "city": "location",
    # date
    "date": "date",
    "year": "date",
    "date_time_constraint": "date",
    # time
    "time": "time",
    "duration": "time",
    "time_constraint": "time",
    "time_period": "time",
    "time_duration": "time",
    "time_offset": "time",
    "timezone": "time",
    "time_zone": "time",
    "rule_time_window": "time",
    # amount
    "amount": "amount",
    "amount_or_quantity": "amount",
    "currency": "amount",
    # seat
    "seat": "seat",
    "seat_number": "seat",
    "seat_preference": "seat",
    "cabin": "seat",
    "cabin_class": "seat",
    "class_of_service": "seat",
    "service_class": "seat",
    "fare_class": "seat",
    # contact
    "email": "contact",
    "email_identifier": "contact",
    "phone": "contact",
    # voucher
    "voucher_code": "voucher",
    "voucher_number": "voucher",
    "travel_credit_code": "voucher",
    # other (explicit mapping so they don't pass through as raw names)
    "gate": "other",
    "number": "other",
    "quantity": "other",
    "count": "other",
    "option_number": "other",
    "checked_bags": "other",
    "passenger_count": "other",
    "connection_count_max": "other",
    "ordinal_number": "other",
    "standby_list_position": "other",
    "selection": "other",
    "organization": "other",
    "organization_or_app": "other",
    "service_request": "other",
    "status_tier": "other",
    "status_level": "other",
    "loyalty_status": "other",
}


def normalize_entity_type(raw_type: str) -> str:
    """Normalize a free-form entity type string to a canonical taxonomy category."""
    key = raw_type.strip().lower()
    return ENTITY_TYPE_TAXONOMY.get(key, "other")


# ---------------------------------------------------------------------------
# Display name mappings — single source of truth for human-friendly labels
# ---------------------------------------------------------------------------

DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "stt": {
        "elevenlabs": "ElevenLabs Scribe v2",
        "deepgram": "Deepgram Nova-3",
        "cartesia": "Cartesia Ink-Whisper",
        "nvidia-baseten": "Nvidia Parakeet CTC 1.1B",
        "openai": "GPT-4o Mini Transcribe",
    },
    "tts": {
        "elevenlabs": "ElevenLabs Turbo v2.5",
        "cartesia": "Cartesia Sonic 3",
        "openai": "OpenAI GPT-4o Mini TTS",
        "nvidia-baseten": "Nvidia Magpie TTS Multilingual 357M",
        "chatterbox": "Chatterbox Turbo",
        "kokoro": "HexGrad KokoroTTS",
    },
    "llm": {
        "gpt-5-mini": "GPT-5 Mini Minimal Reasoning",
        "gpt-5.2": "OpenAI GPT-5.2",
        "gpt-5.2-eval": "GPT-5.2 Low Reasoning",
        "gpt-4o": "OpenAI GPT-4o",
        "gpt-4o-mini": "OpenAI GPT-4o Mini",
        "gpt-realtime": "GPT Realtime",
        "gpt-realtime-mini": "GPT Realtime Mini",
        "gpt-realtime-1.5": "GPT Realtime 1.5",
        "gpt-oss": "OpenAI GPT-OSS",
        "gpt-oss-120b": "OpenAI GPT-OSS 120B",
        "gpt-oss-20b": "OpenAI GPT-OSS 20B",
        "sonnet-4-6": "Anthropic Claude Sonnet 4.6",
        "claude-sonnet-4-6": "Anthropic Claude Sonnet 4.6",
        "opus-4-6": "Anthropic Claude Opus 4.6",
        "claude-opus-4-6": "Anthropic Claude Opus 4.6",
        "gemini-3-pro": "Google Gemini 3 Pro",
        "gemini-3.1-pro-preview": "Google Gemini 3.1 Pro",
        "ultravox": "Ultravox",
        "qwen35-27B": "Qwen 3.5 27B",
        "gemini-live": "Google Gemini Live",
        "gemini-3.1-flash-live-preview": "Google Gemini 3.1 Flash Live",
        "gemini-3-flash-live-preview": "Google Gemini 3 Flash Live",
    },
}

# Component key to DISPLAY_NAMES section
_COMPONENT_TO_SECTION = {
    "llm_model": "llm",
    "stt_model": "stt",
    "tts_model": "tts",
}


def display_name(raw: str, component: str) -> str:
    """Map a raw model identifier to a human-friendly display name.

    Args:
        raw: Raw model key (e.g. "gpt-5-mini", "elevenlabs").
        component: One of "llm", "stt", "tts" (or "llm_model", "stt_model", "tts_model").
    """
    section = _COMPONENT_TO_SECTION.get(component, component)
    table = DISPLAY_NAMES.get(section, {})
    # Exact match, then case-insensitive
    if raw in table:
        return table[raw]
    for key, val in table.items():
        if key.lower() == raw.lower():
            return val
    # Fallback: title-case the raw name
    return raw.replace("-", " ").replace("_", " ").title()


def get_display_combo_label(combo_label: str) -> str:
    """Convert a raw combo label like 'gpt-5-mini / openai / openai' to display names.

    Parts equal to 'n/a' (S2S or Audio-Native pipelines) are omitted.
    Components are joined with ' + ' instead of ' / '.
    """
    parts = combo_label.split(" / ")
    if len(parts) == 3:
        llm, stt, tts = parts
        components = []
        if stt != "n/a":
            components.append(display_name(stt, "stt"))
        components.append(display_name(llm, "llm"))
        if tts != "n/a":
            components.append(display_name(tts, "tts"))
        return " + ".join(components)
    return combo_label


def _format_pipeline_config_block(combo_reports: dict) -> str:
    """Build a text block mapping each pipeline to its STT, LLM, and TTS components."""
    lines = ["### Pipeline Configurations", ""]
    for label, report in combo_reports.items():
        pc = report.get("pipeline_config", {})
        stt = display_name(pc.get("stt_model", "?"), "stt")
        llm = display_name(pc.get("llm_model", "?"), "llm")
        tts = display_name(pc.get("tts_model", "?"), "tts")
        dl = get_display_combo_label(label)
        lines.append(f"- **{dl}**: STT = {stt}, LLM = {llm}, TTS = {tts}")
    return "\n".join(lines)


# Metric descriptions + hints for the generic Pass 1 prompt.
# Hints suggest things to watch for without constraining the analysis.
_METRIC_DESCRIPTIONS: dict[str, str] = {
    "task_completion": (
        "This metric checks whether the agent's tool calls produced the correct final "
        "database state. A perfect score is 1.0 (exact match between expected and actual "
        "database states). Each sample includes: the user's goal, the full tool call "
        "sequence (with parameters and results), the database diff showing expected vs "
        "actual state, and a conversation excerpt.\n\n"
        "Use the tool call sequence to trace exactly where the failure started — was a "
        "tool never called? Called with wrong params? Called in wrong order?\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Missing tool calls vs tool calls with wrong parameters\n"
        "- Partial completions where some but not all required changes were made\n"
        "- Tools called in the wrong order causing cascading state errors\n"
        "- Agent confusion from tool descriptions leading to wrong tool choice"
    ),
    "faithfulness": (
        "This metric evaluates whether the agent remained faithful to tool results, "
        "policies, and user-provided information. Each sample includes per-dimension "
        "ratings across faithfulness dimensions. A perfect score is 1.0 (no violations).\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Fabricating tool parameters not grounded in the conversation\n"
        "- Misrepresenting what a tool returned to the user\n"
        "- Proceeding without disambiguating when information is unclear"
    ),
    "agent_speech_fidelity": (
        "This metric evaluates whether the TTS audio output matches the LLM-generated "
        "text the agent intended to say. Each sample includes per-turn fidelity ratings "
        "and explanations of discrepancies. A perfect score is 1.0.\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Specific entity types that TTS struggles with (names, codes, numbers)\n"
        "- Systematic word substitutions vs random mispronunciations\n"
        "- Whether errors cluster around particular formats or entities"
    ),
    "turn_taking": (
        "This metric evaluates conversation timing — whether the agent responds too "
        "early, too late, or on time. Each sample includes per-turn timing ratings "
        "(On-Time, Late, Early / Interrupting). A perfect score is 1.0.\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Late responses vs early interruptions\n"
        "- Whether timing issues cluster at specific conversation stages"
    ),
    "conversation_progression": (
        "This metric evaluates whether the conversation progresses logically toward "
        "goal completion. Each sample includes dimensional ratings and explanations. "
        "A perfect score is 1.0 (smooth, logical progression).\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Circular dialogue where the agent repeats the same questions\n"
        "- Stalls caused by the agent pursuing a wrong path\n"
        "- Failure to recover after a tool call error"
    ),
    "conciseness": (
        "This metric evaluates whether agent responses are appropriately brief for "
        "voice interactions. Each sample includes per-turn ratings and failure modes. "
        "A perfect score is 1.0 (all turns highly concise).\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Redundant confirmations restating what the user already knows\n"
        "- Over-explanation or excessive preambles before getting to the point\n"
        "- Listing too many options or details in a voice context"
    ),
    "transcription_accuracy_key_entities": (
        "This metric evaluates whether the STT system correctly transcribed key entities "
        "(names, confirmation codes, dates, amounts, phone numbers, etc.) that are critical "
        "for task success. Each sample lists failed entities **grouped by entity type**, showing "
        "the expected value and what STT actually produced. "
        "Score is the ratio of correctly transcribed entities (0-1).\n\n"
        "This metric reveals failures that coarse Word Error Rate (WER) may miss — a sentence "
        "can have low WER but still fail on the one entity that matters for task completion.\n\n"
        "**IMPORTANT: Each issue you create should focus on ONE specific error pattern "
        "(e.g., garbled confirmation codes, or misspelled names). When describing the issue "
        "and citing examples, only reference the specific entity type errors that match "
        "that pattern — do NOT list all errors in the sample. A sample may appear in multiple "
        "issues if it has errors across different entity types.**\n\n"
        "Common patterns to watch for (but not limited to):\n"
        "- Which entity types fail most (codes, names, numbers, dates)\n"
        "- Systematic patterns within each type (e.g., codes always lose leading characters)\n"
        "- Whether certain entity formats are harder (spelled-out vs spoken naturally)"
    ),
}


# ---
# Helper Functions
# ---


def get_available_models() -> list[str]:
    """Get model names from the EVA_MODEL_LIST environment variable."""
    try:
        model_list_json = os.environ.get("EVA_MODEL_LIST", "")
        if model_list_json:
            deployments = json.loads(model_list_json)
            names = [d["model_name"] for d in deployments if "model_name" in d]
            if names:
                return names
    except Exception as e:
        logger.info(f"Failed to parse EVA_MODEL_LIST: {e}")
    return [DEFAULT_JUDGE_MODEL]


def format_metric_name(name: str) -> str:
    """Format a metric name for display, preserving acronyms in upper case."""
    parts = name.split("_")
    return " ".join(p.upper() if p.lower() in _ACRONYMS else p.capitalize() for p in parts)


def load_run_config(run_dir: Path) -> dict:
    """Load config.json for a run, returning empty dict on failure."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.info(f"Failed to load config.json: {e}")
            pass
    return {}


def extract_pipeline_config(run_config: dict) -> dict:
    """Extract llm/stt/tts model names from run config.

    Handles multiple config key formats:
    - pipeline.s2s / pipeline.s2s_params (speech-to-speech, no STT/TTS)
    - pipeline.audio_llm / pipeline.audio_llm_params + pipeline.tts (audio-native, no STT)
    - pipeline.realtime_model / pipeline.llm_model / pipeline.llm (cascade or realtime)
    - pipeline.stt_model / pipeline.stt
    - pipeline.tts_model / pipeline.tts
    """
    pipeline = run_config.get("pipeline") or run_config.get("model") or {}

    # Speech-to-speech pipeline (single end-to-end model, no separate STT/TTS)
    s2s = pipeline.get("s2s") or ""
    if s2s and not pipeline.get("llm") and not pipeline.get("llm_model"):
        s2s_params = pipeline.get("s2s_params") or {}
        s2s_model = s2s_params.get("model") or s2s
        return {
            "llm_model": s2s_model,
            "stt_model": "n/a",
            "tts_model": "n/a",
        }

    # Audio-Native pipeline (audio LLM + TTS, no separate STT)
    audio_llm = pipeline.get("audio_llm") or ""
    if audio_llm:
        audio_llm_params = pipeline.get("audio_llm_params") or {}
        audio_llm_model = audio_llm_params.get("alias") or audio_llm_params.get("model") or audio_llm
        tts_params = pipeline.get("tts_params") or {}
        tts_model = tts_params.get("alias") or tts_params.get("model") or pipeline.get("tts") or "unknown"
        return {
            "llm_model": audio_llm_model,
            "stt_model": "n/a",
            "tts_model": tts_model,
        }

    return {
        "llm_model": pipeline.get("realtime_model") or pipeline.get("llm_model") or pipeline.get("llm") or "unknown",
        "stt_model": pipeline.get("stt_model") or pipeline.get("stt") or "unknown",
        "tts_model": pipeline.get("tts_model") or pipeline.get("tts") or "unknown",
    }


def get_combo_label(pipeline_config: dict) -> str:
    """Generate a human-readable label like 'gpt-5-mini / deepgram / cartesia'."""
    return " / ".join(
        [
            pipeline_config.get("llm_model", "unknown"),
            pipeline_config.get("stt_model", "unknown"),
            pipeline_config.get("tts_model", "unknown"),
        ]
    )


def get_record_directories(run_dir: Path) -> list[Path]:
    """Get all record directories in a run, sorted by record ID."""
    records_dir = run_dir / "records"
    if not records_dir.exists():
        return []
    record_dirs = [d for d in records_dir.iterdir() if d.is_dir()]
    return sorted(record_dirs, key=lambda d: d.name)


def get_valid_trial_dirs(record_dir: Path) -> list[tuple[str, Path]]:
    """Get trial directories, excluding failed_attempt dirs.

    Returns list of (label, path). For non-trial records, returns [("", record_dir)].
    """
    trial_dirs = []
    if record_dir.exists():
        trial_dirs = sorted(
            [
                d
                for d in record_dir.iterdir()
                if d.is_dir()
                and "failed_attempt" not in d.name
                and any(f for f in d.iterdir() if f.suffix in (".json", ".wav", ".jsonl"))
            ],
            key=lambda d: d.name,
        )

    if trial_dirs:
        return [(d.name, d) for d in trial_dirs]
    return [("", record_dir)]


def load_record_metrics(data_path: Path) -> RecordMetrics | None:
    """Load RecordMetrics from metrics.json."""
    metrics_path = data_path / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        return RecordMetrics(**data)
    except Exception as e:
        logger.warning(f"Failed to load metrics.json from {data_path}: {e}")
        return None


def _cache_key(combo_label: str) -> str:
    """Slug-ify a combo label for use as a directory name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", combo_label).strip("_").lower()


# ---
# Data Collection
# ---


def collect_combo_data(
    run_dirs: list[Path],
    max_samples: int | None = None,
) -> dict[str, dict]:
    """Collect all record data grouped by pipeline combo.

    If max_samples is set, limits each combo to that many records (debug mode).
    """
    combos: dict[str, dict] = {}

    for run_dir in run_dirs:
        run_config = load_run_config(run_dir)
        pipeline_config = extract_pipeline_config(run_config)
        combo_label = get_combo_label(pipeline_config)

        if combo_label not in combos:
            combos[combo_label] = {
                "pipeline_config": pipeline_config,
                "run_dirs": [],
                "records": [],
            }
        combos[combo_label]["run_dirs"].append(run_dir)

        for record_dir in get_record_directories(run_dir):
            if max_samples and len(combos[combo_label]["records"]) >= max_samples:
                break
            record_id = record_dir.name
            for trial_label, data_path in get_valid_trial_dirs(record_dir):
                if max_samples and len(combos[combo_label]["records"]) >= max_samples:
                    break
                metrics = load_record_metrics(data_path)
                if metrics is None:
                    continue
                combos[combo_label]["records"].append(
                    {
                        "record_id": record_id,
                        "trial": trial_label,
                        "data_path": data_path,
                        "metrics": metrics,
                        "combo_label": combo_label,
                    }
                )

    return combos


def _get_metric_score(metrics: RecordMetrics, metric_name: str) -> float | None:
    """Get the normalized score for a metric, or raw score as fallback."""
    ms = metrics.metrics.get(metric_name)
    if ms is None or ms.error:
        logger.info(
            "Metric %s not found or had error for record %s",
            metric_name,
            metrics.record_id,
        )
        return None
    return ms.normalized_score if ms.normalized_score is not None else ms.score


def _get_metric_details(metrics: RecordMetrics, metric_name: str) -> dict:
    """Get metric details dict."""
    ms = metrics.metrics.get(metric_name)
    if ms is None:
        return {}
    return ms.details or {}


def _load_tool_call_sequence(data_path: Path) -> str:
    """Extract the tool call sequence from audit_log.json as a compact summary.

    Returns a string like:
    1. get_reservation(confirmation_number="ABC123", last_name="Smith") → success
    2. search_rebooking_options(...) → 3 options returned
    3. rebook_flight(...) → error: fare class mismatch
    """
    audit_path = data_path / "audit_log.json"
    if not audit_path.exists():
        return ""
    try:
        with open(audit_path) as f:
            raw = json.load(f)
        # audit_log can be a dict with "transcript" key or a flat list
        entries = raw.get("transcript", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return ""
    except Exception as e:
        logger.warning("Failed to load tool call sequence from %s: %s", audit_path, e)
        return ""

    tool_calls = []
    pending_call: dict | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("message_type", "")
        if msg_type == "tool_call":
            value = entry.get("value", {})
            tool_name = value.get("tool", "unknown")
            params = value.get("parameters", {})
            param_parts = []
            for k, v in params.items():
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:97] + "..."
                param_parts.append(f'{k}="{v_str}"' if isinstance(v, str) else f"{k}={v_str}")
            param_str = ", ".join(param_parts)
            pending_call = {"tool": tool_name, "params": param_str}
        elif msg_type == "tool_response" and pending_call:
            value = entry.get("value", {})
            if isinstance(value, dict) and "response" in value:
                resp = value["response"]
            else:
                resp = value
            if isinstance(resp, str):
                result_summary = resp[:120]
            elif isinstance(resp, dict):
                status = resp.get("status", "")
                if "error" in str(status).lower():
                    error_msg = resp.get("message", resp.get("error", "unknown error"))
                    result_summary = f"error: {str(error_msg)[:120]}"
                else:
                    msg = resp.get("message", "")
                    count = resp.get("count")
                    if msg:
                        result_summary = f"{status}: {str(msg)[:100]}"
                    elif count is not None:
                        result_summary = f"{status} ({count} results)"
                    else:
                        result_summary = status or "ok"
            else:
                result_summary = str(resp)[:120]
            tool_calls.append(
                f"{len(tool_calls) + 1}. {pending_call['tool']}({pending_call['params']}) → {result_summary}"
            )
            pending_call = None

    return "\n".join(tool_calls) if tool_calls else ""


def _load_conversation_excerpt(data_path: Path, max_turns: int = 40) -> str:
    """Load a compact conversation excerpt from transcript.jsonl.

    Returns the conversation as:
    [USER] I need to cancel my flight...
    [AGENT] I'd be happy to help. Can you provide...
    """
    transcript_path = data_path / "transcript.jsonl"
    if not transcript_path.exists():
        return ""
    try:
        turns = []
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                role = entry.get("role", entry.get("type", "unknown")).upper()
                content = entry.get("content", "")
                if len(content) > 500:
                    content = content[:497] + "..."
                label = "[AGENT]" if role == "ASSISTANT" else f"[{role}]"
                turns.append(f"{label} {content}")
        if len(turns) > max_turns:
            kept_start = 5
            kept_end = max_turns - kept_start
            turns = turns[:kept_start] + [f"  ... ({len(turns) - max_turns} turns omitted) ..."] + turns[-kept_end:]
        return "\n".join(turns)
    except Exception as e:
        logger.info(f"Failed to load conversation excerpt: {e}")
        return ""


def _load_user_goal(data_path: Path) -> str:
    """Extract the high-level user goal from metrics.json context."""
    metrics_path = data_path / "metrics.json"
    if not metrics_path.exists():
        return ""
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        context = data.get("context", {})
        user_goal = context.get("user_goal", {})
        if isinstance(user_goal, dict):
            return user_goal.get("high_level_user_goal", str(user_goal)[:200])
        return str(user_goal)[:200]
    except Exception as e:
        logger.info(f"Failed to load user goal: {e}")
        return ""


def _load_common_context(data_path: Path) -> dict[str, str]:
    """Load user goal, tool call sequence, and conversation excerpt for a record."""
    return {
        "user_goal": _load_user_goal(data_path),
        "tool_calls": _load_tool_call_sequence(data_path),
        "conversation": _load_conversation_excerpt(data_path),
    }


def compute_combo_metric_scores(combo_data: dict) -> dict[str, float]:
    """Compute mean normalized scores per metric for a combo."""
    scores: dict[str, list[float]] = {}
    for rec in combo_data["records"]:
        metrics = rec["metrics"]
        for m in ALL_ANALYSIS_METRICS + DEBUG_METRICS:
            s = _get_metric_score(metrics, m)
            if s is not None:
                scores.setdefault(m, []).append(s)

    return {m: sum(vals) / len(vals) for m, vals in scores.items() if vals}


def compute_pass_at_1(combo_data: dict, metric_group: list[str], threshold: float = 1.0) -> float:
    """Compute pass@1: fraction of samples where ALL metrics in the group meet the threshold."""
    n_pass = 0
    n_total = 0
    for rec in combo_data["records"]:
        metrics = rec["metrics"]
        group_scores = [_get_metric_score(metrics, m) for m in metric_group]
        # Only count samples that have scores for all metrics in the group
        if any(s is None for s in group_scores):
            continue
        n_total += 1
        if all(s >= threshold for s in group_scores):
            n_pass += 1
    return n_pass / n_total if n_total > 0 else 0.0


def _compute_group_metric_means(samples: list[dict], metrics: list[str]) -> dict[str, float]:
    """Compute mean score per metric across a group of samples.

    Each sample is a dict with metric names as keys mapping to Optional[float] scores.
    """
    means = {}
    for m in metrics:
        vals = [s[m] for s in samples if s.get(m) is not None]
        means[m] = sum(vals) / len(vals) if vals else 0.0
    return means


def _compute_failure_rates(samples: list[dict], metrics: list[str], threshold: float) -> dict[str, float]:
    """Compute failure rate per metric: fraction of samples scoring below threshold."""
    rates = {}
    for m in metrics:
        failed = [s for s in samples if s.get(m) is not None and s[m] < threshold]
        total_with_metric = [s for s in samples if s.get(m) is not None]
        rates[m] = len(failed) / len(total_with_metric) if total_with_metric else 0.0
    return rates


def compute_conversation_length_stats(combo_data: dict) -> dict:
    """Compute per-record conversation length and correlate with metric scores.

    Returns stats like:
    - turn count distribution (min, max, mean, median)
    - failure rates bucketed by conversation length (short/medium/long)
    - per-metric score means by conversation length bucket
    """
    records_with_turns: list[dict] = []
    for rec in combo_data["records"]:
        data_path = rec["data_path"]
        result_path = data_path / "result.json"
        num_turns = 0
        if result_path.exists():
            try:
                with open(result_path) as f:
                    result_data = json.load(f)
                num_turns = result_data.get("num_turns", 0)
            except Exception as e:
                logger.info("Failed to load num_turns: %s", e)
                pass
        if num_turns == 0:
            transcript_path = data_path / "transcript.jsonl"
            if transcript_path.exists():
                try:
                    with open(transcript_path) as f:
                        num_turns = sum(1 for line in f if line.strip())
                except Exception as e:
                    logger.info("Failed to load num_turns: %s", e)
                    pass

        records_with_turns.append(
            {
                "record_id": rec["record_id"],
                "trial": rec["trial"],
                "num_turns": num_turns,
                "metrics": rec["metrics"],
            }
        )

    if not records_with_turns:
        return {"turn_distribution": {}, "by_length_bucket": {}}

    turn_counts = [r["num_turns"] for r in records_with_turns]
    sorted_turns = sorted(turn_counts)
    n = len(sorted_turns)

    if n >= 3:
        t1 = sorted_turns[n // 3]
        t2 = sorted_turns[2 * n // 3]
    else:
        t1, t2 = 10, 20

    bucket_labels = {"short": f"≤{t1} turns", "medium": f"{t1 + 1}-{t2} turns", "long": f">{t2} turns"}

    def _bucket(num_turns: int) -> str:
        if num_turns <= t1:
            return "short"
        if num_turns <= t2:
            return "medium"
        return "long"

    bucket_samples: dict[str, list[dict]] = {b: [] for b in ("short", "medium", "long")}
    for rec in records_with_turns:
        b = _bucket(rec["num_turns"])
        row: dict[str, Any] = {}
        for m in ALL_ANALYSIS_METRICS:
            row[m] = _get_metric_score(rec["metrics"], m)
        bucket_samples[b].append(row)

    by_bucket = {}
    for b in ("short", "medium", "long"):
        metric_means = _compute_group_metric_means(bucket_samples[b], ALL_ANALYSIS_METRICS)
        # Only include metrics that actually had values
        metric_means = {m: v for m, v in metric_means.items() if any(s.get(m) is not None for s in bucket_samples[b])}
        by_bucket[b] = {
            "label": bucket_labels[b],
            "count": len(bucket_samples[b]),
            "metric_means": metric_means,
        }

    return {
        "turn_distribution": {
            "min": min(turn_counts),
            "max": max(turn_counts),
            "mean": sum(turn_counts) / len(turn_counts),
            "median": sorted_turns[n // 2],
            "total_records": n,
        },
        "bucket_thresholds": {"short_max": t1, "medium_max": t2},
        "by_length_bucket": by_bucket,
    }


def _format_conversation_length_stats(stats: dict) -> str:
    """Format conversation length stats into a readable string for the LLM prompt."""
    if not stats or not stats.get("turn_distribution"):
        return "No conversation length data available."

    dist = stats["turn_distribution"]
    parts = [
        f"Turn count distribution: min={dist['min']}, max={dist['max']}, "
        f"mean={dist['mean']:.1f}, median={dist['median']} ({dist['total_records']} records)",
        "",
        "Metric scores by conversation length:",
    ]

    by_bucket = stats.get("by_length_bucket", {})
    for b in ("short", "medium", "long"):
        bucket = by_bucket.get(b, {})
        label = bucket.get("label", b)
        count = bucket.get("count", 0)
        metric_means = bucket.get("metric_means", {})
        if not metric_means:
            parts.append(f"  {label} ({count} records): no metric data")
            continue
        scores_str = ", ".join(f"{format_metric_name(m)}={v:.3f}" for m, v in sorted(metric_means.items()))
        parts.append(f"  {label} ({count} records): {scores_str}")

    return "\n".join(parts)


def build_per_sample_dataframe(combos: dict[str, dict]) -> list[dict]:
    """Build a flat list of per-sample rows for plotting.

    Each row has: combo_label, record_id, trial, num_turns, and all metric scores.
    Returns list of dicts (ready for pd.DataFrame).
    """
    rows = []
    for combo_label, combo_data in combos.items():
        for rec in combo_data["records"]:
            data_path = rec["data_path"]

            num_turns = 0
            result_path = data_path / "result.json"
            if result_path.exists():
                try:
                    with open(result_path) as f:
                        result_data = json.load(f)
                    num_turns = result_data.get("num_turns", 0)
                except Exception:
                    pass

            row = {
                "combo_label": combo_label,
                "record_id": rec["record_id"],
                "trial": rec["trial"],
                "num_turns": num_turns,
            }
            for m in ALL_ANALYSIS_METRICS:
                row[m] = _get_metric_score(rec["metrics"], m)

            rows.append(row)
    return rows


# ---
# Metric-Specific Context Extraction
# ---


def _flatten_diff(diff: dict, path: str = "") -> list[str]:
    """Flatten a nested DB diff into readable one-line-per-mismatch summaries.

    Turns deeply nested diff structures into lines like:
      reservations.ABC123.bookings[0].seat: expected "21C", actual null
      journeys.FL_SK402.segments[0].available_seats.main_cabin: expected 6, actual 7
    """
    lines: list[str] = []
    if not isinstance(diff, dict):
        return lines

    for table, table_diff in diff.get("tables_modified", {}).items():
        if not isinstance(table_diff, dict):
            continue
        for record_id, rec_diff in table_diff.get("records_modified", {}).items():
            prefix = f"{table}.{record_id}" if not path else f"{path}.{table}.{record_id}"
            lines.extend(_flatten_fields(rec_diff, prefix))
        for record in table_diff.get("records_added", []):
            rid = record if isinstance(record, str) else str(record)[:80]
            lines.append(f"{table}: unexpected record added: {rid}")
        for record in table_diff.get("records_removed", []):
            rid = record if isinstance(record, str) else str(record)[:80]
            lines.append(f"{table}: expected record missing: {rid}")

    return lines


def _flatten_fields(diff: dict, prefix: str) -> list[str]:
    """Recursively flatten field-level diffs."""
    lines: list[str] = []
    if not isinstance(diff, dict):
        return lines

    for fname, field_diff in diff.get("fields_modified", {}).items():
        field_path = f"{prefix}.{fname}"
        if not isinstance(field_diff, dict):
            continue
        diff_type = field_diff.get("type", "")
        if diff_type in ("value_mismatch", "type_mismatch"):
            expected = field_diff.get("expected")
            actual = field_diff.get("actual")
            lines.append(f"{field_path}: expected {json.dumps(expected)}, actual {json.dumps(actual)}")
        elif diff_type == "list_differences":
            for item in field_diff.get("differences", []):
                idx = item.get("index", "?")
                lines.extend(_flatten_fields(item.get("diff", {}), f"{field_path}[{idx}]"))
        elif diff_type == "list_length_mismatch":
            expected = field_diff.get("expected", [])
            actual = field_diff.get("actual", [])
            lines.append(f"{field_path}: expected {len(expected)} items, actual {len(actual)} items")
            if isinstance(expected, list) and isinstance(actual, list):
                for i, (e, a) in enumerate(zip(expected, actual)):
                    if e != a:
                        lines.append(f"  [{i}] expected: {json.dumps(e, default=str)[:200]}")
                        lines.append(f"  [{i}] actual:   {json.dumps(a, default=str)[:200]}")
        else:
            lines.extend(_flatten_fields(field_diff, field_path))

    for fa in diff.get("fields_added", []):
        lines.append(f"{prefix}.{fa}: unexpected field added")
    for fr in diff.get("fields_removed", []):
        lines.append(f"{prefix}.{fr}: expected field missing")

    return lines


def _extract_per_turn_issues(
    details: dict,
    ratings_key: str,
    explanations_key: str,
    fail_fn: Callable[[Any], bool],
    failure_modes_key: str | None = None,
) -> list[str]:
    """Extract per-turn failure lines from metric details.

    Generalizes the pattern of iterating per-turn ratings/explanations
    and collecting lines for turns that match the fail_fn predicate.

    Returns lines like: "- Turn 3 (rating=0): TTS mispronounced 'ABC123'"
    """
    per_turn_exp = details.get(explanations_key, {})
    per_turn_ratings = details.get(ratings_key, {})
    per_turn_modes = details.get(failure_modes_key, {}) if failure_modes_key else {}
    lines: list[str] = []
    for tid, explanation in per_turn_exp.items():
        rating = per_turn_ratings.get(tid, per_turn_ratings.get(str(tid), "N/A"))
        if fail_fn(rating):
            modes = per_turn_modes.get(tid, per_turn_modes.get(str(tid), []))
            mode_str = f" [{', '.join(modes)}]" if modes else ""
            lines.append(f"- Turn {tid} (rating={rating}){mode_str}: {explanation}")
    return lines


def _build_task_completion_context(details: dict, data_path: Path) -> str:
    """Build context string for task_completion failures.

    Includes: user goal, tool call sequence, DB diff, and conversation excerpt
    so the LLM can trace the causal chain of the failure.
    """
    parts = []
    ctx = _load_common_context(data_path)

    if ctx["user_goal"]:
        parts.append(f"User goal: {ctx['user_goal']}")
    if ctx["tool_calls"]:
        parts.append(f"Tool call sequence:\n{ctx['tool_calls']}")

    # DB diff flattened to readable lines
    if details.get("message"):
        parts.append(f"Result: {details['message']}")
    if details.get("diff"):
        flat_lines = _flatten_diff(details["diff"])
        if flat_lines:
            parts.append("Database diff (expected vs actual):\n" + "\n".join(flat_lines))
        else:
            parts.append(f"Database diff (raw):\n{json.dumps(details['diff'], indent=2)}")

    if ctx["conversation"]:
        parts.append(f"Conversation:\n{ctx['conversation']}")

    return "\n\n".join(parts) if parts else "No detail available"


def _build_faithfulness_context(details: dict, data_path: Path) -> str:
    """Build context for faithfulness failures — extract flagged dimensions + tool call evidence."""
    parts = []
    ctx = _load_common_context(data_path)

    if ctx["user_goal"]:
        parts.append(f"User goal: {ctx['user_goal']}")

    dim_parts = []
    explanation = details.get("explanation", {})
    if isinstance(explanation, dict):
        dimensions = explanation.get("dimensions", explanation)
        for dim_name, dim_data in dimensions.items():
            if isinstance(dim_data, dict):
                if dim_data.get("flagged") or dim_data.get("rating", 3) < 3:
                    evidence = dim_data.get("evidence", "")
                    rating = dim_data.get("rating", "N/A")
                    dim_parts.append(f"- {dim_name} (rating={rating}): {evidence}")
    if not dim_parts:
        exp_str = json.dumps(explanation) if isinstance(explanation, dict) else str(explanation)
        dim_parts.append(exp_str)
    parts.append("Flagged dimensions:\n" + "\n".join(dim_parts))

    if ctx["tool_calls"]:
        parts.append(f"Tool call sequence:\n{ctx['tool_calls']}")
    if ctx["conversation"]:
        parts.append(f"Conversation:\n{ctx['conversation']}")

    return "\n\n".join(parts)


def _build_agent_speech_fidelity_context(details: dict, data_path: Path) -> str:
    """Build context for agent_speech_fidelity failures."""
    # agent_speech_fidelity uses 0/1 scale: 0=low fidelity (fail), 1=high fidelity (pass)
    lines = _extract_per_turn_issues(
        details,
        ratings_key="per_turn_ratings",
        explanations_key="per_turn_explanations",
        fail_fn=lambda rating: rating == 0,
    )
    if not lines:
        lines = ["No per-turn fidelity details available"]
    return "TTS fidelity issues:\n" + "\n".join(lines)


def _build_turn_taking_context(details: dict, data_path: Path) -> str:
    """Build context for turn_taking failures."""

    def _is_timing_failure(rating: Any) -> bool:
        if isinstance(rating, str) and rating.lower() != "on-time":
            return True
        if isinstance(rating, (int, float)) and rating < 0:
            return True
        return False

    lines = _extract_per_turn_issues(
        details,
        ratings_key="per_turn_judge_timing_ratings",
        explanations_key="per_turn_judge_timing_explanations",
        fail_fn=_is_timing_failure,
    )
    if not lines:
        lines = ["No per-turn timing details available"]
    return "Timing issues:\n" + "\n".join(lines)


def compute_turn_taking_stats(combo_data: dict) -> dict:
    """Compute deterministic turn-taking statistics from metric details.

    Returns a pass1-shaped result with distribution tables instead of LLM issues.
    """
    _LABELS = ("On-Time", "Late", "Early / Interrupting", "Other / Indeterminate")
    overall_counts: dict[str, int] = dict.fromkeys(_LABELS, 0)
    per_sample: list[dict] = []
    samples_with_issues: list[str] = []

    for rec in combo_data.get("records", []):
        metrics = rec.get("metrics", {})
        tt = _get_metric_details(metrics, "turn_taking")
        if not tt:
            continue
        ratings = tt.get("per_turn_judge_timing_ratings", {})
        if not ratings:
            continue

        sample_id = rec["record_id"]
        if rec.get("trial"):
            sample_id = f"{rec['record_id']}/{rec['trial']}"

        sample_counts: dict[str, int] = dict.fromkeys(_LABELS, 0)
        n_turns = len(ratings)
        has_issue = False
        for _turn_id, label in ratings.items():
            norm_label = label if label in _LABELS else "Other / Indeterminate"
            sample_counts[norm_label] += 1
            overall_counts[norm_label] += 1
            if norm_label != "On-Time":
                has_issue = True

        if has_issue:
            samples_with_issues.append(sample_id)

        per_sample.append(
            {
                "sample_id": sample_id,
                "num_turns": n_turns,
                **{lbl: sample_counts[lbl] for lbl in _LABELS},
                "pct_on_time": sample_counts["On-Time"] / n_turns * 100 if n_turns else 0,
                "pct_late": sample_counts["Late"] / n_turns * 100 if n_turns else 0,
                "pct_early": sample_counts["Early / Interrupting"] / n_turns * 100 if n_turns else 0,
            }
        )

    total_turns = sum(overall_counts.values())
    pct = {lbl: (c / total_turns * 100 if total_turns else 0) for lbl, c in overall_counts.items()}

    avg_pct_on_time = sum(s["pct_on_time"] for s in per_sample) / len(per_sample) if per_sample else 0
    avg_pct_late = sum(s["pct_late"] for s in per_sample) / len(per_sample) if per_sample else 0
    avg_pct_early = sum(s["pct_early"] for s in per_sample) / len(per_sample) if per_sample else 0

    return {
        "metric_name": "turn_taking",
        "issues": [],
        "metric_summary": (
            f"{pct.get('On-Time', 0):.1f}% on-time, "
            f"{pct.get('Late', 0):.1f}% late, "
            f"{pct.get('Early / Interrupting', 0):.1f}% early across {total_turns} turns"
        ),
        "num_failures": len(samples_with_issues),
        "turn_taking_stats": {
            "overall_counts": overall_counts,
            "overall_pct": pct,
            "total_turns": total_turns,
            "total_samples": len(per_sample),
            "samples_with_issues": len(samples_with_issues),
            "affected_sample_ids": samples_with_issues,
            "avg_pct_on_time": round(avg_pct_on_time, 1),
            "avg_pct_late": round(avg_pct_late, 1),
            "avg_pct_early": round(avg_pct_early, 1),
            "per_sample": per_sample,
        },
    }


def _build_conversation_progression_context(details: dict, data_path: Path) -> str:
    """Build context for conversation_progression failures."""
    parts = []
    ctx = _load_common_context(data_path)

    if ctx["user_goal"]:
        parts.append(f"User goal: {ctx['user_goal']}")

    explanation = details.get("explanation", {})
    if isinstance(explanation, dict):
        dimensions = explanation.get("dimensions", explanation)
        dim_parts = []
        for dim, dim_data in dimensions.items():
            if isinstance(dim_data, dict) and "rating" in dim_data:
                rating = dim_data.get("rating", "N/A")
                flagged = dim_data.get("flagged", False)
                evidence = dim_data.get("evidence", dim_data.get("explanation", ""))
                flag_marker = " [FLAGGED]" if flagged else ""
                dim_parts.append(f"- {dim} (rating={rating}){flag_marker}: {evidence}")
        if dim_parts:
            parts.append("Progression dimensions:\n" + "\n".join(dim_parts))
    else:
        parts.append(f"Explanation: {explanation}")

    if ctx["tool_calls"]:
        parts.append(f"Tool call sequence:\n{ctx['tool_calls']}")
    if ctx["conversation"]:
        parts.append(f"Conversation:\n{ctx['conversation']}")

    return "\n\n".join(parts) if parts else "No detail available"


def _build_conciseness_context(details: dict, data_path: Path) -> str:
    """Build context for conciseness failures."""
    parts = []

    turn_parts = _extract_per_turn_issues(
        details,
        ratings_key="per_turn_ratings",
        explanations_key="per_turn_explanations",
        fail_fn=lambda rating: rating is not None and rating != 3 and rating != "N/A",
        failure_modes_key="per_turn_failure_modes",
    )
    if turn_parts:
        parts.append("Conciseness issues:\n" + "\n".join(turn_parts))
    else:
        parts.append("No per-turn conciseness details available")

    ctx = _load_common_context(data_path)
    if ctx["conversation"]:
        parts.append(f"Conversation:\n{ctx['conversation']}")

    return "\n\n".join(parts)


def _build_stt_wer_context(details: dict, data_path: Path) -> str:
    """Build context string for stt_wer failures."""
    parts = []
    wer = details.get("wer", "?")
    accuracy = details.get("accuracy", "?")
    parts.append(
        f"Overall WER: {wer:.3f}, Accuracy: {accuracy:.3f}" if isinstance(wer, (int, float)) else f"WER: {wer}"
    )

    subs = details.get("total_substitutions", 0)
    dels = details.get("total_deletions", 0)
    ins = details.get("total_insertions", 0)
    if subs or dels or ins:
        parts.append(f"Error breakdown: {subs} substitutions, {dels} deletions, {ins} insertions")

    per_turn_wer = details.get("per_turn_wer", {})
    per_turn_errors = details.get("per_turn_errors", {})
    bad_turns = [(tid, w) for tid, w in per_turn_wer.items() if isinstance(w, (int, float)) and w > 0]
    if bad_turns:
        bad_turns.sort(key=lambda x: x[1], reverse=True)
        turn_lines = []
        for tid, w in bad_turns[:8]:
            errs = per_turn_errors.get(str(tid), per_turn_errors.get(tid, {}))
            if errs:
                sub_list = errs.get("substitutions", [])
                del_list = errs.get("deletions", [])
                ins_list = errs.get("insertions", [])
                err_parts = []
                if sub_list:
                    sub_strs = [f"'{s['expected']}'→'{s['actual']}'" for s in sub_list[:3]]
                    err_parts.append(f"{len(sub_list)}S: {', '.join(sub_strs)}")
                if del_list:
                    err_parts.append(f"{len(del_list)}D: {', '.join(repr(d) for d in del_list[:3])}")
                if ins_list:
                    err_parts.append(f"{len(ins_list)}I: {', '.join(repr(i) for i in ins_list[:3])}")
                err_str = f"({'; '.join(err_parts)})" if err_parts else ""
            else:
                err_str = ""
            turn_lines.append(f"  Turn {tid}: WER={w:.3f} {err_str}")
        parts.append("Turns with errors:\n" + "\n".join(turn_lines))

    return "\n".join(parts) if parts else "No detail available"


def _build_transcription_accuracy_context(details: dict, data_path: Path) -> str:
    """Build context string for transcription_accuracy_key_entities failures.

    Groups errors by entity type so the LLM can isolate patterns per type.
    """
    parts = []
    num_evaluated = details.get("num_evaluated", 0)
    num_na = details.get("num_not_applicable", 0)
    parts.append(f"Turns evaluated: {num_evaluated}, Not applicable (no entities): {num_na}")

    per_turn_details = details.get("per_turn_entity_details", {})
    per_turn_ratings = details.get("per_turn_ratings", {})

    # Collect all failed entities grouped by normalized type
    by_type: dict[str, list[str]] = {}
    for tid, turn_data in per_turn_details.items():
        rating = per_turn_ratings.get(str(tid), per_turn_ratings.get(tid))
        if rating is not None and rating >= 1.0:
            continue
        if rating == -1.0:
            continue
        entities = turn_data.get("entities", [])
        for e in entities:
            if e.get("correct") or e.get("skipped"):
                continue
            etype = normalize_entity_type(e.get("type", "unknown"))
            expected = e.get("value", "?")
            transcribed = e.get("transcribed_value", "?")
            by_type.setdefault(etype, []).append(f"    turn {tid}: expected='{expected}' → got='{transcribed}'")

    if by_type:
        type_lines = []
        for etype, errors in sorted(by_type.items(), key=lambda x: -len(x[1])):
            type_lines.append(f"  [{etype}] ({len(errors)} errors)")
            for line in errors[:8]:
                type_lines.append(line)
            if len(errors) > 8:
                type_lines.append(f"    ... +{len(errors) - 8} more")
        parts.append("Failed entities by type:\n" + "\n".join(type_lines))
    else:
        parts.append("No individual entity failures found in details")

    return "\n".join(parts) if parts else "No detail available"


_CONTEXT_BUILDERS: dict[str, Callable] = {
    "task_completion": _build_task_completion_context,
    "faithfulness": _build_faithfulness_context,
    "agent_speech_fidelity": _build_agent_speech_fidelity_context,
    "turn_taking": _build_turn_taking_context,
    "conversation_progression": _build_conversation_progression_context,
    "conciseness": _build_conciseness_context,
    "stt_wer": _build_stt_wer_context,
    "transcription_accuracy_key_entities": _build_transcription_accuracy_context,
}


def build_metric_context_text(metric_name: str, details: dict, data_path: Path) -> str:
    """Build a context string for a failure sample based on its metric details."""
    builder = _CONTEXT_BUILDERS.get(metric_name)
    if builder:
        return builder(details, data_path)
    detail_str = json.dumps(details, indent=2, default=str)
    if len(detail_str) > 2000:
        detail_str = detail_str[:2000] + "\n... (truncated)"
    return detail_str


def collect_metric_failures(
    combo_data: dict,
    metric_name: str,
    threshold: float = 1.0,
) -> list[dict]:
    """Collect all records where metric score < threshold for a given metric."""
    failures = []
    for rec in combo_data["records"]:
        score = _get_metric_score(rec["metrics"], metric_name)
        if score is None:
            continue
        if score < threshold:
            details = _get_metric_details(rec["metrics"], metric_name)
            context_text = build_metric_context_text(metric_name, details, rec["data_path"])
            sample_id = rec["record_id"]
            if rec["trial"]:
                sample_id = f"{rec['record_id']}/{rec['trial']}"
            failures.append(
                {
                    "sample_id": sample_id,
                    "record_id": rec["record_id"],
                    "trial": rec["trial"],
                    "score": score,
                    "explanation": context_text,
                    "details": details,
                    "data_path": str(rec["data_path"]),
                }
            )
    return failures


# ---
# Caching
# ---


def _get_combo_cache_dir(run_dirs: list[Path], combo_label: str) -> Path:
    """Get cache directory for a combo within the first matching run dir."""
    if run_dirs:
        cache_root = run_dirs[0].parent / ".ai_review_cache"
    else:
        cache_root = Path("output") / ".ai_review_cache"
    return cache_root / _cache_key(combo_label)


def _get_cross_cache_dir(run_dirs: list[Path]) -> Path:
    """Get cache directory for cross-combo results."""
    if run_dirs:
        return run_dirs[0].parent / ".ai_review_cache" / "_cross"
    return Path("output") / ".ai_review_cache" / "_cross"


def _save_cache(cache_dir: Path, data: Any, *path_parts: str) -> None:
    """Save data to a cache JSON file."""
    file_path = cache_dir
    for part in path_parts:
        file_path = file_path / part
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_cache(cache_dir: Path, *path_parts: str) -> Any | None:
    """Load a cached JSON file, returning None if not found."""
    file_path = cache_dir
    for part in path_parts:
        file_path = file_path / part
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception:
        return None


def _compute_metadata_hash(run_dirs: list[Path], threshold: float, batch_size: int) -> str:
    """Compute a hash of pipeline parameters for cache validation."""
    key_parts = [str(d) for d in sorted(run_dirs)] + [str(threshold), str(batch_size)]
    return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]


def _is_cache_valid(
    cache_dir: Path,
    run_dirs: list[Path],
    threshold: float,
    batch_size: int,
    judge_model: str,
) -> bool:
    """Check if pipeline_metadata.json matches current config."""
    meta = _load_cache(cache_dir, "metadata.json")
    if meta is None:
        return False
    expected_hash = _compute_metadata_hash(run_dirs, threshold, batch_size)
    return meta.get("hash") == expected_hash and meta.get("judge_model") == judge_model


def _save_metadata(
    cache_dir: Path,
    run_dirs: list[Path],
    threshold: float,
    batch_size: int,
    judge_model: str,
) -> None:
    """Save pipeline metadata for cache validation."""
    _save_cache(
        cache_dir,
        {
            "hash": _compute_metadata_hash(run_dirs, threshold, batch_size),
            "judge_model": judge_model,
            "threshold": threshold,
            "batch_size": batch_size,
            "run_dirs": [str(d) for d in run_dirs],
        },
        "metadata.json",
    )


# ---
# Cached LLM Call Helper
# ---


async def _cached_llm_call(
    llm_client: LLMClient,
    messages: list[dict],
    cache_dir: Path | None = None,
    cache_parts: tuple[str, ...] = (),
    force_rerun: bool = False,
    label: str = "",
) -> tuple[dict | None, bool]:
    """Check cache then call LLM, returning (parsed_result, was_cached).

    Does NOT save cache — callers save after their own post-processing.
    """
    if cache_dir and not force_rerun and cache_parts:
        cached = _load_cache(cache_dir, *cache_parts)
        if cached is not None:
            logger.info(f"Cache hit for {label}")
            return cached, True

    try:
        response_text = await llm_client.generate_text(messages, response_format={"type": "json_object"})
        parsed = extract_and_load_json(response_text)
    except Exception as e:
        logger.warning(f"LLM call failed for {label}: {e}")
        parsed = None

    return parsed, False


# ---
# Pass 1: Per-Metric Summarization (single LLM call per metric)
# ---


async def run_pass1_per_metric(
    failures: list[dict],
    metric_name: str,
    combo_label: str,
    total_samples: int,
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
) -> dict:
    """Pass 1: Summarize all failures for one metric in a single LLM call."""
    cached = None
    if cache_dir and not force_rerun:
        cached = _load_cache(cache_dir, "pass1", f"{metric_name}.json")
        if cached is not None:
            logger.info(f"Pass 1 cache hit for {metric_name} in {combo_label}")
            return cached

    if not failures:
        result = {
            "metric_name": metric_name,
            "issues": [],
            "metric_summary": f"No failures found for {metric_name}.",
            "num_failures": 0,
        }
        if cache_dir:
            _save_cache(cache_dir, result, "pass1", f"{metric_name}.json")
        return result

    issue_format = prompt_manager.get_prompt("ai_review.issue_format_shared")

    all_issues_parts = []
    for sample in failures:
        all_issues_parts.append(
            f"### Sample: {sample['sample_id']} (score={sample['score']:.3f})\n{sample['explanation']}\n"
        )
    all_issues_text = "\n".join(all_issues_parts)

    prompt = prompt_manager.get_prompt(
        "ai_review.pass1.generic",
        metric_name=format_metric_name(metric_name),
        metric_description=_METRIC_DESCRIPTIONS.get(metric_name, ""),
        batch_size=len(failures),
        batch_issues=all_issues_text,
        issue_format=issue_format,
    )

    messages = [{"role": "user", "content": prompt}]
    parsed, _was_cached = await _cached_llm_call(llm_client, messages, label=f"Pass 1 {metric_name}")

    # Post-hoc validation: verify sample IDs and reconcile counts
    issues = parsed.get("issues", []) if parsed else []
    if issues:
        valid_ids = {s["sample_id"] for s in failures}
        for iss in issues:
            raw_samples = iss.get("affected_samples", [])
            verified = [sid for sid in raw_samples if sid in valid_ids]
            hallucinated = [sid for sid in raw_samples if sid not in valid_ids]
            if hallucinated:
                logger.warning(
                    "Pass 1 %s issue '%s': removed %d hallucinated sample IDs (kept %d/%d): %s",
                    metric_name,
                    iss.get("title", "?"),
                    len(hallucinated),
                    len(verified),
                    len(raw_samples),
                    hallucinated,
                )
                iss["hallucinated_sample_ids"] = hallucinated
            iss["affected_samples"] = verified
            llm_count = iss.get("sample_count", 0)
            try:
                llm_count = int(llm_count)
            except (ValueError, TypeError):
                llm_count = 0
            actual_count = len(verified)
            if llm_count != actual_count:
                logger.info(
                    "Pass 1 %s issue '%s': LLM said sample_count=%s, actual=%d — using actual",
                    metric_name,
                    iss.get("title", "?"),
                    iss.get("sample_count"),
                    actual_count,
                )
            iss["sample_count"] = actual_count

    result = {
        "metric_name": metric_name,
        "issues": issues,
        "metric_summary": parsed.get("batch_summary", "") if parsed else "LLM call failed.",
        "num_failures": len(failures),
    }

    if cache_dir and parsed is not None:
        _save_cache(cache_dir, result, "pass1", f"{metric_name}.json")

    return result


# ---
# Pass 2: Error Category Assignment
# ---

# Metrics whose details already contain structured error categories
_DETERMINISTIC_CATEGORY_METRICS = {
    "faithfulness",
    "conversation_progression",
    "conciseness",
    "turn_taking",
    "stt_wer",
    "transcription_accuracy_key_entities",
}


def _extract_flagged_dimensions(details: dict) -> list[str]:
    """Extract flagged dimension names from metric details with a dimensions dict.

    Works for both faithfulness and conversation_progression, which share the
    same details.explanation.dimensions structure.
    """
    explanation = details.get("explanation", {})
    dimensions = explanation.get("dimensions", {}) if isinstance(explanation, dict) else {}
    flagged = [dim for dim, data in dimensions.items() if isinstance(data, dict) and data.get("flagged")]
    return flagged or ["unknown"]


def _extract_conciseness_categories(details: dict) -> list[str]:
    """Extract dominant failure modes from conciseness per_turn_failure_modes."""
    per_turn_modes = details.get("per_turn_failure_modes", {})
    all_modes: list[str] = []
    for _tid, modes in per_turn_modes.items():
        if isinstance(modes, list):
            all_modes.extend(modes)
    if not all_modes:
        return ["unknown"]
    counter = Counter(all_modes)
    return [mode for mode, _ in counter.most_common()]


def _extract_turn_taking_categories(details: dict) -> list[str]:
    """Extract timing categories from turn_taking per_turn_judge_timing_ratings."""
    ratings = details.get("per_turn_judge_timing_ratings", {})
    _LABEL_MAP = {
        "Late": "late_response",
        "Early / Interrupting": "early_interruption",
        "Other / Indeterminate": "other",
    }
    cats: list[str] = []
    for _tid, label in ratings.items():
        mapped = _LABEL_MAP.get(label)
        if mapped:
            cats.append(mapped)
    if not cats:
        return ["unknown"]
    counter = Counter(cats)
    return [cat for cat, _ in counter.most_common()]


def _extract_stt_wer_categories(details: dict) -> list[str]:
    """Extract dominant error types from stt_wer details."""
    counts = [
        ("word_substitution", details.get("total_substitutions", 0)),
        ("word_deletion", details.get("total_deletions", 0)),
        ("word_insertion", details.get("total_insertions", 0)),
    ]
    cats = [name for name, count in sorted(counts, key=lambda x: x[1], reverse=True) if count > 0]
    return cats or ["unknown"]


def _extract_transcription_accuracy_categories(details: dict) -> list[str]:
    """Extract failed entity types from transcription_accuracy_key_entities details."""
    per_turn_details = details.get("per_turn_entity_details", {})
    failed_types: list[str] = []
    for _tid, turn_data in per_turn_details.items():
        for entity in turn_data.get("entities", []):
            if not entity.get("correct") and not entity.get("skipped"):
                etype = normalize_entity_type(entity.get("type", "unknown"))
                failed_types.append(etype)
    if not failed_types:
        return ["unknown"]
    counter = Counter(failed_types)
    return [etype for etype, _ in counter.most_common()]


_DETERMINISTIC_EXTRACTORS: dict[str, Callable] = {
    "faithfulness": _extract_flagged_dimensions,
    "conversation_progression": _extract_flagged_dimensions,
    "conciseness": _extract_conciseness_categories,
    "turn_taking": _extract_turn_taking_categories,
    "stt_wer": _extract_stt_wer_categories,
    "transcription_accuracy_key_entities": _extract_transcription_accuracy_categories,
}


def _run_pass2_deterministic(
    failures: list[dict],
    metric_name: str,
) -> list[dict]:
    """Extract error categories deterministically from metric details (no LLM)."""
    extractor = _DETERMINISTIC_EXTRACTORS[metric_name]
    results: list[dict] = []
    for f in failures:
        categories = extractor(f.get("details", {}))
        results.append(
            {
                "sample_id": f["sample_id"],
                "trial": f.get("trial", ""),
                "metric_name": metric_name,
                "score": f["score"],
                "categories": categories,
                "confidence": "high",
                "reasoning": f"Extracted from metric details: {', '.join(categories)}",
            }
        )
    return results


async def _run_pass2_llm(
    failures: list[dict],
    metric_name: str,
    prompt_manager: PromptManager,
    llm_client: LLMClient,
) -> list[dict]:
    """Categorize failures via a single batched LLM call."""
    category_list = prompt_manager.get_prompt(f"ai_review.pass2.categories.{metric_name}")

    samples_text_parts = []
    for f in failures:
        samples_text_parts.append(f"### Sample: {f['sample_id']} | Score: {f['score']:.3f}\n{f['explanation']}\n")
    all_samples_text = "\n".join(samples_text_parts)

    prompt = prompt_manager.get_prompt(
        "ai_review.pass2.batch_categorizer",
        metric=format_metric_name(metric_name),
        num_samples=len(failures),
        samples=all_samples_text,
        category_list=category_list,
    )

    messages = [{"role": "user", "content": prompt}]
    parsed, _was_cached = await _cached_llm_call(llm_client, messages, label=f"Pass 2 {metric_name}")

    results: list[dict] = []
    if parsed and isinstance(parsed.get("assignments"), list):
        failure_lookup = {f["sample_id"]: f for f in failures}
        for assignment in parsed["assignments"]:
            sid = assignment.get("sample_id", "")
            f_data = failure_lookup.get(sid, {})
            cats = assignment.get("categories", ["unknown"])
            if isinstance(cats, str):
                cats = [cats]
            results.append(
                {
                    "sample_id": sid,
                    "trial": f_data.get("trial", ""),
                    "metric_name": metric_name,
                    "score": f_data.get("score", 0.0),
                    "categories": cats,
                    "confidence": assignment.get("confidence", "low"),
                    "reasoning": assignment.get("reasoning", ""),
                }
            )
    else:
        for f in failures:
            results.append(
                {
                    "sample_id": f["sample_id"],
                    "trial": f.get("trial", ""),
                    "metric_name": metric_name,
                    "score": f["score"],
                    "categories": ["llm_error"],
                    "confidence": "low",
                    "reasoning": "Batch categorization failed",
                }
            )
    return results


async def run_pass2_error_categories(
    failures: list[dict],
    metric_name: str,
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
) -> list[dict]:
    """Pass 2: Categorize failures — deterministic for metrics with structured details, LLM for others."""
    if cache_dir and not force_rerun:
        cached = _load_cache(cache_dir, "pass2", f"{metric_name}.json")
        if cached is not None:
            logger.info(f"Pass 2 cache hit for {metric_name}")
            return cached

    if not failures:
        result: list[dict] = []
        if cache_dir:
            _save_cache(cache_dir, result, "pass2", f"{metric_name}.json")
        return result

    if metric_name in _DETERMINISTIC_CATEGORY_METRICS:
        logger.info(f"Pass 2 deterministic extraction for {metric_name} ({len(failures)} failures)")
        results = _run_pass2_deterministic(failures, metric_name)
    else:
        logger.info(f"Pass 2 LLM categorization for {metric_name} ({len(failures)} failures)")
        results = await _run_pass2_llm(failures, metric_name, prompt_manager, llm_client)

    if cache_dir:
        _save_cache(cache_dir, results, "pass2", f"{metric_name}.json")

    return results


# ---
# Pass 3: Cross-Metric Synthesis
# ---


def _format_category_distribution(pass2_results: dict[str, list[dict]]) -> str:
    """Format error category counts across all metrics for the synthesis prompt."""
    parts = []
    for metric_name, assignments in pass2_results.items():
        if not assignments:
            continue
        counter: Counter = Counter()
        for a in assignments:
            for cat in a.get("categories", []):
                counter[cat] += 1
        lines = [f"  - {cat}: {count}" for cat, count in counter.most_common()]
        parts.append(f"**{format_metric_name(metric_name)}:**\n" + "\n".join(lines))
    return "\n\n".join(parts) if parts else "No error categories assigned."


async def run_pass3_synthesis(
    pass1_results: dict[str, dict],
    pass2_results: dict[str, list[dict]],
    combo_label: str,
    total_samples: int,
    eva_a_mean: float,
    eva_x_mean: float,
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
    pipeline_config: dict | None = None,
) -> dict:
    """Pass 3: Cross-metric synthesis for one combo."""
    per_metric_parts = []
    for m in ALL_ANALYSIS_METRICS:
        p1 = pass1_results.get(m, {})
        issues = p1.get("issues", [])
        n_failures = p1.get("num_failures", 0)
        if not issues:
            continue
        lines = [f"### {format_metric_name(m)} ({n_failures} failures)"]
        for i, iss in enumerate(issues, 1):
            title = iss.get("title", "Untitled")
            desc = iss.get("description", "")
            samples = iss.get("affected_samples", [])
            count = iss.get("sample_count", len(samples))
            causal = iss.get("causal_chain", "")
            sample_preview = ", ".join(str(s) for s in samples[:5])
            if len(samples) > 5:
                sample_preview += f", ... (+{len(samples) - 5} more)"
            lines.append(f"- **Issue {i}: {title}** ({count} samples: {sample_preview})")
            if desc:
                lines.append(f"  Description: {desc}")
            if causal:
                lines.append(f"  Causal chain: {causal}")
        per_metric_parts.append("\n".join(lines))

    category_distribution = _format_category_distribution(pass2_results)

    pc = pipeline_config or {}
    pc_block = _format_pipeline_config_block({combo_label: {"pipeline_config": pc}}) if pc else ""
    metric_reference = prompt_manager.get_prompt("ai_review.metric_reference", pipeline_config=pc_block)
    prompt = prompt_manager.get_prompt(
        "ai_review.pass3.cross_metric_synthesis",
        combo_label=get_display_combo_label(combo_label),
        metric_reference=metric_reference,
        per_metric_issues="\n\n".join(per_metric_parts) if per_metric_parts else "No issues found.",
        category_distribution=category_distribution,
        total_samples=total_samples,
        eva_a_mean=eva_a_mean,
        eva_x_mean=eva_x_mean,
    )

    messages = [{"role": "user", "content": prompt}]
    parsed, was_cached = await _cached_llm_call(
        llm_client,
        messages,
        cache_dir=cache_dir,
        cache_parts=("pass3_synthesis.json",),
        force_rerun=force_rerun,
        label=f"Pass 3 {combo_label}",
    )

    if was_cached:
        return parsed

    result = parsed or {
        "systemic_issues": [],
        "overall_assessment": ["Analysis failed (LLM call error — likely rate limited)."],
    }

    if cache_dir:
        _save_cache(cache_dir, result, "pass3_synthesis.json")

    return result


# ---
# Error Propagation Analysis
# ---


def compute_error_propagation_stats(
    combo_data: dict,
    threshold: float = 1.0,
) -> dict:
    """Compute quantitative error propagation stats for a combo.

    Splits samples into entity-error vs no-entity-error groups and compares
    downstream metric means. Also computes co-failure rates and per-entity-type breakdowns.
    """
    target_metric = "transcription_accuracy_key_entities"
    downstream_metrics = [
        "task_completion",
        "faithfulness",
        "conversation_progression",
        "conciseness",
        "turn_taking",
        "agent_speech_fidelity",
    ]

    entity_error_samples: list[dict] = []
    clean_samples: list[dict] = []

    for rec in combo_data["records"]:
        entity_score = _get_metric_score(rec["metrics"], target_metric)
        if entity_score is None:
            continue

        row = {"record_id": rec["record_id"], "trial": rec["trial"]}
        row[target_metric] = entity_score
        for m in downstream_metrics:
            row[m] = _get_metric_score(rec["metrics"], m)

        row["_entity_details"] = _get_metric_details(rec["metrics"], target_metric)
        row["_data_path"] = rec["data_path"]
        row["_metrics"] = rec["metrics"]

        if entity_score < threshold:
            entity_error_samples.append(row)
        else:
            clean_samples.append(row)

    if not entity_error_samples and not clean_samples:
        return {"available": False}

    error_means = _compute_group_metric_means(entity_error_samples, downstream_metrics)
    clean_means = _compute_group_metric_means(clean_samples, downstream_metrics)
    cofailure_rates = _compute_failure_rates(entity_error_samples, downstream_metrics, threshold)
    baseline_rates = _compute_failure_rates(clean_samples, downstream_metrics, threshold)

    # Per entity type: group failed entity types and compute downstream means
    entity_type_impact: dict[str, dict] = {}
    for sample in entity_error_samples:
        details = sample.get("_entity_details", {})
        per_turn = details.get("per_turn_entity_details", {})
        failed_types: set[str] = set()
        for _tid, turn_data in per_turn.items():
            for entity in turn_data.get("entities", []):
                if not entity.get("correct") and not entity.get("skipped"):
                    failed_types.add(normalize_entity_type(entity.get("type", "unknown")))
        for etype in failed_types:
            if etype not in entity_type_impact:
                entity_type_impact[etype] = {"samples": [], "count": 0}
            entity_type_impact[etype]["samples"].append(sample)
            entity_type_impact[etype]["count"] += 1

    for _etype, data in entity_type_impact.items():
        data["downstream_means"] = _compute_group_metric_means(data["samples"], downstream_metrics)
        data["cofailure_rates"] = _compute_failure_rates(data["samples"], downstream_metrics, threshold)
        data["sample_count"] = len(data["samples"])
        del data["samples"]

    # Collect co-failure sample contexts (entity error + task_completion failure)
    cofailure_contexts: list[dict] = []
    for sample in entity_error_samples:
        tc_score = sample.get("task_completion")
        if tc_score is not None and tc_score < threshold:
            sample_id = sample["record_id"]
            if sample["trial"]:
                sample_id = f"{sample['record_id']}/{sample['trial']}"

            entity_ctx = _build_transcription_accuracy_context(
                sample.get("_entity_details", {}), sample.get("_data_path", "")
            )
            tc_details = _get_metric_details(sample["_metrics"], "task_completion")
            tc_ctx = _build_task_completion_context(tc_details, sample.get("_data_path", ""))

            cofailure_contexts.append(
                {
                    "sample_id": sample_id,
                    "entity_score": sample[target_metric],
                    "task_completion_score": tc_score,
                    "entity_context": entity_ctx[:800],
                    "task_context": tc_ctx[:800],
                }
            )

    return {
        "available": True,
        "n_entity_errors": len(entity_error_samples),
        "n_clean": len(clean_samples),
        "error_means": error_means,
        "clean_means": clean_means,
        "cofailure_rates": cofailure_rates,
        "baseline_rates": baseline_rates,
        "entity_type_impact": {
            k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in entity_type_impact.items()
        },
        "cofailure_contexts": cofailure_contexts[:15],
    }


async def run_error_propagation_analysis(
    combo_label: str,
    propagation_stats: dict,
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
) -> dict:
    """Run LLM analysis on error propagation patterns for a combo."""
    if not propagation_stats.get("available") or not propagation_stats.get("cofailure_contexts"):
        result = {"causal_chains": [], "overall_finding": "Insufficient co-failure data for analysis."}
        if cache_dir:
            _save_cache(cache_dir, result, "error_propagation.json")
        return result

    stats = propagation_stats
    quant_lines = [
        f"Samples with entity errors: {stats['n_entity_errors']}",
        f"Samples without entity errors: {stats['n_clean']}",
        "",
        "Mean metric scores — entity errors vs clean:",
    ]
    for m in stats["error_means"]:
        err = stats["error_means"].get(m, 0)
        clean = stats["clean_means"].get(m, 0)
        delta = clean - err
        quant_lines.append(f"  {format_metric_name(m)}: {err:.3f} (errors) vs {clean:.3f} (clean) — delta={delta:+.3f}")

    quant_lines.append("")
    quant_lines.append("Co-failure rates (% of entity-error samples that also fail each metric):")
    for m, rate in stats["cofailure_rates"].items():
        baseline = stats["baseline_rates"].get(m, 0)
        quant_lines.append(f"  {format_metric_name(m)}: {rate:.0%} (vs {baseline:.0%} baseline)")

    if stats.get("entity_type_impact"):
        quant_lines.append("")
        quant_lines.append("Impact by entity type:")
        for etype, data in sorted(stats["entity_type_impact"].items(), key=lambda x: x[1]["count"], reverse=True):
            tc_rate = data.get("cofailure_rates", {}).get("task_completion", 0)
            quant_lines.append(f"  {etype}: {data['count']} samples, task_completion co-failure={tc_rate:.0%}")

    sample_parts = []
    for ctx in stats["cofailure_contexts"][:10]:
        sample_parts.append(
            f"### Sample: {ctx['sample_id']} (entity={ctx['entity_score']:.3f}, task_completion={ctx['task_completion_score']:.3f})\n"
            f"**Entity errors:**\n{ctx['entity_context']}\n\n"
            f"**Task completion context:**\n{ctx['task_context']}\n"
        )

    prompt = prompt_manager.get_prompt(
        "ai_review.error_propagation.entity_error_analysis",
        quantitative_summary="\n".join(quant_lines),
        cofailure_samples="\n".join(sample_parts) if sample_parts else "No co-failure samples available.",
    )

    messages = [{"role": "user", "content": prompt}]
    parsed, was_cached = await _cached_llm_call(
        llm_client,
        messages,
        cache_dir=cache_dir,
        cache_parts=("error_propagation.json",),
        force_rerun=force_rerun,
        label=f"Error propagation {combo_label}",
    )

    if was_cached:
        return parsed

    result = parsed or {
        "causal_chains": [],
        "overall_finding": "LLM analysis failed.",
    }

    if cache_dir and parsed is not None:
        _save_cache(cache_dir, result, "error_propagation.json")

    return result


# ---
# Model-Level Reports
# ---


def _format_combo_profile(
    label: str,
    report: dict,
    include_issues: bool = True,
    include_categories: bool = True,
    include_convo_stats: bool = False,
    max_issues_per_metric: int = 3,
) -> str:
    """Build a text profile block for a combo, used across report generators."""
    ms = report.get("metric_scores", {})
    score_line = ", ".join(f"{format_metric_name(m)}={ms.get(m, 0):.3f}" for m in ALL_ANALYSIS_METRICS if m in ms)

    display_label = get_display_combo_label(label)
    block = (
        f"### {display_label}\n"
        f"EVA-A={report.get('eva_a_mean', 0):.3f}, EVA-X={report.get('eva_x_mean', 0):.3f}\n"
        f"Metric scores: {score_line}\n"
    )

    if include_issues:
        issue_lines = []
        pass1 = report.get("pass1", {})
        for m in ALL_ANALYSIS_METRICS:
            p1 = pass1.get(m, {})
            issues = p1.get("issues", [])
            if issues:
                top_titles = [iss.get("title", "?") for iss in issues[:max_issues_per_metric]]
                n_fail = p1.get("num_failures", 0)
                issue_lines.append(f"  {format_metric_name(m)} ({n_fail} failures): {'; '.join(top_titles)}")
        if issue_lines:
            block += "\nTop issues:\n" + "\n".join(issue_lines) + "\n"

    if include_categories:
        cat_summary = _format_category_distribution(report.get("pass2", {}))
        if cat_summary and cat_summary != "No error categories assigned.":
            block += f"\nError categories:\n{cat_summary}\n"

    if include_convo_stats:
        convo_stats = _format_conversation_length_stats(report.get("convo_length_stats", {}))
        if convo_stats:
            block += f"\nConversation length analysis:\n{convo_stats}\n"

    return block


async def generate_combo_report(
    combo_label: str,
    combo_data: dict,
    run_dirs: list[Path],
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_FAILURE_THRESHOLD,
    force_rerun: set[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Run Passes 1-3 + model report for a single combo. Returns ComboReport as dict."""
    force_rerun = force_rerun or set()
    cache_dir = _get_combo_cache_dir(run_dirs, combo_label)

    metric_scores = compute_combo_metric_scores(combo_data)
    total_samples = len(combo_data["records"])
    convo_length_stats = compute_conversation_length_stats(combo_data)

    eva_a_scores = [metric_scores.get(m, 0.0) for m in EVA_A_METRICS if m in metric_scores]
    eva_x_scores = [metric_scores.get(m, 0.0) for m in EVA_X_METRICS if m in metric_scores]
    eva_a_mean = sum(eva_a_scores) / len(eva_a_scores) if eva_a_scores else 0.0
    eva_x_mean = sum(eva_x_scores) / len(eva_x_scores) if eva_x_scores else 0.0
    eva_a_pass1 = compute_pass_at_1(combo_data, EVA_A_METRICS)
    eva_x_pass1 = compute_pass_at_1(combo_data, EVA_X_METRICS)

    failures_per_metric = {m: collect_metric_failures(combo_data, m, threshold) for m in ALL_ANALYSIS_METRICS}

    if progress_callback:
        progress_callback(0.1, f"Pass 1 — summarizing all metrics for {get_display_combo_label(combo_label)}...")

    # Turn-taking uses deterministic stats, not LLM
    llm_metrics = [m for m in ALL_ANALYSIS_METRICS if m != "turn_taking"]
    pass1_tasks = {
        m: run_pass1_per_metric(
            failures=failures_per_metric[m],
            metric_name=m,
            combo_label=combo_label,
            total_samples=total_samples,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            cache_dir=cache_dir,
            force_rerun="pass1" in force_rerun,
        )
        for m in llm_metrics
    }
    pass1_values = await asyncio.gather(*pass1_tasks.values(), return_exceptions=True)
    pass1_results: dict[str, dict] = {}
    for m, result in zip(pass1_tasks.keys(), pass1_values):
        if isinstance(result, Exception):
            logger.warning(f"Pass 1 failed for {m}: {result}")
            pass1_results[m] = {"metric_name": m, "issues": [], "metric_summary": f"Error: {result}", "num_failures": 0}
        else:
            pass1_results[m] = result

    # Deterministic turn-taking analysis
    tt_stats = compute_turn_taking_stats(combo_data)
    if cache_dir:
        _save_cache(cache_dir, tt_stats, "pass1", "turn_taking.json")
    pass1_results["turn_taking"] = tt_stats

    if progress_callback:
        progress_callback(0.3, f"Pass 2 — categorizing errors for {get_display_combo_label(combo_label)}...")

    pass2_tasks = {
        m: run_pass2_error_categories(
            failures=failures_per_metric[m],
            metric_name=m,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            cache_dir=cache_dir,
            force_rerun="pass2" in force_rerun,
        )
        for m in ALL_ANALYSIS_METRICS
    }
    pass2_values = await asyncio.gather(*pass2_tasks.values(), return_exceptions=True)
    pass2_results: dict[str, list[dict]] = {}
    for m, result in zip(pass2_tasks.keys(), pass2_values):
        if isinstance(result, Exception):
            logger.warning(f"Pass 2 failed for {m}: {result}")
            pass2_results[m] = []
        else:
            pass2_results[m] = result

    pass3_result = await run_pass3_synthesis(
        pass1_results=pass1_results,
        pass2_results=pass2_results,
        combo_label=combo_label,
        total_samples=total_samples,
        eva_a_mean=eva_a_mean,
        eva_x_mean=eva_x_mean,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        cache_dir=cache_dir,
        force_rerun="pass3" in force_rerun,
        pipeline_config=combo_data.get("pipeline_config"),
    )

    _save_metadata(cache_dir, run_dirs, threshold, batch_size, llm_client.model)

    failures_compact: dict[str, dict[str, dict]] = {}
    for m, failures in failures_per_metric.items():
        failures_compact[m] = {f["sample_id"]: {"score": f["score"], "explanation": f["explanation"]} for f in failures}

    if progress_callback:
        progress_callback(0.55, f"Analyzing error propagation for {get_display_combo_label(combo_label)}...")
    propagation_stats = compute_error_propagation_stats(combo_data, threshold)
    propagation_analysis = await run_error_propagation_analysis(
        combo_label=combo_label,
        propagation_stats=propagation_stats,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        cache_dir=cache_dir,
        force_rerun="pass1" in force_rerun,  # Rerun with pass1 since it depends on same data
    )

    metrics_summary: dict = {}
    for rd in run_dirs:
        ms_path = rd / "metrics_summary.json"
        if ms_path.exists():
            try:
                with open(ms_path) as f:
                    metrics_summary = json.load(f)
                break
            except Exception:
                pass

    return {
        "combo_label": combo_label,
        "pipeline_config": combo_data["pipeline_config"],
        "run_dirs": [str(d) for d in run_dirs],
        "pass1": pass1_results,
        "pass2": pass2_results,
        "pass3": pass3_result,
        "metric_scores": metric_scores,
        "total_samples": total_samples,
        "eva_a_mean": eva_a_mean,
        "eva_x_mean": eva_x_mean,
        "eva_a_pass1": eva_a_pass1,
        "eva_x_pass1": eva_x_pass1,
        "convo_length_stats": convo_length_stats,
        "failures": failures_compact,
        "metrics_summary": metrics_summary,
        "error_propagation": {
            "stats": {k: v for k, v in propagation_stats.items() if k != "cofailure_contexts"},
            "analysis": propagation_analysis,
        },
    }


async def generate_cross_combo_report(
    combo_reports: dict[str, dict],
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
) -> dict:
    """Cross-combo comparison report."""
    profiles = []
    for label, report in combo_reports.items():
        profiles.append(
            _format_combo_profile(label, report, include_issues=True, include_categories=True, include_convo_stats=True)
        )

    pc_block = _format_pipeline_config_block(combo_reports)
    metric_reference = prompt_manager.get_prompt("ai_review.metric_reference", pipeline_config=pc_block)
    prompt = prompt_manager.get_prompt(
        "ai_review.model_reports.cross_combo_comparison",
        metric_reference=metric_reference,
        combo_profiles="\n\n".join(profiles),
    )
    messages = [{"role": "user", "content": prompt}]
    parsed, was_cached = await _cached_llm_call(
        llm_client,
        messages,
        cache_dir=cache_dir,
        cache_parts=("cross_combo.json",),
        force_rerun=force_rerun,
        label="Cross-combo report",
    )

    if was_cached:
        return parsed

    result = parsed or {}
    result.setdefault("best_overall", "unknown")
    result.setdefault("comparison_summary", "Analysis failed (LLM error).")

    if cache_dir:
        _save_cache(cache_dir, result, "cross_combo.json")
    return result


async def generate_component_comparison(
    combo_reports: dict[str, dict],
    component: str,
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
) -> dict:
    """Compare combos that differ only in one component (llm_model, stt_model, tts_model)."""
    other_components = [c for c in ("llm_model", "stt_model", "tts_model") if c != component]
    groups: dict[str, list[tuple[str, dict]]] = {}
    for label, report in combo_reports.items():
        pc = report.get("pipeline_config", {})
        group_key = f"{pc.get(other_components[0], '')} / {pc.get(other_components[1], '')}"
        groups.setdefault(group_key, []).append((label, report))

    comparison_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not comparison_groups:
        result = {"summary": f"No comparable groups found for {component} isolation."}
        if cache_dir:
            _save_cache(cache_dir, result, f"component_{component}.json")
        return result

    config_results_parts = []
    component_values = set()
    for _group_key, combos in comparison_groups.items():
        for label, report in combos:
            pc = report.get("pipeline_config", {})
            val = pc.get(component, "unknown")
            component_values.add(val)
            block = _format_combo_profile(
                f"{label} ({component}={val})",
                report,
                include_issues=False,
                include_categories=True,
            )
            config_results_parts.append(block)

    pc_block = _format_pipeline_config_block(combo_reports)
    metric_reference = prompt_manager.get_prompt("ai_review.metric_reference", pipeline_config=pc_block)
    prompt = prompt_manager.get_prompt(
        "ai_review.model_reports.component_comparison",
        metric_reference=metric_reference,
        component_name=component.replace("_model", "").upper(),
        component_values=", ".join(sorted(component_values)),
        config_results="\n\n".join(config_results_parts),
    )
    messages = [{"role": "user", "content": prompt}]
    parsed, was_cached = await _cached_llm_call(
        llm_client,
        messages,
        cache_dir=cache_dir,
        cache_parts=(f"component_{component}.json",),
        force_rerun=force_rerun,
        label=f"Component comparison {component}",
    )

    if was_cached:
        return parsed

    result = parsed or {"summary": "Analysis failed."}

    if cache_dir:
        _save_cache(cache_dir, result, f"component_{component}.json")
    return result


async def generate_pattern_summary(
    combo_reports: dict[str, dict],
    cross_combo: dict,
    component_comparisons: dict[str, dict],
    prompt_manager: PromptManager,
    llm_client: LLMClient,
    cache_dir: Path | None = None,
    force_rerun: bool = False,
    component_focus: str = "all",
) -> dict:
    """Final executive summary across all models."""
    all_profiles_parts = []
    for label, report in combo_reports.items():
        block = _format_combo_profile(
            label, report, include_issues=True, include_categories=False, max_issues_per_metric=2
        )
        all_profiles_parts.append(block)

    comp_parts = []
    for comp_name, comp_report in component_comparisons.items():
        summary = comp_report.get("summary", "N/A")
        comp_parts.append(f"**{comp_name}:** {summary}")

    pc_block = _format_pipeline_config_block(combo_reports)
    metric_reference = prompt_manager.get_prompt("ai_review.metric_reference", pipeline_config=pc_block)
    prompt = prompt_manager.get_prompt(
        "ai_review.model_reports.pattern_summary",
        metric_reference=metric_reference,
        all_profiles="\n\n".join(all_profiles_parts),
        cross_combo=json.dumps(cross_combo, indent=2, default=str),
        component_comparisons="\n\n".join(comp_parts),
    )

    # Append component focus filter if not "all"
    focus_instructions = {
        "stt_tts": (
            "\n\n## IMPORTANT: Component Focus\n"
            "The intended audience for this report is the **STT/TTS vendor**. "
            "Only include findings that are relevant to speech-to-text or text-to-speech "
            "performance — entity transcription accuracy, speech fidelity, pronunciation, "
            "WER, turn-taking timing, and similar audio/speech concerns. "
            "Exclude findings that are purely about LLM reasoning, task completion logic, "
            "faithfulness to retrieved data, or conversation flow unless they are directly "
            "caused by or worsened by STT/TTS errors."
        ),
        "llm": (
            "\n\n## IMPORTANT: Component Focus\n"
            "The intended audience for this report is the **LLM provider**. "
            "Only include findings relevant to LLM reasoning and conversation quality — "
            "task completion, faithfulness, conversation progression, conciseness, tool "
            "calling accuracy. Exclude findings about STT transcription errors, TTS "
            "pronunciation, speech fidelity, or WER unless they are caused by the LLM."
        ),
    }
    if component_focus in focus_instructions:
        prompt += focus_instructions[component_focus]

    cache_suffix = f"_focus_{component_focus}" if component_focus != "all" else ""
    cache_file = f"pattern_summary{cache_suffix}.json"

    messages = [{"role": "user", "content": prompt}]
    parsed, was_cached = await _cached_llm_call(
        llm_client,
        messages,
        cache_dir=cache_dir,
        cache_parts=(cache_file,),
        force_rerun=force_rerun,
        label=f"Pattern summary (focus={component_focus})",
    )

    if was_cached:
        return parsed

    result = parsed or {"key_findings": []}

    if cache_dir:
        _save_cache(cache_dir, result, cache_file)
    return result


# ---
# Full Pipeline Orchestrator
# ---


async def run_full_pipeline(
    run_dirs: list[Path],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    force_rerun: set[str] | None = None,
    max_samples: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    result_callback: Callable[[dict], None] | None = None,
    component_focus: str = "all",
) -> dict:
    """Orchestrate the full multi-pass pipeline across all combos.

    Returns a PipelineResult-shaped dict.
    """
    force_rerun = force_rerun or set()

    prompts_dir = Path(__file__).parent / "prompts"
    prompt_manager = PromptManager(prompts_path=prompts_dir)
    llm_client = LLMClient(model=judge_model, params={"temperature": 0.0})

    if progress_callback:
        progress_callback(0.0, "Collecting data from run directories...")
    combo_data = collect_combo_data(run_dirs, max_samples=max_samples)

    if not combo_data:
        return {"combos": {}, "cross_combo": {}, "component_isolation": {}, "executive_summary": {}}

    combo_reports: dict[str, dict] = {}
    combo_labels = list(combo_data.keys())
    total_combos = len(combo_labels)

    for idx, combo_label in enumerate(combo_labels):
        if progress_callback:
            progress_callback(
                idx / total_combos * 0.7,
                f"Analyzing combo {idx + 1}/{total_combos}: {get_display_combo_label(combo_label)}",
            )

        report = await generate_combo_report(
            combo_label=combo_label,
            combo_data=combo_data[combo_label],
            run_dirs=combo_data[combo_label].get("run_dirs", run_dirs),
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            batch_size=batch_size,
            threshold=failure_threshold,
            force_rerun=force_rerun,
            progress_callback=progress_callback,
        )
        combo_reports[combo_label] = report

        if result_callback:
            result_callback(
                {
                    "combos": dict(combo_reports),
                    "cross_combo": {},
                    "component_isolation": {},
                    "executive_summary": {},
                }
            )

    cross_combo: dict = {}
    component_isolation: dict = {}
    executive_summary: dict = {}

    cross_cache_dir = _get_cross_cache_dir(run_dirs)

    if len(combo_reports) > 1:
        if progress_callback:
            progress_callback(0.7, "Generating cross-combo comparison...")

        cross_combo = await generate_cross_combo_report(
            combo_reports=combo_reports,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            cache_dir=cross_cache_dir,
            force_rerun="reports" in force_rerun,
        )

        if result_callback:
            result_callback(
                {
                    "combos": combo_reports,
                    "cross_combo": cross_combo,
                    "component_isolation": {},
                    "executive_summary": {},
                }
            )

        if progress_callback:
            progress_callback(0.8, "Running component-isolated comparisons...")

        for component in ("llm_model", "stt_model", "tts_model"):
            comp_report = await generate_component_comparison(
                combo_reports=combo_reports,
                component=component,
                prompt_manager=prompt_manager,
                llm_client=llm_client,
                cache_dir=cross_cache_dir,
                force_rerun="reports" in force_rerun,
            )
            component_isolation[component] = comp_report

        if progress_callback:
            progress_callback(0.9, "Generating executive summary...")

        executive_summary = await generate_pattern_summary(
            combo_reports=combo_reports,
            cross_combo=cross_combo,
            component_comparisons=component_isolation,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            cache_dir=cross_cache_dir,
            force_rerun="reports" in force_rerun,
            component_focus=component_focus,
        )

    if progress_callback:
        progress_callback(1.0, "Pipeline complete!")

    per_sample_rows = build_per_sample_dataframe(combo_data)

    final_result = {
        "combos": combo_reports,
        "cross_combo": cross_combo,
        "component_isolation": component_isolation,
        "executive_summary": executive_summary,
        "per_sample_data": per_sample_rows,
    }

    if result_callback:
        result_callback(final_result)

    return final_result
