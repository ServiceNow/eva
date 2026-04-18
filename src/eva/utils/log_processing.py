"""Voice agent log processing utilities."""

import copy
import itertools
from enum import StrEnum
from typing import Any


class AnnotationLabel(StrEnum):
    """Annotation labels inserted into transcript text to mark interruption events."""

    # Prefixes: placed at the start of an entry to signal who is barging in
    ASSISTANT_INTERRUPTS = "[assistant interrupts]"
    USER_INTERRUPTS = "[user interrupts]"

    # Inline separators: placed between chunks of the same speaker's text
    CUT_OFF_BY_USER = "[likely cut off by user]"
    CUT_OFF_BY_ASSISTANT = "[likely cut off by assistant]"
    CUT_OFF_ON_ITS_OWN = "[speaker likely cut itself off]"
    LIKELY_INTERRUPTION = "[likely interruption]"
    PAUSE_TOOL_CALL = "[pause]"


_DATA_URI_PREFIX = "data:"
_TRUNCATION_SUFFIX = "...[truncated]"
_MAX_DATA_URI_LEN = 256

MIN_PREFIX_MATCH_CHARS = 20


def strip_labels(text: str) -> str:
    """Remove all interruption/annotation labels from text and normalize whitespace."""
    for label in AnnotationLabel:
        text = text.replace(label, "")
    return " ".join(text.split())


def normalize_for_comparison(text: str) -> str:
    """Normalize text for robust comparison by keeping only alphanumeric characters.

    Removes all punctuation, whitespace, and special Unicode characters
    (quotes, hyphens, non-breaking spaces, etc.) to handle inconsistencies
    between different text sources (pipecat llm_response vs audit_log entries).
    This prevents false-negative substring matches caused by Unicode variations.
    """
    return "".join(c.lower() for c in text if c.isalnum())


def truncate_data_uris(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep copy of *messages* with long data-URI strings truncated.

    Audio messages embed base64 WAV data in ``audio_url.url`` fields which can
    be hundreds of KB each.  For audit logging we only need a prefix to identify
    the content type; the full payload is not useful in logs.
    """
    truncated = copy.deepcopy(messages)
    for msg in truncated:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            inner = item.get("audio_url")
            if isinstance(inner, dict):
                url = inner.get("url", "")
                if url.startswith(_DATA_URI_PREFIX) and len(url) > _MAX_DATA_URI_LEN:
                    inner["url"] = url[:_MAX_DATA_URI_LEN] + _TRUNCATION_SUFFIX
    return truncated


def truncate_to_spoken(audit_text: str, pipecat_segments: list[str]) -> str | None:
    """Truncate audit_log text to the portion that was actually spoken.

    The audit log records the full LLM response, but the turn may have been
    interrupted before TTS finished — so only a prefix was actually spoken.
    The pipecat segments (from tts_text events) reflect what was sent to TTS.

    Args:
        audit_text: Full LLM response from audit log.
        pipecat_segments: List of raw TTS text segments for this turn.

    Returns the (possibly truncated) audit text, or ``None`` if no
    meaningful overlap is found (the entry should be filtered).
    """
    norm_audit = normalize_for_comparison(strip_labels(" ".join(audit_text.split())))
    # Normalize each segment start for prefix matching
    norm_segments = [normalize_for_comparison(strip_labels(" ".join(seg.split()))) for seg in pipecat_segments]
    norm_segments = [s for s in norm_segments if s]

    if not norm_audit or not norm_segments:
        return None

    # Full match — entire audit text appears in any segment
    if any(norm_audit in seg for seg in norm_segments):
        return audit_text

    # Find longest word-prefix of the audit text that starts any pipecat
    # segment.  Matching at the START of a segment (not at an arbitrary
    # position) prevents spurious short matches (e.g. "Just to" matching
    # "just to" inside a different sentence).
    # Binary search: if N words match, N-1 words also match (prefix property).
    # We split the *original* text into words and normalize each candidate
    # prefix, because normalize_for_comparison strips all whitespace so
    # splitting the normalized string would yield a single token.
    def is_prefix_of_any_segment(norm_candidate: str) -> bool:
        return any(seg.startswith(norm_candidate) for seg in norm_segments)

    orig_words = strip_labels(" ".join(audit_text.split())).split()
    lo, hi = 0, len(orig_words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = normalize_for_comparison(" ".join(orig_words[:mid]))
        if is_prefix_of_any_segment(candidate):
            lo = mid
        else:
            hi = mid - 1

    if lo == 0:
        return None  # No overlap at all

    # Require the match to cover enough characters to be meaningful.
    # Short matches (e.g. "I" or "I'm") are almost certainly coincidental
    # when two different LLM responses start with the same common word.
    matched_normalized = normalize_for_comparison(" ".join(orig_words[:lo]))
    if len(matched_normalized) < MIN_PREFIX_MATCH_CHARS:
        return None

    # Map back to original text: take the first `lo` whitespace-separated
    # tokens from the original audit_text (preserves labels and formatting).
    raw_words = audit_text.split()
    return " ".join(raw_words[:lo])


def append_turn_text(
    target: dict[int, str],
    turn_idx: int,
    new_text: str,
    separator: str = "",
) -> None:
    """Append *new_text* to ``target[turn_idx]`` with *separator* between existing and new."""
    existing = target.get(turn_idx, "")
    if existing:
        target[turn_idx] = f"{existing}{separator}{new_text}"
    else:
        target[turn_idx] = new_text


def annotate_last_entry(
    entries: list[dict],
    turn_id: int,
    role: str,
    entry_type: str,
    suffix: str,
) -> None:
    """Append *suffix* to the last entry in *entries* matching turn/role/type."""
    for prev in reversed(entries):
        if prev.get("turn_id") != turn_id:
            break
        if prev.get("role") == role and prev.get("type") == entry_type:
            prev["content"] += f" {suffix}"
            break


def align_turn_keys(
    transcribed_turns: dict[int, str],
    intended_turns: dict[int, str],
    audio_timestamps: dict[int, list[tuple[float, float]] | None],
    text_default: str = "",
    timestamps_default: list[tuple[float, float]] | None = None,
) -> None:
    """Ensure transcribed, intended, and audio_timestamps share the same keys.

    Missing entries are filled with the provided defaults (mutates in place).
    """
    all_keys = transcribed_turns.keys() | intended_turns.keys() | audio_timestamps.keys()
    for key in all_keys:
        transcribed_turns.setdefault(key, text_default)
        intended_turns.setdefault(key, text_default)
        audio_timestamps.setdefault(key, timestamps_default)


def aggregate_pipecat_logs_by_type(pipecat_logs: list[dict]) -> list[dict]:
    """Aggregate consecutive pipecat logs of the same type.

    Only tts_text/llm_response entries can appear consecutively and need aggregation;
    turn_start/turn_end are single events that pass through unchanged.

    tts_text chunks (cascade) are joined with a space; llm_response chunks (S2S)
    already contain proper spacing and are joined without a separator.

    Args:
        pipecat_logs: Filtered pipecat logs (tts_text, llm_response, turn_start, turn_end).

    Returns:
        A list of aggregated log dictionaries.
    """
    if not pipecat_logs:
        return []

    aggregated: list[dict] = []
    current = pipecat_logs[0]
    data_key = next(iter(current["data"]))
    text = current["data"][data_key]
    min_ts = max_ts = current.get("timestamp", 0)

    for log in pipecat_logs[1:]:
        if log["type"] == current["type"]:
            # Consecutive text chunks — tts_text needs space, llm_response already has spacing
            sep = " " if current["type"] == "tts_text" else ""
            text += f"{sep}{log['data'][data_key]}"
            if log.get("timestamp"):
                max_ts = max(max_ts, log["timestamp"])
        else:
            aggregated.append(
                {
                    "type": current["type"],
                    "start_timestamp": min_ts,
                    "end_timestamp": max_ts,
                    "data": {data_key: text},
                }
            )
            current = log
            data_key = next(iter(current["data"]))
            text = current["data"][data_key]
            min_ts = max_ts = current.get("timestamp", 0)

    aggregated.append(
        {
            "type": current["type"],
            "start_timestamp": min_ts,
            "end_timestamp": max_ts,
            "data": {data_key: text},
        }
    )

    return aggregated


def get_entry_for_audit_log(event: dict, turn_id: int) -> dict:
    """Parse tool information from the audit logs.

    Adapted for voice-bench audit log format without agent tracking.

    Args:
        event: Event from logs
        turn_id: Turn id associated to that event

    Returns:
        List of parsed entries with role, content, tool calls, and tool responses
    """
    if event["event_type"] == "user":
        return {
            "role": "user",
            "content": event["data"],
            "timestamp": event["timestamp_ms"],
            "type": "transcribed",
            "turn_id": turn_id,
        }
    elif event["event_type"] == "assistant":
        return {
            "role": "assistant",
            "content": event["data"],
            "timestamp": event["timestamp_ms"],
            "type": "intended",
            "turn_id": turn_id,
        }
    elif event["event_type"] in ("tts_text", "llm_response"):
        return {
            "role": "assistant",
            "content": event["data"]["frame"].strip(),
            "timestamp": event["timestamp_ms"],
            "type": "intended",
            "turn_id": turn_id,
        }
    elif event["event_type"] == "tool_call":
        return {
            "tool_name": event["data"]["tool"],
            "parameters": event["data"]["parameters"],
            "timestamp": event["timestamp_ms"],
            "type": "tool_call",
            "turn_id": turn_id,
        }
    elif event["event_type"] == "tool_response":
        return {
            "tool_name": event["data"]["tool"],
            "tool_response": event["data"].get("response"),
            "timestamp": event["timestamp_ms"],
            "type": "tool_response",
            "turn_id": turn_id,
        }
    return {}


def group_consecutive_turns(turns: list[dict]) -> list[dict]:
    """Group consecutive turns of the same role *and* turn_id into a single turn.

    Entries at different turn_ids are never merged, even if they share the same role.
    Preserves entries without role field (tool calls, tool responses, etc.).
    """
    grouped = []
    for key, group in itertools.groupby(turns, key=lambda turn: (turn.get("role"), turn.get("turn_id"))):
        role, _ = key
        if role in ("user", "assistant"):
            group_tuple = tuple(group)
            first_turn = group_tuple[0].copy()
            first_turn["content"] = " ".join(content for turn in group_tuple if (content := turn.get("content")))
            grouped.append(first_turn)
        else:
            grouped.extend(group)
    return grouped


def extract_tool_params_and_responses(
    conversation_trace: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Extract tool_params and tool_responses from a conversation trace.

    Returns (tool_params, tool_responses) where:
      - tool_params: list of {"tool_name": str, "tool_parameters": dict}
      - tool_responses: list of {"tool_name": str, "tool_response": dict}
    """
    tool_params = [
        {"tool_name": d.get("tool_name"), "tool_parameters": d.get("parameters")}
        for d in conversation_trace
        if d.get("tool_name") and d.get("parameters") is not None
    ]
    tool_responses = [
        {"tool_name": d.get("tool_name"), "tool_response": d.get("tool_response")}
        for d in conversation_trace
        if d.get("tool_response") is not None
    ]
    return tool_params, tool_responses


def filter_empty_responses(logs: list[dict]) -> list[dict]:
    """Filter out empty or meaningless user responses from elevenlabs logs.

    Removes user_speech events that are empty or only contain punctuation/whitespace.
    """
    filtered = []
    for log in logs:
        if log.get("type") == "user_speech":
            text = log.get("data", {}).get("text", "").strip()
            # Filter out empty responses or responses that are just punctuation
            if text and text not in ["...", "......", ".", ",", "?", "!"]:
                filtered.append(log)
        else:
            filtered.append(log)
    return filtered
