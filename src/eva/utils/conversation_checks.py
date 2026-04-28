"""Lightweight conversation validation checks.

Provides fast, synchronous checks for conversation quality without
requiring the full metrics pipeline or MetricContext construction.
"""

import json
from pathlib import Path

from eva.utils.logging import get_logger

logger = get_logger(__name__)


def check_conversation_finished(output_dir: Path) -> bool:
    """Check if a conversation ended properly with a goodbye.

    Replicates the logic from ConversationFinishedMetric.compute() but
    returns a simple boolean. No LLM calls, just file parsing.

    Args:
        output_dir: Path to the record output directory containing
            elevenlabs_events.jsonl

    Returns:
        True if the conversation ended with a goodbye, False otherwise
    """
    events_path = output_dir / "elevenlabs_events.jsonl"

    if not events_path.exists():
        return False

    try:
        with open(events_path) as f:
            lines = f.readlines()
    except OSError:
        return False

    if not lines:
        return False

    last_line = lines[-1].strip()
    if not last_line:
        return False

    try:
        last_event = json.loads(last_line)
    except json.JSONDecodeError:
        return False

    if last_event.get("type") != "connection_state":
        return False

    data = last_event.get("data", {})
    details = data.get("details", {})
    return details.get("reason") == "goodbye"


LLM_GENERIC_ERROR_MESSAGE = "I'm sorry, I encountered an error processing your request."


def find_records_with_llm_generic_error(output_dir: Path, record_ids: set[str] | list[str]) -> list[str]:
    """Find records that have the LLM generic error message in pipecat_logs.jsonl."""
    affected = []
    for record_id in record_ids:
        pipecat_logs_path = output_dir / "records" / record_id / "framework_logs.jsonl"
        if not pipecat_logs_path.exists():
            pipecat_logs_path = output_dir / "records" / record_id / "pipecat_logs.jsonl"
        if not pipecat_logs_path.exists():
            continue
        with open(pipecat_logs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "llm_response" and LLM_GENERIC_ERROR_MESSAGE in entry.get("data", {}).get(
                    "frame", ""
                ):
                    affected.append(record_id)
                    break
    return sorted(affected)
