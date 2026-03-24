"""Unit tests for conversation_checks utility."""

import json

import pytest

from eva.utils.conversation_checks import check_conversation_finished, find_records_with_llm_generic_error


@pytest.fixture
def record_dir(temp_dir):
    """Create a record output directory."""
    record_path = temp_dir / "record_1"
    record_path.mkdir()
    return record_path


def test_check_conversation_finished_success(record_dir):
    """Test check returns True when conversation ended with goodbye."""
    events = [
        {"type": "audio", "data": {}},
        {"type": "connection_state", "data": {"details": {"reason": "goodbye"}}},
    ]
    events_path = record_dir / "elevenlabs_events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    assert check_conversation_finished(record_dir) is True


def test_check_conversation_finished_no_goodbye(record_dir):
    """Test check returns False when last event is not goodbye."""
    events = [
        {"type": "connection_state", "data": {"details": {"reason": "timeout"}}},
    ]
    events_path = record_dir / "elevenlabs_events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    assert check_conversation_finished(record_dir) is False


def test_check_conversation_finished_no_file(record_dir):
    """Test check returns False when events file doesn't exist."""
    assert check_conversation_finished(record_dir) is False


def test_check_conversation_finished_empty_file(record_dir):
    """Test check returns False when events file is empty."""
    events_path = record_dir / "elevenlabs_events.jsonl"
    events_path.write_text("")

    assert check_conversation_finished(record_dir) is False


def test_check_conversation_finished_wrong_type(record_dir):
    """Test check returns False when last event is not connection_state."""
    events = [
        {"type": "audio", "data": {}},
    ]
    events_path = record_dir / "elevenlabs_events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    assert check_conversation_finished(record_dir) is False


def test_check_conversation_finished_invalid_json(record_dir):
    """Test check returns False when last line is invalid JSON."""
    events_path = record_dir / "elevenlabs_events.jsonl"
    events_path.write_text("not valid json\n")

    assert check_conversation_finished(record_dir) is False


def test_check_conversation_finished_no_details(record_dir):
    """Test check returns False when connection_state has no details."""
    events = [
        {"type": "connection_state", "data": {}},
    ]
    events_path = record_dir / "elevenlabs_events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    assert check_conversation_finished(record_dir) is False


def _write_pipecat_logs(output_dir, record_id, entries):
    """Helper to write pipecat_logs.jsonl for a record."""
    record_path = output_dir / "records" / record_id
    record_path.mkdir(parents=True, exist_ok=True)
    logs_path = record_path / "pipecat_logs.jsonl"
    logs_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_find_records_with_llm_generic_error_detects_error(temp_dir):
    """Test that records containing the generic LLM error are detected."""
    _write_pipecat_logs(
        temp_dir,
        "1.1.1",
        [
            {"type": "llm_response", "data": {"frame": "Hello, how can I help you?"}},
            {"type": "llm_response", "data": {"frame": "I'm sorry, I encountered an error processing your request."}},
        ],
    )
    _write_pipecat_logs(
        temp_dir,
        "1.1.2",
        [
            {"type": "llm_response", "data": {"frame": "Sure, let me look that up."}},
        ],
    )

    result = find_records_with_llm_generic_error(temp_dir, ["1.1.1", "1.1.2"])
    assert result == ["1.1.1"]


def test_find_records_with_llm_generic_error_no_errors(temp_dir):
    """Test that no records are returned when there are no generic LLM errors."""
    _write_pipecat_logs(
        temp_dir,
        "1.1.1",
        [
            {"type": "llm_response", "data": {"frame": "Hello, how can I help you?"}},
        ],
    )
    _write_pipecat_logs(
        temp_dir,
        "1.1.2",
        [
            {"type": "transcript", "data": {"frame": "I'm sorry, I encountered an error processing your request."}},
        ],
    )

    result = find_records_with_llm_generic_error(temp_dir, ["1.1.1", "1.1.2"])
    assert result == []


def test_find_records_with_llm_generic_error_missing_logs(temp_dir):
    """Test that records without pipecat_logs.jsonl are skipped."""
    result = find_records_with_llm_generic_error(temp_dir, ["nonexistent"])
    assert result == []
