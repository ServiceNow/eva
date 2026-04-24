"""Tests for eva.user_simulator.event_logger module."""

import json

import pytest

from eva.user_simulator.event_logger import ElevenLabsEventLogger


@pytest.fixture
def logger(tmp_path):
    """Create an event logger with a temp output path."""
    return ElevenLabsEventLogger(output_path=tmp_path / "events.jsonl")


class TestEventLogger:
    def test_init(self, logger):
        assert logger._events == []
        assert logger._sequence == 0

    def test_log_event_creates_correct_structure(self, logger):
        logger.log_event("test_type", {"key": "value"})
        assert len(logger._events) == 1
        event = logger._events[0]
        assert event["type"] == "test_type"
        assert event["data"] == {"key": "value"}
        assert event["sequence"] == 1
        assert isinstance(event["timestamp"], int)

    def test_sequence_increments(self, logger):
        logger.log_event("a", {})
        logger.log_event("b", {})
        logger.log_event("c", {})
        assert [e["sequence"] for e in logger._events] == [1, 2, 3]

    def test_log_user_speech(self, logger):
        logger.log_user_speech("hello world", is_final=True)
        event = logger._events[0]
        assert event["type"] == "user_speech"
        assert event["data"]["text"] == "hello world"
        assert event["data"]["is_final"] is True

    def test_log_user_speech_not_final(self, logger):
        logger.log_user_speech("hel", is_final=False)
        assert logger._events[0]["data"]["is_final"] is False

    def test_log_assistant_speech(self, logger):
        logger.log_assistant_speech("Hi there!")
        event = logger._events[0]
        assert event["type"] == "assistant_speech"
        assert event["data"]["text"] == "Hi there!"

    def test_log_audio_sent(self, logger):
        logger.log_audio_sent(4096)
        assert logger._events[0]["data"]["size_bytes"] == 4096

    def test_log_audio_received(self, logger):
        logger.log_audio_received(8192)
        assert logger._events[0]["data"]["size_bytes"] == 8192

    def test_log_connection_state(self, logger):
        logger.log_connection_state("connected", {"url": "ws://localhost"})
        event = logger._events[0]
        assert event["type"] == "connection_state"
        assert event["data"]["state"] == "connected"
        assert event["data"]["details"]["url"] == "ws://localhost"

    def test_log_connection_state_no_details(self, logger):
        logger.log_connection_state("disconnected")
        assert logger._events[0]["data"]["details"] == {}

    def test_log_error(self, logger):
        logger.log_error("Connection timeout", {"retry": 3})
        event = logger._events[0]
        assert event["type"] == "error"
        assert event["data"]["error"] == "Connection timeout"
        assert event["data"]["details"]["retry"] == 3

    def test_log_error_no_details(self, logger):
        logger.log_error("Oops")
        assert logger._events[0]["data"]["details"] == {}

    def test_log_audio_start_structure(self, logger):
        """Audio events have different structure: event_type/user instead of type/data."""
        logger.log_audio_start("elevenlabs_user")
        event = logger._events[0]
        assert event["event_type"] == "audio_start"
        assert event["user"] == "elevenlabs_user"
        assert isinstance(event["audio_timestamp"], float)
        assert event["sequence"] == 1
        assert "type" not in event
        assert "data" not in event

    def test_log_audio_end_structure(self, logger):
        logger.log_audio_end("framework_agent")
        event = logger._events[0]
        assert event["event_type"] == "audio_end"
        assert event["user"] == "framework_agent"
        assert isinstance(event["audio_timestamp"], float)

    def test_save_creates_jsonl(self, logger):
        logger.log_user_speech("hello")
        logger.log_assistant_speech("hi")
        logger.save()

        lines = logger.output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        event0 = json.loads(lines[0])
        assert event0["type"] == "user_speech"
        event1 = json.loads(lines[1])
        assert event1["type"] == "assistant_speech"

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "events.jsonl"
        log = ElevenLabsEventLogger(output_path=nested)
        log.log_event("test", {})
        log.save()
        assert nested.exists()

    def test_get_events_all(self, logger):
        logger.log_user_speech("a")
        logger.log_assistant_speech("b")
        events = logger.get_events()
        assert len(events) == 2
        # Returns a copy
        events.clear()
        assert len(logger._events) == 2

    def test_get_events_filtered(self, logger):
        logger.log_user_speech("a")
        logger.log_assistant_speech("b")
        logger.log_user_speech("c")
        events = logger.get_events(event_type="user_speech")
        assert len(events) == 2

    def test_get_summary(self, logger):
        logger.log_user_speech("a")
        logger.log_assistant_speech("b")
        logger.log_user_speech("c")
        summary = logger.get_summary()
        assert summary["total_events"] == 3
        assert summary["event_counts"]["user_speech"] == 2
        assert summary["event_counts"]["assistant_speech"] == 1

    def test_clear(self, logger):
        logger.log_event("a", {})
        logger.log_event("b", {})
        logger.clear()
        assert logger._events == []
        assert logger._sequence == 0
        # New events start at sequence 1 again
        logger.log_event("c", {})
        assert logger._events[0]["sequence"] == 1
