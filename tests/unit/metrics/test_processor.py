"""Parameterized tests for _extract_turns_from_history and _reconcile_transcript_with_tools.

Each test case is a minimal history (list of events) with expected outputs,
loaded from tests/fixtures/processor_histories.json.
"""

import json
from pathlib import Path

import pytest

from eva.metrics.processor import (
    MetricsContextProcessor,
    _normalize_event_for_processor,
    _ProcessorContext,
    _validate_conversation_trace,
)
from eva.models.config import PipelineType

FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "processor_histories.json"

with open(FIXTURES_PATH) as f:
    TEST_CASES = json.load(f)

TEST_IDS = [case["id"] for case in TEST_CASES]


def _int_keys(d: dict) -> dict:
    """Convert string keys to int keys (JSON only supports string keys)."""
    return {int(k): v for k, v in d.items()}


def _convert_expected_value(key: str, value):
    """Convert expected values from JSON format to Python format.

    - Dict attributes (turns_*) need int keys.
    - Audio timestamp values are lists in JSON but tuples in Python.
    - Set attributes (*_interrupted_turns) are JSON lists → Python sets.
    """
    if isinstance(value, dict):
        converted = _int_keys(value)
        if "audio_timestamps" in key:
            converted = {k: [tuple(seg) for seg in v] if v is not None else None for k, v in converted.items()}
        return converted
    if key.endswith("_interrupted_turns"):
        return set(value)
    return value


@pytest.fixture(params=TEST_CASES, ids=TEST_IDS)
def case(request):
    return request.param


class TestExtractTurnsFromHistory:
    """Test _extract_turns_from_history with minimal synthetic histories."""

    def test_expected_outputs(self, case):
        ctx = _ProcessorContext()
        ctx.record_id = case["id"]
        ctx.history = case["history"]
        pipeline_type_str = case.get("pipeline_type", "cascade")
        ctx.pipeline_type = PipelineType(pipeline_type_str)

        MetricsContextProcessor._extract_turns_from_history(ctx)
        MetricsContextProcessor._reconcile_transcript_with_tools(ctx)

        for key, expected_value in case["expected"].items():
            actual = getattr(ctx, key)
            expected = _convert_expected_value(key, expected_value)
            if key == "conversation_trace":
                # Strip timestamps for comparison (exact ms values are brittle)
                actual = [{k: v for k, v in entry.items() if k != "timestamp"} for entry in actual]
            assert actual == expected, (
                f"Case '{case['id']}', attribute '{key}':\n  expected: {expected}\n  actual:   {actual}"
            )


class TestValidateConversationTraceFallback:
    """_validate_conversation_trace preserves fallback nudge entries with no matching TTS segment.

    Consecutive turn-end fallback nudges can share a single merged pipecat segment on a
    neighboring turn, leaving the second nudge's turn with no segments. Without preservation
    the nudge is dropped, collapsing the two surrounding user turns together.
    """

    def _ctx(self, segments):
        ctx = _ProcessorContext()
        ctx.record_id = "test"
        ctx._intended_assistant_segments = segments
        ctx.intended_assistant_turns = {k: " ".join(v) for k, v in segments.items()}
        return ctx

    def test_fallback_entry_kept_when_no_segment_match(self):
        ctx = self._ctx({4: ["I'm sorry, I didn't quite catch that."]})  # turn 5 has no segments
        trace = [
            {
                "role": "assistant",
                "content": "I'm sorry, I didn't quite catch that.",
                "turn_id": 4,
                "_audit_source": True,
            },
            {
                "role": "assistant",
                "content": "Are you still there?",
                "turn_id": 5,
                "_audit_source": True,
                "_fallback_nudge": True,
            },
        ]
        out = _validate_conversation_trace(trace, ctx)
        assert [e["content"] for e in out] == ["I'm sorry, I didn't quite catch that.", "Are you still there?"]
        # Bookkeeping keys are stripped from the output.
        assert all("_fallback_nudge" not in e and "_audit_source" not in e for e in out)

    def test_non_fallback_entry_still_dropped_when_unsaid(self):
        ctx = self._ctx({4: ["I'm sorry, I didn't quite catch that."]})
        trace = [
            {"role": "assistant", "content": "Some text never spoken.", "turn_id": 5, "_audit_source": True},
        ]
        out = _validate_conversation_trace(trace, ctx)
        assert out == []


class TestNormalizeEventForProcessor:
    """_normalize_event_for_processor maps legacy role names to neutral names for old elevenlabs_events.jsonl files."""

    def test_legacy_elevenlabs_user_mapped_to_neutral(self):
        event = {"event_type": "audio_start", "user": "elevenlabs_user", "audio_timestamp": 1.0}
        result = _normalize_event_for_processor(event)
        assert result["user"] == "simulated_user"

    def test_legacy_framework_agent_mapped_to_neutral(self):
        event = {"event_type": "audio_start", "user": "framework_agent", "audio_timestamp": 1.0}
        result = _normalize_event_for_processor(event)
        assert result["user"] == "assistant"

    def test_legacy_pipecat_agent_mapped_to_neutral(self):
        event = {"event_type": "audio_start", "user": "pipecat_agent", "audio_timestamp": 1.0}
        result = _normalize_event_for_processor(event)
        assert result["user"] == "assistant"

    def test_neutral_role_passes_through_unchanged(self):
        event = {"event_type": "audio_start", "user": "simulated_user", "audio_timestamp": 1.0}
        result = _normalize_event_for_processor(event)
        assert result["user"] == "simulated_user"

    def test_event_without_user_field_unchanged(self):
        event = {"type": "user_speech", "data": {"text": "hello"}}
        result = _normalize_event_for_processor(event)
        assert result == event

    def test_original_event_not_mutated(self):
        event = {"event_type": "audio_start", "user": "simulated_user"}
        _normalize_event_for_processor(event)
        assert event["user"] == "simulated_user"
