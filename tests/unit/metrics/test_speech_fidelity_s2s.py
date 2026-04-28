"""Tests for agent_speech_fidelity S2S variant (entity-focused evaluation)."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from eva.metrics.accuracy.agent_speech_fidelity_s2s import AgentSpeechFidelityS2SMetric
from eva.models.config import PipelineType

from .conftest import make_judge_metric, make_metric_context


def make_judge_response(turns: list[dict]) -> str:
    """Create a JSON judge response with a ``turns`` wrapper."""
    return json.dumps({"turns": turns})


@pytest.fixture
def s2s_metric():
    return make_judge_metric(
        AgentSpeechFidelityS2SMetric,
        mock_llm=True,
        logger_name="test_agent_speech_fidelity_s2s",
    )


# --- Sample conversation traces ---

# Conversation trace entries use different schemas:
# - user/assistant: have "role" + "content" + "type" (intended/transcribed)
# - tool entries: have "type" (tool_call/tool_response) + "tool_name" + data fields, no "role"

SIMPLE_TRACE = [
    {"role": "user", "content": "Check reservation ABC123, last name Smith", "type": "intended", "turn_id": 0},
    {"role": "assistant", "content": "Looking that up for you.", "type": "transcribed", "turn_id": 1},
    {
        "tool_name": "get_reservation",
        "parameters": {"confirmation_number": "ABC123"},
        "type": "tool_call",
        "turn_id": 1,
    },
    {
        "tool_name": "get_reservation",
        "tool_response": {"confirmation_number": "ABC123", "last_name": "Smith", "flight": "UA456"},
        "type": "tool_response",
        "turn_id": 1,
    },
    {"role": "assistant", "content": "Your flight is UA456.", "type": "transcribed", "turn_id": 1},
    {"role": "user", "content": "Thanks", "type": "intended", "turn_id": 2},
    {"role": "assistant", "content": "You're welcome!", "type": "transcribed", "turn_id": 3},
]

MULTI_ASSISTANT_SAME_TURN_TRACE = [
    {"role": "user", "content": "Book me a flight", "type": "intended", "turn_id": 0},
    {"role": "assistant", "content": "Let me search.", "type": "transcribed", "turn_id": 1},
    {"tool_name": "search_flights", "parameters": {}, "type": "tool_call", "turn_id": 1},
    {"tool_name": "search_flights", "tool_response": {"flights": ["SW302"]}, "type": "tool_response", "turn_id": 1},
    {"role": "assistant", "content": "I found flight SW302.", "type": "transcribed", "turn_id": 1},
    {"role": "user", "content": "Great, book it", "type": "intended", "turn_id": 2},
    {"role": "assistant", "content": "Done!", "type": "transcribed", "turn_id": 3},
]

NO_TOOL_TRACE = [
    {"role": "user", "content": "Hello", "type": "intended", "turn_id": 0},
    {"role": "assistant", "content": "Hi there!", "type": "transcribed", "turn_id": 1},
]


def _default_context(**overrides):
    """Context for S2S speech fidelity tests."""
    defaults = {
        "audio_assistant_path": "/fake/audio_assistant.wav",
        "audio_user_path": "/fake/audio_user.wav",
        "pipeline_type": PipelineType.S2S,
        "conversation_trace": SIMPLE_TRACE,
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


class TestClassAttributes:
    def test_s2s_metric_attributes(self, s2s_metric):
        assert s2s_metric.name == "agent_speech_fidelity"
        assert s2s_metric.category == "accuracy"
        assert s2s_metric.role == "assistant"
        assert s2s_metric.rating_scale == (0, 1)
        assert s2s_metric.pass_at_k_threshold == 0.95


class TestBuildRedactedTrace:
    def test_assistant_entries_are_redacted(self, s2s_metric):
        redacted = s2s_metric._build_redacted_trace(_default_context())
        assistant_entries = [e for e in redacted if e["role"] == "assistant"]
        for entry in assistant_entries:
            assert entry.get("redacted") is True
            assert "content" not in entry

    def test_user_entries_preserved(self, s2s_metric):
        redacted = s2s_metric._build_redacted_trace(_default_context())
        user_entries = [e for e in redacted if e["role"] == "user"]
        assert len(user_entries) == 2
        assert user_entries[0]["content"] == "Check reservation ABC123, last name Smith"
        assert user_entries[1]["content"] == "Thanks"

    def test_tool_responses_preserved(self, s2s_metric):
        redacted = s2s_metric._build_redacted_trace(_default_context())
        tool_entries = [e for e in redacted if e["role"] == "tool_response"]
        assert len(tool_entries) == 1
        assert tool_entries[0]["tool_name"] == "get_reservation"
        assert tool_entries[0]["content"]["confirmation_number"] == "ABC123"
        assert tool_entries[0]["content"]["flight"] == "UA456"

    def test_tool_calls_dropped(self, s2s_metric):
        """Tool call entries (type=tool_call, no role) should not appear in redacted trace."""
        redacted = s2s_metric._build_redacted_trace(_default_context())
        tool_call_entries = [e for e in redacted if e.get("type") == "tool_call" or e.get("role") == "tool_call"]
        assert len(tool_call_entries) == 0

    def test_multiple_assistant_entries_same_turn_deduplicated(self, s2s_metric):
        """Multiple assistant entries in the same turn should produce one placeholder."""
        context = _default_context(conversation_trace=MULTI_ASSISTANT_SAME_TURN_TRACE)
        redacted = s2s_metric._build_redacted_trace(context)
        assistant_entries = [e for e in redacted if e["role"] == "assistant"]
        # Turn 1 has two assistant entries, but should be deduplicated to one
        turn_1_entries = [e for e in assistant_entries if e["turn_id"] == 1]
        assert len(turn_1_entries) == 1

    def test_empty_trace(self, s2s_metric):
        context = _default_context(conversation_trace=[])
        redacted = s2s_metric._build_redacted_trace(context)
        assert redacted == []

    def test_none_trace(self, s2s_metric):
        context = _default_context(conversation_trace=None)
        redacted = s2s_metric._build_redacted_trace(context)
        assert redacted == []


class TestGetAssistantTurnIds:
    def test_extracts_unique_turn_ids(self, s2s_metric):
        redacted = s2s_metric._build_redacted_trace(_default_context())
        turn_ids = s2s_metric._get_assistant_turn_ids(redacted)
        assert turn_ids == [1, 3]

    def test_deduplicates_same_turn(self, s2s_metric):
        context = _default_context(conversation_trace=MULTI_ASSISTANT_SAME_TURN_TRACE)
        redacted = s2s_metric._build_redacted_trace(context)
        turn_ids = s2s_metric._get_assistant_turn_ids(redacted)
        assert turn_ids == [1, 3]

    def test_empty_trace(self, s2s_metric):
        turn_ids = s2s_metric._get_assistant_turn_ids([])
        assert turn_ids == []


class TestFormatRedactedTrace:
    def test_format_simple_trace(self, s2s_metric):
        redacted = s2s_metric._build_redacted_trace(_default_context())
        formatted = s2s_metric._format_redacted_trace(redacted)
        lines = formatted.split("\n")

        assert lines[0] == "Turn 0 - User: Check reservation ABC123, last name Smith"
        assert lines[1] == "Turn 1 - [Assistant speaks]"
        assert "Turn 1 - Tool Response (get_reservation):" in lines[2]
        assert '"confirmation_number": "ABC123"' in lines[2]
        assert lines[3] == "Turn 2 - User: Thanks"
        assert lines[4] == "Turn 3 - [Assistant speaks]"

    def test_format_no_duplicate_assistant_lines(self, s2s_metric):
        """Even with multiple assistant entries per turn, only one line appears."""
        context = _default_context(conversation_trace=MULTI_ASSISTANT_SAME_TURN_TRACE)
        redacted = s2s_metric._build_redacted_trace(context)
        formatted = s2s_metric._format_redacted_trace(redacted)
        assert formatted.count("Turn 1 - [Assistant speaks]") == 1


class TestNoAudio:
    @pytest.mark.asyncio
    async def test_no_audio_returns_error(self, s2s_metric):
        context = _default_context(audio_assistant_path=None)
        result = await s2s_metric.compute(context)
        assert result.score == 0.0
        assert "No assistant audio" in result.error


class TestNoAssistantTurns:
    @pytest.mark.asyncio
    async def test_no_assistant_turns_returns_error(self, s2s_metric):
        trace = [
            {"role": "user", "content": "Hello", "type": "intended", "turn_id": 0},
        ]
        context = _default_context(conversation_trace=trace)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            result = await s2s_metric.compute(context)
        assert result.score == 0.0
        assert "No assistant turns" in result.error


class TestNoJudgeResponse:
    @pytest.mark.asyncio
    async def test_no_response_returns_error(self, s2s_metric):
        s2s_metric.llm_client.generate_text.return_value = (None, None)
        context = _default_context()
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                result = await s2s_metric.compute(context)
        assert result.score == 0.0
        assert result.error == "No response from judge"


class TestS2SCompute:
    @pytest.mark.asyncio
    async def test_all_high_fidelity(self, s2s_metric):
        """All turns rated 1 -> perfect score."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 1, "explanation": "All entities correct"},
                {"turn_id": 3, "rating": 1, "explanation": "No entities to check"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        assert result.score == 1.0
        assert result.normalized_score == 1.0
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["variant"] == "s2s"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_all_low_fidelity(self, s2s_metric):
        """All turns rated 0 -> zero score."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 0, "explanation": "Said UA465 instead of UA456"},
                {"turn_id": 3, "rating": 0, "explanation": "Wrong name"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        assert result.score == 0.0
        assert result.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_mixed_ratings(self, s2s_metric):
        """One turn correct, one incorrect -> 0.5."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 1, "explanation": "Correct"},
                {"turn_id": 3, "rating": 0, "explanation": "Wrong entity"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        assert result.score == 0.5
        assert result.normalized_score == 0.5

    @pytest.mark.asyncio
    async def test_invalid_rating_excluded(self, s2s_metric):
        """Invalid ratings are excluded from aggregation."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 1, "has_entities": True, "explanation": "Good"},
                {"turn_id": 3, "rating": 5, "has_entities": True, "explanation": "Invalid"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        assert result.details["num_evaluated"] == 1
        assert result.details["per_turn_ratings"][3] is None
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_entity_turns_excluded_from_score(self, s2s_metric):
        """Turns with has_entities=false should not count toward the score."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 0, "has_entities": False, "explanation": "Greeting, no entities"},
                {"turn_id": 3, "rating": 1, "has_entities": True, "explanation": "Flight number correct"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        # Only turn 3 (has_entities=True) should be evaluated
        assert result.details["num_evaluated"] == 1
        assert result.details["num_skipped_no_entities"] == 1
        assert result.score == 1.0
        assert result.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_all_turns_no_entities(self, s2s_metric):
        """If all turns have no entities, it is not an error — scores are None."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 1, "has_entities": False, "explanation": "No entities"},
                {"turn_id": 3, "rating": 1, "has_entities": False, "explanation": "No entities"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        assert result.details["num_evaluated"] == 0
        assert result.details["num_skipped_no_entities"] == 2
        assert result.score is None
        assert result.normalized_score is None
        assert result.error is None
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_has_entities_defaults_to_true(self, s2s_metric):
        """If has_entities is missing from response, default to True (include in scoring)."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 1, "explanation": "Good"},
                {"turn_id": 3, "rating": 0, "explanation": "Wrong entity"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await s2s_metric.compute(context)

        assert result.details["num_evaluated"] == 2
        assert result.details["num_skipped_no_entities"] == 0
        assert result.score == 0.5


class TestTurnCountMismatch:
    @pytest.mark.asyncio
    async def test_fewer_turns_returned(self, s2s_metric, caplog):
        """Fewer turns than expected logs a warning but still computes."""
        response = make_judge_response(
            [
                {"turn_id": 1, "rating": 1, "explanation": "Good"},
            ]
        )
        s2s_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(s2s_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(s2s_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                with caplog.at_level(logging.WARNING):
                    result = await s2s_metric.compute(context)

        assert "Expected 2 ratings" in caplog.text
        assert result.details["num_evaluated"] == 1
        assert result.score == 1.0


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_exception_returns_error_score(self, s2s_metric):
        with patch.object(s2s_metric, "load_role_audio", side_effect=RuntimeError("boom")):
            context = _default_context()
            result = await s2s_metric.compute(context)

        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "boom" in result.error
