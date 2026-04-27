"""Tests for agent_speech_fidelity and user_speech_fidelity metrics."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions

from eva.metrics.accuracy.agent_speech_fidelity import AgentSpeechFidelityMetric
from eva.metrics.validation.user_speech_fidelity import UserSpeechFidelityMetric

from .conftest import make_judge_metric, make_metric_context


def make_judge_response(turns: list[dict]) -> str:
    """Create a JSON judge response with a ``turns`` wrapper."""
    return json.dumps({"turns": turns})


@pytest.fixture
def agent_metric():
    return make_judge_metric(
        AgentSpeechFidelityMetric,
        mock_llm=True,
        logger_name="test_agent_speech_fidelity",
    )


@pytest.fixture
def user_metric():
    return make_judge_metric(
        UserSpeechFidelityMetric,
        mock_llm=True,
        logger_name="test_user_speech_fidelity",
    )


def _default_context(**overrides):
    """Context with default turns for speech fidelity tests."""
    defaults = {
        "transcribed_user_turns": {0: "Hi", 1: "Help me"},
        "transcribed_assistant_turns": {0: "Hello", 1: "Sure"},
        "intended_user_turns": {0: "Hi", 1: "Help me"},
        "intended_assistant_turns": {0: "Hello", 1: "Sure thing"},
        "audio_assistant_path": "/fake/audio_assistant.wav",
        "audio_user_path": "/fake/audio_user.wav",
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


class TestClassAttributes:
    """Verify subclass metadata is set correctly."""

    def test_agent_metric_attributes(self, agent_metric):
        assert agent_metric.name == "agent_speech_fidelity"
        assert agent_metric.category == "accuracy"
        assert agent_metric.role == "assistant"
        assert agent_metric.rating_scale == (0, 1)
        assert agent_metric.pass_at_k_threshold == 0.95

    def test_user_metric_attributes(self, user_metric):
        assert user_metric.name == "user_speech_fidelity"
        assert user_metric.category == "validation"
        assert user_metric.role == "user"
        assert user_metric.rating_scale == (1, 3)


class TestNoAudio:
    """Compute returns error when audio file is missing."""

    @pytest.mark.asyncio
    async def test_agent_no_audio(self, agent_metric):
        context = _default_context(audio_assistant_path=None)
        result = await agent_metric.compute(context)
        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "No assistant audio" in result.error

    @pytest.mark.asyncio
    async def test_user_no_audio(self, user_metric):
        context = _default_context(audio_user_path=None)
        result = await user_metric.compute(context)
        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "No user audio" in result.error


class TestNoJudgeResponse:
    """Compute returns error when judge returns None."""

    @pytest.mark.asyncio
    async def test_agent_no_response(self, agent_metric):
        agent_metric.llm_client.generate_text.return_value = (None, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await agent_metric.compute(context)
        assert result.score == 0.0
        assert result.error == "No response from judge"

    @pytest.mark.asyncio
    async def test_user_no_response(self, user_metric):
        user_metric.llm_client.generate_text.return_value = (None, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)
        assert result.score == 0.0
        assert result.error == "No response from judge"


class TestAgentCompute:
    """Test agent speech fidelity compute with 0/1 ratings."""

    @pytest.mark.asyncio
    async def test_all_high_fidelity(self, agent_metric):
        """All turns rated 1 -> perfect score."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Accurate"},
                {"turn_id": 1, "rating": 1, "explanation": "Accurate"},
            ]
        )
        agent_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await agent_metric.compute(context)

        assert result.score == 1.0
        assert result.normalized_score == 1.0
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.error is None

    @pytest.mark.asyncio
    async def test_all_low_fidelity(self, agent_metric):
        """All turns rated 0 → zero score."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 0, "explanation": "Mismatch"},
                {"turn_id": 1, "rating": 0, "explanation": "Mismatch"},
            ]
        )
        agent_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await agent_metric.compute(context)

        assert result.score == 0.0
        assert result.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_mixed_ratings(self, agent_metric):
        """One turn 0, one turn 1 → avg 0.5."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Good"},
                {"turn_id": 1, "rating": 0, "explanation": "Bad"},
            ]
        )
        agent_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await agent_metric.compute(context)

        assert result.score == 0.5
        assert result.normalized_score == 0.5

    @pytest.mark.asyncio
    async def test_invalid_rating_excluded(self, agent_metric):
        """Invalid ratings are excluded from aggregation."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Good"},
                {"turn_id": 1, "rating": 5, "explanation": "Invalid"},
            ]
        )
        agent_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await agent_metric.compute(context)

        assert result.details["num_evaluated"] == 1
        assert result.details["per_turn_ratings"][1] is None
        assert result.score == 1.0  # Only the valid rating of 1

    @pytest.mark.asyncio
    async def test_no_per_turn_normalized_in_details(self, agent_metric):
        """Agent (0/1 scale) should NOT include per_turn_normalized in details."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Good"},
                {"turn_id": 1, "rating": 1, "explanation": "Good"},
            ]
        )
        agent_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await agent_metric.compute(context)

        assert "per_turn_normalized" not in result.details


class TestUserSpeechFidelityCompute:
    """Test user speech fidelity compute with 1-3 ratings."""

    @pytest.mark.asyncio
    async def test_all_high_fidelity(self, user_metric):
        """All turns rated 3 → perfect normalized score."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 3, "explanation": "Excellent"},
                {"turn_id": 1, "rating": 3, "explanation": "Excellent"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)

        assert result.score == 3.0
        assert result.normalized_score == 1.0
        assert result.details["num_evaluated"] == 2
        assert result.error is None

    @pytest.mark.asyncio
    async def test_all_poor_fidelity(self, user_metric):
        """All turns rated 1 → zero normalized score."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Poor"},
                {"turn_id": 1, "rating": 1, "explanation": "Poor"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)

        assert result.score == 1.0
        assert result.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_acceptable_rating(self, user_metric):
        """All turns rated 2 → 0.5 normalized."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 2, "explanation": "Ok"},
                {"turn_id": 1, "rating": 2, "explanation": "Ok"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)

        assert result.score == 2.0
        assert result.normalized_score == 0.5

    @pytest.mark.asyncio
    async def test_mixed_ratings(self, user_metric):
        """Rating 1 and 3 → avg raw 2.0, avg normalized 0.5."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 3, "explanation": "Good"},
                {"turn_id": 1, "rating": 1, "explanation": "Bad"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)

        assert result.score == 2.0
        assert result.normalized_score == 0.5

    @pytest.mark.asyncio
    async def test_invalid_rating_excluded(self, user_metric):
        """Rating outside 1-3 is excluded."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 3, "explanation": "Good"},
                {"turn_id": 1, "rating": 0, "explanation": "Invalid"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)

        assert result.details["num_evaluated"] == 1
        assert result.details["per_turn_ratings"][1] is None

    @pytest.mark.asyncio
    async def test_per_turn_normalized_in_details(self, user_metric):
        """User (1-3 scale) should include per_turn_normalized in details."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 3, "explanation": "Good"},
                {"turn_id": 1, "rating": 2, "explanation": "Ok"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                result = await user_metric.compute(context)

        assert "per_turn_normalized" in result.details
        assert result.details["per_turn_normalized"][0] == 1.0  # (3-1)/(3-1) = 1.0
        assert result.details["per_turn_normalized"][1] == 0.5  # (2-1)/(3-1) = 0.5


class TestTurnCountMismatch:
    """Warns when judge returns different number of turns than expected."""

    @pytest.mark.asyncio
    async def test_agent_fewer_turns_returned(self, agent_metric, caplog):
        """Fewer turns than expected logs a warning but still computes."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Good"},
            ]
        )
        agent_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(agent_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(agent_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                with caplog.at_level(logging.WARNING):
                    result = await agent_metric.compute(context)

        assert "Expected 2 ratings" in caplog.text
        assert result.details["num_evaluated"] == 1
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_user_extra_turns_returned(self, user_metric, caplog):
        """More turns than expected logs a warning but still computes."""
        response = make_judge_response(
            [
                {"turn_id": 0, "rating": 3, "explanation": "Good"},
                {"turn_id": 1, "rating": 2, "explanation": "Ok"},
                {"turn_id": 99, "rating": 3, "explanation": "Extra"},
            ]
        )
        user_metric.llm_client.generate_text.return_value = (response, None)
        with patch.object(user_metric, "load_role_audio", return_value=MagicMock()):
            with patch.object(user_metric, "encode_audio_segment", return_value="base64audio"):
                context = _default_context()
                with caplog.at_level(logging.WARNING):
                    result = await user_metric.compute(context)

        assert "Expected 2 ratings" in caplog.text
        # Should still have 2 valid ratings (turn 99 is not in tts_turn_ids)
        assert result.details["num_evaluated"] == 2


class TestErrorHandling:
    """Exceptions during compute return error MetricScore."""

    @pytest.mark.asyncio
    async def test_agent_exception_returns_error_score(self, agent_metric):
        with patch.object(agent_metric, "load_role_audio", side_effect=RuntimeError("boom")):
            context = _default_context()
            result = await agent_metric.compute(context)

        assert result.score == 0.0
        assert result.normalized_score == 0.0
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_user_exception_returns_error_score(self, user_metric):
        with patch.object(user_metric, "load_role_audio", side_effect=RuntimeError("boom")):
            context = _default_context()
            result = await user_metric.compute(context)

        assert result.score == 0.0
        assert "boom" in result.error


class TestCallAndParseRetry:
    @pytest.mark.asyncio
    async def test_retries_on_empty_turns(self, agent_metric):
        """Should retry when judge returns empty turns list."""
        empty_response = json.dumps({"turns": []})
        good_response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Good"},
            ]
        )
        agent_metric.llm_client.generate_text.side_effect = [(empty_response, None), (good_response, None)]

        context = _default_context()
        dummy_audio = MagicMock()
        with patch("eva.metrics.speech_fidelity_base.asyncio.sleep", new_callable=AsyncMock):
            response_text, turns = await agent_metric._call_and_parse([], context, dummy_audio, "prompt")

        assert len(turns) == 1
        assert agent_metric.llm_client.generate_text.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_no_audio_explanation(self, agent_metric):
        """Should retry when Gemini reports 'no audio', falling back to file upload via google.genai."""
        no_audio_response = json.dumps({"explanation": "I detected no audio in the input"})
        good_response = make_judge_response(
            [
                {"turn_id": 0, "rating": 1, "explanation": "Good"},
            ]
        )
        # First call (inline) returns no audio, second call goes through google.genai
        agent_metric.llm_client.generate_text.return_value = (no_audio_response, None)

        # Mock the uploaded file object returned by google.genai
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/abc123"
        mock_uploaded_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"

        # Mock the generate_content response
        mock_genai_response = MagicMock()
        mock_genai_response.text = good_response

        context = _default_context()
        dummy_audio = MagicMock()
        with (
            patch("eva.metrics.speech_fidelity_base.asyncio.sleep", new_callable=AsyncMock),
            patch.object(
                agent_metric,
                "_upload_audio_file",
                new_callable=AsyncMock,
                return_value=mock_uploaded_file,
            ),
            patch.object(
                agent_metric,
                "_generate_with_file",
                new_callable=AsyncMock,
                return_value=good_response,
            ),
        ):
            response_text, turns = await agent_metric._call_and_parse([], context, dummy_audio, "prompt")

        assert len(turns) == 1
        assert agent_metric.llm_client.generate_text.call_count == 1  # Only the initial inline call

    @pytest.mark.asyncio
    async def test_returns_none_on_null_response(self, agent_metric):
        """Returns (None, []) if LLM returns None."""
        agent_metric.llm_client.generate_text.return_value = (None, None)
        context = _default_context()
        dummy_audio = MagicMock()
        response_text, turns = await agent_metric._call_and_parse([], context, dummy_audio, "prompt")

        assert response_text is None
        assert turns == []


class TestIntendedTurnsSelection:
    """Verify correct turns are selected based on role."""

    def test_agent_uses_assistant_turns(self, agent_metric):
        context = _default_context(
            intended_assistant_turns={0: "Hello", 1: "Sure thing"},
            intended_user_turns={0: "Hi", 1: "Help me"},
        )
        turns = agent_metric._get_intended_turns(context)
        assert turns == {0: "Hello", 1: "Sure thing"}

    def test_user_uses_user_turns(self, user_metric):
        context = _default_context(
            intended_assistant_turns={0: "Hello", 1: "Sure thing"},
            intended_user_turns={0: "Hi", 1: "Help me"},
        )
        turns = user_metric._get_intended_turns(context)
        assert turns == {0: "Hi", 1: "Help me"}


class TestFormatIntendedTurns:
    def test_format(self, agent_metric):
        turns = {0: "Hello there", 2: "Sure thing"}
        formatted = agent_metric._format_intended_turns(turns)
        assert formatted == "Turn 0: Hello there\nTurn 2: Sure thing"

    def test_empty(self, agent_metric):
        assert agent_metric._format_intended_turns({}) == ""


class TestGenerateWithFileRetry:
    """Test retry logic in _generate_with_file for transient Google API errors."""

    @staticmethod
    def _make_mock_client(generate_content_side_effect):
        """Create a mock genai.Client with aio.models.generate_content configured."""
        mock_generate = AsyncMock(side_effect=generate_content_side_effect)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = mock_generate
        return mock_client, mock_generate

    @pytest.mark.asyncio
    async def test_retries_transient_error_then_succeeds(self, agent_metric):
        """Should retry on transient errors and return the response on success."""
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.uri = "https://example.com/files/abc123"

        mock_response = MagicMock()
        mock_response.text = '{"turns": [{"turn_id": 0, "rating": 1, "explanation": "Good"}]}'

        mock_client, mock_generate = self._make_mock_client(
            [
                google_exceptions.ServiceUnavailable("Service unavailable"),
                google_exceptions.InternalServerError("Internal error"),
                mock_response,
            ]
        )

        context = _default_context()
        with (
            patch("eva.metrics.speech_fidelity_base.genai.Client", return_value=mock_client),
            patch("eva.metrics.speech_fidelity_base.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await agent_metric._generate_with_file(mock_uploaded_file, "prompt", context)

        assert result == mock_response.text
        assert mock_generate.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, agent_metric):
        """Should raise after exhausting all retry attempts."""
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.uri = "https://example.com/files/abc123"

        mock_client, mock_generate = self._make_mock_client(google_exceptions.ServiceUnavailable("Service unavailable"))

        context = _default_context()
        with (
            patch("eva.metrics.speech_fidelity_base.genai.Client", return_value=mock_client),
            patch("eva.metrics.speech_fidelity_base.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(google_exceptions.ServiceUnavailable),
        ):
            await agent_metric._generate_with_file(mock_uploaded_file, "prompt", context, max_retries=2)

        # 1 initial + 2 retries = 3 total attempts
        assert mock_generate.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self, agent_metric):
        """Should raise immediately for non-retryable errors (e.g., InvalidArgument)."""
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.uri = "https://example.com/files/abc123"

        mock_client, mock_generate = self._make_mock_client(google_exceptions.InvalidArgument("Invalid argument"))

        context = _default_context()
        with (
            patch("eva.metrics.speech_fidelity_base.genai.Client", return_value=mock_client),
            patch("eva.metrics.speech_fidelity_base.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(google_exceptions.InvalidArgument),
        ):
            await agent_metric._generate_with_file(mock_uploaded_file, "prompt", context)

        assert mock_generate.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self, agent_metric):
        """Should retry on TooManyRequests / ResourceExhausted."""
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.uri = "https://example.com/files/abc123"

        mock_response = MagicMock()
        mock_response.text = '{"turns": []}'

        mock_client, mock_generate = self._make_mock_client(
            [
                google_exceptions.TooManyRequests("Rate limited"),
                mock_response,
            ]
        )

        context = _default_context()
        with (
            patch("eva.metrics.speech_fidelity_base.genai.Client", return_value=mock_client),
            patch("eva.metrics.speech_fidelity_base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await agent_metric._generate_with_file(mock_uploaded_file, "prompt", context)

        assert result == mock_response.text
        assert mock_sleep.await_count == 1
