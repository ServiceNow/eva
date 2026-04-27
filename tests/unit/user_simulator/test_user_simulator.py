"""Unit tests for UserSimulator client.

Focuses on non-trivial logic: conversation end idempotency, keep-alive
inactivity detection, end_call API polling with backoff, and error handling.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eva.user_simulator.client import UserSimulator


def _make_simulator(tmp_path: Path, **overrides) -> UserSimulator:
    """Create a UserSimulator with minimal config for testing."""
    defaults = {
        "current_date_time": "2026-03-23 10:00:00",
        "persona_config": {
            "user_persona_id": "1",
            "user_persona": "Friendly caller",
        },
        "goal": {
            "high_level_user_goal": "Rebook a flight",
            "starting_utterance": "Hi, I need to change my flight",
            "information_required": "confirmation number ABC123",
            "decision_tree": {
                "must_have_criteria": "Same destination",
                "escalation_behavior": "Ask for supervisor",
                "nice_to_have_criteria": "Window seat",
                "negotiation_behavior": "Accept first offer",
                "resolution_condition": "Flight rebooked",
                "failure_condition": "No availability",
                "edge_cases": "None",
            },
        },
        "server_url": "ws://localhost:9999",
        "output_dir": tmp_path,
        "agent_id": "agent_airline",
        "timeout": 60,
    }
    defaults.update(overrides)
    return UserSimulator(**defaults)


def _make_conv_details(transcript=None, status="done"):
    """Create a mock ElevenLabs conversation details response."""
    details = SimpleNamespace(
        transcript=transcript,
        status=status,
    )
    details.model_dump = lambda: {"transcript": transcript, "status": status}
    return details


def _make_turn(tool_results=None):
    """Create a mock transcript turn."""
    return SimpleNamespace(tool_results=tool_results)


def _make_tool_result(tool_name):
    """Create a mock tool result with a tool_name attribute."""
    return SimpleNamespace(tool_name=tool_name)


class TestOnConversationEndIdempotency:
    """First call to _on_conversation_end wins — subsequent calls are ignored."""

    def test_first_reason_wins(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim._on_conversation_end("goodbye")
        sim._on_conversation_end("transfer")
        sim._on_conversation_end("error")
        assert sim._end_reason == "goodbye"
        assert sim._conversation_done.is_set()

    def test_event_set_only_once(self, tmp_path):
        """Event.set() is idempotent but _end_reason should not change."""
        sim = _make_simulator(tmp_path)
        sim._on_conversation_end("transfer")
        # Manually check the event was set
        assert sim._conversation_done.is_set()
        # Second call should not change reason
        sim._on_conversation_end("timeout")
        assert sim._end_reason == "transfer"


class TestCallbacksResetKeepalive:
    """_on_user_speaks and _on_assistant_speaks must reset the inactivity counter."""

    def test_user_speech_resets_counter(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim._consecutive_keepalive_count = 11  # One away from timeout
        sim.event_logger = MagicMock()
        sim._on_user_speaks("hello")
        assert sim._consecutive_keepalive_count == 0

    def test_assistant_speech_resets_counter(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim._consecutive_keepalive_count = 11
        sim.event_logger = MagicMock()
        sim._on_assistant_speaks("How can I help?")
        assert sim._consecutive_keepalive_count == 0

    def test_user_speech_logs_correct_event_structure(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim.event_logger = MagicMock()
        sim._on_user_speaks("I need help")
        sim.event_logger.log_event.assert_called_once_with(
            "user_speech",
            {"text": "I need help", "source": "elevenlabs_agent"},
        )

    def test_assistant_speech_logs_correct_event_structure(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim.event_logger = MagicMock()
        sim._on_assistant_speaks("Sure thing")
        sim.event_logger.log_event.assert_called_once_with(
            "assistant_speech",
            {"text": "Sure thing", "source": "pipecat_assistant"},
        )


class TestRunConversation:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_valueerror(self, tmp_path):
        sim = _make_simulator(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
                await sim.run_conversation()

    @pytest.mark.asyncio
    async def test_elevenlabs_error_returns_error_and_saves_log(self, tmp_path):
        """When _run_elevenlabs_conversation raises, we get 'error' and the log is saved."""
        sim = _make_simulator(tmp_path)
        sim.event_logger = MagicMock()

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            with patch.object(sim, "_run_elevenlabs_conversation", side_effect=RuntimeError("ws connect failed")):
                result = await sim.run_conversation()

        assert result == "error"
        assert sim._end_reason == "error"
        sim.event_logger.log_error.assert_called_once_with("ws connect failed")
        # Event log must always be saved, even on error
        sim.event_logger.save.assert_called_once()


class TestCheckEndCallViaApi:
    """Tests for the polling/backoff logic in _check_end_call_via_api."""

    @pytest.mark.asyncio
    async def test_finds_end_call_in_transcript(self, tmp_path):
        """Returns True when end_call tool is found in the transcript."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        end_call_result = _make_tool_result("end_call")
        turn_with_end_call = _make_turn(tool_results=[end_call_result])
        details = _make_conv_details(transcript=[turn_with_end_call])
        sim._client.conversational_ai.conversations.get.return_value = details

        with patch("eva.user_simulator.client.asyncio.sleep", new_callable=AsyncMock):
            result = await sim._check_end_call_via_api("conv-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_no_end_call_in_transcript(self, tmp_path):
        """Returns False when transcript exists but has no end_call."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        other_tool = _make_tool_result("lookup_flight")
        turn = _make_turn(tool_results=[other_tool])
        details = _make_conv_details(transcript=[turn])
        sim._client.conversational_ai.conversations.get.return_value = details

        with patch("eva.user_simulator.client.asyncio.sleep", new_callable=AsyncMock):
            result = await sim._check_end_call_via_api("conv-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_turn_with_no_tool_results(self, tmp_path):
        """Returns False when transcript turns have no tool_results."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        turn = _make_turn(tool_results=None)
        details = _make_conv_details(transcript=[turn])
        sim._client.conversational_ai.conversations.get.return_value = details

        with patch("eva.user_simulator.client.asyncio.sleep", new_callable=AsyncMock):
            result = await sim._check_end_call_via_api("conv-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_retries_when_transcript_empty_then_succeeds(self, tmp_path):
        """Polls with backoff when transcript is empty, succeeds on later attempt."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        empty_details = _make_conv_details(transcript=None, status="in-progress")
        end_call_result = _make_tool_result("end_call")
        populated_details = _make_conv_details(transcript=[_make_turn(tool_results=[end_call_result])])

        # First two calls return empty transcript, third returns populated
        sim._client.conversational_ai.conversations.get.side_effect = [
            empty_details,
            empty_details,
            populated_details,
        ]

        sleep_delays = []

        async def track_sleep(duration):
            sleep_delays.append(duration)

        with patch("eva.user_simulator.client.asyncio.sleep", side_effect=track_sleep):
            result = await sim._check_end_call_via_api("conv-123")

        assert result is True
        assert sim._client.conversational_ai.conversations.get.call_count == 3
        # Verify exponential backoff: 2.0, 4.0, 8.0
        assert sleep_delays == [2.0, 4.0, 8.0]

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, tmp_path):
        """Returns False after exhausting all retry attempts."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        empty_details = _make_conv_details(transcript=None, status="in-progress")
        sim._client.conversational_ai.conversations.get.return_value = empty_details

        with patch("eva.user_simulator.client.asyncio.sleep", new_callable=AsyncMock):
            result = await sim._check_end_call_via_api("conv-123")

        assert result is False
        # Should have tried exactly 5 times (max_attempts)
        assert sim._client.conversational_ai.conversations.get.call_count == 5

    @pytest.mark.asyncio
    async def test_backoff_caps_at_10_seconds(self, tmp_path):
        """Delay should not exceed 10 seconds even after many retries."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        empty_details = _make_conv_details(transcript=None, status="in-progress")
        sim._client.conversational_ai.conversations.get.return_value = empty_details

        sleep_delays = []

        async def track_sleep(duration):
            sleep_delays.append(duration)

        with patch("eva.user_simulator.client.asyncio.sleep", side_effect=track_sleep):
            await sim._check_end_call_via_api("conv-123")

        # Delays: 2.0, 4.0, 8.0, 10.0 (capped), 10.0 (capped)
        assert sleep_delays == [2.0, 4.0, 8.0, 10.0, 10.0]

    @pytest.mark.asyncio
    async def test_writes_conversation_details_to_file(self, tmp_path):
        """Conversation details are dumped to JSON for debugging."""
        sim = _make_simulator(tmp_path)
        sim._client = MagicMock()

        turn = _make_turn(tool_results=None)
        details = _make_conv_details(transcript=[turn])
        sim._client.conversational_ai.conversations.get.return_value = details

        with patch("eva.user_simulator.client.asyncio.sleep", new_callable=AsyncMock):
            await sim._check_end_call_via_api("conv-123")

        details_path = tmp_path / "elevenlabs_conversation_details.json"
        assert details_path.exists()


class TestKeepAliveTask:
    @pytest.mark.asyncio
    async def test_inactivity_timeout_ends_conversation(self, tmp_path):
        """Conversation ends after max consecutive keepalives without activity."""
        sim = _make_simulator(tmp_path)
        sim._conversation = MagicMock()
        sim._conversation.register_user_activity = MagicMock()
        sim._audio_interface = MagicMock()
        sim._audio_interface._assistant_audio_active = False
        sim._max_consecutive_keepalives = 3

        call_count = 0

        async def fake_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 20:
                sim._conversation_done.set()

        async def fake_run_in_executor(*args):
            return None

        with patch("eva.user_simulator.client.asyncio.sleep", side_effect=fake_sleep):
            with patch("eva.user_simulator.client.asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = fake_run_in_executor
                await sim._keep_alive_task()

        assert sim._conversation_done.is_set()
        assert sim._end_reason == "inactivity_timeout"
        assert sim._consecutive_keepalive_count >= 3

    @pytest.mark.asyncio
    async def test_active_audio_resets_inactivity(self, tmp_path):
        """Assistant audio activity prevents inactivity timeout."""
        sim = _make_simulator(tmp_path)
        sim._conversation = MagicMock()
        sim._conversation.register_user_activity = MagicMock()
        sim._audio_interface = MagicMock()
        sim._max_consecutive_keepalives = 3

        iteration = 0

        async def fake_sleep(duration):
            nonlocal iteration
            iteration += 1
            # Simulate assistant speaking on every other iteration so counter
            # never reaches 3 consecutive keepalives (resets at 2, 4, 6, ...)
            if iteration % 2 == 0:
                sim._audio_interface._assistant_audio_active = True
            else:
                sim._audio_interface._assistant_audio_active = False
            # End via conversation_done after enough iterations
            if iteration >= 10:
                sim._conversation_done.set()

        async def fake_run_in_executor(*args):
            return None

        with patch("eva.user_simulator.client.asyncio.sleep", side_effect=fake_sleep):
            with patch("eva.user_simulator.client.asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = fake_run_in_executor
                await sim._keep_alive_task()

        # Should have ended via conversation_done, NOT inactivity_timeout
        assert sim._end_reason != "inactivity_timeout"

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim._conversation = MagicMock()

        with patch("eva.user_simulator.client.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await sim._keep_alive_task()


class TestRecordAndRetrieveAudio:
    """Test the full record → retrieve flow with interleaved audio."""

    def test_interleaved_audio_preserved_in_order(self, tmp_path):
        sim = _make_simulator(tmp_path)
        sim._record_audio("user", b"\x01\x02")
        sim._record_audio("assistant", b"\xaa")
        sim._record_audio("user", b"\x03")
        sim._record_audio("assistant", b"\xbb\xcc")

        user_audio, assistant_audio = sim.get_recorded_audio()
        assert user_audio == b"\x01\x02\x03"
        assert assistant_audio == b"\xaa\xbb\xcc"
