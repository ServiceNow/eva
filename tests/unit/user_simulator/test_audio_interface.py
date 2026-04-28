"""Unit tests for BotToBotAudioInterface.

Focuses on: PCM→μ-law conversion correctness, silence detection state machine,
_receive_from_assistant message dispatch, audio state transitions, and
WebSocket lifecycle.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eva.user_simulator.audio_interface import (
    ASSISTANT_SAMPLE_RATE,
    SEND_CHUNK_SIZE_PCM,
    BotToBotAudioInterface,
)


def _make_interface(**overrides) -> BotToBotAudioInterface:
    """Create a BotToBotAudioInterface with defaults for testing."""
    defaults = {
        "websocket_uri": "ws://localhost:9999",
        "conversation_id": "test-conv-123",
        "record_callback": None,
        "event_logger": None,
        "conversation_done_callback": None,
    }
    defaults.update(overrides)
    return BotToBotAudioInterface(**defaults)


class TestConvertPcmToMulaw:
    """PCM 16kHz 16-bit → μ-law 8kHz conversion is the audio backbone."""

    def test_20ms_chunk_produces_correct_output_size(self):
        """640 bytes PCM (20ms @ 16kHz) → 160 bytes μ-law (20ms @ 8kHz)."""
        pcm_20ms = b"\x00" * SEND_CHUNK_SIZE_PCM  # 640 bytes
        result = BotToBotAudioInterface._convert_pcm_to_mulaw(pcm_20ms)
        expected_mulaw_len = int(ASSISTANT_SAMPLE_RATE * 0.02)  # 160
        assert len(result) == expected_mulaw_len

    def test_non_silence_audio_differs_from_silence(self):
        """Actual audio data should produce different μ-law than silence."""
        silence = b"\x00" * SEND_CHUNK_SIZE_PCM
        # Sawtooth-ish pattern (loud enough to differ from silence in μ-law)
        loud = bytes([(i * 50) % 256 for i in range(SEND_CHUNK_SIZE_PCM)])
        mulaw_silence = BotToBotAudioInterface._convert_pcm_to_mulaw(silence)
        mulaw_loud = BotToBotAudioInterface._convert_pcm_to_mulaw(loud)
        assert mulaw_silence != mulaw_loud

    def test_odd_size_input_does_not_crash(self):
        """Gracefully handles misaligned input (odd byte count)."""
        # 3 bytes is not sample-aligned (16-bit = 2 bytes per sample)
        # audioop may truncate or error; we just want no crash
        result = BotToBotAudioInterface._convert_pcm_to_mulaw(b"\x01\x02\x03")
        assert isinstance(result, bytes)


class TestSilenceDetectionStateMachine:
    """_should_send_assistant_silence and _should_send_user_silence"""

    def test_assistant_silence_when_user_ended_waiting_for_response(self):
        iface = _make_interface()
        iface._user_audio_ended_time = 100.0
        assert iface._should_send_assistant_silence() is True

    def test_no_assistant_silence_during_active_speech(self):
        """Neither party speaking should suppress silence if someone is active."""
        iface = _make_interface()
        iface._user_audio_ended_time = 100.0
        # User still actively speaking
        iface._user_audio_active = True
        assert iface._should_send_assistant_silence() is False
        # Assistant actively speaking
        iface._user_audio_active = False
        iface._assistant_audio_active = True
        assert iface._should_send_assistant_silence() is False

    def test_interruption_assistant_silence_based_on_who_ended_last(self):
        """When both ended (interruption), silence type follows who ended last."""
        iface = _make_interface()
        # User ended after assistant → send assistant silence (waiting for assistant)
        iface._user_audio_ended_time = 200.0
        iface._assistant_audio_ended_time = 100.0
        assert iface._should_send_assistant_silence() is True
        assert iface._should_send_user_silence() is False

    def test_user_silence_when_assistant_ended_waiting_for_user(self):
        iface = _make_interface()
        iface._assistant_audio_ended_time = 100.0
        assert iface._should_send_user_silence() is True

    def test_interruption_user_silence_based_on_who_ended_last(self):
        """When both ended (interruption), user silence if assistant ended later."""
        iface = _make_interface()
        iface._assistant_audio_ended_time = 200.0
        iface._user_audio_ended_time = 100.0
        assert iface._should_send_user_silence() is True
        assert iface._should_send_assistant_silence() is False

    def test_idle_state_no_silence_sent(self):
        """When nobody has spoken yet, no silence should be sent."""
        iface = _make_interface()
        assert iface._should_send_assistant_silence() is False
        assert iface._should_send_user_silence() is False


class TestAudioStateTransitions:
    """Test that audio start/end callbacks correctly update state and timestamps."""

    @pytest.mark.asyncio
    async def test_user_start_clears_assistant_ended_time(self):
        """When user starts speaking, we stop waiting for assistant."""
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface._assistant_audio_ended_time = 50.0  # Was waiting for user

        with patch("eva.user_simulator.audio_interface.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 100.0
            await iface._on_user_audio_start()

        assert iface._user_audio_active is True
        assert iface._user_audio_ended_time is None
        assert iface._assistant_audio_ended_time is None  # Cleared!

    def test_assistant_start_clears_user_ended_time(self):
        """When assistant starts speaking, we stop waiting for user."""
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface._user_audio_ended_time = 50.0

        with patch("eva.user_simulator.audio_interface.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 100.0
            iface._on_assistant_audio_start()

        assert iface._assistant_audio_active is True
        assert iface._user_audio_ended_time is None  # Cleared!

    @pytest.mark.asyncio
    async def test_user_end_records_timestamp(self):
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface._user_audio_active = True

        await iface._on_user_audio_end(150.0)

        assert iface._user_audio_active is False
        assert iface._user_audio_ended_time == 150.0
        event_logger.log_audio_end.assert_called_once_with("elevenlabs_user")

    @pytest.mark.asyncio
    async def test_assistant_end_records_timestamp(self):
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface._assistant_audio_active = True

        with patch("eva.user_simulator.audio_interface.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 200.0
            await iface._on_assistant_audio_end()

        assert iface._assistant_audio_active is False
        assert iface._assistant_audio_ended_time == 200.0


class TestReceiveFromAssistant:
    """Test _receive_from_assistant WebSocket message dispatch."""

    @pytest.mark.asyncio
    async def test_media_message_buffers_audio_and_triggers_start(self):
        """Media event decodes base64, buffers audio, and fires audio_start."""
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface.running = True

        raw_audio = b"\xff" * 160
        payload = base64.b64encode(raw_audio).decode("utf-8")
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        # Mock websocket as an async iterator yielding one message then stopping
        async def ws_messages():
            yield message
            iface.running = False  # Stop after one message

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: ws_messages()
        iface.websocket = mock_ws

        with patch("eva.user_simulator.audio_interface.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 100.0
            await iface._receive_from_assistant()

        # Audio should be in the buffer
        assert iface.audio_buffer.qsize() == 1
        buffered = await iface.audio_buffer.get()
        assert buffered == raw_audio
        # _on_assistant_audio_start was called (audio_start logged)
        # Note: _assistant_audio_active is reset in the finally block on disconnect,
        # so we verify the start event was logged instead
        event_logger.log_audio_start.assert_called_once_with("framework_agent")

    @pytest.mark.asyncio
    async def test_empty_payload_ignored(self):
        """Media event with empty payload should not buffer anything."""
        iface = _make_interface()
        iface.running = True

        message = json.dumps({"event": "media", "media": {"payload": ""}})

        async def ws_messages():
            yield message
            iface.running = False

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: ws_messages()
        iface.websocket = mock_ws

        await iface._receive_from_assistant()

        assert iface.audio_buffer.qsize() == 0
        assert iface._assistant_audio_active is False

    @pytest.mark.asyncio
    async def test_malformed_json_skipped(self):
        """Non-JSON messages should be silently skipped."""
        iface = _make_interface()
        iface.running = True

        async def ws_messages():
            yield "not json at all"
            yield "}{also bad"
            iface.running = False

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: ws_messages()
        iface.websocket = mock_ws

        await iface._receive_from_assistant()

        assert iface.audio_buffer.qsize() == 0

    @pytest.mark.asyncio
    async def test_disconnect_signals_conversation_end(self):
        """WebSocket close during active conversation signals elevenlabs_disconnect."""
        callback = MagicMock()
        iface = _make_interface(conversation_done_callback=callback)
        iface.running = True

        # Empty iterator — simulates immediate disconnect
        async def ws_messages():
            return
            yield  # Make it an async generator

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: ws_messages()
        iface.websocket = mock_ws

        await iface._receive_from_assistant()

        callback.assert_called_once_with("elevenlabs_disconnect")

    @pytest.mark.asyncio
    async def test_disconnect_closes_active_assistant_audio(self):
        """If assistant was speaking when WS closes, audio_end is logged."""
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface.running = True
        iface._assistant_audio_active = True

        async def ws_messages():
            return
            yield

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: ws_messages()
        iface.websocket = mock_ws

        await iface._receive_from_assistant()

        assert iface._assistant_audio_active is False
        event_logger.log_audio_end.assert_called_once_with("framework_agent")


class TestSendAudioFrame:
    """Test guard conditions and message format for _send_audio_frame."""

    @pytest.mark.asyncio
    async def test_sends_correct_json_structure(self):
        from websockets.protocol import State as WebSocketState

        iface = _make_interface()
        mock_ws = AsyncMock()
        mock_ws.state = WebSocketState.OPEN
        iface.websocket = mock_ws

        result = await iface._send_audio_frame(b"\xff\x00\xab")
        assert result is True

        sent_msg = json.loads(mock_ws.send.call_args[0][0])
        assert sent_msg["event"] == "media"
        assert sent_msg["conversation_id"] == "test-conv-123"
        # Verify payload is valid base64
        decoded = base64.b64decode(sent_msg["media"]["payload"])
        assert decoded == b"\xff\x00\xab"

    @pytest.mark.asyncio
    async def test_guards_prevent_sending(self):
        """No websocket, empty data, or stopping state should all return False."""
        iface = _make_interface()

        # No websocket
        assert await iface._send_audio_frame(b"\xff") is False

        # Empty data
        iface.websocket = MagicMock()
        assert await iface._send_audio_frame(b"") is False

        # Stopping
        iface._stopping = True
        assert await iface._send_audio_frame(b"\xff") is False


class TestStopCallback:
    def test_stop_signals_session_ended(self):
        """stop() should signal conversation_done_callback with 'session_ended'."""
        callback = MagicMock()
        iface = _make_interface(conversation_done_callback=callback)
        iface.running = True
        iface.stop()
        assert iface.running is False
        callback.assert_called_once_with("session_ended")

    def test_output_rejected_after_stop(self):
        """Audio output should be silently dropped once running=False."""
        iface = _make_interface()
        iface.running = True
        iface.stop()
        iface.output(b"\x01\x02")
        assert iface.send_queue.qsize() == 0


class TestStopAsync:
    @pytest.mark.asyncio
    async def test_cleans_up_tasks_and_websocket(self):
        iface = _make_interface()
        mock_ws = AsyncMock()
        iface.websocket = mock_ws
        iface.running = True

        async def cancelled_coro():
            raise asyncio.CancelledError()

        iface.receive_task = asyncio.ensure_future(cancelled_coro())
        iface.send_task = asyncio.ensure_future(cancelled_coro())
        iface.input_stream_task = asyncio.ensure_future(cancelled_coro())
        await asyncio.sleep(0)

        await iface.stop_async()

        assert iface.running is False
        assert iface._stopping is True
        assert iface.websocket is None
        mock_ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_no_websocket_or_tasks(self):
        """Graceful when nothing was started."""
        iface = _make_interface()
        await iface.stop_async()
        assert iface._stopping is True
