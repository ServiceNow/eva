"""Unit tests for BotToBotAudioBridge and ElevenLabsAudioInterface.

Tests are written against ElevenLabsAudioInterface (the concrete subclass) but
most exercise logic defined in BotToBotAudioBridge: PCM→μ-law conversion,
silence detection state machine, audio state transitions, and WebSocket lifecycle.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from eva.user_simulator.audio_bridge import (
    ASSISTANT_SAMPLE_RATE,
    SEND_CHUNK_SIZE_PCM,
    USER_TRAILING_NOISE_SNR_DB,
    USER_TRAILING_SILENCE_CHUNKS,
    ElevenLabsAudioInterface,
    _white_noise_pcm,
)


def _make_interface(**overrides) -> ElevenLabsAudioInterface:
    """Create a ElevenLabsAudioInterface with defaults for testing."""
    defaults = {
        "websocket_uri": "ws://localhost:9999",
        "conversation_id": "test-conv-123",
        "record_callback": None,
        "event_logger": None,
        "conversation_done_callback": None,
    }
    defaults.update(overrides)
    return ElevenLabsAudioInterface(**defaults)


class TestConvertPcmToMulaw:
    """PCM 16kHz 16-bit → μ-law 8kHz conversion is the audio backbone."""

    def test_20ms_chunk_produces_correct_output_size(self):
        """640 bytes PCM (20ms @ 16kHz) → 160 bytes μ-law (20ms @ 8kHz)."""
        pcm_20ms = b"\x00" * SEND_CHUNK_SIZE_PCM  # 640 bytes
        result = ElevenLabsAudioInterface._convert_pcm_to_mulaw(pcm_20ms)
        expected_mulaw_len = int(ASSISTANT_SAMPLE_RATE * 0.02)  # 160
        assert len(result) == expected_mulaw_len

    def test_non_silence_audio_differs_from_silence(self):
        """Actual audio data should produce different μ-law than silence."""
        silence = b"\x00" * SEND_CHUNK_SIZE_PCM
        # Sawtooth-ish pattern (loud enough to differ from silence in μ-law)
        loud = bytes([(i * 50) % 256 for i in range(SEND_CHUNK_SIZE_PCM)])
        mulaw_silence = ElevenLabsAudioInterface._convert_pcm_to_mulaw(silence)
        mulaw_loud = ElevenLabsAudioInterface._convert_pcm_to_mulaw(loud)
        assert mulaw_silence != mulaw_loud

    def test_odd_size_input_does_not_crash(self):
        """Gracefully handles misaligned input (odd byte count)."""
        # 3 bytes is not sample-aligned (16-bit = 2 bytes per sample)
        # audioop may truncate or error; we just want no crash
        result = ElevenLabsAudioInterface._convert_pcm_to_mulaw(b"\x01\x02\x03")
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

        with patch("eva.user_simulator.audio_bridge.asyncio.get_event_loop") as mock_loop:
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

        with patch("eva.user_simulator.audio_bridge.asyncio.get_event_loop") as mock_loop:
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
        args, _ = event_logger.log_audio_end.call_args
        assert args[0] == "simulated_user"
        assert isinstance(args[1], float)

    @pytest.mark.asyncio
    async def test_assistant_end_records_timestamp(self):
        event_logger = MagicMock()
        iface = _make_interface(event_logger=event_logger)
        iface._assistant_audio_active = True

        with patch("eva.user_simulator.audio_bridge.asyncio.get_event_loop") as mock_loop:
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

        with patch("eva.user_simulator.audio_bridge.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 100.0
            await iface._receive_from_assistant()

        # Audio should be in the buffer
        assert iface.audio_buffer.qsize() == 1
        buffered = await iface.audio_buffer.get()
        assert buffered == raw_audio
        # _on_assistant_audio_start was called (audio_start logged)
        # Note: _assistant_audio_active is reset in the finally block on disconnect,
        # so we verify the start event was logged instead
        event_logger.log_audio_start.assert_called_once_with("assistant")

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
    async def test_disconnect_reason_is_provider_configurable(self):
        callback = MagicMock()
        iface = _make_interface(
            conversation_done_callback=callback,
            disconnect_reason="assistant_disconnect",
        )
        iface.running = True

        async def ws_messages():
            return
            yield

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: ws_messages()
        iface.websocket = mock_ws

        await iface._receive_from_assistant()

        callback.assert_called_once_with("assistant_disconnect")

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
        event_logger.log_audio_end.assert_called_once_with("assistant")


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


class TestTrailingWhiteNoise:
    """Trailing white-noise block streamed to the assistant after each user turn.

    Only the wire receives white noise; the saved user_clean track records plain
    silence, so end-of-speech detection improves without polluting audio used by
    speech-fidelity metrics.
    """

    def test_noise_chunk_size_and_level(self):
        """Noise chunk matches the send size and sits at ~-45 dBFS (reads as silence to VAD)."""
        noise = _white_noise_pcm()
        assert len(noise) == SEND_CHUNK_SIZE_PCM
        arr = np.frombuffer(noise, dtype=np.int16).astype(np.float64)
        rms = float(np.sqrt((arr**2).mean()))
        expected_rms = 32768.0 * (10 ** (-USER_TRAILING_NOISE_SNR_DB / 20.0))
        assert rms == pytest.approx(expected_rms, rel=0.1)

    def test_noise_is_deterministic_and_not_silence(self):
        """Fixed seed → identical cached buffer; and it differs from digital silence."""
        assert _white_noise_pcm() == _white_noise_pcm()
        assert _white_noise_pcm() != b"\x00" * SEND_CHUNK_SIZE_PCM

    @pytest.mark.asyncio
    async def test_user_end_schedules_trailing_noise(self):
        """Ending a user turn queues a fixed block of trailing noise frames."""
        iface = _make_interface()
        iface._user_audio_active = True
        await iface._on_user_audio_end(150.0)
        assert iface._pending_trailing_noise_chunks == USER_TRAILING_SILENCE_CHUNKS

    @pytest.mark.asyncio
    async def test_new_user_turn_cancels_pending_noise(self):
        """A new user turn supersedes any trailing noise still queued."""
        iface = _make_interface()
        iface._pending_trailing_noise_chunks = 42
        with patch("eva.user_simulator.audio_bridge.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 100.0
            await iface._on_user_audio_start()
        assert iface._pending_trailing_noise_chunks == 0

    def test_assistant_response_cancels_pending_noise(self):
        """Once the assistant starts responding, trailing noise stops (no bleed into its turn)."""
        iface = _make_interface(event_logger=MagicMock())
        iface._pending_trailing_noise_chunks = 42
        with patch("eva.user_simulator.audio_bridge.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 100.0
            iface._on_assistant_audio_start()
        assert iface._pending_trailing_noise_chunks == 0

    @pytest.mark.asyncio
    async def test_wire_gets_noise(self):
        """_send_white_noise_frame puts non-silent μ-law on the wire."""
        iface = _make_interface()
        sent: list[bytes] = []
        iface._send_audio_frame = AsyncMock(side_effect=lambda frame: sent.append(frame) or True)
        assert await iface._send_white_noise_frame() is True
        silence_mulaw = ElevenLabsAudioInterface._convert_pcm_to_mulaw(b"\x00" * SEND_CHUNK_SIZE_PCM)
        assert sent and sent[0] != silence_mulaw

    @pytest.mark.asyncio
    async def test_drain_records_silence_not_noise_into_user_clean(self):
        """Draining the noise block sends noise to the wire but records only silence."""
        recorded: list[tuple[str, bytes]] = []
        iface = _make_interface(record_callback=lambda source, data: recorded.append((source, data)))
        iface.websocket = MagicMock()
        iface.running = True
        # One frame queued; user finished and assistant has not spoken yet.
        iface._pending_trailing_noise_chunks = 1
        iface._user_audio_active = False
        iface._user_audio_ended_time = 10.0

        wire: list[bytes] = []
        iface._send_audio_frame = AsyncMock(side_effect=lambda frame: wire.append(frame) or True)

        async def stop_after_one_frame():
            await asyncio.sleep(0.05)
            iface.running = False

        await asyncio.gather(iface._send_to_assistant(), stop_after_one_frame())

        # Exactly the scheduled frame drained.
        assert iface._pending_trailing_noise_chunks == 0
        # Wire carried noise, not silence.
        silence_mulaw = ElevenLabsAudioInterface._convert_pcm_to_mulaw(b"\x00" * SEND_CHUNK_SIZE_PCM)
        assert wire and any(frame != silence_mulaw for frame in wire)
        # Every recorded user_clean frame is pure silence (never the noise buffer).
        user_clean = [data for source, data in recorded if source == "user_clean"]
        assert user_clean, "expected user_clean frames recorded for alignment"
        assert all(data == b"\x00" * SEND_CHUNK_SIZE_PCM for data in user_clean)
        assert _white_noise_pcm() not in user_clean


class TestMidUtteranceUnderrun:
    """A stall inside a user utterance (a barge-in's opener played out before its reply) stays real time."""

    @pytest.mark.asyncio
    async def test_stall_is_filled_with_silence_without_ending_or_bursting(self, monkeypatch):
        monkeypatch.setattr("eva.user_simulator.audio_bridge.USER_END_DETECTION_DELAY_INTERVALS", 2)
        recorded: list[bytes] = []
        iface = _make_interface(record_callback=lambda source, data: recorded.append(data))
        iface.websocket = MagicMock()
        iface.running = True
        loop = asyncio.get_running_loop()
        wire: list[tuple[float, bytes]] = []
        iface._send_audio_frame = AsyncMock(side_effect=lambda frame: wire.append((loop.time(), frame)) or True)
        iface._on_user_audio_end = AsyncMock(side_effect=lambda now: setattr(iface, "_user_audio_active", False))
        loud = b"\x00\x40" * (SEND_CHUNK_SIZE_PCM // 2)

        async def caller():
            iface.output(loud * 4 + loud[: SEND_CHUNK_SIZE_PCM // 2])  # 90ms opener, ending mid-frame
            await asyncio.sleep(0.4)  # the reply is still being generated
            assert iface._user_audio_active  # the stall did not end the utterance
            iface.output(loud * 5)
            iface.notify_user_utterance_complete()
            await asyncio.sleep(0.3)
            iface.running = False

        await asyncio.gather(iface._send_to_assistant(), caller())

        start = wire[0][0]
        # Never ahead of real time: the late reply is not burst out to catch up the schedule.
        for index, (sent_at, _) in enumerate(wire):
            assert index * 0.02 <= sent_at - start + 0.03
        silence = ElevenLabsAudioInterface._convert_pcm_to_mulaw(b"\x00" * SEND_CHUNK_SIZE_PCM)
        loud_mulaw = ElevenLabsAudioInterface._convert_pcm_to_mulaw(loud)
        frames = [frame for _, frame in wire]
        first_reply = frames.index(loud_mulaw, 5)
        assert frames[5:first_reply].count(silence) >= 5  # the stall went out as silence
        assert frames[4] not in (silence, loud_mulaw)  # the opener's last half-frame, padded
        iface._on_user_audio_end.assert_awaited_once()  # one utterance, ended once it was complete
        assert b"".join(recorded).count(b"\x00" * SEND_CHUNK_SIZE_PCM) >= 5  # the gap is in user_clean

    @pytest.mark.asyncio
    async def test_tts_streaming_a_little_behind_real_time_is_not_chopped(self):
        iface = _make_interface()
        iface.websocket = MagicMock()
        iface.running = True
        wire: list[bytes] = []
        iface._send_audio_frame = AsyncMock(side_effect=lambda frame: wire.append(frame) or True)
        loud = b"\x00\x40" * (SEND_CHUNK_SIZE_PCM // 2)

        async def caller():
            piece = (loud * 2)[:400]  # 12.5ms of speech every 20ms: the queue keeps running dry briefly
            for _ in range(32):
                iface.output(piece)
                await asyncio.sleep(0.02)
            iface.notify_user_utterance_complete()
            await asyncio.sleep(0.1)
            iface.running = False

        await asyncio.gather(iface._send_to_assistant(), caller())

        loud_mulaw = ElevenLabsAudioInterface._convert_pcm_to_mulaw(loud)
        assert len(wire) >= 19
        assert [frame for frame in wire if frame != loud_mulaw] == []  # no silence padded into the speech

    @pytest.mark.asyncio
    async def test_tts_streaming_behind_real_time_after_a_stall_is_not_chopped(self):
        iface = _make_interface()
        iface.websocket = MagicMock()
        iface.running = True
        wire: list[bytes] = []
        iface._send_audio_frame = AsyncMock(side_effect=lambda frame: wire.append(frame) or True)
        loud = b"\x00\x40" * (SEND_CHUNK_SIZE_PCM // 2)

        async def caller():
            iface.output(loud * 4)  # the opener
            await asyncio.sleep(0.3)  # the reply is still being generated: padded with silence
            piece = (loud * 2)[:400]  # then its TTS trickles in a little behind real time
            for _ in range(32):
                iface.output(piece)
                await asyncio.sleep(0.02)
            iface.notify_user_utterance_complete()
            await asyncio.sleep(0.1)
            iface.running = False

        await asyncio.gather(iface._send_to_assistant(), caller())

        loud_mulaw = ElevenLabsAudioInterface._convert_pcm_to_mulaw(loud)
        reply = wire[wire.index(loud_mulaw, 4) :]
        assert len(reply) >= 15
        # The stall's grace is not carried into the reply: its brief starves are held, not padded.
        assert [frame for frame in reply if frame != loud_mulaw] == []


class TestInputTick:
    """The assistant's audio reaches input_callback within one tick of arriving, at the tick's rate."""

    @pytest.mark.asyncio
    async def test_short_tick_forwards_audio_as_it_arrives_without_drift(self):
        iface = _make_interface(input_chunk_duration=0.02)
        iface.running = True
        iface._send_silence_frame = AsyncMock()
        loop = asyncio.get_running_loop()
        heard: list[tuple[float, bytes]] = []
        iface.input_callback = lambda chunk: heard.append((loop.time(), chunk))
        speech = b"\x10" * 320  # 40ms of mu-law, as the assistant's transport sends it

        async def assistant():
            await asyncio.sleep(0.1)
            sent_at = loop.time()
            iface._on_assistant_audio_start()
            iface._assistant_audio_ended_time = loop.time()
            await iface.audio_buffer.put(speech)
            await asyncio.sleep(0.4)
            iface.running = False
            return sent_at

        _, sent_at = await asyncio.gather(iface._continuous_input_stream(), assistant())

        arrived = next(at for at, chunk in heard if chunk == speech)
        assert arrived - sent_at < 0.03  # not held for a 250ms batch

    @pytest.mark.asyncio
    async def test_short_tick_keeps_its_rate_without_drift(self):
        iface = _make_interface(input_chunk_duration=0.02)
        iface.running = True
        iface._send_silence_frame = AsyncMock()
        ticks: list[bytes] = []
        iface.input_callback = ticks.append

        async def stop():
            await asyncio.sleep(0.4)
            iface.running = False

        await asyncio.gather(iface._continuous_input_stream(), stop())

        # Absolute schedule: ~20 ticks over 0.4s, not fewer from accumulated overruns
        assert len(ticks) >= 18


class TestAssistantAudioEndCatchup:
    """Catch-up silence after the assistant stops is never interleaved with the caller's speech."""

    @pytest.mark.asyncio
    async def test_no_catchup_silence_while_the_caller_is_talking(self):
        iface = _make_interface()
        iface._send_silence_frame = AsyncMock()
        iface._assistant_audio_active = True
        iface._user_audio_active = True  # a barge-in's opener is playing as the agent stops

        await iface._on_assistant_audio_end()

        iface._send_silence_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_catchup_silence_when_the_caller_is_silent(self, monkeypatch):
        monkeypatch.setattr("eva.user_simulator.audio_bridge.SEND_CHUNK_DURATION_MS", 0)
        iface = _make_interface()
        iface._send_silence_frame = AsyncMock()
        iface._assistant_audio_active = True

        await iface._on_assistant_audio_end()

        assert iface._send_silence_frame.await_count == 10


class TestPlayBackChannel:
    """A back-channel reaches the assistant as audio only: it is never a caller utterance or an interruption."""

    @staticmethod
    def _bridge(perturbator=None):
        from websockets.protocol import State as WebSocketState

        event_logger = MagicMock()
        record_callback = MagicMock()
        iface = _make_interface(event_logger=event_logger, record_callback=record_callback)
        iface._perturbator = perturbator
        iface.websocket = AsyncMock(state=WebSocketState.OPEN)
        iface.running = True
        return iface, event_logger, record_callback

    @staticmethod
    async def _play(iface, audio: bytes):
        sender = asyncio.create_task(iface._send_to_assistant())
        try:
            return await asyncio.wait_for(iface.play_back_channel(audio), timeout=2)
        finally:
            iface.running = False
            sender.cancel()

    @staticmethod
    def _sent(iface) -> list[dict]:
        return [json.loads(call.args[0]) for call in iface.websocket.send.call_args_list]

    @pytest.mark.asyncio
    async def test_sends_only_media_frames_and_no_speech_signals(self):
        iface, event_logger, _ = self._bridge()

        await self._play(iface, b"\x00\x40" * SEND_CHUNK_SIZE_PCM)  # 2 frames

        assert [message["event"] for message in self._sent(iface)] == ["media", "media"]
        event_logger.log_audio_start.assert_not_called()
        event_logger.log_audio_end.assert_not_called()

    @pytest.mark.asyncio
    async def test_leaves_utterance_state_untouched(self):
        iface, _, record_callback = self._bridge()

        started_at, ended_at = await self._play(iface, b"\x00\x40" * SEND_CHUNK_SIZE_PCM)

        assert ended_at > started_at
        assert iface.user_audio_starts == 0
        assert iface._user_audio_active is False
        assert iface._user_audio_ended_time is None
        assert iface._pending_trailing_noise_chunks == 0
        assert iface._back_channel_active is False
        # Silence in the clean user track, which holds what the caller's turns said.
        assert {call.args for call in record_callback.call_args_list} == {("user_clean", bytes(SEND_CHUNK_SIZE_PCM))}

    @pytest.mark.asyncio
    async def test_partial_frame_is_padded_before_the_perturbator(self):
        perturbator = MagicMock(has_ambient_noise=False)
        perturbator.apply.side_effect = lambda pcm: pcm
        iface, _, _ = self._bridge(perturbator)

        await self._play(iface, b"\x00\x40" * (SEND_CHUNK_SIZE_PCM // 4))  # half a frame

        perturbator.apply.assert_called_once_with(
            (b"\x00\x40" * (SEND_CHUNK_SIZE_PCM // 4)).ljust(SEND_CHUNK_SIZE_PCM, b"\x00")
        )

    @pytest.mark.asyncio
    async def test_caller_counts_as_playing_until_it_finishes(self):
        iface, _, _ = self._bridge()
        during: list[bool] = []
        send = iface._send_audio_frame

        async def record(frame):
            during.append(iface.is_caller_playing())
            return await send(frame)

        iface._send_audio_frame = record
        assert not iface.is_caller_playing()

        await self._play(iface, b"\x00\x40" * SEND_CHUNK_SIZE_PCM)

        assert during == [True, True]
        assert not iface.is_caller_playing()

    @pytest.mark.asyncio
    async def test_a_real_utterance_cuts_it_off(self):
        iface, _, _ = self._bridge()
        playing = asyncio.create_task(iface.play_back_channel(b"\x00\x40" * SEND_CHUNK_SIZE_PCM * 10))
        await asyncio.sleep(0)

        await iface._on_user_audio_start()

        await asyncio.wait_for(playing, timeout=1)
        assert iface._back_channel_audio == b""

    @pytest.mark.asyncio
    async def test_fails_when_the_connection_is_closed(self):
        iface, _, _ = self._bridge()
        iface._send_audio_frame = AsyncMock(return_value=False)

        with pytest.raises(RuntimeError, match="connection is closed"):
            await self._play(iface, b"\x00\x40" * SEND_CHUNK_SIZE_PCM)

        assert iface._back_channel_active is False
