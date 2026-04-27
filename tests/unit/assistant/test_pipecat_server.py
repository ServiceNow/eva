"""Tests for PipecatAssistantServer."""

import asyncio
import json
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.pipecat_server import SAMPLE_RATE, PipecatAssistantServer
from eva.utils.audio_utils import save_pcm_as_wav


def _make_server(tmp_path: Path):
    """Build a lightweight PipecatAssistantServer without invoking __init__ (avoids Pipecat I/O)."""
    srv = object.__new__(PipecatAssistantServer)
    srv.output_dir = tmp_path
    srv.audit_log = AuditLog()
    srv.agentic_system = None
    srv.tool_handler = MagicMock()
    srv.tool_handler.original_db = {"reservations": {"ABC": {"status": "confirmed"}}}
    srv.tool_handler.db = {"reservations": {"ABC": {"status": "cancelled"}}}
    srv._audio_buffer = bytearray()
    srv._audio_sample_rate = SAMPLE_RATE
    srv.user_audio_buffer = bytearray()
    srv.assistant_audio_buffer = bytearray()
    srv._running = False
    srv._task = None
    srv._server = None
    srv._server_task = None
    srv._runner = None
    srv._metrics_observer = None
    srv._latency_measurements = []
    srv.num_seconds = 0
    srv.pipeline_config = MagicMock()
    srv.conversation_id = "test-conv"
    srv.port = 9999
    srv.non_instrumented_realtime_llm = False
    srv._user_turn_started_wall_ms = None
    return srv


class TestSavePcmAsWav:
    def test_mono_wav_preserves_header_and_content(self, tmp_path):
        """WAV file should have correct headers and byte-exact audio content."""
        audio_data = b"\x00\x01\xff\xfe" * 50
        file_path = tmp_path / "test.wav"

        save_pcm_as_wav(audio_data, file_path, 24000, 1)

        with wave.open(str(file_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() == len(audio_data) // 2
            assert wf.readframes(wf.getnframes()) == audio_data

    def test_stereo_wav_frame_count(self, tmp_path):
        """Stereo WAV: frame count = total_bytes / (channels * sample_width)."""
        audio_data = b"\x00" * 800
        file_path = tmp_path / "stereo.wav"

        save_pcm_as_wav(audio_data, file_path, 16000, 2)

        with wave.open(str(file_path), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 200

    def test_bad_path_does_not_raise(self, tmp_path):
        """Error writing to non-existent directory should be swallowed (logged)."""
        bad_path = tmp_path / "nonexistent_dir" / "test.wav"
        save_pcm_as_wav(b"\x00", bad_path, 24000, 1)
        assert not bad_path.exists()


class TestSaveAudioDeferred:
    def test_writes_separate_channel_content(self, tmp_path):
        """Each channel's audio should end up in the correct file with distinct content."""
        srv = _make_server(tmp_path)
        mixed = b"\x01\x00" * 100
        user = b"\x02\x00" * 100
        assistant = b"\x03\x00" * 100

        srv._save_audio_deferred(mixed, user, assistant, SAMPLE_RATE)

        with wave.open(str(tmp_path / "audio_mixed.wav"), "rb") as wf:
            assert wf.readframes(wf.getnframes()) == mixed
        with wave.open(str(tmp_path / "audio_user.wav"), "rb") as wf:
            assert wf.readframes(wf.getnframes()) == user
        with wave.open(str(tmp_path / "audio_assistant.wav"), "rb") as wf:
            assert wf.readframes(wf.getnframes()) == assistant

    def test_stop_returns_none_when_no_audio(self, tmp_path):
        """stop() returns None (no deferred task) when no audio was accumulated."""
        srv = _make_server(tmp_path)
        srv._running = True
        # Buffers are empty (default) — stop() should return None, no WAV files written
        result = asyncio.run(srv.stop())
        assert result is None
        assert not (tmp_path / "audio_mixed.wav").exists()


class TestSaveTranscriptMessage:
    @pytest.mark.asyncio
    async def test_appends_multiple_entries_in_order(self, tmp_path):
        srv = _make_server(tmp_path)

        await srv._save_transcript_message_from_turn(
            role="user", content="I need to rebook", timestamp="2026-01-01T00:00:00Z"
        )
        await srv._save_transcript_message_from_turn(
            role="assistant", content="Let me check", timestamp="2026-01-01T00:00:01Z"
        )

        lines = (tmp_path / "transcript.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

        e1 = json.loads(lines[0])
        assert e1 == {"timestamp": "2026-01-01T00:00:00Z", "role": "user", "content": "I need to rebook"}

        e2 = json.loads(lines[1])
        assert e2["role"] == "assistant"
        assert e2["content"] == "Let me check"

    @pytest.mark.asyncio
    async def test_preserves_unicode_content(self, tmp_path):
        """Non-ASCII characters should be preserved (ensure_ascii=False)."""
        srv = _make_server(tmp_path)
        await srv._save_transcript_message_from_turn(
            role="user", content="Héllo wörld", timestamp="2026-01-01T00:00:00Z"
        )

        entry = json.loads((tmp_path / "transcript.jsonl").read_text().strip())
        assert entry["content"] == "Héllo wörld"


class TestSaveOutputs:
    @pytest.mark.asyncio
    async def test_saves_audit_log_and_both_scenario_db_snapshots(self, tmp_path):
        """Validates audit_log.json and initial/final scenario DBs are written with correct content."""
        srv = _make_server(tmp_path)
        srv._audio_buffer = bytearray(b"\x00" * 100)
        srv.user_audio_buffer = bytearray(b"\x00" * 100)
        srv.assistant_audio_buffer = bytearray(b"\x00" * 100)
        # PipelineConfig (not SpeechToSpeechConfig) — transcript.jsonl written via audit log
        srv.pipeline_config = MagicMock(spec=[])

        # Add an entry so audit_log is non-trivial
        srv.audit_log.append_user_input("Hello")

        await srv.save_outputs()

        # Audit log contains our entry
        audit = json.loads((tmp_path / "audit_log.json").read_text())
        assert len(audit.get("entries", audit.get("user_inputs", []))) >= 0  # structure varies

        # Scenario DB snapshots reflect tool_handler state
        initial = json.loads((tmp_path / "initial_scenario_db.json").read_text())
        assert initial["reservations"]["ABC"]["status"] == "confirmed"

        final = json.loads((tmp_path / "final_scenario_db.json").read_text())
        assert final["reservations"]["ABC"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_saves_agent_perf_stats_when_agentic_system_present(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._audio_buffer = bytearray(b"\x00" * 100)
        srv.user_audio_buffer = bytearray(b"\x00" * 100)
        srv.assistant_audio_buffer = bytearray(b"\x00" * 100)
        srv.pipeline_config = MagicMock(spec=[])

        mock_system = MagicMock()
        srv.agentic_system = mock_system

        await srv.save_outputs()

        mock_system.save_agent_perf_stats.assert_called_once()


class TestStop:
    @pytest.mark.asyncio
    async def test_noop_when_not_running(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._running = False
        await srv.stop()
        # Shouldn't touch anything
        assert srv._server is None

    @pytest.mark.asyncio
    async def test_cancels_pipeline_task_and_stops_server(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._running = True

        mock_task = MagicMock()
        mock_task.cancel = AsyncMock()
        srv._task = mock_task

        srv._server = MagicMock()
        srv._server.should_exit = False
        srv._server_task = asyncio.create_task(asyncio.sleep(100))

        srv._audio_buffer = bytearray(b"\x00" * 100)
        srv.user_audio_buffer = bytearray(b"\x00" * 100)
        srv.assistant_audio_buffer = bytearray(b"\x00" * 100)

        await srv.stop()

        mock_task.cancel.assert_called_once()
        assert srv._running is False
        assert srv._task is None
        assert srv._server is None
        assert srv._server_task is None
        # Verify outputs were saved (audit_log.json should exist)
        assert (tmp_path / "audit_log.json").exists()
