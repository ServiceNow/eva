"""Unit tests for the telephony bridge server."""

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import WebSocketDisconnect

from eva.assistant.telephony_bridge import (
    AudioSegment,
    BaseTelephonyTransport,
    TelephonyBridgeServer,
    TelephonyBridgeConfig,
    _SessionState,
)


class _FakeWebSocket:
    def __init__(self, messages: list[dict]):
        self._messages = iter(json.dumps(message) for message in messages)
        self.sent: list[dict] = []

    async def receive_text(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise WebSocketDisconnect() from exc

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class _MockTransport(BaseTelephonyTransport):
    def __init__(self, sip_uri: str, conversation_id: str, webhook_base_url: str):
        super().__init__(sip_uri, conversation_id, webhook_base_url)
        self.started = False
        self.stopped = False
        self.sent_audio: list[bytes] = []

    async def start(self) -> None:
        self.started = True
        await self.emit_audio(b"\xff" * 160)

    async def stop(self) -> None:
        self.stopped = True

    async def send_audio(self, audio_data: bytes) -> None:
        self.sent_audio.append(audio_data)
        await self.emit_audio(audio_data)


def _write_agent_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {
                        "id": "get_reservation",
                        "name": "Get reservation",
                        "description": "Look up a reservation",
                        "required_parameters": [{"name": "confirmation_number", "type": "string"}],
                    }
                ]
            }
        )
    )


def _write_scenario_db(path: Path) -> None:
    path.write_text(json.dumps({"reservations": {"ABC123": {"status": "confirmed"}}}))


def _make_bridge(
    tmp_path: Path,
    *,
    transport: _MockTransport | None = None,
    transcriber: object | None = None,
) -> tuple[TelephonyBridgeServer, _MockTransport]:
    agent_config_path = tmp_path / "agent.yaml"
    scenario_db_path = tmp_path / "scenario.json"
    _write_agent_config(agent_config_path)
    _write_scenario_db(scenario_db_path)

    transport_instance = transport or _MockTransport(
        sip_uri="sip:test@example.com",
        conversation_id="conv-1",
        webhook_base_url="https://example.com",
    )

    bridge = TelephonyBridgeServer(
        current_date_time="2026-01-01 00:00 UTC",
        bridge_config=TelephonyBridgeConfig(
            sip_uri="sip:test@example.com",
            webhook_base_url="https://example.com",
            stt="deepgram",
            stt_params={"api_key": "test-key", "model": "nova-2"},
        ),
        agent=MagicMock(tool_module_path="eva.assistant.tools.airline_tools"),
        agent_config_path=str(agent_config_path),
        scenario_db_path=str(scenario_db_path),
        output_dir=tmp_path / "output",
        port=9999,
        conversation_id="conv-1",
        transport_factory=lambda config, conversation_id: transport_instance,
        segment_transcriber=transcriber,
    )
    return bridge, transport_instance


class TestTelephonyBridgeServer:
    @pytest.mark.asyncio
    async def test_handles_websocket_audio_bridge(self, tmp_path: Path):
        transcriber = MagicMock()
        transcriber.transcribe = AsyncMock(return_value="")
        bridge, transport = _make_bridge(tmp_path, transcriber=transcriber)

        user_audio = b"\xff" * 160
        websocket = _FakeWebSocket(
            [
                {"event": "connected", "conversation_id": "conv-1"},
                {"event": "start", "conversation_id": "conv-1"},
                {
                    "event": "media",
                    "conversation_id": "conv-1",
                    "media": {"payload": base64.b64encode(user_audio).decode("utf-8")},
                },
                {"event": "stop", "conversation_id": "conv-1"},
            ]
        )

        await bridge._handle_session(websocket)

        assert transport.started is True
        assert transport.stopped is True
        assert transport.sent_audio == [user_audio]
        assert websocket.sent
        assert websocket.sent[0]["event"] == "media"
        assert bridge._session_state is not None
        assert bridge._session_state.user_audio
        assert bridge._session_state.assistant_audio

    @pytest.mark.asyncio
    async def test_save_outputs_writes_transcript_audit_audio_and_db_snapshots(self, tmp_path: Path):
        transcriber = MagicMock()
        transcriber.transcribe = AsyncMock(side_effect=["I need help", "Sure, let me check"])
        bridge, _transport = _make_bridge(tmp_path, transcriber=transcriber)

        bridge._session_state = _SessionState(started_at=datetime(2026, 1, 1, tzinfo=UTC))
        bridge._session_state.user_segments.append(
            AudioSegment(role="user", started_at=0.0, ended_at=0.1, audio=bytearray(b"\x00\x00" * 2400))
        )
        bridge._session_state.assistant_segments.append(
            AudioSegment(role="assistant", started_at=1.0, ended_at=1.1, audio=bytearray(b"\x01\x01" * 2400))
        )
        bridge._session_state.user_audio.extend(b"\x00\x00" * 2400)
        bridge._session_state.assistant_audio.extend(b"\x01\x01" * 2400)
        bridge._session_state.mixed_audio.extend(b"\x01\x00" * 2400)

        await bridge._save_outputs()

        transcript_lines = (bridge.output_dir / "transcript.jsonl").read_text().strip().splitlines()
        assert len(transcript_lines) == 2
        first_entry = json.loads(transcript_lines[0])
        second_entry = json.loads(transcript_lines[1])
        assert first_entry["role"] == "user"
        assert first_entry["content"] == "I need help"
        assert second_entry["role"] == "assistant"
        assert second_entry["content"] == "Sure, let me check"

        audit_log = json.loads((bridge.output_dir / "audit_log.json").read_text())
        transcript_types = [entry["message_type"] for entry in audit_log["transcript"]]
        assert transcript_types == ["user", "assistant"]

        assert (bridge.output_dir / "audio_user.wav").exists()
        assert (bridge.output_dir / "audio_assistant.wav").exists()
        assert (bridge.output_dir / "audio_mixed.wav").exists()

        initial_db = json.loads((bridge.output_dir / "initial_scenario_db.json").read_text())
        final_db = json.loads((bridge.output_dir / "final_scenario_db.json").read_text())
        assert initial_db["reservations"]["ABC123"]["status"] == "confirmed"
        assert final_db["reservations"]["ABC123"]["status"] == "confirmed"
