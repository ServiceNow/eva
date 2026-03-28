"""Unit tests for the telephony bridge server."""

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import WebSocketDisconnect

from eva.assistant.telephony_bridge import BaseTelephonyTransport, TelephonyBridgeConfig, TelephonyBridgeServer, _SessionState


class _FakeObserver:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.user_chunks: list[tuple[bytes, float]] = []
        self.assistant_chunks: list[tuple[bytes, float]] = []
        self.finalize_reasons: list[str | None] = []
        self.transcript_entries: list[dict[str, str]] = []

    async def feed_user_audio(self, pcm_bytes: bytes, timestamp: float) -> None:
        self.user_chunks.append((pcm_bytes, timestamp))

    async def feed_assistant_audio(self, pcm_bytes: bytes, timestamp: float) -> None:
        self.assistant_chunks.append((pcm_bytes, timestamp))

    async def finalize(self, *, reason: str | None = None, details: dict | None = None) -> None:
        self.finalize_reasons.append(reason)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "pipecat_logs.jsonl").write_text("", encoding="utf-8")
        with open(self.output_dir / "response_latencies.json", "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "latencies": [0.9],
                    "mean": 0.9,
                    "max": 0.9,
                    "min": 0.9,
                    "count": 1,
                    "source": "vad_observer",
                },
                file_obj,
                indent=2,
            )

    def get_transcript_entries(self) -> list[dict[str, str]]:
        return list(self.transcript_entries)


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
        import asyncio
        self._disconnected_event = asyncio.Event()

    async def start(self) -> None:
        self.started = True
        await self.emit_audio(b"\xff" * 160)

    async def stop(self) -> None:
        self.stopped = True

    async def send_audio(self, audio_data: bytes) -> None:
        self.sent_audio.append(audio_data)
        await self.emit_audio(audio_data)


class _FakeAiohttpResponse:
    def __init__(self, *, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)


class _FakeAiohttpSession:
    def __init__(self, *, response: _FakeAiohttpResponse, capture: dict, **kwargs):
        self._response = response
        self._capture = capture
        self._capture["session_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str):
        self._capture["url"] = url
        return self._response


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
    observer: _FakeObserver | None = None,
) -> tuple[TelephonyBridgeServer, _MockTransport, _FakeObserver]:
    agent_config_path = tmp_path / "agent.yaml"
    scenario_db_path = tmp_path / "scenario.json"
    _write_agent_config(agent_config_path)
    _write_scenario_db(scenario_db_path)

    transport_instance = transport or _MockTransport(
        sip_uri="sip:test@example.com",
        conversation_id="conv-1",
        webhook_base_url="https://example.com",
    )
    observer_instance = observer or _FakeObserver(tmp_path / "output")

    bridge = TelephonyBridgeServer(
        current_date_time="2026-01-01 00:00 UTC",
        bridge_config=TelephonyBridgeConfig(
            sip_uri="sip:test@example.com",
            telnyx_api_key="telnyx-key",
            call_control_app_id="app-123",
            call_control_from="+15551234567",
            webhook_base_url="https://example.ngrok-free.dev",
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
        observer_factory=lambda output_dir, transcriber, audit_log: observer_instance,
    )
    return bridge, transport_instance, observer_instance


class TestTelephonyBridgeServer:
    @pytest.mark.asyncio
    async def test_handles_websocket_audio_bridge_and_finalizes_observer(self, tmp_path: Path):
        bridge, transport, observer = _make_bridge(tmp_path)

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
                {"event": "stop", "conversation_id": "conv-1", "reason": "goodbye"},
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
        assert observer.user_chunks and observer.user_chunks[0][0] == user_audio
        assert observer.assistant_chunks
        assert observer.finalize_reasons == ["goodbye"]

    @pytest.mark.asyncio
    async def test_save_outputs_writes_transcript_audit_audio_and_db_snapshots(self, tmp_path: Path):
        bridge, _transport, observer = _make_bridge(tmp_path)

        bridge._session_state = _SessionState(started_at=datetime(2026, 1, 1, tzinfo=UTC))
        bridge._session_state.user_audio.extend(b"\x00\x00" * 2400)
        bridge._session_state.assistant_audio.extend(b"\x01\x01" * 2400)
        bridge._session_state.mixed_audio.extend(b"\x01\x00" * 2400)

        observer.transcript_entries = [
            {"timestamp": "2026-01-01T00:00:00+00:00", "role": "user", "content": "I need help"},
            {"timestamp": "2026-01-01T00:00:01+00:00", "role": "assistant", "content": "Sure, let me check"},
        ]
        bridge._vad_observer = observer
        await observer.finalize(reason="goodbye")

        bridge.audit_log.append_user_input("I need help", timestamp_ms="1767225600000")
        bridge.audit_log.append_assistant_output("Sure, let me check", timestamp_ms="1767225601000")

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

        latencies = json.loads((bridge.output_dir / "response_latencies.json").read_text())
        assert latencies["count"] == 1
        assert latencies["latencies"] == [0.9]
        assert latencies["source"] == "vad_observer"

        assert (bridge.output_dir / "pipecat_logs.jsonl").read_text() == ""

    @pytest.mark.asyncio
    async def test_fetch_intended_assistant_speech_filters_and_orders_messages(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        bridge, _transport, _observer = _make_bridge(tmp_path)
        capture: dict[str, object] = {}
        payload = {
            "data": [
                {
                    "role": "assistant",
                    "text": "Second response",
                    "sent_at": "2026-03-28T00:30:13Z",
                },
                {
                    "role": "tool",
                    "text": "{\"status\":\"success\"}",
                    "sent_at": "2026-03-28T00:30:12Z",
                },
                {
                    "role": "assistant",
                    "text": "",
                    "sent_at": "2026-03-28T00:30:11Z",
                },
                {
                    "role": "assistant",
                    "text": "First response",
                    "sent_at": "2026-03-28T00:30:10Z",
                },
            ]
        }
        response = _FakeAiohttpResponse(status=200, payload=payload)

        def _session_factory(*args, **kwargs):
            return _FakeAiohttpSession(response=response, capture=capture, **kwargs)

        monkeypatch.setattr("eva.assistant.telephony_bridge.aiohttp.ClientSession", _session_factory)

        messages = await bridge._fetch_intended_assistant_speech("tel-conv-123")

        assert capture["url"] == "https://api.telnyx.com/v2/ai/conversations/tel-conv-123/messages"
        assert capture["session_kwargs"]["headers"] == {"Authorization": "Bearer telnyx-key"}
        assert messages == [
            {
                "text": "First response",
                "timestamp_ms": 1774657810000,
            },
            {
                "text": "Second response",
                "timestamp_ms": 1774657813000,
            },
        ]

    @pytest.mark.asyncio
    async def test_fetch_intended_assistant_speech_returns_empty_on_http_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        bridge, _transport, _observer = _make_bridge(tmp_path)
        response = _FakeAiohttpResponse(status=500, payload={"errors": [{"detail": "boom"}]})

        monkeypatch.setattr(
            "eva.assistant.telephony_bridge.aiohttp.ClientSession",
            lambda *args, **kwargs: _FakeAiohttpSession(response=response, capture={}, **kwargs),
        )

        messages = await bridge._fetch_intended_assistant_speech("tel-conv-123")

        assert messages == []

    @pytest.mark.asyncio
    async def test_save_outputs_writes_pipecat_tts_events_from_telnyx_messages(self, tmp_path: Path):
        bridge, _transport, _observer = _make_bridge(tmp_path)
        bridge._eva_call_id = "eva-call-123"
        bridge._telnyx_conversation_lookup = lambda eva_call_id: "tel-conv-123" if eva_call_id == "eva-call-123" else None
        bridge._fetch_intended_assistant_speech = AsyncMock(
            return_value=[
                {"text": "Hello there", "timestamp_ms": 1774657810000},
                {"text": "How can I help?", "timestamp_ms": 1774657813000},
            ]
        )

        await bridge._save_outputs()

        lines = (bridge.output_dir / "pipecat_logs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line) for line in lines] == [
            {
                "type": "tts_text",
                "start_timestamp": 1774657810000,
                "data": {"frame": "Hello there"},
            },
            {
                "type": "tts_text",
                "start_timestamp": 1774657813000,
                "data": {"frame": "How can I help?"},
            },
        ]
        bridge._fetch_intended_assistant_speech.assert_awaited_once_with("tel-conv-123")

    def test_default_transport_factory_creates_call_control_transport(self, tmp_path: Path, monkeypatch):
        bridge, _transport, _observer = _make_bridge(tmp_path)
        bridge.bridge_config = TelephonyBridgeConfig(
            sip_uri="sip:test@example.com",
            telnyx_api_key="telnyx-key",
            call_control_app_id="app-123",
            call_control_from="+15551234567",
            webhook_base_url="https://example.ngrok-free.dev",
        )

        created: dict[str, str] = {}

        class _FakeCallControlTransport:
            def __init__(
                self,
                api_key: str,
                to: str,
                app_id: str,
                from_number: str,
                conversation_id: str,
                webhook_base_url: str,
            ):
                created["api_key"] = api_key
                created["to"] = to
                created["app_id"] = app_id
                created["from_number"] = from_number
                created["conversation_id"] = conversation_id
                created["webhook_base_url"] = webhook_base_url

        monkeypatch.setattr("eva.assistant.transports.CallControlTransport", _FakeCallControlTransport)

        transport = bridge._default_transport_factory(bridge.bridge_config, "conv-1")

        assert isinstance(transport, _FakeCallControlTransport)
        assert created == {
            "api_key": "telnyx-key",
            "to": "sip:test@example.com",
            "app_id": "app-123",
            "from_number": "+15551234567",
            "conversation_id": "conv-1",
            "webhook_base_url": "https://example.ngrok-free.dev",
        }
