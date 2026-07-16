"""Unit tests for Telnyx hosted assistant helpers."""

import json
import time
from typing import Any, Self

import pytest

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.telnyx_server import (
    TelnyxAssistantManager,
    TelnyxAssistantServer,
    TelnyxDirectSessionConfig,
    TelnyxWebRTCClient,
    _extract_tool_params,
    parse_telnyx_intended_speech_payload,
)
from eva.models.agents import AgentConfig
from eva.orchestrator.worker import _get_server_class


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)


class _FakeSession:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.closed = False

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        self.requests.append(("POST", url, json))
        return _FakeResponse(201, {"id": "assistant-123"})

    def delete(self, url: str) -> _FakeResponse:
        self.requests.append(("DELETE", url, None))
        return _FakeResponse(200, {"id": "assistant-123", "deleted": True})

    async def close(self) -> None:
        self.closed = True


def _make_agent_config() -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "id": "airline-agent",
            "name": "Airline Agent",
            "description": "Assist callers with airline reservations.",
            "role": "Reservation specialist",
            "instructions": "Verify the reservation before making changes.",
            "personality": "Calm and efficient.",
            "tool_module_path": "eva.assistant.tools.airline_tools",
            "tools": [
                {
                    "id": "get_reservation",
                    "name": "Get Reservation",
                    "description": "Look up a reservation by confirmation number",
                    "required_parameters": [
                        {
                            "name": "confirmation_number",
                            "type": "string",
                            "description": "The booking confirmation number",
                        }
                    ],
                }
            ],
        }
    )


class TestTelnyxAssistantManager:
    @pytest.mark.asyncio
    async def test_create_benchmark_assistant_builds_webhook_tool_payload(self) -> None:
        session = _FakeSession()
        manager = TelnyxAssistantManager(api_key="test-key", session=session)  # type: ignore[arg-type]

        assistant_id = await manager.create_benchmark_assistant(
            agent_config=_make_agent_config(),
            agent_config_path="configs/agents/airline_agent.yaml",
            webhook_base_url="https://example.ngrok-free.app/",
            model="telnyx-llm-gpt-4o",
            voice="alloy",
        )

        assert assistant_id == "assistant-123"
        assert len(session.requests) == 1
        method, url, payload = session.requests[0]
        assert method == "POST"
        assert url == "https://api.telnyx.com/v2/ai/assistants"
        assert payload is not None
        assert payload["model"] == "telnyx-llm-gpt-4o"
        assert payload["voice_settings"]["voice"] == "alloy"
        assert payload["telephony_settings"]["supports_unauthenticated_web_calls"] is True

        webhook_tool = payload["tools"][0]
        assert webhook_tool["type"] == "webhook"
        webhook = webhook_tool["webhook"]
        assert webhook["name"] == "get_reservation"
        assert webhook["url"] == "https://example.ngrok-free.app/tools/{{eva_call_id}}/get_reservation"
        assert webhook["method"] == "POST"
        assert "confirmation_number" in webhook["body_parameters"]["properties"]
        assert "Role: Reservation specialist" in payload["instructions"]
        assert "Calm and efficient." in payload["instructions"]

    @pytest.mark.asyncio
    async def test_delete_assistant_uses_delete_endpoint(self) -> None:
        session = _FakeSession()
        manager = TelnyxAssistantManager(api_key="test-key", session=session)  # type: ignore[arg-type]

        await manager.delete_assistant("assistant-123")

        assert session.requests == [
            ("DELETE", "https://api.telnyx.com/v2/ai/assistants/assistant-123", None),
        ]


class TestTelnyxPayloadHelpers:
    def test_webhook_tool_payload_uses_eva_call_id(self) -> None:
        manager = object.__new__(TelnyxAssistantManager)
        payload = manager._build_assistant_payload(
            agent_config=_make_agent_config(),
            agent_config_path="configs/agents/airline_agent.yaml",
            webhook_base_url="https://public.example",
            model="model-1",
            voice="voice-1",
        )

        webhook_tools = [tool for tool in payload["tools"] if tool["type"] == "webhook"]
        assert len(webhook_tools) == 1
        assert "{{eva_call_id}}" in webhook_tools[0]["webhook"]["url"]
        assert "{{call_control_id}}" not in webhook_tools[0]["webhook"]["url"]
        assert "hangup" in [tool["type"] for tool in payload["tools"]]

    def test_direct_assistant_payload_does_not_require_public_webhook(self) -> None:
        manager = object.__new__(TelnyxAssistantManager)
        payload = manager._build_assistant_payload(
            agent_config=_make_agent_config(),
            agent_config_path="configs/agents/airline_agent.yaml",
            webhook_base_url=None,
            model="model-1",
            voice="voice-1",
        )

        assert payload["tools"] == [
            {
                "type": "hangup",
                "hangup": {
                    "description": (
                        "To be used whenever the conversation has ended "
                        "and it would be appropriate to hang up the call."
                    )
                },
            }
        ]
        assert "dynamic_variables_webhook_url" not in payload
        assert payload["telephony_settings"]["supports_unauthenticated_web_calls"] is True

    def test_extract_tool_params_accepts_telnyx_wrapper(self) -> None:
        assert _extract_tool_params({"function_params": {"pnr": "ABC123"}}) == {"pnr": "ABC123"}

    def test_extract_tool_params_accepts_plain_body(self) -> None:
        assert _extract_tool_params({"pnr": "ABC123"}) == {"pnr": "ABC123"}

    def test_parse_intended_speech_reverses_telnyx_newest_first_payload(self) -> None:
        payload = {
            "data": [
                {
                    "role": "assistant",
                    "text": "Second response",
                    "sent_at": "2026-01-01T00:00:02Z",
                },
                {"role": "user", "text": "Ignored", "sent_at": "2026-01-01T00:00:01Z"},
                {
                    "role": "assistant",
                    "text": "First response",
                    "sent_at": "2026-01-01T00:00:00Z",
                },
            ]
        }

        assert parse_telnyx_intended_speech_payload(payload) == [
            {"text": "First response", "timestamp_ms": 1767225600000},
            {"text": "Second response", "timestamp_ms": 1767225602000},
        ]


class TestTelnyxDirectWebRTC:
    def test_anonymous_login_payload_targets_ai_assistant(self) -> None:
        config = TelnyxDirectSessionConfig(
            assistant_id="assistant-123",
            target_version_id="version-1",
            target_params={"locale": "en-US"},
            user_variables={"eva_record_id": "record-1"},
        )

        payload = TelnyxWebRTCClient.build_anonymous_login_params(config)

        assert payload["target_type"] == "ai_assistant"
        assert payload["target_id"] == "assistant-123"
        assert payload["target_version_id"] == "version-1"
        assert payload["target_params"] == {"locale": "en-US"}
        assert payload["userVariables"] == {"eva_record_id": "record-1"}
        assert payload["reconnection"] is False
        assert payload["User-Agent"]["sdkVersion"].startswith("EVA-TelnyxWebRTC/")

    def test_invite_payload_uses_verto_dialog_shape(self) -> None:
        async def noop(_: bytes) -> None:
            return None

        config = TelnyxDirectSessionConfig(
            assistant_id="assistant-123",
            caller_name="EVA Caller",
            caller_number="+15550001000",
            destination_number="ignored-by-ai-assistant",
            client_state="state-1",
            custom_headers=[{"name": "X-Eva-Record-Id", "value": "record-1"}],
        )
        client = TelnyxWebRTCClient(config=config, audio_handler=noop)
        client.session_id = "sess-123"
        client.call_id = "call-123"

        payload = client.build_invite_params("v=0\r\n")

        assert payload["sessid"] == "sess-123"
        assert payload["sdp"] == "v=0\r\n"
        assert payload["User-Agent"].startswith("EVA-TelnyxWebRTC/")
        assert payload["dialogParams"] == {
            "callID": "call-123",
            "caller_id_name": "EVA Caller",
            "caller_id_number": "+15550001000",
            "destination_number": "ignored-by-ai-assistant",
            "audio": True,
            "client_state": "state-1",
            "custom_headers": [{"name": "X-Eva-Record-Id", "value": "record-1"}],
        }

    @pytest.mark.asyncio
    async def test_start_requires_aiortc_when_runtime_media_opens(self, monkeypatch) -> None:
        async def noop(_: bytes) -> None:
            return None

        import eva.assistant.telnyx_server as telnyx_server

        def missing_aiortc() -> tuple[Any, Any, Any, Any]:
            raise ValueError("Direct Telnyx WebRTC mode requires aiortc at runtime")

        monkeypatch.setattr(telnyx_server, "_load_aiortc_modules", missing_aiortc)
        client = TelnyxWebRTCClient(
            config=TelnyxDirectSessionConfig(assistant_id="assistant-123"),
            audio_handler=noop,
        )

        with pytest.raises(ValueError, match="requires aiortc"):
            await client.start()


class TestTelnyxServer:
    def test_dynamic_variables_returns_eva_call_id_and_records_conversation_mapping(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._conversation_map = {}
        server._transport = None
        server.conversation_id = "record-1"
        server._model = "model-1"

        response = server._handle_dynamic_variables_payload(
            {
                "data": {
                    "payload": {
                        "eva_call_id": "eva-call-1",
                        "telnyx_conversation_id": "telnyx-conv-1",
                        "call_control_id": "call-control-1",
                    }
                }
            }
        )

        assert server._conversation_map == {"eva-call-1": "telnyx-conv-1"}
        assert response["dynamic_variables"] == {"eva_call_id": "eva-call-1"}
        assert response["conversation"]["metadata"]["eva_record_id"] == "record-1"

    def test_validate_runtime_config_requires_assistant_id_only_for_direct_mode(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._assistant_id = ""
        server._api_key = ""
        server.s2s_params = {}
        server._webhook_base_url = ""
        server._transport_pref = ""
        server._create_assistant = False

        with pytest.raises(ValueError) as exc:
            server._validate_runtime_config()

        message = str(exc.value)
        assert "assistant_id or assistant_agent_id" in message
        assert "webhook_base_url" in message
        assert "connection_id" in message
        assert "to/sip_uri" in message

    def test_validate_runtime_config_does_not_require_call_control_fields(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._assistant_id = "assistant-123"
        server._api_key = ""
        server.s2s_params = {}
        server._webhook_base_url = ""
        server._transport_pref = ""
        server._create_assistant = False

        server._validate_runtime_config()

    def test_validate_runtime_config_call_control_mode_requires_credentials(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._assistant_id = ""
        server._api_key = ""
        server.s2s_params = {"connection_id": "conn-123", "webhook_base_url": "https://public.example.com"}
        server._webhook_base_url = "https://public.example.com"
        server._transport_pref = ""
        server._create_assistant = False

        with pytest.raises(ValueError) as exc:
            server._validate_runtime_config()

        message = str(exc.value)
        assert "api_key or telnyx_api_key" in message
        assert "from_number or caller_number" in message
        assert "to or to_number or destination_number" in message
        assert "Call Control mode" in message

    def test_validate_runtime_config_call_control_mode_passes_with_all_fields(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._assistant_id = ""
        server._api_key = "key0123"
        server.s2s_params = {
            "connection_id": "conn-123",
            "webhook_base_url": "https://public.example.com",
            "from_number": "+15550001234",
            "to": "+15550005678",
        }
        server._webhook_base_url = "https://public.example.com"
        server._transport_pref = ""
        server._create_assistant = False

        # Should not raise
        server._validate_runtime_config()

    def test_media_streaming_transport_selects_call_control_without_connection_id(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._transport_pref = "media_streaming"
        server.s2s_params = {}
        server._webhook_base_url = ""
        assert server._use_call_control is True

    def test_validate_runtime_config_call_control_waives_to_when_creating_assistant(self) -> None:
        # create_assistant derives the SIP `to` from the freshly-created assistant id, so `to`
        # is not required up front; connection_id + from_number are filled by auto-provision.
        server = object.__new__(TelnyxAssistantServer)
        server._assistant_id = ""
        server._api_key = "key0123"
        server.s2s_params = {
            "connection_id": "conn-123",
            "webhook_base_url": "https://public.example.com",
            "from_number": "+15550001234",
        }
        server._webhook_base_url = "https://public.example.com"
        server._transport_pref = "media_streaming"
        server._create_assistant = True

        # Should not raise even though no `to`/`to_number`/`destination_number` is supplied.
        server._validate_runtime_config()

    def test_build_direct_session_config_accepts_aliases_and_metadata(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._assistant_id = "assistant-123"
        server._websocket_host = "wss://rtcdev.telnyx.com"
        server._target_version_id = "version-1"
        server._target_params = {"locale": "en-US"}
        server._user_variables = {"eva_record_id": "record-1"}
        server._caller_name = "EVA Caller"
        server._caller_number = "+15550001000"
        server._destination_number = "ignored"
        server._client_state = "state-1"
        server._custom_headers = [{"name": "X-Eva-Record-Id", "value": "record-1"}]

        config = server._build_direct_session_config()

        assert config.assistant_id == "assistant-123"
        assert config.host == "wss://rtcdev.telnyx.com"
        assert config.target_version_id == "version-1"
        assert config.target_params == {"locale": "en-US"}
        assert config.user_variables == {"eva_record_id": "record-1"}
        assert config.custom_headers == [{"name": "X-Eva-Record-Id", "value": "record-1"}]

    def test_tool_stub_maps_agent_tools_without_public_webhook(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server.agent = _make_agent_config()

        tools = server._build_telnyx_tool_definitions()

        assert tools == [
            {
                "type": "client_tool",
                "name": "get_reservation",
                "description": "Look up a reservation by confirmation number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmation_number": {
                            "type": "string",
                            "description": "The booking confirmation number",
                        }
                    },
                    "required": ["confirmation_number"],
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_conversation_event_records_assistant_text(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server.audit_log = AuditLog()
        server._fw_log = None

        await server._handle_telnyx_conversation_event(
            {
                "type": "conversation.item.created",
                "item": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello, how can I help?"}],
                },
            }
        )

        assert server.audit_log.transcript[-1]["message_type"] == "assistant"
        assert server.audit_log.transcript[-1]["value"] == "Hello, how can I help?"

    def test_assistant_audio_active_uses_recent_audio_window(self) -> None:
        server = object.__new__(TelnyxAssistantServer)
        server._last_assistant_audio_monotonic = time.monotonic()
        assert server._is_assistant_audio_active() is True

        server._last_assistant_audio_monotonic = time.monotonic() - 1.0
        assert server._is_assistant_audio_active() is False

    def test_worker_registers_telnyx_framework(self) -> None:
        assert _get_server_class("telnyx") is TelnyxAssistantServer

    @pytest.mark.parametrize(
        "events_filename",
        # The user simulator writes user_simulator_events.jsonl; elevenlabs_events.jsonl
        # is the legacy name still accepted by resolve_user_simulator_events_path().
        ["user_simulator_events.jsonl", "elevenlabs_events.jsonl"],
    )
    def test_user_event_enrichment_is_idempotent(self, tmp_path, events_filename: str) -> None:
        (tmp_path / events_filename).write_text(
            json.dumps(
                {
                    "timestamp": 1767225600000,
                    "type": "user_speech",
                    "data": {"text": "I need help with my reservation."},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        server = object.__new__(TelnyxAssistantServer)
        server.output_dir = tmp_path
        server.audit_log = AuditLog()
        server._user_events_enriched = False

        server._append_user_events_from_simulator()
        server._append_user_events_from_simulator()

        user_entries = [entry for entry in server.audit_log.transcript if entry["message_type"] == "user"]
        assert len(user_entries) == 1
        assert user_entries[0]["value"] == "I need help with my reservation."

    @pytest.mark.asyncio
    async def test_hangup_active_call_stops_transport(self) -> None:
        class FakeTransport:
            def __init__(self) -> None:
                self.stopped = False

            async def stop(self) -> None:
                self.stopped = True

        server = object.__new__(TelnyxAssistantServer)
        server._transport = FakeTransport()

        await server._hangup_active_call()

        assert server._transport.stopped is True
