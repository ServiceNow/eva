"""Telnyx hosted AI Assistant server for EVA.

Expected ``pipeline_config.s2s_params`` keys:

Required for direct Telnyx WebRTC calls:
  * ``assistant_id`` or ``assistant_agent_id``: Telnyx AI Assistant target id.

Optional for direct Telnyx WebRTC calls:
  * ``target_version_id``: specific Telnyx AI Assistant version.
  * ``host`` or ``websocket_host``: Verto websocket host, defaults to
    ``wss://rtc.telnyx.com``.
  * ``target_params`` and ``user_variables``: anonymous-login metadata.
  * ``caller_name``, ``caller_number``, ``client_state``, ``custom_headers``:
    dialog metadata sent with the WebRTC invite.

Optional assistant setup:
  * ``create_assistant``: create a Telnyx AI Assistant before the run.
  * ``api_key`` or ``telnyx_api_key``: Telnyx API key, required only for REST
    assistant creation/deletion or transcript fetch helpers.
  * ``api_base``: defaults to ``https://api.telnyx.com``; ``/v2`` is appended
    automatically for v2 endpoints unless already present.
  * ``model``, ``voice``, and ``stt_model``: used when creating an assistant.

The server exposes the upstream EVA assistant-server contract to the local user
simulator while bridging to Telnyx's anonymous Verto/WebRTC AI Assistant path.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_16k,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    sync_buffer_to_position,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.models.agents import AgentConfig, AgentTool
from eva.utils.conversation_checks import resolve_user_simulator_events_path
from eva.utils.culture import get_initial_message
from eva.utils.logging import get_logger

logger = get_logger(__name__)

TELNYX_DEFAULT_API_BASE = "https://api.telnyx.com"
TELNYX_PROD_RTC_HOST = "wss://rtc.telnyx.com"
TELNYX_DEV_RTC_HOST = "wss://rtcdev.telnyx.com"
TELNYX_SAMPLE_RATE = 16000
RECORDING_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2
MULAW_CHUNK_SIZE = 160
MULAW_CHUNK_DURATION_S = 0.02
# Telnyx's inter-turn frames decode to digital zero; anything at or below this RMS is
# silence and must not be forwarded, or the user simulator never gets a turn.
ASSISTANT_SILENCE_RMS = 1
TELNYX_USER_AGENT = "EVA-TelnyxWebRTC/0.1"
# Mirrors DEFAULT_PROD_ICE_SERVERS from @telnyx/webrtc 2.27.4
# (STUN + TURN UDP/3478 + TURN TCP/3478 + TURNS on 443). The TURNS/443 entry
# is the last-resort TLS fallback for networks that block both 3478 transports.
TELNYX_DEFAULT_ICE_SERVERS = [
    {"urls": "stun:stun.telnyx.com:3478"},
    {"urls": "stun:stun.l.google.com:19302"},
    {
        "urls": "turn:turn.telnyx.com:3478?transport=udp",
        "username": "testuser",
        "credential": "testpassword",
    },
    {
        "urls": "turn:turn.telnyx.com:3478?transport=tcp",
        "username": "testuser",
        "credential": "testpassword",
    },
    {
        "urls": "turns:turn2.telnyx.com:443",
        "username": "testuser",
        "credential": "testpassword",
    },
]


def _wall_ms() -> str:
    return str(int(round(time.time() * 1000)))


def _normalize_api_v2_base(api_base: str | None) -> str:
    base = (api_base or TELNYX_DEFAULT_API_BASE).rstrip("/")
    if base.endswith("/v2"):
        return base
    return f"{base}/v2"


def _pcm16_16k_to_pcm16_24k(pcm_16k: bytes) -> bytes:
    if not pcm_16k:
        return b""
    pcm_24k, _ = audioop.ratecv(pcm_16k, PCM_SAMPLE_WIDTH, 1, 16000, 24000, None)
    return pcm_24k


def _pcm16_16k_to_mulaw_8k(pcm_16k: bytes) -> bytes:
    if not pcm_16k:
        return b""
    pcm_8k, _ = audioop.ratecv(pcm_16k, PCM_SAMPLE_WIDTH, 1, 16000, 8000, None)
    return audioop.lin2ulaw(pcm_8k, PCM_SAMPLE_WIDTH)


def _iso8601_to_epoch_ms(value: str) -> int:
    dt = datetime.fromisoformat(value)
    return int(dt.timestamp() * 1000)


def _is_epoch_ms(value: str | None) -> bool:
    if not value:
        return False
    try:
        return int(value) > 1_000_000_000_000
    except (TypeError, ValueError):
        return False


def _extract_tool_params(body: Any) -> dict[str, Any]:
    """Normalize Telnyx webhook bodies to the function argument dict."""
    if body is None:
        return {}
    if not isinstance(body, dict):
        raise ValueError("Tool webhook body must be a JSON object")

    function_params = body.get("function_params")
    if function_params is None:
        return body
    if not isinstance(function_params, dict):
        raise ValueError("function_params must be a JSON object")
    return function_params


def _message_text(message: dict[str, Any]) -> str:
    text = message.get("text")
    if isinstance(text, str):
        return text.strip()

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                part_text = item.get("text") or item.get("content")
                if isinstance(part_text, str):
                    parts.append(part_text)
        return "".join(parts).strip()
    return ""


def _conversation_item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text") or part.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def parse_telnyx_intended_speech_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract chronological assistant messages from Telnyx conversation messages."""
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []

    intended_speech: list[dict[str, Any]] = []
    for message in reversed(data):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = _message_text(message)
        if not text:
            continue
        sent_at = message.get("sent_at") or message.get("created_at")
        if not isinstance(sent_at, str) or not sent_at.strip():
            continue
        try:
            timestamp_ms = _iso8601_to_epoch_ms(sent_at)
        except ValueError:
            continue
        intended_speech.append({"text": text, "timestamp_ms": timestamp_ms})
    return intended_speech


def extract_user_speech_events(events_path: Path) -> list[dict[str, Any]]:
    """Read simulated-user transcript events saved by ``UserSimulator``."""
    if not events_path.exists():
        return []

    turns: list[dict[str, Any]] = []
    with open(events_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "user_speech":
                continue
            data = event.get("data", {})
            if not isinstance(data, dict):
                continue
            text = str(data.get("text", "")).strip()
            if text:
                turns.append({"text": text, "timestamp_ms": str(event.get("timestamp", ""))})
    return turns


def extract_assistant_speech_events(events_path: Path) -> list[dict[str, Any]]:
    """Read assistant speech heard by the user simulator as a fallback transcript."""
    if not events_path.exists():
        return []

    turns: list[dict[str, Any]] = []
    with open(events_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "assistant_speech":
                continue
            data = event.get("data", {})
            if not isinstance(data, dict):
                continue
            text = str(data.get("text", "")).strip()
            if text:
                turns.append({"text": text, "timestamp_ms": int(event.get("timestamp", 0))})
    return turns


class TelnyxAssistantManager:
    """Create and manage Telnyx AI assistants for benchmark runs."""

    DEFAULT_MODEL = "moonshotai/Kimi-K2.5"
    DEFAULT_VOICE = "Telnyx.Ultra.a7a59115-2425-4192-844c-1e98ec7d6877"
    DEFAULT_STT_MODEL = "deepgram/flux"

    def __init__(
        self,
        api_key: str,
        api_base: str = TELNYX_DEFAULT_API_BASE,
        session: aiohttp.ClientSession | None = None,
    ):
        self.api_key = api_key
        self.api_base = _normalize_api_v2_base(api_base)
        self._session_owner = session is None
        self.session = session or aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30.0),
        )

    async def create_benchmark_assistant(
        self,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        stt_model: str | None = None,
    ) -> str:
        payload = self._build_assistant_payload(
            agent_config=agent_config,
            agent_config_path=agent_config_path,
            webhook_base_url=webhook_base_url,
            model=model or self.DEFAULT_MODEL,
            voice=voice or self.DEFAULT_VOICE,
            stt_model=stt_model or self.DEFAULT_STT_MODEL,
        )
        async with self.session.post(f"{self.api_base}/ai/assistants", json=payload) as response:
            response_payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to create Telnyx assistant: {response.status} "
                    f"{json.dumps(response_payload, sort_keys=True)}"
                )

        assistant_id = str(response_payload.get("id") or response_payload.get("data", {}).get("id") or "").strip()
        if not assistant_id:
            raise RuntimeError(f"Telnyx assistant creation response did not include an id: {response_payload}")
        return assistant_id

    async def delete_assistant(self, assistant_id: str) -> None:
        async with self.session.delete(f"{self.api_base}/ai/assistants/{assistant_id}") as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to delete Telnyx assistant {assistant_id}: "
                    f"{response.status} {json.dumps(payload, sort_keys=True)}"
                )

    async def get_assistant_model(self, assistant_id: str) -> str:
        async with self.session.get(f"{self.api_base}/ai/assistants/{assistant_id}") as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to get Telnyx assistant {assistant_id}: "
                    f"{response.status} {json.dumps(payload, sort_keys=True)}"
                )
            return str(payload.get("model") or payload.get("data", {}).get("model", ""))

    async def update_assistant_model(self, assistant_id: str, model: str) -> None:
        async with self.session.patch(
            f"{self.api_base}/ai/assistants/{assistant_id}", json={"model": model}
        ) as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to update Telnyx assistant model: {response.status} {json.dumps(payload, sort_keys=True)}"
                )

    async def close(self) -> None:
        if self._session_owner and not self.session.closed:
            await self.session.close()

    @staticmethod
    async def _parse_response_json(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            payload = await response.json()
        except aiohttp.ContentTypeError:
            payload = {"message": await response.text()}
        return payload if isinstance(payload, dict) else {"data": payload}

    def _build_assistant_payload(
        self,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str | None,
        model: str,
        voice: str,
        stt_model: str | None = None,
    ) -> dict[str, Any]:
        normalized_webhook_base = webhook_base_url.rstrip("/") if webhook_base_url else ""
        tools = (
            [self._build_webhook_tool(tool, normalized_webhook_base) for tool in agent_config.tools]
            if normalized_webhook_base
            else []
        )
        if not any(tool.id == "end_call" for tool in agent_config.tools):
            tools.append(
                {
                    "type": "hangup",
                    "hangup": {
                        "description": (
                            "To be used whenever the conversation has ended "
                            "and it would be appropriate to hang up the call."
                        ),
                    },
                }
            )

        payload = {
            "name": f"EVA Benchmark - {agent_config.name}",
            "description": (
                f"Auto-generated by EVA from {Path(agent_config_path).name} for benchmark agent {agent_config.id}."
            ),
            "instructions": self._build_system_prompt(agent_config),
            "model": model,
            "greeting": getattr(agent_config, "greeting", None) or get_initial_message("en"),
            "tools": tools,
            "enabled_features": ["telephony"],
            "voice_settings": {
                "voice": voice,
                "voice_speed": 1.2,
                "expressive_mode": False,
                "language_boost": "English",
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True,
                "background_audio": {
                    "type": "predefined_media",
                    "value": "silence",
                    "volume": 0.5,
                },
            },
            "transcription": {
                "model": stt_model or self.DEFAULT_STT_MODEL,
                "language": "en",
                "settings": {
                    "eot_threshold": 0.9,
                    "eot_timeout_ms": 5000,
                    "eager_eot_threshold": 0.9,
                },
            },
            "telephony_settings": {
                "supports_unauthenticated_web_calls": True,
                "time_limit_secs": 600,
                "user_idle_timeout_secs": 60,
            },
            "interruption_settings": {
                "enable": True,
                "start_speaking_plan": {
                    "wait_seconds": 0.1,
                    "transcription_endpointing_plan": {
                        "on_punctuation_seconds": 0.1,
                        "on_no_punctuation_seconds": 0.1,
                        "on_number_seconds": 0.1,
                    },
                },
            },
            "privacy_settings": {
                "data_retention": True,
                "pii_redaction": "disabled",
            },
            "dynamic_variables": {
                "eva_call_id": None,
            },
        }
        if normalized_webhook_base:
            payload["dynamic_variables_webhook_url"] = f"{normalized_webhook_base}/dynamic-variables"
        return payload

    @staticmethod
    def _build_system_prompt(agent_config: AgentConfig) -> str:
        sections = [
            agent_config.description.strip(),
            f"Role: {agent_config.role.strip()}",
            agent_config.personality.strip() if agent_config.personality else "",
            agent_config.instructions.strip(),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _build_webhook_tool(tool: AgentTool, webhook_base_url: str) -> dict[str, Any]:
        return {
            "type": "webhook",
            "timeout_ms": 5000,
            "webhook": {
                "name": tool.id,
                "description": tool.description,
                "url": f"{webhook_base_url}/tools/{{{{eva_call_id}}}}/{tool.id}",
                "method": "POST",
                "path_parameters": {"type": "object", "properties": {}},
                "query_parameters": {"type": "object", "properties": {}},
                "body_parameters": {
                    "type": "object",
                    "properties": tool.get_parameter_properties(),
                    "required": tool.get_required_param_names(),
                },
                "headers": [],
            },
        }


@dataclass
class TelnyxDirectSessionConfig:
    """Configuration for Telnyx anonymous Verto/WebRTC AI Assistant sessions."""

    assistant_id: str
    host: str = TELNYX_PROD_RTC_HOST
    target_version_id: str | None = None
    target_params: dict[str, Any] = field(default_factory=dict)
    user_variables: dict[str, Any] = field(default_factory=dict)
    caller_name: str = "EVA User"
    caller_number: str = "+15555550100"
    destination_number: str = "ai_assistant"
    client_state: str | None = None
    custom_headers: list[dict[str, str]] = field(default_factory=list)
    ice_servers: list[dict[str, Any]] = field(default_factory=lambda: list(TELNYX_DEFAULT_ICE_SERVERS))


_dtls_session_tickets_disabled = False


def _disable_dtls_session_tickets() -> None:
    """Stop aiortc's DTLS server from emitting a NewSessionTicket.

    OpenSSL's DTLS server sends NewSessionTicket + ChangeCipherSpec + Finished as its
    final flight. aiortc never sets SSL_OP_NO_TICKET, so that ticket (~1.4 kB) has to be
    fragmented across several records. Telnyx's FreeSWITCH does not process the fragmented
    ticket, so it never consumes our Finished: it re-sends its own final flight, never
    completes the handshake, and never derives SRTP keys. OpenSSL considers the handshake
    done on our side, so aiortc reports "connected" while media is silently dead in BOTH
    directions -- the assistant hears nothing and sends us nothing.

    Session resumption is meaningless for a single DTLS-SRTP call, and browsers do not
    offer tickets here either, so disabling them costs nothing and makes the final flight
    a single unfragmented CCS + Finished.
    """
    global _dtls_session_tickets_disabled
    if _dtls_session_tickets_disabled:
        return

    from aiortc.rtcdtlstransport import RTCCertificate
    from OpenSSL import SSL

    original = RTCCertificate._create_ssl_context

    def _create_ssl_context(self: Any, srtp_profiles: Any) -> Any:
        ctx = original(self, srtp_profiles)
        ctx.set_options(SSL.OP_NO_TICKET)
        return ctx

    RTCCertificate._create_ssl_context = _create_ssl_context  # type: ignore[method-assign]
    _dtls_session_tickets_disabled = True


def _load_aiortc_modules() -> tuple[Any, Any, Any, Any]:
    try:
        from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
        from av import AudioFrame
    except ImportError as exc:
        raise ValueError(
            "Direct Telnyx WebRTC mode requires aiortc at runtime. "
            "Install aiortc in the EVA environment to open the Telnyx media session."
        ) from exc
    _disable_dtls_session_tickets()
    return MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, AudioFrame


def _normalize_telnyx_headers(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [{"name": str(name), "value": str(header_value)} for name, header_value in value.items()]
    if not isinstance(value, list):
        raise ValueError("custom_headers must be a mapping or a list of name/value objects")

    headers: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("custom_headers list entries must be objects")
        name = item.get("name")
        header_value = item.get("value")
        if name is None or header_value is None:
            raise ValueError("custom_headers entries must include name and value")
        headers.append({"name": str(name), "value": str(header_value)})
    return headers


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


class TelnyxVertoJsonRpcClient:
    """Minimal Verto JSON-RPC client for Telnyx anonymous WebRTC sessions."""

    def __init__(self, host: str):
        self.host = host
        self._ws: Any | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._receive_task: asyncio.Task[None] | None = None
        self._request_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self, request_handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise ValueError("Direct Telnyx WebRTC mode requires the websockets package") from exc

        self._request_handler = request_handler
        self._ws = await websockets.connect(self.host)
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Telnyx Verto websocket is not connected")

        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        await self._ws.send(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}))
        return await future

    async def reply(self, request_id: Any, result: dict[str, Any]) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}))

    async def _receive_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON Telnyx Verto message")
                    continue
                await self._dispatch_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
            logger.info("Telnyx Verto websocket closed: %s", exc)

    async def _dispatch_message(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("id", ""))
        if request_id in self._pending and ("result" in message or "error" in message):
            future = self._pending.pop(request_id)
            if "error" in message:
                future.set_exception(RuntimeError(f"Telnyx Verto error: {message['error']}"))
            else:
                result = message.get("result")
                future.set_result(result if isinstance(result, dict) else {})
            return

        if "method" in message and self._request_handler is not None:
            await self._request_handler(message)


class TelnyxWebRTCClient:
    """Direct Telnyx anonymous WebRTC client shaped after the Telnyx JS SDK."""

    def __init__(
        self,
        *,
        config: TelnyxDirectSessionConfig,
        audio_handler: Callable[[bytes], Awaitable[None]],
    ):
        self.config = config
        self.audio_handler = audio_handler

        self.session_id: str | None = None
        self.call_id = str(uuid.uuid4())
        self.disconnected_event = asyncio.Event()

        self._rpc = TelnyxVertoJsonRpcClient(config.host)
        self._peer: Any | None = None
        self._audio_track: Any | None = None
        self._remote_audio_task: asyncio.Task[None] | None = None
        self._remote_description_set = asyncio.Event()
        self._conversation_event_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    @property
    def eva_call_id(self) -> str:
        return self.call_id

    @staticmethod
    def build_anonymous_login_params(config: TelnyxDirectSessionConfig) -> dict[str, Any]:
        params: dict[str, Any] = {
            "target_type": "ai_assistant",
            "target_id": config.assistant_id,
            "userVariables": config.user_variables,
            "reconnection": False,
            "User-Agent": {
                "sdkVersion": TELNYX_USER_AGENT,
                "data": "EVA assistant server",
            },
        }
        if config.target_version_id:
            params["target_version_id"] = config.target_version_id
        if config.target_params:
            params["target_params"] = config.target_params
        return params

    def build_invite_params(self, sdp: str) -> dict[str, Any]:
        if not self.session_id:
            raise RuntimeError("Telnyx session id is not available")

        dialog_params: dict[str, Any] = {
            "callID": self.call_id,
            "caller_id_name": self.config.caller_name,
            "caller_id_number": self.config.caller_number,
            "destination_number": self.config.destination_number,
            "audio": True,
        }
        if self.config.client_state:
            dialog_params["client_state"] = self.config.client_state
        if self.config.custom_headers:
            dialog_params["custom_headers"] = self.config.custom_headers

        return {
            "sessid": self.session_id,
            "sdp": sdp,
            "dialogParams": dialog_params,
            "User-Agent": TELNYX_USER_AGENT,
        }

    async def start(self) -> None:
        _, RTCPeerConnection, RTCSessionDescription, AudioFrame = _load_aiortc_modules()
        await self._rpc.connect(self._handle_verto_request)

        login_result = await self._rpc.request(
            "anonymous_login",
            self.build_anonymous_login_params(self.config),
        )
        self.session_id = str(login_result.get("sessid") or "").strip()
        if not self.session_id:
            raise RuntimeError(f"Telnyx anonymous_login response did not include sessid: {login_result}")

        self._peer = self._create_peer_connection(RTCPeerConnection)
        self._audio_track = self._build_audio_track(AudioFrame)
        self._peer.addTrack(self._audio_track)

        @self._peer.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                logger.info("Telnyx WebRTC remote audio track received")
                self._remote_audio_task = asyncio.create_task(self._receive_remote_audio(track))

        @self._peer.on("iceconnectionstatechange")
        def on_ice_connection_state_change() -> None:
            logger.info(
                "Telnyx WebRTC iceConnectionState=%s",
                getattr(self._peer, "iceConnectionState", None),
            )

        @self._peer.on("connectionstatechange")
        def on_connection_state_change() -> None:
            state = getattr(self._peer, "connectionState", None)
            logger.info("Telnyx WebRTC connectionState=%s", state)
            # A failed/closed transport will never deliver media; surface it
            # immediately instead of idling until the far end hangs up.
            if state in {"failed", "closed"}:
                self.disconnected_event.set()

        offer = await self._peer.createOffer()
        await self._peer.setLocalDescription(offer)
        await self._wait_for_ice_gathering_complete()

        if self._peer is None:
            # stop() ran while we were gathering ICE (it nulls _peer). The session was torn
            # down before the invite went out -- e.g. the user simulator failed to start --
            # so there is nothing to invite. Exit quietly instead of dereferencing None.
            logger.info("Telnyx WebRTC session stopped during startup; skipping invite")
            return

        local_description = self._peer.localDescription
        logger.info("Telnyx WebRTC SDP offer:\n%s", local_description.sdp)
        invite_result = await self._rpc.request("telnyx_rtc.invite", self.build_invite_params(local_description.sdp))
        result_sdp = invite_result.get("sdp")
        logger.info("Telnyx WebRTC invite result keys: %s", list(invite_result.keys()))
        if isinstance(result_sdp, str) and result_sdp:
            logger.info("Telnyx WebRTC SDP answer (from invite):\n%s", result_sdp)
            await self._set_remote_description(RTCSessionDescription, result_sdp)
        else:
            logger.warning("Telnyx WebRTC invite returned no SDP; waiting for media/answer event")

    def _create_peer_connection(self, RTCPeerConnection: Any) -> Any:
        from aiortc import RTCConfiguration, RTCIceServer

        ice_servers = [
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in self.config.ice_servers
        ]
        return RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

    async def stop(self) -> None:
        if self.session_id:
            try:
                await self._rpc.request(
                    "telnyx_rtc.bye",
                    {
                        "sessid": self.session_id,
                        "dialogParams": {"callID": self.call_id},
                    },
                )
            except Exception as exc:
                logger.debug("Ignoring Telnyx bye error: %s", exc)

        if self._remote_audio_task is not None:
            self._remote_audio_task.cancel()
            try:
                await self._remote_audio_task
            except asyncio.CancelledError:
                pass
            self._remote_audio_task = None

        if self._peer is not None:
            await self._peer.close()
            self._peer = None

        await self._rpc.close()
        self.disconnected_event.set()

    async def send_audio(self, pcm_16k: bytes) -> None:
        if self._audio_track is None or not pcm_16k:
            return
        await self._audio_track.enqueue_pcm(pcm_16k)

    async def _handle_verto_request(self, message: dict[str, Any]) -> None:
        method = str(message.get("method") or "")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        request_id = message.get("id")

        if request_id is not None:
            await self._rpc.reply(request_id, {"method": method})

        if method in {"telnyx_rtc.media", "telnyx_rtc.answer"}:
            sdp = params.get("sdp")
            if isinstance(sdp, str) and sdp:
                _, _, RTCSessionDescription, _ = _load_aiortc_modules()
                await self._set_remote_description(RTCSessionDescription, sdp)
        elif method == "telnyx_rtc.bye":
            self.disconnected_event.set()
        elif method == "ai_conversation":
            if self._conversation_event_handler is not None:
                await self._conversation_event_handler(params)
        elif method == "telnyx_rtc.ping":
            logger.debug("Received Telnyx Verto ping")
        else:
            logger.debug("Unhandled Telnyx Verto method: %s", method)

    async def _set_remote_description(self, RTCSessionDescription: Any, sdp: str) -> None:
        if self._peer is None or self._remote_description_set.is_set():
            return
        logger.info("Telnyx WebRTC SDP answer (setting remote description):\n%s", sdp)
        await self._peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
        self._remote_description_set.set()
        logger.info(
            "Telnyx WebRTC remote description set. "
            "Receivers: %s, Senders: %s, Transceivers: %s",
            len(self._peer.getReceivers()),
            len(self._peer.getSenders()),
            len(self._peer.getTransceivers()),
        )
        for i, t in enumerate(self._peer.getTransceivers()):
            mid = getattr(t, "mid", None)
            direction = getattr(t, "currentDirection", None) or getattr(t, "direction", None)
            stopped = getattr(t, "stopped", False)
            receiver_track = getattr(t.receiver, "track", None) if t.receiver else None
            sender_track = getattr(t.sender, "track", None) if t.sender else None
            logger.info(
                "  Transceiver[%s] mid=%s direction=%s stopped=%s "
                "receiver_track=%s sender_track=%s",
                i, mid, direction, stopped,
                getattr(receiver_track, "kind", None),
                getattr(sender_track, "kind", None),
            )
        self._ensure_remote_audio_task()

    def _ensure_remote_audio_task(self) -> None:
        if self._peer is None or self._remote_audio_task is not None:
            return
        for receiver in self._peer.getReceivers():
            track = getattr(receiver, "track", None)
            if track is not None and getattr(track, "kind", None) == "audio":
                logger.info("Telnyx WebRTC remote audio receiver attached")
                self._remote_audio_task = asyncio.create_task(self._receive_remote_audio(track))
                return

    async def _wait_for_ice_gathering_complete(self, timeout_seconds: float = 10.0) -> None:
        if self._peer is None or self._peer.iceGatheringState == "complete":
            return

        complete = asyncio.Event()

        @self._peer.on("icegatheringstatechange")
        def on_ice_gathering_state_change() -> None:
            if self._peer and self._peer.iceGatheringState == "complete":
                complete.set()

        try:
            await asyncio.wait_for(complete.wait(), timeout=timeout_seconds)
        except TimeoutError:
            logger.warning("Timed out waiting for Telnyx WebRTC ICE gathering to complete; sending current SDP")

    def _build_audio_track(self, AudioFrame: Any) -> Any:
        MediaStreamTrack, _, _, _ = _load_aiortc_modules()

        class _QueuedPcmAudioTrack(MediaStreamTrack):
            kind = "audio"

            def __init__(self) -> None:
                super().__init__()
                self._queue: asyncio.Queue[bytes] = asyncio.Queue()
                self._pts = 0
                self._sample_rate = TELNYX_SAMPLE_RATE
                self._silence_frame = b"\x00" * int(TELNYX_SAMPLE_RATE * MULAW_CHUNK_DURATION_S * PCM_SAMPLE_WIDTH)

            async def enqueue_pcm(self, pcm_16k: bytes) -> None:
                await self._queue.put(pcm_16k)

            async def recv(self) -> Any:
                try:
                    pcm_16k = await asyncio.wait_for(self._queue.get(), timeout=MULAW_CHUNK_DURATION_S)
                except TimeoutError:
                    pcm_16k = self._silence_frame
                samples = max(1, len(pcm_16k) // PCM_SAMPLE_WIDTH)
                frame = AudioFrame(format="s16", layout="mono", samples=samples)
                frame.planes[0].update(pcm_16k)
                frame.sample_rate = self._sample_rate
                frame.pts = self._pts
                frame.time_base = Fraction(1, self._sample_rate)
                self._pts += samples
                return frame

        return _QueuedPcmAudioTrack()

    async def _receive_remote_audio(self, track: Any) -> None:
        frames = 0
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=10.0)
                except TimeoutError:
                    # recv() blocks when no RTP is arriving. If the media path
                    # never establishes this repeats until the far end hangs up,
                    # producing empty audio buffers and a silent conversation.
                    logger.warning(
                        "Telnyx WebRTC: no remote audio frame in 10s "
                        "(iceConnectionState=%s, frames_so_far=%d) — media not flowing",
                        getattr(self._peer, "iceConnectionState", None),
                        frames,
                    )
                    # Log receiver stats to distinguish "no RTP arriving" vs
                    # "arriving but not decoding".
                    if self._peer is not None:
                        for r in self._peer.getReceivers():
                            try:
                                stats = await r.getStats()
                                for report in stats.values():
                                    if hasattr(report, "packetsReceived"):
                                        logger.info(
                                            "  Receiver stats: packetsReceived=%s "
                                            "bytesReceived=%s packetsLost=%s "
                                            "jitter=%s kind=%s",
                                            getattr(report, "packetsReceived", None),
                                            getattr(report, "bytesReceived", None),
                                            getattr(report, "packetsLost", None),
                                            getattr(report, "jitter", None),
                                            getattr(report, "kind", None),
                                        )
                            except Exception as exc:
                                logger.debug("  Receiver stats error: %s", exc)
                    continue
                frames += 1
                if frames == 1:
                    logger.info("Telnyx WebRTC first remote audio frame received "
                                "(sample_rate=%s, samples=%s)",
                                getattr(frame, "sample_rate", None),
                                getattr(frame, "samples", None))
                pcm_16k = self._frame_to_pcm16_16k(frame)
                if pcm_16k:
                    await self.audio_handler(pcm_16k)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.info("Telnyx remote audio track stopped after %d frames: %s", frames, exc)
            self.disconnected_event.set()

    @staticmethod
    def _frame_to_pcm16_16k(frame: Any) -> bytes:
        sample_rate = int(getattr(frame, "sample_rate", TELNYX_SAMPLE_RATE) or TELNYX_SAMPLE_RATE)
        try:
            pcm = frame.to_ndarray().astype("int16").tobytes()
        except Exception:
            pcm = bytes(frame.planes[0])
        if sample_rate != TELNYX_SAMPLE_RATE:
            pcm, _ = audioop.ratecv(pcm, PCM_SAMPLE_WIDTH, 1, sample_rate, TELNYX_SAMPLE_RATE, None)
        return pcm


class TelnyxCallControlTransport:
    """Telnyx Call Control transport using bidirectional media streaming."""

    def __init__(
        self,
        *,
        api_key: str,
        to: str,
        connection_id: str,
        from_number: str,
        conversation_id: str,
        webhook_base_url: str,
        api_v2_base: str,
        audio_handler: Callable[[bytes], Awaitable[None]],
        connect_timeout_seconds: float = 60.0,
    ):
        self.api_key = api_key
        self.to = to
        self.connection_id = connection_id
        self.from_number = from_number
        self.conversation_id = conversation_id
        self.webhook_base_url = webhook_base_url.rstrip("/")
        self.api_v2_base = api_v2_base.rstrip("/")
        self.audio_handler = audio_handler
        self.connect_timeout_seconds = connect_timeout_seconds

        self._session: aiohttp.ClientSession | None = None
        self._stream_ws: WebSocket | None = None
        self._call_control_id: str | None = None
        self._call_session_id: str | None = None
        self._call_leg_id: str | None = None
        self._eva_call_id: str | None = None
        self._stream_id: str | None = None
        self._connected_event = asyncio.Event()
        self._disconnected_event = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._inbound_encoding = "L16"
        self._inbound_sample_rate = TELNYX_SAMPLE_RATE

    @property
    def call_control_id(self) -> str | None:
        return self._call_control_id

    @property
    def call_session_id(self) -> str | None:
        return self._call_session_id

    @property
    def call_leg_id(self) -> str | None:
        return self._call_leg_id

    @property
    def eva_call_id(self) -> str | None:
        return self._eva_call_id

    @property
    def disconnected_event(self) -> asyncio.Event:
        return self._disconnected_event

    async def start(self) -> None:
        if self._session is not None:
            logger.warning("Telnyx transport already started for %s", self.to)
            return

        self._connected_event.clear()
        self._disconnected_event.clear()
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30.0),
        )
        self._eva_call_id = str(uuid.uuid4())
        payload = self.build_call_payload()
        logger.info("Placing Telnyx Call Control call to %s, eva_call_id=%s", self.to, self._eva_call_id)

        response = await self._post("/calls", payload)
        data = response.get("data", {})
        self._call_control_id = data.get("call_control_id")
        self._call_session_id = data.get("call_session_id")
        self._call_leg_id = data.get("call_leg_id")
        if not self._call_control_id:
            raise RuntimeError("Telnyx call creation response did not include data.call_control_id")
        if not self._call_session_id:
            raise RuntimeError("Telnyx call creation response did not include data.call_session_id")

        await asyncio.wait_for(self._connected_event.wait(), timeout=self.connect_timeout_seconds)
        logger.info("Telnyx media stream connected for conversation %s", self.conversation_id)

    def build_call_payload(self) -> dict[str, Any]:
        stream_base = self.webhook_base_url.replace("https://", "wss://").replace("http://", "ws://")
        stream_wss_url = f"{stream_base}/media-stream/{self.conversation_id}"
        return {
            "connection_id": self.connection_id,
            "to": self.to,
            "from": self.from_number,
            "stream_url": stream_wss_url,
            "stream_track": "both_tracks",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "L16",
            "stream_bidirectional_sampling_rate": TELNYX_SAMPLE_RATE,
            "custom_headers": [
                {
                    "name": "X-Eva-Call-Id",
                    "value": self._eva_call_id or "",
                }
            ],
        }

    async def stop(self) -> None:
        if self._call_control_id and not self._disconnected_event.is_set():
            try:
                await self._post(f"/calls/{self._call_control_id}/actions/hangup", {})
            except Exception as exc:
                logger.warning("Failed to hang up Telnyx call %s: %s", self._call_control_id, exc)

        if self._stream_ws is not None:
            try:
                await self._stream_ws.close()
            except Exception as exc:
                logger.debug("Ignoring Telnyx media stream close error: %s", exc)
            self._stream_ws = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        self._connected_event.clear()

    async def send_audio(self, pcm_16k: bytes) -> None:
        if not pcm_16k:
            return
        async with self._send_lock:
            if self._stream_ws is None:
                logger.debug("Dropping outbound Telnyx audio because media stream is not connected")
                return
            try:
                outbound = self._convert_outbound_audio(pcm_16k)
                await self._stream_ws.send_text(
                    json.dumps(
                        {
                            "event": "media",
                            "media": {
                                "payload": base64.b64encode(outbound).decode("ascii"),
                            },
                        }
                    )
                )
            except Exception:
                logger.info("Telnyx media stream disconnected while sending audio for %s", self.to)

    async def handle_media_stream(self, websocket: WebSocket) -> None:
        self._stream_ws = websocket
        self._connected_event.set()
        self._disconnected_event.clear()

        try:
            while True:
                raw_message = await websocket.receive_text()
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON Telnyx media stream message")
                    continue

                event_type = message.get("event")
                if event_type == "media":
                    payload_b64 = message.get("media", {}).get("payload")
                    if payload_b64:
                        raw_audio = base64.b64decode(payload_b64)
                        await self.audio_handler(self._convert_inbound_audio(raw_audio))
                elif event_type == "start":
                    self._stream_id = message.get("stream_id")
                    media_format = message.get("start", {}).get("media_format", {})
                    if isinstance(media_format, dict) and media_format:
                        self._inbound_encoding = media_format.get("encoding", "L16")
                        self._inbound_sample_rate = int(media_format.get("sample_rate", TELNYX_SAMPLE_RATE))
                    logger.info(
                        "Telnyx media stream started: stream_id=%s codec=%s@%s",
                        self._stream_id,
                        self._inbound_encoding,
                        self._inbound_sample_rate,
                    )
                elif event_type == "stop":
                    logger.info("Telnyx media stream stopped for conversation %s", self.conversation_id)
                    break
                elif event_type == "connected":
                    logger.debug("Telnyx media stream connected event for %s", self.conversation_id)
                else:
                    logger.debug("Unknown Telnyx media stream event: %s", event_type)
        finally:
            self._stream_ws = None
            self._disconnected_event.set()

    def _convert_inbound_audio(self, raw_audio: bytes) -> bytes:
        enc = self._inbound_encoding
        rate = self._inbound_sample_rate
        if enc in {"PCMU", "audio/x-mulaw"}:
            pcm_audio = audioop.ulaw2lin(raw_audio, PCM_SAMPLE_WIDTH)
            if rate and rate != TELNYX_SAMPLE_RATE:
                pcm_audio, _ = audioop.ratecv(pcm_audio, PCM_SAMPLE_WIDTH, 1, rate, TELNYX_SAMPLE_RATE, None)
            return pcm_audio
        if enc in {"L16", "audio/L16"}:
            if rate and rate != TELNYX_SAMPLE_RATE:
                pcm_audio, _ = audioop.ratecv(raw_audio, PCM_SAMPLE_WIDTH, 1, rate, TELNYX_SAMPLE_RATE, None)
                return pcm_audio
            return raw_audio
        return raw_audio

    def _convert_outbound_audio(self, pcm_16k: bytes) -> bytes:
        if self._inbound_sample_rate >= TELNYX_SAMPLE_RATE:
            return pcm_16k
        converted, _ = audioop.ratecv(
            pcm_16k,
            PCM_SAMPLE_WIDTH,
            1,
            TELNYX_SAMPLE_RATE,
            self._inbound_sample_rate,
            None,
        )
        return converted

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Telnyx HTTP session is not initialized")
        async with self._session.post(f"{self.api_v2_base}{path}", json=payload) as response:
            try:
                body = await response.json()
            except aiohttp.ContentTypeError:
                body = {"message": await response.text()}
            if response.status >= 400:
                raise RuntimeError(f"Telnyx API {response.status} at {path}: {json.dumps(body)}")
            return body if isinstance(body, dict) else {"data": body}


class TelnyxAssistantServer(AbstractAssistantServer):
    """Bridge EVA's Twilio-framed local simulator to a hosted Telnyx assistant."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audio_sample_rate = RECORDING_SAMPLE_RATE

        self.s2s_params = self.pipeline_config.s2s_params or {}
        self._api_key = self.s2s_params.get("api_key") or self.s2s_params.get("telnyx_api_key") or ""
        self._api_v2_base = _normalize_api_v2_base(self.s2s_params.get("api_base"))
        self._webhook_base_url = (
            self.s2s_params.get("webhook_base_url") or self.s2s_params.get("public_base_url") or ""
        ).rstrip("/")
        self._assistant_id = self.s2s_params.get("assistant_id") or self.s2s_params.get("assistant_agent_id") or ""
        self._target_version_id = self.s2s_params.get("target_version_id") or self.s2s_params.get("targetVersionId")
        self._websocket_host = (
            self.s2s_params.get("host")
            or self.s2s_params.get("websocket_host")
            or self.s2s_params.get("base_websocket_host")
            or TELNYX_PROD_RTC_HOST
        )
        self._target_params = _require_dict(
            self.s2s_params.get("target_params") or self.s2s_params.get("targetParams"),
            "target_params",
        )
        self._user_variables = _require_dict(
            self.s2s_params.get("user_variables") or self.s2s_params.get("userVariables"),
            "user_variables",
        )
        self._caller_name = self.s2s_params.get("caller_name") or "EVA User"
        self._caller_number = (
            self.s2s_params.get("caller_number") or self.s2s_params.get("from_number") or "+15555550100"
        )
        self._destination_number = self.s2s_params.get("destination_number") or "ai_assistant"
        self._client_state = self.s2s_params.get("client_state") or None
        self._custom_headers = _normalize_telnyx_headers(self.s2s_params.get("custom_headers"))
        self._model = self.s2s_params.get("model") or TelnyxAssistantManager.DEFAULT_MODEL
        self._voice = self.s2s_params.get("voice") or TelnyxAssistantManager.DEFAULT_VOICE
        self._stt_model = self.s2s_params.get("stt_model") or TelnyxAssistantManager.DEFAULT_STT_MODEL

        self._transport: TelnyxWebRTCClient | TelnyxCallControlTransport | None = None
        self._completed_transport: TelnyxWebRTCClient | TelnyxCallControlTransport | None = None
        self._created_assistant_id: str | None = None
        self._conversation_map: dict[str, str] = {}
        self._transcript_enriched = False
        self._user_events_enriched = False
        self._stream_sid = self.conversation_id
        self._user_speaking = False
        self._last_user_speech_stop_ts: str | None = None
        self._last_assistant_audio_monotonic: float | None = None

    async def start(self) -> None:
        if self._running:
            logger.warning("Telnyx server already running")
            return

        self._validate_runtime_config()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        if self.s2s_params.get("create_assistant"):
            if not self._api_key:
                raise ValueError("Telnyx create_assistant requires api_key or telnyx_api_key")
            manager = TelnyxAssistantManager(
                self._api_key, api_base=self.s2s_params.get("api_base") or TELNYX_DEFAULT_API_BASE
            )
            try:
                self._created_assistant_id = await manager.create_benchmark_assistant(
                    agent_config=self.agent,
                    agent_config_path=self.agent_config_path,
                    webhook_base_url=self._webhook_base_url or None,
                    model=self._model,
                    voice=self._voice,
                    stt_model=self._stt_model,
                )
                self._assistant_id = self._created_assistant_id
                logger.info("Created Telnyx assistant %s", self._created_assistant_id)
            finally:
                await manager.close()

        self._app = FastAPI()
        self._register_routes(self._app)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info("Telnyx server started on ws://localhost:%s", self.port)

    async def _shutdown(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._transport is not None:
            await self._transport.stop()
            self._completed_transport = self._transport
            self._transport = None

        if self._created_assistant_id:
            manager = TelnyxAssistantManager(
                self._api_key,
                api_base=self.s2s_params.get("api_base") or TELNYX_DEFAULT_API_BASE,
            )
            try:
                await manager.delete_assistant(self._created_assistant_id)
                logger.info("Deleted Telnyx assistant %s", self._created_assistant_id)
            except Exception:
                logger.warning("Failed to delete Telnyx assistant %s", self._created_assistant_id, exc_info=True)
            finally:
                await manager.close()

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        logger.info("Telnyx server stopped on port %s", self.port)

    async def save_outputs(self) -> None:
        await self._enrich_audit_log_from_external_sources()
        await super().save_outputs()

    def get_conversation_stats(self) -> dict[str, Any]:
        self._append_user_events_from_simulator()
        return super().get_conversation_stats()

    @property
    def _use_call_control(self) -> bool:
        """Whether to use Call Control transport instead of WebRTC."""
        return bool(self.s2s_params.get("connection_id") and self._webhook_base_url)

    def _validate_runtime_config(self) -> None:
        missing: list[str] = []
        if self._use_call_control:
            if not self._api_key:
                missing.append("api_key or telnyx_api_key")
            if not self.s2s_params.get("connection_id"):
                missing.append("connection_id")
            if not (self.s2s_params.get("from_number") or self.s2s_params.get("caller_number")):
                missing.append("from_number or caller_number")
            if not (
                self.s2s_params.get("to")
                or self.s2s_params.get("to_number")
                or self.s2s_params.get("destination_number")
            ):
                missing.append("to or to_number or destination_number")
            if not self._webhook_base_url:
                missing.append("webhook_base_url or public_base_url")
            if missing:
                raise ValueError(
                    "Telnyx Call Control mode requires s2s_params: "
                    + ", ".join(missing)
                    + "."
                )
        else:
            if not self._assistant_id and not self.s2s_params.get("create_assistant"):
                missing.append("assistant_id or assistant_agent_id")
            if self.s2s_params.get("create_assistant") and not self._api_key:
                missing.append("api_key or telnyx_api_key")
            if missing:
                raise ValueError(
                    "Telnyx direct WebRTC mode requires s2s_params: "
                    + ", ".join(missing)
                    + ". Call Control fields such as webhook_base_url, connection_id, from_number, "
                    "and to/sip_uri are not required for direct mode."
                )

    def _register_routes(self, app: FastAPI) -> None:
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        @app.websocket("/")
        async def websocket_root(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        @app.websocket("/media-stream/{conversation_id:path}")
        async def media_stream(websocket: WebSocket, conversation_id: str) -> None:
            if not isinstance(self._transport, TelnyxCallControlTransport) or conversation_id != self.conversation_id:
                await websocket.close(code=1008, reason="No active Telnyx transport for this conversation")
                return
            await websocket.accept()
            try:
                await self._transport.handle_media_stream(websocket)
            except WebSocketDisconnect:
                logger.info("Telnyx media stream WebSocket disconnected for %s", conversation_id)

        @app.post("/tools/{eva_call_id}/{tool_name}")
        async def invoke_tool(eva_call_id: str, tool_name: str, request: Request) -> Any:
            transport_call_id = self._completed_transport.eva_call_id if self._completed_transport else None
            active_call_id = self._transport.eva_call_id if self._transport else transport_call_id
            if active_call_id and eva_call_id != active_call_id:
                logger.warning("Tool webhook route id mismatch: got=%s expected=%s", eva_call_id, active_call_id)
                raise HTTPException(status_code=404, detail=f"Unknown call_id: {eva_call_id}")

            try:
                body = await request.json()
                params = _extract_tool_params(body)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON request body") from exc

            result = await self.execute_tool(tool_name, params)
            if tool_name == "end_call":
                logger.info("end_call tool invoked for Telnyx call %s", eva_call_id)
                asyncio.create_task(self._hangup_active_call())
            return result

        @app.post("/dynamic-variables")
        async def dynamic_variables(request: Request) -> dict[str, Any]:
            try:
                body = await request.json()
            except Exception:
                body = {}
            return self._handle_dynamic_variables_payload(body)

        @app.post("/call-control-events")
        async def call_control_events(request: Request) -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

    async def _handle_session(self, websocket: WebSocket) -> None:
        logger.info("Client connected to Telnyx server")
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        async def on_telnyx_audio(pcm_16k: bytes) -> None:
            await self._on_telnyx_audio(pcm_16k, audio_output_queue)

        if self._use_call_control:
            logger.info(
                "Using Telnyx Call Control transport (connection_id=%s, to=%s)",
                self.s2s_params.get("connection_id"),
                self.s2s_params.get("to") or self.s2s_params.get("to_number") or self._destination_number,
            )
            self._transport = TelnyxCallControlTransport(
                api_key=self._api_key,
                to=self.s2s_params.get("to") or self.s2s_params.get("to_number") or self._destination_number,
                connection_id=self.s2s_params["connection_id"],
                from_number=self._caller_number,
                conversation_id=self.conversation_id,
                webhook_base_url=self._webhook_base_url,
                api_v2_base=self._api_v2_base,
                audio_handler=on_telnyx_audio,
            )
        else:
            self._transport = TelnyxWebRTCClient(
                config=self._build_direct_session_config(),
                audio_handler=on_telnyx_audio,
            )
            self._transport._conversation_event_handler = self._handle_telnyx_conversation_event
        self._completed_transport = self._transport

        pacer_task = asyncio.create_task(self._pace_audio_output(websocket, audio_output_queue))
        forward_task: asyncio.Task[None] | None = None
        disconnect_task: asyncio.Task[bool] | None = None

        try:
            await self._transport.start()
            if self._transport.eva_call_id:
                logger.info("Telnyx direct WebRTC call id: %s", self._transport.eva_call_id)

            forward_task = asyncio.create_task(self._forward_user_audio(websocket, self._transport))
            disconnect_task = asyncio.create_task(self._transport.disconnected_event.wait())
            done, pending = await asyncio.wait(
                [forward_task, disconnect_task, pacer_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            for task in done:
                task_exc = task.exception()
                if task is not pacer_task and task_exc is not None:
                    raise task_exc
        except Exception as exc:
            logger.error("Telnyx session error: %s", exc, exc_info=True)
        finally:
            if forward_task is not None and not forward_task.done():
                forward_task.cancel()
            if disconnect_task is not None and not disconnect_task.done():
                disconnect_task.cancel()
            pacer_task.cancel()
            try:
                await pacer_task
            except asyncio.CancelledError:
                pass
            if self._transport is not None:
                await self._transport.stop()
            logger.info("Client disconnected from Telnyx server")

    async def _handle_telnyx_conversation_event(self, params: dict[str, Any]) -> None:
        event_type = params.get("type")
        item = params.get("item") if isinstance(params.get("item"), dict) else {}
        if event_type != "conversation.item.created" or not item:
            return

        role = item.get("role")
        text = _conversation_item_text(item)
        if not text:
            return

        timestamp_ms = _wall_ms()
        if role == "assistant":
            self.audit_log.append_assistant_output(text, timestamp_ms=timestamp_ms)
            if self._fw_log:
                self._fw_log.llm_response(text, timestamp_ms=int(timestamp_ms))
                self._fw_log.turn_end(was_interrupted=False)
        elif role == "user":
            self.audit_log.append_user_input(text, timestamp_ms=timestamp_ms)
        else:
            logger.debug("Ignoring Telnyx conversation item role=%s text=%s", role, text[:80])

    def _build_direct_session_config(self) -> TelnyxDirectSessionConfig:
        if not self._assistant_id:
            raise ValueError("Telnyx direct WebRTC mode requires assistant_id or assistant_agent_id")
        return TelnyxDirectSessionConfig(
            assistant_id=self._assistant_id,
            host=self._websocket_host,
            target_version_id=self._target_version_id,
            target_params=self._target_params,
            user_variables=self._user_variables,
            caller_name=str(self._caller_name),
            caller_number=str(self._caller_number),
            destination_number=str(self._destination_number),
            client_state=str(self._client_state) if self._client_state else None,
            custom_headers=self._custom_headers,
        )

    def _build_telnyx_tool_definitions(self) -> list[dict[str, Any]]:
        return [self._build_telnyx_tool_definition(tool) for tool in self.agent.tools]

    @staticmethod
    def _build_telnyx_tool_definition(tool: AgentTool) -> dict[str, Any]:
        return {
            "type": "client_tool",
            "name": tool.id,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.get_parameter_properties(),
                "required": tool.get_required_param_names(),
            },
        }

    async def _execute_telnyx_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        logger.warning("Telnyx direct client tool execution is stubbed and not yet wired to Verto events")
        return await self.execute_tool(tool_name, arguments)

    async def _forward_user_audio(
        self,
        websocket: WebSocket,
        transport: TelnyxWebRTCClient | TelnyxCallControlTransport,
    ) -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = data.get("event")

                if event_type == "start":
                    self._stream_sid = data.get("start", {}).get("streamSid") or self.conversation_id
                    continue
                if event_type == "stop":
                    break
                if event_type == "user_speech_start":
                    self._user_speaking = True
                    timestamp_ms = data.get("timestamp_ms")
                    if self._fw_log and _is_epoch_ms(timestamp_ms):
                        self._fw_log.turn_start(timestamp_ms=int(timestamp_ms))
                    elif self._fw_log:
                        self._fw_log.turn_start()
                    continue
                if event_type == "user_speech_stop":
                    self._user_speaking = False
                    timestamp_ms = str(data.get("timestamp_ms") or "")
                    self._last_user_speech_stop_ts = timestamp_ms if _is_epoch_ms(timestamp_ms) else _wall_ms()
                    continue
                if event_type != "media":
                    continue

                mulaw_bytes = parse_twilio_media_message(raw)
                if not mulaw_bytes:
                    continue

                pcm_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)
                if not self._is_assistant_audio_active():
                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                self.user_audio_buffer.extend(pcm_24k)

                pcm_16k = mulaw_8k_to_pcm16_16k(mulaw_bytes)
                await transport.send_audio(pcm_16k)
        except WebSocketDisconnect:
            logger.debug("Telnyx local WebSocket disconnected")
        except asyncio.CancelledError:
            pass

    async def _on_telnyx_audio(self, pcm_16k: bytes, audio_output_queue: asyncio.Queue[bytes]) -> None:
        if not pcm_16k:
            return

        # Telnyx streams RTP continuously (~50 pps) for the whole call, so between turns we
        # decode frames of pure silence. The user simulator treats EVERY media frame as
        # assistant speech -- it never inspects the payload, and infers the turn ended only
        # from the ABSENCE of frames (audio_bridge._receive_from_assistant). Forwarding the
        # silence therefore pins is_assistant_playing() on for the entire call: the caller's
        # turn never settles, it never speaks, and the record fails with zero user turns.
        # Drop silent frames so the simulator sees a real gap and can take its turn.
        if audioop.rms(pcm_16k, PCM_SAMPLE_WIDTH) <= ASSISTANT_SILENCE_RMS:
            return

        self._last_assistant_audio_monotonic = time.monotonic()
        first_audio_wall_ms = _wall_ms()
        if self._last_user_speech_stop_ts and self._metrics_log:
            latency_ms = int(first_audio_wall_ms) - int(self._last_user_speech_stop_ts)
            if 0 < latency_ms < 30_000:
                self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)
            self._last_user_speech_stop_ts = None

        pcm_24k = _pcm16_16k_to_pcm16_24k(pcm_16k)
        if not self._user_speaking:
            sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
        self.assistant_audio_buffer.extend(pcm_24k)

        mulaw = _pcm16_16k_to_mulaw_8k(pcm_16k)
        offset = 0
        while offset < len(mulaw):
            chunk = mulaw[offset : offset + MULAW_CHUNK_SIZE]
            offset += MULAW_CHUNK_SIZE
            await audio_output_queue.put(chunk)

    async def _pace_audio_output(self, websocket: WebSocket, audio_output_queue: asyncio.Queue[bytes]) -> None:
        next_send_time = time.monotonic()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                await websocket.send_text(create_twilio_media_message(self._stream_sid, chunk))
                now = time.monotonic()
                if next_send_time <= now:
                    next_send_time = now
                next_send_time += MULAW_CHUNK_DURATION_S
                sleep_duration = next_send_time - time.monotonic()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.info("Telnyx audio pacer stopped: %s", exc)

    def _handle_dynamic_variables_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = body.get("data", {}).get("payload", {}) if isinstance(body, dict) else {}
        if not isinstance(payload, dict):
            payload = {}

        eva_call_id = payload.get("eva_call_id") or (self._transport.eva_call_id if self._transport else None)
        telnyx_conversation_id = payload.get("telnyx_conversation_id")
        call_control_id = payload.get("call_control_id")

        logger.info(
            "Telnyx dynamic variables: eva_call_id=%s conversation_id=%s call_control_id=%s",
            eva_call_id,
            telnyx_conversation_id,
            call_control_id,
        )

        if eva_call_id and telnyx_conversation_id:
            self._conversation_map[str(eva_call_id)] = str(telnyx_conversation_id)

        response: dict[str, Any] = {"dynamic_variables": {}}
        if eva_call_id:
            response["dynamic_variables"]["eva_call_id"] = str(eva_call_id)
            metadata = {"eva_call_id": str(eva_call_id), "eva_record_id": self.conversation_id}
            if self._model:
                metadata["eva_llm_model"] = self._model
            response["conversation"] = {"metadata": metadata}
        return response

    async def _hangup_active_call(self) -> None:
        if self._transport is not None:
            await self._transport.stop()

    def _is_assistant_audio_active(self) -> bool:
        if self._last_assistant_audio_monotonic is None:
            return False
        return time.monotonic() - self._last_assistant_audio_monotonic < 0.5

    async def _enrich_audit_log_from_external_sources(self) -> None:
        if self._transcript_enriched:
            return
        self._transcript_enriched = True

        self._append_user_events_from_simulator()

        events_path = resolve_user_simulator_events_path(self.output_dir)
        assistant_messages = await self._fetch_telnyx_intended_speech()
        if not assistant_messages and events_path is not None:
            assistant_messages = extract_assistant_speech_events(events_path)

        for message in assistant_messages:
            text = str(message.get("text", "")).strip()
            timestamp_ms = str(message.get("timestamp_ms", ""))
            if not text:
                continue
            self.audit_log.append_assistant_output(text, timestamp_ms=timestamp_ms)
            if self._fw_log:
                self._fw_log.llm_response(text, timestamp_ms=int(timestamp_ms) if timestamp_ms.isdigit() else None)
                self._fw_log.turn_end(was_interrupted=False)

    def _append_user_events_from_simulator(self) -> None:
        if self._user_events_enriched:
            return
        self._user_events_enriched = True

        events_path = resolve_user_simulator_events_path(self.output_dir)
        if events_path is None:
            logger.warning("No user simulator events file found in %s; audit log will have no user turns", self.output_dir)
            return

        for user_turn in extract_user_speech_events(events_path):
            timestamp_ms = user_turn.get("timestamp_ms")
            self.audit_log.append_user_input(user_turn["text"], timestamp_ms=timestamp_ms)

    async def _fetch_telnyx_intended_speech(self) -> list[dict[str, Any]]:
        if not self._api_key:
            logger.warning("No Telnyx API key configured; skipping intended speech fetch")
            return []

        telnyx_conversation_id = self._resolve_telnyx_conversation_id()
        if not telnyx_conversation_id:
            logger.warning("No Telnyx conversation_id captured; skipping intended speech fetch")
            return []

        url = f"{self._api_v2_base}/ai/conversations/{telnyx_conversation_id}/messages?page[size]=100"
        timeout = aiohttp.ClientTimeout(total=30.0)
        try:
            async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._api_key}"}, timeout=timeout
            ) as session:
                async with session.get(url) as response:
                    try:
                        payload = await response.json()
                    except aiohttp.ContentTypeError:
                        payload = {"message": await response.text()}
                    if response.status >= 400:
                        logger.warning(
                            "Failed to fetch Telnyx conversation messages for %s: %s %s",
                            telnyx_conversation_id,
                            response.status,
                            payload,
                        )
                        return []
        except Exception as exc:
            logger.warning("Failed to fetch Telnyx conversation messages for %s: %s", telnyx_conversation_id, exc)
            return []

        if not isinstance(payload, dict):
            return []
        return parse_telnyx_intended_speech_payload(payload)

    def _resolve_telnyx_conversation_id(self) -> str | None:
        transport = self._completed_transport or self._transport
        route_id = transport.eva_call_id if transport else None
        if route_id:
            return self._conversation_map.get(route_id)
        if len(self._conversation_map) == 1:
            return next(iter(self._conversation_map.values()))
        return None


__all__ = [
    "TelnyxAssistantManager",
    "TelnyxAssistantServer",
    "TelnyxCallControlTransport",
    "TelnyxDirectSessionConfig",
    "TelnyxVertoJsonRpcClient",
    "TelnyxWebRTCClient",
    "extract_assistant_speech_events",
    "extract_user_speech_events",
    "parse_telnyx_intended_speech_payload",
]
