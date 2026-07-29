"""aai host-mode session: the wire protocol behind AAIAssistantServer.

aai agents normally execute their tools server-side in a sandbox. Host mode
inverts that: this client supplies the system prompt, greeting, and tool
*schemas* in its first ``config`` frame, and aai relays every tool *call* back
as a ``tool_call`` frame that resolves when the matching ``tool_result``
arrives. That inversion is what lets EVA's ToolExecutor mutate the scenario
database — without it the accuracy metrics would score an unchanged database.

Host mode is selected by the ``?host=1`` query parameter and gated server-side
by ``AAI_ALLOW_HOST``.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import websockets

from eva.assistant.aai_events import (
    AAIAgentTranscriptEvent,
    AAIAudioDoneEvent,
    AAIConfigEvent,
    AAIErrorEvent,
    AAIIdleTimeoutEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAIToolCallEvent,
    AAIUserTranscriptEvent,
    BaseAAIEvent,
    parse_aai_event,
)
from eva.assistant.bridge_events import (
    AssistantTranscript,
    AudioChunk,
    BackendError,
    BridgeEvent,
    SpeechStarted,
    ToolCall,
    TurnDone,
    UserTranscript,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_AAI_WS_URL = "ws://localhost:3000/websocket"
DEFAULT_AAI_INPUT_SAMPLE_RATE = 16000
DEFAULT_AAI_OUTPUT_SAMPLE_RATE = 24000

#: aai host mode is endpoint-determined: the model is whatever the host runs.
DEFAULT_AAI_MODEL = "aai-host"

CONFIG_ACK_TIMEOUT_S = 15.0


class AAIHostSessionError(Exception):
    """The aai host refused or failed the host-mode handshake."""


def with_host_flag(url: str) -> str:
    """Return *url* with ``host=1``, activating host mode on the aai server."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params["host"] = ["1"]
    return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))


def _explain_connect_error(exc: Exception) -> str:
    """Append a cause to the HTTP rejections a host-mode upgrade can hit.

    The platform answers a host-mode upgrade with 401 when the bearer token is
    absent and 403 when it is valid but does not own the slug; both arrive as an
    opaque ``InvalidStatus`` otherwise.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    hints = {
        401: (
            "host mode on a deployed agent requires the owner's API key — set "
            'EVA_MODEL__S2S_PARAMS \'{"api_key": "..."}\' or AAI_API_KEY '
            "(a local `aai dev` host uses AAI_ALLOW_HOST instead)"
        ),
        403: "the API key is valid but does not own this agent slug",
    }
    hint = hints.get(status) if isinstance(status, int) else None
    return f"{exc} — {hint}" if hint else str(exc)


def build_host_config_message(
    *,
    system_prompt: str,
    tools: list[dict],
    greeting: str | None,
    input_rate: int,
    output_rate: int,
) -> dict:
    """Build the host-mode handshake frame.

    Mirrors ``HostConfigMessageSchema`` in the aai SDK: the audio negotiation
    fields ride alongside the ``host`` block in a single ``config`` frame.
    """
    host: dict[str, Any] = {"systemPrompt": system_prompt, "tools": list(tools)}
    if greeting:
        host["greeting"] = greeting
    return {
        "type": "config",
        "audioFormat": "pcm16",
        "sampleRate": input_rate,
        "ttsSampleRate": output_rate,
        "host": host,
    }


def map_aai_event(event: BaseAAIEvent) -> BridgeEvent | None:
    """Translate an aai wire event into a bridge event, or None to ignore it.

    Both ``reply_done`` and ``audio_done`` map to ``TurnDone``: audio completion
    is the truer end-of-turn signal for a voice call, but a text-only reply
    produces no ``audio_done``, so honoring both avoids a stuck turn. The
    bridge's turn completion is idempotent, making the duplicate harmless.
    """
    if isinstance(event, AAIAgentTranscriptEvent):
        return AssistantTranscript(text=event.text)
    if isinstance(event, AAIUserTranscriptEvent):
        return UserTranscript(text=event.text)
    if isinstance(event, AAIToolCallEvent):
        return ToolCall(call_id=event.tool_call_id, name=event.tool_name, arguments=event.args)
    if isinstance(event, AAISpeechStartedEvent):
        return SpeechStarted()
    if isinstance(event, AAIReplyDoneEvent | AAIAudioDoneEvent):
        return TurnDone()
    if isinstance(event, AAIIdleTimeoutEvent):
        return BackendError(message="aai session idle timeout", fatal=True)
    if isinstance(event, AAIErrorEvent):
        return BackendError(message=f"{event.code or 'error'}: {event.message or ''}".strip())
    return None


class AAIHostSession:
    """One host-mode conversation with an aai voice agent.

    Satisfies ``VoiceBackendSession``. Audio is raw binary PCM16 frames;
    everything else is JSON.
    """

    def __init__(self, ws, input_rate: int, output_rate: int):
        self._ws = ws
        self.backend_input_rate = input_rate
        self.backend_output_rate = output_rate

    @classmethod
    async def connect(
        cls,
        *,
        ws_url: str,
        system_prompt: str,
        tools: list[dict],
        greeting: str | None = None,
        input_rate: int = DEFAULT_AAI_INPUT_SAMPLE_RATE,
        output_rate: int = DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
        api_key: str | None = None,
    ) -> "AAIHostSession":
        """Open a host-mode session and complete the handshake.

        *api_key* is the agent owner's platform API key, sent as a bearer token
        on the upgrade. A deployed agent's WebSocket is otherwise
        unauthenticated, so the platform requires proof of slug ownership before
        honoring prompt and tool overrides — without it, host mode would turn
        any deployed agent into an open LLM proxy billed to its owner. A local
        ``aai dev`` host gates on ``AAI_ALLOW_HOST`` instead and needs no key.

        Raises:
            AAIHostSessionError: the host rejected the handshake, or did not
                acknowledge it within ``CONFIG_ACK_TIMEOUT_S``.
        """
        url = with_host_flag(ws_url)
        logger.info(f"Connecting to aai host at {url} ({len(tools)} tools, api_key={'yes' if api_key else 'no'})")
        # Header only: a query parameter would leak the caller's whole platform
        # credential into proxy logs and Referer headers.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            # max_size=None: TTS audio frames can exceed the 1 MiB default.
            ws = await websockets.connect(url, max_size=None, additional_headers=headers)
        except Exception as e:
            raise AAIHostSessionError(f"Could not connect to aai host at {url}: {_explain_connect_error(e)}") from e

        try:
            await ws.send(
                json.dumps(
                    build_host_config_message(
                        system_prompt=system_prompt,
                        tools=tools,
                        greeting=greeting,
                        input_rate=input_rate,
                        output_rate=output_rate,
                    )
                )
            )
            await cls._await_config_ack(ws, timeout_s=CONFIG_ACK_TIMEOUT_S)
        except Exception:
            with contextlib.suppress(Exception):
                await ws.close()
            raise

        logger.info("aai host-mode handshake complete")
        return cls(ws, input_rate=input_rate, output_rate=output_rate)

    @staticmethod
    async def _await_config_ack(ws, timeout_s: float) -> None:
        """Block until the host acknowledges the config frame."""
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except TimeoutError as e:
                raise AAIHostSessionError(
                    f"aai host did not acknowledge the host-mode config within {timeout_s}s (is AAI_ALLOW_HOST set?)"
                ) from e

            if isinstance(raw, bytes | bytearray):
                continue  # Audio can precede the ack; ignore it.

            try:
                event = parse_aai_event(json.loads(raw))
            except json.JSONDecodeError:
                continue

            if isinstance(event, AAIConfigEvent):
                return
            if isinstance(event, AAIErrorEvent):
                raise AAIHostSessionError(f"aai host rejected host mode: {event.code}: {event.message}")

    async def send_audio(self, pcm: bytes) -> None:
        await self._ws.send(pcm)

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        """Relay a tool result. ``result`` travels as a JSON string, which aai unwraps."""
        await self._ws.send(
            json.dumps(
                {
                    "type": "tool_result",
                    "toolCallId": call_id,
                    "result": json.dumps(result, default=str),
                }
            )
        )

    async def events(self) -> AsyncIterator[BridgeEvent]:
        async for raw in self._ws:
            if isinstance(raw, bytes | bytearray):
                yield AudioChunk(pcm=bytes(raw))
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Discarding non-JSON text frame from aai host")
                continue
            mapped = map_aai_event(parse_aai_event(data))
            if mapped is not None:
                yield mapped

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.close()
