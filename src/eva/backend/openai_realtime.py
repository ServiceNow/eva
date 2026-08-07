"""OpenAI Realtime ``Backend``: normalizing adapter for one provider session.

Wraps a single OpenAI Realtime API session behind the role-agnostic ``Backend``
contract (see ``eva.backend.base``). It knows nothing about whether it drives
an ``AssistantRole`` or a ``UserRole``; both use the *same* backend and differ
only in the ``session_config`` the worker constructs it with and in how they
interpret the clean events it emits.

Responsibilities (all provider-specific work lives here, so roles stay
generic):
- session lifecycle: connect / ``session.update`` / stream audio in / events
  out / tool results / close;
- session-config assembly (voice, VAD, formats, transcription) from the
  worker-supplied ``session_config`` -- roles never build provider config;
- tool-schema translation: generic ``{name, description, parameters}`` specs
  -> OpenAI Realtime ``session.tools`` shape;
- **full event normalization**: every provider event is parsed here and
  surfaced as a clean ``BackendEvent`` (typed fields + normalized ``metadata``
  scalars). Roles never see a raw OpenAI event. Provider-specific bookkeeping
  (output-transcript accumulation, final-text selection, interruption
  detection, token-usage extraction) happens here, with per-turn state carried
  on the ``OpenAIRealtimeSession`` handle (the backend object stays stateless).

Not covered here (role concerns): the transport to the counterparty (the
Twilio WS server / audio-bridge client), audio format conversion for that
transport, recording, prompt building, and tool execution.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from openai import AsyncOpenAI

from eva.backend.base import (
    Backend,
    BackendEvent,
    BackendEventType,
    BackendSession,
    ToolCallRequest,
    ToolCallResult,
)
from eva.backend.capabilities import BackendCapabilities
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE = 24000
PCMU_SAMPLE_RATE = 8000


@dataclass
class OpenAIRealtimeSession(BackendSession):
    """Live state for one OpenAI Realtime session.

    Carries the SDK client, the entered realtime-connection context manager +
    connection, and the per-turn accumulators the normalizer needs (output
    transcript parts, whether a response is in flight, whether it produced tool
    calls). All of this is session state, deliberately off the backend object.
    """

    client: AsyncOpenAI
    conn_cm: Any
    conn: Any
    responding: bool = False
    output_transcript_parts: list[str] = field(default_factory=list)
    output_transcript_done: str = ""
    has_function_calls: bool = False


class OpenAIRealtimeBackend(Backend):
    """One OpenAI Realtime session behind the role-agnostic ``Backend`` contract.

    Construction is cheap and network-free (client + connection are created in
    ``open()``), matching ``BackendFactory.create``'s "not-yet-opened" contract.

    Takes a single flat ``config`` of "config things" and assembles the
    OpenAI ``session.update`` structure itself -- the caller (worker via the
    factory) never hand-builds the provider JSON. Recognized keys:

    - ``model`` (required). ``api_key`` (optional): falls back to the
      ``OPENAI_API_KEY`` env var if not provided. ``base_url`` (optional).
    - ``accent``: if set, rejected -- this backend can't honor accents (they
      are realized via ElevenLabs agent IDs). Fails loud, mirroring the old
      ``OpenAIRealtimeUserSimulator`` guard, now backend-side.
    - ``voice`` (default ``"marin"``), ``output_sample_rate`` (default 24000).
    - ``input_format``: ``"pcm"`` (default) or ``"pcmu"`` (telephony/caller).
    - ``vad_settings``: turn-detection tunables (``type`` / ``threshold`` /
      ``prefix_padding_ms`` / ``silence_duration_ms``), defaults applied. Named
      to match EVA's ``s2s_params["vad_settings"]`` so an assistant can pass its
      provider params straight through.
    - ``manual_turn_taking``: when True, the model does not auto-create
      responses -- adds ``create_response``/``interrupt_response`` false and an
      idle timeout (a caller gates replies itself via ``trigger_response``).
    - ``transcription_model`` (default ``"whisper-1"``),
      ``transcription_language`` (optional).
    - ``reasoning_effort`` (optional), ``parallel_tool_calls`` (optional).
    """

    _CAPABILITIES = BackendCapabilities(
        emits_continuous_audio=True,
        supports_streaming_interruption=True,
        owns_playout_clock=False,
    )

    # Session handle class ``open()`` instantiates. Subclasses for API-compatible
    # providers (e.g. Grok Voice) override this to carry extra per-turn state.
    _SESSION_CLS: ClassVar[type[OpenAIRealtimeSession]] = OpenAIRealtimeSession

    # Env var the api_key falls back to when not supplied in config. Subclasses
    # for other OpenAI-compatible providers override it (e.g. Grok -> XAI_API_KEY).
    _API_KEY_ENV: ClassVar[str] = "OPENAI_API_KEY"

    # Extra turn-detection fields when the caller gates responses manually.
    _MANUAL_TURN_DETECTION = {"create_response": False, "interrupt_response": False, "idle_timeout_ms": 15_000}

    def __init__(self, *, config: dict[str, Any]) -> None:
        api_key = config.get("api_key") or os.environ.get(self._API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires an api_key (config['api_key'] or {self._API_KEY_ENV})")
        if config.get("accent") is not None:
            raise ValueError("OpenAI Realtime backend does not support accent variants")
        self._model: str = config.get("model") or ""
        if not self._model:
            raise ValueError(f"{type(self).__name__} requires a 'model' (config['model'])")
        self._api_key = api_key
        self._base_url = config.get("base_url")
        self._input_format: str = config.get("input_format", "pcm")
        self._output_sample_rate = int(config.get("output_sample_rate", DEFAULT_SAMPLE_RATE))
        self._session_config = self._assemble_session_config(config)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES

    @property
    def output_sample_rate(self) -> int:
        """Sample rate (Hz) of ``AUDIO_OUTPUT`` payloads."""
        return self._output_sample_rate

    @property
    def input_sample_rate(self) -> int:
        """Sample rate (Hz) the session expects for ``send(audio=...)`` input."""
        return PCMU_SAMPLE_RATE if self._input_format == "pcmu" else self._output_sample_rate

    def _assemble_session_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build the OpenAI session-shaping block (minus type/instructions/tools) from flat config."""
        input_fmt: dict[str, Any] = (
            {"type": "audio/pcmu"}
            if self._input_format == "pcmu"
            else {"type": "audio/pcm", "rate": self._output_sample_rate}
        )
        vad = config.get("vad_settings") or {}
        turn_detection = {
            "type": vad.get("type", "server_vad"),
            "threshold": vad.get("threshold", 0.5),
            "prefix_padding_ms": vad.get("prefix_padding_ms", 300),
            "silence_duration_ms": vad.get("silence_duration_ms", 200),
        }
        if config.get("manual_turn_taking"):
            turn_detection.update(self._MANUAL_TURN_DETECTION)

        transcription: dict[str, Any] = {"model": config.get("transcription_model", "whisper-1")}
        if config.get("transcription_language"):
            transcription["language"] = config["transcription_language"]

        session_config: dict[str, Any] = {
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "voice": config.get("voice", "marin"),
                    "format": {"type": "audio/pcm", "rate": self._output_sample_rate},
                },
                "input": {"format": input_fmt, "turn_detection": turn_detection, "transcription": transcription},
            },
        }
        if config.get("reasoning_effort"):
            session_config["reasoning"] = {"effort": config["reasoning_effort"]}
        if config.get("parallel_tool_calls") is not None:
            session_config["parallel_tool_calls"] = config["parallel_tool_calls"]
        return session_config

    # ── Session lifecycle ─────────────────────────────────────────────

    async def open(self, *, system_prompt: str, tools: list[dict[str, Any]] | None) -> OpenAIRealtimeSession:
        """Connect and configure a new OpenAI Realtime session; return its handle."""
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url
        client = AsyncOpenAI(**client_kwargs)

        conn_cm = client.realtime.connect(model=self._model)
        conn = await conn_cm.__aenter__()

        session_update = self._build_session_update(system_prompt, tools)
        await conn.session.update(session=session_update)  # type: ignore[arg-type]
        logger.info(f"OpenAI Realtime session opened (model={self._model})")
        return self._SESSION_CLS(client=client, conn_cm=conn_cm, conn=conn)

    def _build_session_update(self, system_prompt: str, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Finalize the ``session.update`` payload for ``open()``.

        Takes the session-shaping block assembled at construction and stamps the
        per-open fields the backend owns (``type`` / ``instructions`` /
        ``tools``), translating generic tool specs to the provider shape.
        """
        session_update: dict[str, Any] = dict(self._session_config)
        session_update["type"] = "realtime"
        session_update["instructions"] = system_prompt
        session_update["tools"] = self._format_tools(tools)
        return session_update

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Translate generic ``{name, description, parameters}`` specs to OpenAI schema."""
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in (tools or [])
        ]

    async def send(
        self,
        session: BackendSession,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        tool_result: ToolCallResult | None = None,
    ) -> None:
        """Push audio / a text turn / a tool result to ``session`` (exactly one)."""
        provided = [x is not None for x in (audio, text, tool_result)]
        if sum(provided) != 1:
            raise ValueError("send() requires exactly one of audio, text, tool_result")
        conn = self._conn(session)

        if audio is not None:
            await conn.input_audio_buffer.append(audio=base64.b64encode(audio).decode("ascii"))
            return

        if text is not None:
            await conn.conversation.item.create(
                item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
            )
            await conn.response.create()
            return

        assert tool_result is not None  # exactly-one check above
        await conn.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": tool_result.call_id,
                "output": json.dumps(tool_result.result, ensure_ascii=False),
            }
        )
        await conn.response.create()

    async def trigger_response(self, session: BackendSession) -> None:
        """Manually request a model response (caller-gated turn-taking)."""
        await self._conn(session).response.create()

    async def receive(self, session: BackendSession) -> AsyncIterator[BackendEvent]:
        """Yield normalized events from ``session``'s connection until it ends."""
        s = self._session(session)
        async for event in s.conn:
            for be in self._map_event(s, event):
                yield be

    @staticmethod
    def _session(session: BackendSession) -> OpenAIRealtimeSession:
        if not isinstance(session, OpenAIRealtimeSession):
            raise TypeError(f"expected OpenAIRealtimeSession, got {type(session).__name__}")
        return session

    @classmethod
    def _conn(cls, session: BackendSession) -> Any:
        return cls._session(session).conn

    @staticmethod
    def _map_event(session: OpenAIRealtimeSession, event: Any) -> list[BackendEvent]:
        """Normalize one provider event into zero or more clean ``BackendEvent``s.

        Stateful (accumulates output transcript / turn flags on ``session``) so
        it can surface a fully-selected final transcript and detect
        interruption without the role touching raw provider data. Pure of I/O,
        so unit-testable with a fake session + event.
        """
        event_type = getattr(event, "type", "")
        out: list[BackendEvent] = []

        match event_type:
            case "response.created":
                session.responding = True
                session.output_transcript_parts = []
                session.output_transcript_done = ""
                session.has_function_calls = False
                out.append(BackendEvent(event_type=BackendEventType.OUTPUT_TURN_STARTED))

            case "response.output_audio.delta":
                delta_b64 = getattr(event, "delta", "") or ""
                if delta_b64:
                    out.append(
                        BackendEvent(event_type=BackendEventType.AUDIO_OUTPUT, audio=base64.b64decode(delta_b64))
                    )

            case "response.output_audio_transcript.delta":
                session.output_transcript_parts.append(getattr(event, "delta", "") or "")

            case "response.output_audio_transcript.done":
                done = (getattr(event, "transcript", "") or "").strip()
                session.output_transcript_done = done
                text = done or "".join(session.output_transcript_parts).strip()
                out.append(
                    BackendEvent(
                        event_type=BackendEventType.TRANSCRIPT,
                        transcript=text,
                        metadata={"stream": "output", "final": True},
                    )
                )

            case "conversation.item.input_audio_transcription.completed":
                transcript = (getattr(event, "transcript", "") or "").strip()
                if transcript:
                    out.append(
                        BackendEvent(
                            event_type=BackendEventType.TRANSCRIPT,
                            transcript=transcript,
                            metadata={"stream": "input", "final": True},
                        )
                    )

            case "conversation.item.input_audio_transcription.failed":
                out.append(
                    BackendEvent(
                        event_type=BackendEventType.TRANSCRIPT,
                        transcript="",
                        metadata={"stream": "input", "failed": True},
                    )
                )

            case "input_audio_buffer.speech_started":
                # If the inbound party barges in over a response that has already
                # produced text, flush that partial as an interrupted turn before
                # signaling the speech start (mirrors the old flush-then-new-turn order).
                if session.responding and session.output_transcript_parts:
                    partial = "".join(session.output_transcript_parts)
                    out.append(
                        BackendEvent(
                            event_type=BackendEventType.TURN_END,
                            transcript=partial,
                            metadata={
                                "interrupted": True,
                                "cancelled": False,
                                "has_function_calls": session.has_function_calls,
                                "usage": None,
                            },
                        )
                    )
                    session.responding = False
                    session.output_transcript_parts = []
                    session.output_transcript_done = ""
                out.append(BackendEvent(event_type=BackendEventType.INPUT_SPEECH_STARTED))

            case "input_audio_buffer.speech_stopped":
                out.append(BackendEvent(event_type=BackendEventType.INPUT_SPEECH_STOPPED))

            case "response.function_call_arguments.done":
                session.has_function_calls = True
                arguments_str = getattr(event, "arguments", "{}") or "{}"
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {}
                out.append(
                    BackendEvent(
                        event_type=BackendEventType.TOOL_CALL_REQUEST,
                        tool_call_request=ToolCallRequest(
                            call_id=getattr(event, "call_id", "") or "",
                            name=getattr(event, "name", "") or "",
                            arguments=arguments,
                        ),
                    )
                )

            case "response.output_audio.done":
                out.append(BackendEvent(event_type=BackendEventType.OUTPUT_AUDIO_DONE))

            case "response.done":
                response = getattr(event, "response", None)
                cancelled = bool(response and getattr(response, "status", None) == "cancelled")
                final_text = (
                    session.output_transcript_done
                    or "".join(session.output_transcript_parts).strip()
                    or OpenAIRealtimeBackend._extract_response_text(event)
                )
                has_fc = OpenAIRealtimeBackend._response_has_function_calls(event) or session.has_function_calls
                out.append(
                    BackendEvent(
                        event_type=BackendEventType.TURN_END,
                        transcript=final_text,
                        metadata={
                            "cancelled": cancelled,
                            "interrupted": False,
                            "has_function_calls": has_fc,
                            "usage": OpenAIRealtimeBackend._extract_usage(response),
                        },
                    )
                )
                session.responding = False
                session.output_transcript_parts = []
                session.output_transcript_done = ""
                session.has_function_calls = False

            case "error":
                error_data = getattr(event, "error", None)
                code = getattr(error_data, "code", None) if error_data is not None else None
                out.append(
                    BackendEvent(
                        event_type=BackendEventType.ERROR,
                        error=str(error_data) if error_data is not None else "unknown error",
                        metadata={"code": code},
                    )
                )

            case _:
                # session.created/updated, interim transcription deltas, etc.: no
                # cross-role meaning -> drop.
                pass

        return out

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int] | None:
        if not response:
            return None
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        return {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
        }

    @staticmethod
    def _response_has_function_calls(event: Any) -> bool:
        response = getattr(event, "response", None)
        if not response:
            return False
        output_items = getattr(response, "output", None) or []
        return any(getattr(item, "type", "") == "function_call" for item in output_items)

    @staticmethod
    def _extract_response_text(event: Any) -> str:
        response = getattr(event, "response", None)
        if not response:
            return ""
        output_items = getattr(response, "output", None) or []
        text_parts: list[str] = []
        for item in output_items:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") in ("audio", "text"):
                    transcript = getattr(part, "transcript", None) or getattr(part, "text", None) or ""
                    if transcript:
                        text_parts.append(transcript)
        return "".join(text_parts).strip()

    async def close(self, session: BackendSession) -> None:
        """Tear down ``session``. Idempotent: safe to call more than once."""
        s = self._session(session)
        if s.conn_cm is not None:
            try:
                await s.conn_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing OpenAI Realtime connection: {e}")
            finally:
                s.conn_cm = None
                s.conn = None
        if s.client is not None:
            try:
                await s.client.close()
            except Exception as e:
                logger.debug(f"Error closing OpenAI client: {e}")
            finally:
                s.client = None  # type: ignore[assignment]
