"""ElevenLabs ``Backend``: end-to-end Conversational AI behind the role contract.

ElevenLabs Agents are a *thin, end-to-end* provider (see ``eva.backend.base``):
the agent owns ASR, dialogue policy, turn-taking, and TTS server-side, so this
backend exposes far fewer knobs than the OpenAI Realtime family. It carries the
same shared config core -- ``model`` (a metrics label here), ``api_key`` (with an
``ELEVENLABS_API_KEY`` env fallback) -- plus the one shared *speaker* field:
``speaker_id`` (an ElevenLabs agent id here, the realtime backends' voice analog).
Everything else (VAD / transcription / formats / reasoning) lives inside the
ElevenLabs agent and has no config surface here.

Adapting the SDK to the ``Backend`` contract requires bridging three shape
differences; all of them are absorbed here so ``AssistantRole`` stays generic
and unchanged:

- **callbacks -> event stream**: the SDK is callback-driven
  (agent-response / user-transcript / end-session). Each callback pushes a
  normalized ``BackendEvent`` onto an internal queue that ``receive()`` drains.
- **audio rate**: the SDK speaks 8 kHz mulaw in / 16 kHz PCM out, but the role's
  Twilio transport is uniformly 24 kHz PCM. The backend declares 24 kHz in/out
  and converts internally (24 kHz PCM -> 8 kHz mulaw for the SDK; 16 kHz PCM ->
  24 kHz PCM for ``AUDIO_OUTPUT``), so the role's audio path is untouched.
- **tool execution**: the SDK runs tools itself via ``ClientTools`` handlers.
  To keep tool execution role-side (matching OpenAI/Grok), each handler emits a
  ``TOOL_CALL_REQUEST`` event and awaits a future the role resolves via
  ``send(tool_result=...)`` -- so the ElevenLabs agent's tool call round-trips
  through the same ``AssistantRole`` seam as every other backend.

Greeting: ``AssistantRole`` triggers the opening line with
``send(text="Say: '<greeting>'")`` right after ``open()``. ElevenLabs greets
from an ``initial_message`` dynamic variable set at session start, so the actual
``start_session()`` is deferred to that first ``send(text=...)`` -- the greeting
is unwrapped and injected as the dynamic variable, matching the legacy server.
"""

from __future__ import annotations

import asyncio
import audioop
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    AsyncConversation,
    ClientTools,
    ConversationInitiationData,
)

from eva.assistant.elevenlabs_audio_interface import TwilioAudioBridge
from eva.backend.base import (
    Backend,
    BackendEvent,
    BackendEventType,
    BackendSession,
    ToolCallRequest,
    ToolCallResult,
)
from eva.backend.capabilities import BackendCapabilities
from eva.utils.audio_utils import pcm16_24k_to_mulaw_8k
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "elevenlabs"
ROLE_SAMPLE_RATE = 24000  # the rate AssistantRole's Twilio transport works in
ELEVENLABS_OUTPUT_RATE = 16000  # the SDK delivers assistant audio at 16 kHz


def _pcm16_24k_to_mulaw_8k(pcm_24k: bytes) -> bytes:
    """Convert the role's 24 kHz PCM input into the SDK's 8 kHz mulaw."""
    return pcm16_24k_to_mulaw_8k(pcm_24k)


def _pcm16_16k_to_pcm16_24k(pcm_16k: bytes) -> bytes:
    """Resample the SDK's 16 kHz PCM output up to the role's 24 kHz."""
    pcm_24k, _ = audioop.ratecv(pcm_16k, 2, 1, ELEVENLABS_OUTPUT_RATE, ROLE_SAMPLE_RATE, None)
    return pcm_24k


def _unwrap_greeting(text: str) -> str:
    """Strip the role's ``Say: '<greeting>'`` wrapper down to the raw greeting."""
    prefix = "Say: '"
    if text.startswith(prefix) and text.endswith("'"):
        return text[len(prefix) : -1]
    return text


@dataclass
class ElevenLabsSession(BackendSession):
    """Live state for one ElevenLabs Conversational AI session.

    Holds the SDK client + audio bridge, the queue that ``receive()`` drains, the
    pending tool-call futures (resolved by the role via ``send(tool_result=...)``),
    and the background task pumping SDK output audio into ``AUDIO_OUTPUT`` events.
    The ``AsyncConversation`` itself is created lazily on the first
    ``send(text=...)`` (see the module docstring on greeting/deferred start).
    """

    client: Any
    bridge: Any
    system_prompt: str
    client_tools: Any
    events: asyncio.Queue[BackendEvent] = field(default_factory=asyncio.Queue)
    pending_tools: dict[str, asyncio.Future[ToolCallResult]] = field(default_factory=dict)
    conversation: Any = None
    output_task: asyncio.Task[None] | None = None
    ended: asyncio.Event = field(default_factory=asyncio.Event)


class ElevenLabsBackend(Backend):
    """One ElevenLabs Conversational AI session behind the ``Backend`` contract.

    Recognized ``config`` keys:

    - ``model`` (optional, default ``"elevenlabs"``): a metrics label only --
      ElevenLabs selects the actual model inside the agent.
    - ``api_key`` (optional): falls back to the ``ELEVENLABS_API_KEY`` env var.
    - ``speaker_id`` (**required**): the shared speaker-identifier key (an ElevenLabs
      *agent id* here, the realtime backends' voice-name analog).
    """

    _API_KEY_ENV: ClassVar[str] = "ELEVENLABS_API_KEY"

    # End-to-end provider: continuous audio, no first-class streaming-interruption
    # seam exposed to us, and it does not own the role's playout clock.
    _CAPABILITIES = BackendCapabilities(
        emits_continuous_audio=True, supports_streaming_interruption=False, owns_playout_clock=False
    )

    def __init__(self, *, config: dict[str, Any]) -> None:
        errors: list[str] = []
        self._api_key = self._resolve_api_key(config, errors)
        self._agent_id: str = self._require(config, "speaker_id", errors)
        self._raise_config_errors(errors)
        self._model: str = config.get("model") or DEFAULT_MODEL

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES

    @property
    def input_sample_rate(self) -> int:
        # Declared at the role's rate; the SDK's 8 kHz mulaw conversion is internal.
        return ROLE_SAMPLE_RATE

    @property
    def output_sample_rate(self) -> int:
        # AUDIO_OUTPUT is resampled up to the role's rate (from the SDK's 16 kHz).
        return ROLE_SAMPLE_RATE

    # ── Session lifecycle ─────────────────────────────────────────────

    async def open(self, *, system_prompt: str, tools: list[dict[str, Any]] | None) -> ElevenLabsSession:
        """Build the client, audio bridge, and tool bridge; defer session start.

        The ElevenLabs ``AsyncConversation`` is created on the first
        ``send(text=...)`` (the greeting), which supplies the ``initial_message``
        dynamic variable the agent greets from.
        """
        client = ElevenLabs(api_key=self._api_key, timeout=30.0, httpx_client=httpx.Client(verify=False, timeout=30.0))
        session = ElevenLabsSession(
            client=client,
            bridge=TwilioAudioBridge(),
            system_prompt=system_prompt,
            client_tools=None,
        )
        session.client_tools = self._build_client_tools(session, tools)
        logger.info(f"ElevenLabs backend prepared (agent_id={self._agent_id})")
        return session

    def _build_client_tools(self, session: ElevenLabsSession, tools: list[dict[str, Any]] | None) -> Any:
        """Register each generic tool spec as an SDK ``ClientTool`` that bridges to the role.

        Each handler emits a ``TOOL_CALL_REQUEST`` event and blocks on a future the
        role resolves via ``send(tool_result=...)``, so tool execution stays role-side.

        The ``ClientTools`` is bound to THIS (the caller's) event loop. Without a
        ``loop``, the SDK spins its own loop in a separate thread and runs handlers
        there -- our handler would then touch the main-loop ``session.events`` queue and
        the ``send(tool_result=...)`` future cross-loop, raise, and the SDK would report
        the tool as failed (``is_error``) to the agent. Binding it to the running loop
        keeps the whole tool round-trip on one loop.
        """
        if not tools:
            return None
        client_tools = ClientTools(loop=asyncio.get_running_loop())
        for spec in tools:
            name = spec["name"]

            async def _handle(parameters: dict[str, Any], _name: str = name) -> str:
                return await self._bridge_tool_call(session, _name, parameters)

            client_tools.register(name, _handle, is_async=True)
        return client_tools

    async def _bridge_tool_call(self, session: ElevenLabsSession, name: str, parameters: dict[str, Any]) -> str:
        """Surface an SDK tool call as a role event; await and return the role's result."""
        # The SDK injects tool_call_id into parameters; use it to correlate the result.
        call_id = str(parameters.get("tool_call_id") or f"{name}-{len(session.pending_tools)}")
        arguments = {k: v for k, v in parameters.items() if k != "tool_call_id"}
        future: asyncio.Future[ToolCallResult] = asyncio.get_running_loop().create_future()
        session.pending_tools[call_id] = future
        await session.events.put(
            BackendEvent(
                event_type=BackendEventType.TOOL_CALL_REQUEST,
                tool_call_request=ToolCallRequest(call_id=call_id, name=name, arguments=arguments),
            )
        )
        result = await future
        return json.dumps(result.result, ensure_ascii=False) if isinstance(result.result, dict) else str(result.result)

    async def send(
        self,
        session: BackendSession,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        tool_result: ToolCallResult | None = None,
    ) -> None:
        """Push audio / the greeting / a tool result (exactly one)."""
        provided = [x is not None for x in (audio, text, tool_result)]
        if sum(provided) != 1:
            raise ValueError("send() requires exactly one of audio, text, tool_result")
        s = self._session(session)

        if audio is not None:
            # 24 kHz PCM from the role -> 8 kHz mulaw for the SDK's audio interface.
            await s.bridge.feed_user_audio(_pcm16_24k_to_mulaw_8k(audio))
            return

        if text is not None:
            # The greeting: start the deferred session with it as initial_message.
            await self._start_session(s, greeting=_unwrap_greeting(text))
            return

        assert tool_result is not None  # exactly-one check above
        future = s.pending_tools.pop(tool_result.call_id, None)
        if future is not None and not future.done():
            future.set_result(tool_result)

    async def _start_session(self, session: ElevenLabsSession, *, greeting: str) -> None:
        """Create and start the ElevenLabs conversation, then pump its output audio."""
        if session.conversation is not None:
            return
        conv_config = ConversationInitiationData(
            dynamic_variables={"system_prompt": session.system_prompt, "initial_message": greeting},
        )

        # The SDK expects awaitable callbacks; each just enqueues a normalized event.
        async def _on_agent_response(text: str) -> None:
            self._enqueue(session, self._agent_response_event(text))

        async def _on_agent_response_correction(original: str, corrected: str) -> None:
            self._enqueue(session, self._correction_event(corrected))

        async def _on_user_transcript(text: str) -> None:
            self._enqueue(session, self._user_transcript_event(text))

        async def _on_end_session() -> None:
            session.ended.set()

        session.conversation = AsyncConversation(
            session.client,
            self._agent_id,
            requires_auth=True,
            audio_interface=session.bridge,
            config=conv_config,
            client_tools=session.client_tools,
            callback_agent_response=_on_agent_response,
            callback_agent_response_correction=_on_agent_response_correction,
            callback_user_transcript=_on_user_transcript,
            callback_end_session=_on_end_session,
        )
        await session.conversation.start_session()
        session.output_task = asyncio.create_task(self._pump_output(session))
        logger.info("ElevenLabs conversation session started")

    @staticmethod
    def _enqueue(session: ElevenLabsSession, event: BackendEvent) -> None:
        """Push a normalized event from an SDK callback onto the receive queue."""
        session.events.put_nowait(event)

    @staticmethod
    def _agent_response_event(text: str) -> BackendEvent:
        """A completed assistant turn -> ``TURN_END`` (thin provider: no tool/usage metadata)."""
        return BackendEvent(
            event_type=BackendEventType.TURN_END,
            transcript=(text or "").strip(),
            metadata={"interrupted": False, "cancelled": False, "has_function_calls": False, "usage": None},
        )

    @staticmethod
    def _correction_event(corrected: str) -> BackendEvent:
        """An interruption-corrected assistant turn -> interrupted ``TURN_END``."""
        return BackendEvent(
            event_type=BackendEventType.TURN_END,
            transcript=(corrected or "").strip(),
            metadata={"interrupted": True, "cancelled": False, "has_function_calls": False, "usage": None},
        )

    @staticmethod
    def _user_transcript_event(text: str) -> BackendEvent:
        """A finalized user transcript -> input ``TRANSCRIPT``."""
        return BackendEvent(
            event_type=BackendEventType.TRANSCRIPT,
            transcript=(text or "").strip(),
            metadata={"stream": "input", "final": True},
        )

    async def _pump_output(self, session: ElevenLabsSession) -> None:
        """Drain the bridge's 16 kHz PCM output and emit 24 kHz ``AUDIO_OUTPUT`` events."""
        try:
            while not session.ended.is_set():
                pcm_16k = await session.bridge.get_output_audio(timeout=1.0)
                if not pcm_16k or len(pcm_16k) < 4:
                    continue
                session.events.put_nowait(
                    BackendEvent(event_type=BackendEventType.AUDIO_OUTPUT, audio=_pcm16_16k_to_pcm16_24k(pcm_16k))
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 -- background task must not crash the session
            logger.error(f"ElevenLabs output pump error: {e}", exc_info=True)

    async def receive(self, session: BackendSession) -> AsyncIterator[BackendEvent]:
        """Yield normalized events until the SDK signals end-of-session."""
        s = self._session(session)
        while True:
            if s.ended.is_set() and s.events.empty():
                return
            try:
                event = await asyncio.wait_for(s.events.get(), timeout=0.5)
            except TimeoutError:
                continue
            yield event

    async def close(self, session: BackendSession) -> None:
        """Tear down the SDK session and background pump. Idempotent."""
        s = self._session(session)
        s.ended.set()
        if s.output_task is not None:
            s.output_task.cancel()
            try:
                await s.output_task
            except asyncio.CancelledError:
                pass
            s.output_task = None
        # Unblock any tool handler still awaiting a result so the SDK task can exit.
        for future in s.pending_tools.values():
            if not future.done():
                future.cancel()
        s.pending_tools.clear()
        if s.conversation is not None:
            try:
                await s.conversation.end_session()
                await s.conversation.wait_for_session_end()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error ending ElevenLabs session: {e}")
            finally:
                s.conversation = None
        if s.client is not None:
            try:
                s.client = None
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error closing ElevenLabs client: {e}")

    @staticmethod
    def _session(session: BackendSession) -> ElevenLabsSession:
        if not isinstance(session, ElevenLabsSession):
            raise TypeError(f"expected ElevenLabsSession, got {type(session).__name__}")
        return session
