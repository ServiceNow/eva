"""Abstract ``Backend`` contract: pure API/session exchange, no role knowledge.

DESIGN ONLY (Step 1 of the refactor, see docs/refactor-step1.md). This module
defines shapes, not behavior -- every method body is a stub. Nothing in
``eva.assistant`` or ``eva.user_simulator`` depends on this yet.

A ``Backend`` wraps exactly one provider integration (OpenAI Realtime, Gemini
Live, ElevenLabs Agents, a cascade STT->LLM->TTS pipeline, ...) and exposes a
uniform send/receive surface for exchanging audio, text, and tool-call
traffic with that provider. It has **no opinion about role** -- it does not
know whether it is being driven by an ``AssistantRole`` or a ``UserRole``,
and it does not decide *what* prompt or tools to use (the ``Role`` supplies
those at open-time and owns tool execution).

Symmetry note (per the design doc): a ``Backend`` must not assume it is "the
network server side" or "the client side" of a connection. Today,
assistant-side backends happen to be reached by an inbound WebSocket
connection (Twilio-framed) and user-side backends happen to dial out to a
provider or to the assistant's socket. Both are just implementation details
of a concrete subclass's ``open()``/``send()``/``receive()`` -- the abstract
contract itself is direction-agnostic so that a later mediator can sit
between two ``Backend`` instances (Backend <-> mediator <-> Backend) without
requiring either side to be "the server."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from eva.backend.capabilities import BackendCapabilities


class BackendEventType(StrEnum):
    """Kinds of events a ``Backend`` can surface via ``receive()``.

    Not every ``Backend`` implementation will emit every event type -- a thin,
    end-to-end backend (e.g. ElevenLabs Agents) may only ever emit
    ``AUDIO_OUTPUT``, ``TRANSCRIPT``, ``TURN_END``, and ``ERROR``, because it
    has no separable tool-calling seam of its own that the caller can observe
    (tool calls, if any, happen inside the provider and are not surfaced).
    Consumers must treat unhandled event types as ignorable, not as errors.
    """

    AUDIO_OUTPUT = "audio_output"
    """A chunk of output audio from the backend (assistant speech, or the
    simulated user's speech, depending on which role's Backend this is)."""

    TRANSCRIPT = "transcript"
    """A (possibly partial) transcript of something spoken -- either the
    backend's own output or, for backends that provide it, the other party's
    input as heard by this backend's ASR."""

    TOOL_CALL_REQUEST = "tool_call_request"
    """The backend's model wants to invoke a tool. Only emitted by backends
    that expose a separable tool-calling seam (native S2S realtime APIs,
    cascade LLM backends). The owning ``Role`` is responsible for executing
    the tool and returning the result via ``send(tool_result=...)`` -- the
    ``Backend`` never executes tools itself (see docs/refactor-step1.md,
    "tool execution stays role-side")."""

    TURN_END = "turn_end"
    """The backend's model has finished its current turn (end-of-utterance /
    end-of-response signal)."""

    ERROR = "error"
    """A provider-level error occurred (connection drop, API error, etc.)."""


@dataclass
class ToolCallRequest:
    """A tool invocation requested by a backend's underlying model.

    Surfaced via a ``BackendEvent`` of type ``TOOL_CALL_REQUEST``. The owning
    ``Role`` executes the tool (via its own ``ToolExecutor``) and reports the
    outcome back to the backend with ``Backend.send(tool_result=...)`` so the
    provider's tool-calling loop can continue.
    """

    call_id: str
    """Provider-assigned identifier correlating the request to its result."""

    name: str
    """Tool name as requested by the model."""

    arguments: dict[str, Any]
    """Parsed tool call arguments."""


@dataclass
class ToolCallResult:
    """The outcome of executing a ``ToolCallRequest``.

    To be sent back to the backend so its underlying model can continue the
    tool-calling loop.
    """

    call_id: str
    """Must match the ``call_id`` of the originating ``ToolCallRequest``."""

    result: Any
    """JSON-serializable tool result payload."""


@dataclass
class BackendEvent:
    """A single event surfaced by ``Backend.receive()``.

    Exactly one of the optional payload fields is populated, matching
    ``event_type``. This is intentionally a loose envelope (rather than a
    tagged union of dataclasses) so that thin backends can populate only the
    fields they support without needing empty placeholder subclasses.
    """

    event_type: BackendEventType
    audio: bytes | None = None
    transcript: str | None = None
    tool_call_request: ToolCallRequest | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific extras (e.g. raw event name, timestamps) that don't
    warrant a first-class field. Consumers should not rely on specific keys
    being present across providers.

    Convention (not enforced by this contract): a backend that proactively
    re-engages after an idle period (see ``AssistantRole``'s
    ``self_nudge_timeout_seconds``) may set ``metadata["is_nudge"] = True`` on
    the ``AUDIO_OUTPUT``/``TRANSCRIPT`` event it emits for that turn, purely
    so callers that want to distinguish a self-initiated nudge from an
    ordinary model turn (e.g. for audit logging) can do so. This is *not* a
    new event type -- a nudge is just an ordinary turn from the backend's
    model, triggered by the backend noticing its own idle timeout rather than
    by new input; it flows through the same ``receive()`` surface as
    anything else."""


class Backend(ABC):
    """Pure API/session exchange with one provider. No role knowledge.

    Lifecycle: ``open()`` establishes the session, ``send()`` pushes audio /
    text / tool results to the provider, ``receive()`` yields events back,
    and ``close()`` tears the session down. A ``Role`` (see
    ``eva.role.base``) owns one ``Backend`` instance and drives it.

    Implementations are expected to fall along a spectrum:

    - **Native speech-to-speech** (OpenAI Realtime, Gemini Live): a single
      persistent duplex session; ``send(audio=...)`` streams mic audio in,
      ``receive()`` yields interleaved ``AUDIO_OUTPUT``/``TRANSCRIPT``/
      ``TOOL_CALL_REQUEST``/``TURN_END`` events as the provider produces them.
    - **Cascade** (STT -> LLM -> TTS, e.g. a Pipecat pipeline): internally
      composed of separate provider calls, but from the caller's perspective
      still just one ``Backend`` -- it decides internally when to run STT,
      call the LLM, and synthesize TTS, and surfaces the same event shape.
    - **End-to-end / thin** (ElevenLabs Agents): the provider handles
      everything (ASR, dialogue policy, TTS) opaquely. Such a ``Backend``
      may only ever emit ``AUDIO_OUTPUT``/``TRANSCRIPT``/``TURN_END``/
      ``ERROR`` and may treat ``send(tool_result=...)`` as a no-op or raise
      ``NotImplementedError`` -- callers must consult ``capabilities`` and
      not assume every method does something on every backend.

    Symmetry: this contract says nothing about which side dials out and
    which side is dialed into -- see the module docstring.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Static capability flags for this backend (see ``BackendCapabilities``).

        Must be available even before ``open()`` is called (i.e. it describes
        the provider integration, not live session state).
        """
        ...

    @abstractmethod
    async def open(self, *, system_prompt: str, tools: list[dict[str, Any]] | None, config: dict[str, Any]) -> None:
        """Establish the provider session.

        Args:
            system_prompt: Fully-built system prompt for this session, as
                assembled by the owning ``Role`` (``Role.build_prompt()``).
                A thin end-to-end backend still receives this even if it
                maps it onto a different provider concept (e.g. ElevenLabs
                agent overrides).
            tools: Tool schemas to expose to the provider's model, in
                whatever wire format the concrete backend needs to translate
                from the agent's tool definitions. ``None`` or ``[]`` for
                backends/roles that don't expose tool calling (e.g. a
                ``UserRole`` that only needs an ``end_call`` tool would still
                pass that single tool here; a backend with no tool-calling
                seam at all may simply ignore this argument).
            config: Provider-specific configuration blob (model name, voice,
                sample rate, turn-detection parameters, etc.). Deliberately
                untyped here -- each concrete ``Backend`` defines and
                validates its own config shape; the abstract contract does
                not prescribe one, since a native S2S config and a cascade
                config share little structure. An ``AssistantRole`` backend
                configured to self-nudge (see
                ``AssistantRole.self_nudge_timeout_seconds``) reads its
                threshold from this blob (e.g. a
                ``config["self_nudge_timeout_seconds"]`` key) the same way --
                self-nudging needs no dedicated typed parameter or new
                ``Backend`` method, since the resulting nudge is just an
                ordinary outbound turn (see ``BackendEvent.metadata``).

        Must be safe to call exactly once per ``Backend`` instance. Must not
        block on the other party being ready to exchange data -- readiness to
        *accept* traffic is enough (mirrors today's
        ``AbstractAssistantServer.start()`` contract: non-blocking, returns
        once ready).
        """
        ...

    @abstractmethod
    async def send(
        self,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        tool_result: ToolCallResult | None = None,
    ) -> None:
        """Push data to the provider. Exactly one of the keyword args is set.

        Args:
            audio: Raw input audio chunk (format/sample-rate is whatever this
                backend's ``open(config=...)`` declared; format conversion is
                the caller's responsibility via the shared audio utilities,
                not this method's).
            text: A text turn to inject directly (e.g. a starting utterance,
                or a cascade backend's synthesized user/assistant text before
                TTS). Backends that are audio-only end-to-end (no text
                injection seam) may raise ``NotImplementedError``.
            tool_result: The result of executing a previously-surfaced
                ``ToolCallRequest``, to be relayed back into the provider's
                tool-calling loop so it can continue. Backends with no
                tool-calling seam (see ``capabilities``) may raise
                ``NotImplementedError``.

        This is intentionally the single, symmetric outbound method for both
        "network-server-like" and "client-like" backends -- see the module
        docstring on symmetry. A future mediator sitting between two
        ``Backend`` instances would call this same method on each side.
        """
        ...

    @abstractmethod
    def receive(self) -> AsyncIterator[BackendEvent]:
        """Yield events from the provider as they arrive.

        The single, symmetric inbound stream for both "network-server-like"
        and "client-like" backends. Must be an async generator (or return an
        object implementing ``__aiter__``/``__anext__``) that yields until
        the session ends (``close()`` is called, the provider disconnects,
        or a terminal ``ERROR``/``TURN_END``-with-hangup event occurs --
        exact termination semantics are provider-specific and left to each
        concrete backend).
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Tear down the provider session.

        Must be safe to call even if ``open()`` was never called or the
        session already ended on its own (idempotent). Concrete backends are
        responsible for their own provider-specific teardown (closing
        websockets, cancelling tasks, flushing buffers); this method does not
        itself define audio/output persistence -- that remains a ``Role``
        concern (see ``eva.role.base``).
        """
        ...
