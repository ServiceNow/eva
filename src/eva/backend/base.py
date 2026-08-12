"""Abstract ``Backend`` contract: pure API/session exchange, no role knowledge.

This is the live ``Backend`` contract --
implemented by ``eva.backend.openai_realtime`` and driven by ``AssistantRole``
/ ``UserRole``. The worker builds one per conversation via ``BackendFactory``
for every provider the factory supports.

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

    Every event is normalized: typed fields (``audio`` / ``transcript`` /
    ``tool_call_request`` / ``error``) plus normalized ``metadata`` scalars.
    Backends never surface raw provider event objects -- all provider-specific
    parsing happens inside the backend, so a ``Role`` consuming these stays
    fully provider-agnostic.

    Not every ``Backend`` implementation emits every event type -- a thin,
    end-to-end backend (e.g. ElevenLabs Agents) may only ever emit
    ``AUDIO_OUTPUT``, ``TRANSCRIPT``, ``TURN_END``, and ``ERROR``, because it
    has no separable tool-calling seam of its own that the caller can observe.
    Consumers must treat unhandled event types as ignorable, not as errors.
    """

    AUDIO_OUTPUT = "audio_output"
    """A chunk of output audio from the backend (assistant speech, or the
    simulated user's speech, depending on which role's Backend this is)."""

    TRANSCRIPT = "transcript"
    """A finalized transcript of something spoken. ``transcript`` holds the
    text; ``metadata`` carries normalized descriptors: ``stream`` is
    ``"input"`` (what the backend heard from the inbound party) or ``"output"``
    (what the backend's own model said), and for input transcripts
    ``metadata["failed"] = True`` marks a transcription failure (empty text).
    These are normalized scalars, not raw provider payloads."""

    TOOL_CALL_REQUEST = "tool_call_request"
    """The backend's model wants to invoke a tool. The owning ``Role`` is responsible for executing
    the tool and returning the result via ``send(tool_result=...)``"""

    TURN_END = "turn_end"
    """The backend's model finished a response turn. informs whether turn was cancelled, interrupted, etc."""

    INPUT_SPEECH_STARTED = "input_speech_started"
    """The backend's VAD detected that the *inbound* party (whoever is talking
    *to* this backend's model) started speaking. Role-agnostic: for an
    ``AssistantRole`` backend the inbound party is the caller, for a
    ``UserRole`` backend it is the assistant. Emitted only by backends whose
    provider surfaces input-side voice-activity boundaries (native S2S realtime
    APIs)."""

    INPUT_SPEECH_STOPPED = "input_speech_stopped"
    """The backend's VAD detected that the inbound party stopped speaking. The
    end-of-speech counterpart to ``INPUT_SPEECH_STARTED`` (see its docstring)."""

    OUTPUT_TURN_STARTED = "output_turn_started"
    """The backend's model began a response turn. Needed by a manually-sequencing
    consumer (e.g. a ``UserRole`` gating replies) to know a response is now in
    flight; consumers that don't care ignore it."""

    OUTPUT_AUDIO_DONE = "output_audio_done"
    """The backend's model finished emitting output audio for the current turn
    (the audio stream is drained, distinct from ``TURN_END`` which also covers
    the text/tool bookkeeping). Lets a consumer that paces or gates on playout
    flush trailing output; ignorable otherwise."""

    ERROR = "error"
    """A provider-level error occurred (connection drop, API error, etc.)"""


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
    """Normalized descriptors for this event -- plain scalars/dicts, never a raw
    provider event object. Which keys are present depends on ``event_type`` and
    is documented on each ``BackendEventType`` member (e.g. ``stream`` /
    ``failed`` for ``TRANSCRIPT``; ``cancelled`` / ``interrupted`` /
    ``has_function_calls`` / ``usage`` for ``TURN_END``; ``code`` for
    ``ERROR``). A ``Role`` consumes these normalized fields only, so it stays
    provider-agnostic: all provider-specific event parsing happens inside the
    backend before emission."""


class BackendSession:
    """Opaque handle to one live provider session, returned by ``Backend.open``.

    The ``Backend`` itself is stateless beyond its construction config (model,
    key, endpoint): *all* per-exchange state -- the live connection, any
    provider-side accumulators -- lives on the session handle, not on the
    backend. The caller (a ``Role``, or later a mediator) holds this handle and
    passes it back into ``send`` / ``receive`` / ``close``. Concrete backends
    subclass this with whatever they need to carry; consumers treat it as
    opaque and never introspect it.

    Keeping session state off the backend is deliberate (see
    docs/refactor-step1.md discussion): one ``Backend`` instance can then serve
    many independent sessions/conversations concurrently, and no exchange data
    is smuggled into the backend object.
    """


class Backend(ABC):
    """Stateless adapter to one provider's API. No role knowledge, no session state.

    Lifecycle: ``open()`` establishes a session and returns a
    ``BackendSession`` handle, ``send()`` pushes audio / text / tool results to
    the provider on a given session, ``receive()`` yields events back for a
    session, and ``close()`` tears a session down. The backend holds only its
    construction config; the caller holds the session handle. A ``Role`` (see
    ``eva.role.base``) is given a ``Backend`` instance (constructed by the
    worker via a ``BackendFactory`` the worker owns) and drives it.

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

    @property
    def input_sample_rate(self) -> int:
        """Sample rate (Hz) of PCM the backend expects via ``send(audio=...)``.

        Lets a role convert/record counterparty audio without knowing the
        provider. Defaults to 24 kHz (the common realtime rate); backends on a
        different rate override. Describes the session's audio format, not turn
        state.
        """
        return 24000

    @property
    def output_sample_rate(self) -> int:
        """Sample rate (Hz) of PCM carried in ``AUDIO_OUTPUT`` events.

        See ``input_sample_rate``; defaults to 24 kHz, overridden per backend.
        """
        return 24000

    @abstractmethod
    async def open(self, *, system_prompt: str, tools: list[dict[str, Any]] | None) -> BackendSession:
        """Establish a provider session and return its opaque handle.

        The role supplies only the two things that are genuinely its own -- the
        prompt and the tool catalog. All provider-specific session shaping
        (model, voice, sample rate, turn-detection, audio formats, ...) is the
        backend's own construction config, injected by the worker via the
        ``BackendFactory``; the role neither builds nor sees it. That is what
        keeps a single generic ``Role`` usable with any backend.

        Args:
            system_prompt: Fully-built system prompt/instructions for this
                session (the role's ``build_prompt()`` output). A thin
                end-to-end backend still receives this even if it maps it onto a
                different provider concept (e.g. ElevenLabs agent overrides).
            tools: Provider-agnostic tool specs -- a list of
                ``{"name", "description", "parameters"}`` dicts -- which the
                backend translates into its provider's tool-schema wire format.
                ``None`` or ``[]`` for roles that expose no tools; a backend
                with no tool-calling seam may ignore this argument.

        Each call returns a fresh, independent ``BackendSession``; because the
        backend carries no session state, a single ``Backend`` instance may be
        opened many times (e.g. one session per conversation). Must not block
        on the other party being ready to exchange data -- readiness to
        *accept* traffic is enough (mirrors today's
        ``AbstractAssistantServer.start()`` contract: non-blocking, returns
        once ready).
        """
        ...

    @abstractmethod
    async def send(
        self,
        session: BackendSession,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        tool_result: ToolCallResult | None = None,
    ) -> None:
        """Push data to the provider on ``session``. Exactly one kwarg is set.

        Args:
            session: The handle returned by ``open()`` for this exchange.
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
    def receive(self, session: BackendSession) -> AsyncIterator[BackendEvent]:
        """Yield events from the provider on ``session`` as they arrive.

        The single, symmetric inbound stream for both "network-server-like"
        and "client-like" backends. Must be an async generator (or return an
        object implementing ``__aiter__``/``__anext__``) that yields until
        the session ends (``close()`` is called, the provider disconnects,
        or a terminal ``ERROR``/``TURN_END``-with-hangup event occurs --
        exact termination semantics are provider-specific and left to each
        concrete backend).
        """
        ...

    async def trigger_response(self, session: BackendSession) -> None:
        """Ask the provider to generate a response now, with no new input.

        Only meaningful for backends whose turn detection is configured *not*
        to auto-create responses, so the caller sequences replies itself (e.g.
        a ``UserRole`` that gates when the simulated caller speaks). Backends
        with no such control -- thin end-to-end providers, or any backend where
        responses are always driven by input/tool-results -- leave this as the
        default ``NotImplementedError``; consult ``capabilities`` / provider
        docs before calling. Not abstract, so those backends need not implement
        it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support trigger_response()")

    @abstractmethod
    async def close(self, session: BackendSession) -> None:
        """Tear down the given provider ``session``.

        Must be safe to call even if the session already ended on its own
        (idempotent). Concrete backends are
        responsible for their own provider-specific teardown (closing
        websockets, cancelling tasks, flushing buffers); this method does not
        itself define audio/output persistence -- that remains a ``Role``
        concern (see ``eva.role.base``).
        """
        ...
