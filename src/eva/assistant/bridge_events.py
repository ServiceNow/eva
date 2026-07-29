"""Backend-agnostic events and the session protocol used by the bridge server.

A voice backend (aai host mode, a hosted realtime API, ...) speaks its own wire
protocol. ``WebSocketBridgeAssistantServer`` never sees that protocol: an
adapter translates it into the small event vocabulary below, and the bridge
turns those events into what EVA's evaluation contract requires.

Keeping this module free of EVA imports means a new backend adapter can be
written and unit-tested without pulling in the server.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AudioChunk:
    """PCM16 audio from the backend, at its declared ``backend_output_rate``."""

    pcm: bytes


@dataclass(frozen=True)
class AssistantTranscript:
    """The assistant's reply text.

    Full-text semantics: each event supersedes the previous one for the current
    turn rather than appending to it.
    """

    text: str


@dataclass(frozen=True)
class UserTranscript:
    """A user turn as transcribed by the backend."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A tool the backend wants executed. The bridge runs it and replies."""

    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class SpeechStarted:
    """Backend VAD detected user speech. Triggers barge-in if the assistant is speaking."""


@dataclass(frozen=True)
class TurnDone:
    """The assistant's turn is complete. Safe to deliver more than once per turn."""


@dataclass(frozen=True)
class BackendError:
    """A backend-reported problem. ``fatal`` ends the session."""

    message: str
    fatal: bool = False


BridgeEvent = AudioChunk | AssistantTranscript | UserTranscript | ToolCall | SpeechStarted | TurnDone | BackendError


@runtime_checkable
class VoiceBackendSession(Protocol):
    """One live conversation with a voice backend.

    Implementations own their wire protocol and nothing else: no Twilio framing,
    no recording buffers, no audit logging.
    """

    #: Sample rate of the PCM16 audio this session expects from ``send_audio``.
    backend_input_rate: int
    #: Sample rate of the PCM16 audio this session emits in ``AudioChunk``.
    backend_output_rate: int

    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 user audio at ``backend_input_rate``."""
        ...

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        """Resolve the relayed tool call identified by ``call_id``."""
        ...

    def events(self) -> AsyncIterator[BridgeEvent]:
        """Yield events until the backend closes the session."""
        ...

    async def aclose(self) -> None:
        """Close the session. Must be safe to call more than once."""
        ...
