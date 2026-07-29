"""Pydantic models for aai voice-agent wire events.

The aai host sends camelCase JSON frames over the session WebSocket; binary
frames carry PCM16 audio and are handled by the session, not here.

Ported from the tau2-bench aai provider. Parsing never raises: an unrecognized
or malformed frame becomes an ``AAIUnknownEvent`` so a single bad frame cannot
abort a conversation mid-evaluation.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAAIEvent(BaseModel):
    """Base class for all aai events.

    ``extra="ignore"`` so that new server-side fields never break parsing.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: str


class AAIConfigEvent(BaseAAIEvent):
    """Server acknowledgment of the client's config frame — the handshake completing."""

    type: Literal["config"] = "config"


class AAISpeechStartedEvent(BaseAAIEvent):
    """aai's VAD detected the start of user speech."""

    type: Literal["speech_started"] = "speech_started"


class AAISpeechStoppedEvent(BaseAAIEvent):
    """aai's VAD detected the end of user speech."""

    type: Literal["speech_stopped"] = "speech_stopped"


class AAIUserTranscriptEvent(BaseAAIEvent):
    """Transcript of a user turn, as heard by aai."""

    type: Literal["user_transcript"] = "user_transcript"
    text: str
    turn_order: int | None = Field(default=None, alias="turnOrder")


class AAIAgentTranscriptEvent(BaseAAIEvent):
    """Transcript of the agent's reply. Carries the full text, not a delta."""

    type: Literal["agent_transcript"] = "agent_transcript"
    text: str


class AAIToolCallEvent(BaseAAIEvent):
    """A relayed tool call awaiting a ``tool_result`` from this client."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: dict = Field(default_factory=dict)


class AAIToolCallDoneEvent(BaseAAIEvent):
    """Acknowledgment that a relayed tool call was resolved."""

    type: Literal["tool_call_done"] = "tool_call_done"
    tool_call_id: str = Field(alias="toolCallId")
    result: str = ""


class AAIReplyDoneEvent(BaseAAIEvent):
    """The agent's reply is complete."""

    type: Literal["reply_done"] = "reply_done"


class AAIAudioDoneEvent(BaseAAIEvent):
    """The agent's audio output is complete."""

    type: Literal["audio_done"] = "audio_done"


class AAICancelledEvent(BaseAAIEvent):
    """An in-flight reply was cancelled."""

    type: Literal["cancelled"] = "cancelled"


class AAIResetEvent(BaseAAIEvent):
    """Session state was reset."""

    type: Literal["reset"] = "reset"


class AAIIdleTimeoutEvent(BaseAAIEvent):
    """The session was closed for inactivity."""

    type: Literal["idle_timeout"] = "idle_timeout"


class AAIErrorEvent(BaseAAIEvent):
    """An error reported by the aai host."""

    type: Literal["error"] = "error"
    code: str | None = None
    message: str | None = None


class AAICustomEvent(BaseAAIEvent):
    """An application-defined event emitted by the agent."""

    type: Literal["custom_event"] = "custom_event"
    event: str
    data: Any | None = None


class AAIUnknownEvent(BaseAAIEvent):
    """An unrecognized or unparseable frame, preserved for logging."""

    type: str
    raw: dict | None = None


_EVENT_TYPE_MAP: dict[str, type[BaseAAIEvent]] = {
    "config": AAIConfigEvent,
    "speech_started": AAISpeechStartedEvent,
    "speech_stopped": AAISpeechStoppedEvent,
    "user_transcript": AAIUserTranscriptEvent,
    "agent_transcript": AAIAgentTranscriptEvent,
    "tool_call": AAIToolCallEvent,
    "tool_call_done": AAIToolCallDoneEvent,
    "reply_done": AAIReplyDoneEvent,
    "audio_done": AAIAudioDoneEvent,
    "cancelled": AAICancelledEvent,
    "reset": AAIResetEvent,
    "idle_timeout": AAIIdleTimeoutEvent,
    "error": AAIErrorEvent,
    "custom_event": AAICustomEvent,
}


def parse_aai_event(data: dict) -> BaseAAIEvent:
    """Parse a raw aai frame into a typed event.

    Never raises. Unknown types and validation failures both yield
    ``AAIUnknownEvent`` with the original payload attached.
    """
    event_type = data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return AAIUnknownEvent(type=event_type, raw=data)

    try:
        return event_class.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse aai event {event_type}: {e}")
        return AAIUnknownEvent(type=event_type, raw=data)
