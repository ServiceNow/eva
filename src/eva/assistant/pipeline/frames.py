"""Custom Pipecat frame types for the benchmark pipeline."""

from dataclasses import dataclass

from pipecat.frames.frames import (
    ControlFrame,
    DataFrame,
    ProposedUserStoppedSpeakingFrame,
    SystemFrame,
    TranscriptionFrame,
    UninterruptibleFrame,
)


@dataclass
class UninterruptibleTranscriptionFrame(TranscriptionFrame, UninterruptibleFrame):
    """A final TranscriptionFrame that survives an InterruptionFrame queue flush.

    When a late interim opens the user turn, the aggregator broadcasts an InterruptionFrame
    and pipecat discards its queued data/control frames — including a final transcript queued
    right behind the interim — so the turn never closes (eva-internal#561 follow-up).
    """


@dataclass
class UninterruptibleProposedUserStoppedSpeakingFrame(ProposedUserStoppedSpeakingFrame, UninterruptibleFrame):
    """A ProposedUserStoppedSpeakingFrame that survives an InterruptionFrame queue flush.

    Companion to UninterruptibleTranscriptionFrame: the stop proposal is queued behind the
    final transcript and would be flushed along with it.
    """


@dataclass
class TurnTimestampFrame(ControlFrame):
    """A frame that contains the start and end intervals of a turn in seconds.

    Attributes:
        role: The role of the speaker.
        start_timestamp: The start time of the turn in seconds (user timeline).
        end_timestamp: The end time of the turn in seconds (user timeline).
        end_reason: The reason for why the turn ended. None if the turn ended normally.
    """

    role: str
    start_timestamp: float
    end_timestamp: float
    end_reason: str | None = None


@dataclass
class SpokenMessageFrame(DataFrame):
    """A system frame to signal that the initial message has been generated."""

    text: str


@dataclass
class VADBufferFrame(SystemFrame):
    """A frame containing VAD buffer metrics for a user turn."""

    vad_buffer_ms: int  # Time in milliseconds between last transcription and VAD firing


@dataclass
class UserMessageFrame(SystemFrame):
    """Mirrors TranscriptionFrame, but flows through the rest of the pipeline without getting consumed by the context aggregator."""

    text: str


@dataclass
class LLMMessageFrame(SystemFrame):
    """Sends LLM message through the flow without getting sent to TTS."""

    text: str


@dataclass
class UserContextFrame(SystemFrame):
    """Mirrors TranscriptionFrame, but flows through the rest of the pipeline without getting consumed by the context aggregator."""

    messages: list[dict]
