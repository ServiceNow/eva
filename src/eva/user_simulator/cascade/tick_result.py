"""Per-tick exchange record between the scheduler and an adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from eva.user_simulator.cascade.constants import BYTES_PER_TICK, SILENCE_BYTE


@dataclass(frozen=True)
class TickResult:
    """What one tick of the conversation produced."""

    tick_number: int
    """Monotonic simulation-clock index; the ordering key for events, unlike wall_clock_ms."""

    assistant_audio: bytes
    """Exactly one tick's worth of PCM16, silence-padded when the assistant was quiet."""

    assistant_audio_raw_bytes: int
    """Real audio bytes received before padding. Zero means the assistant was silent."""

    wall_clock_ms: int
    """Unix ms at the tick's I/O boundary. For latency metrics only, never for ordering."""

    @property
    def has_assistant_speech(self) -> bool:
        """Whether any real assistant audio arrived this tick."""
        return self.assistant_audio_raw_bytes > 0


class TickAudioSplit(NamedTuple):
    """Result of splitting audio at a tick boundary."""

    chunk: bytes
    """Always exactly bytes_per_tick, silence-padded when the input was short."""

    overflow: bytes
    """Any remainder past bytes_per_tick, carried to the next tick."""


def split_tick_audio(audio: bytes, bytes_per_tick: int = BYTES_PER_TICK) -> TickAudioSplit:
    """Split audio into exactly one tick's worth plus overflow, padding short input with silence."""
    if len(audio) >= bytes_per_tick:
        return TickAudioSplit(audio[:bytes_per_tick], audio[bytes_per_tick:])
    return TickAudioSplit(audio + SILENCE_BYTE * (bytes_per_tick - len(audio)), b"")
