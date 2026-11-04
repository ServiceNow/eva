"""State of a cascade caller barge-in, and the silence gap between its opener and reply."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import audioop
except ImportError:  # Python 3.13+
    import audioop_lts as audioop

from eva.models.results import BargeInRecord
from eva.user_simulator.cascade.tts import TTS_SAMPLE_RATE

SILENCE_RMS = 200
"""PCM16 RMS per window below which caller audio counts as silence (opener trimming and the gap)."""
WINDOW_BYTES = TTS_SAMPLE_RATE * 2 // 100
"""10ms of PCM16 mono."""
MIN_GAP_MS = 130.0
"""Least silence between the opener and the reply; a reply butted right against it sounds spliced."""


def pcm_ms(num_bytes: int) -> float:
    """Duration of PCM16 mono at the caller TTS rate."""
    return num_bytes / (TTS_SAMPLE_RATE * 2) * 1000.0


def loud_windows(pcm: bytes) -> list[int]:
    """Byte offsets of the 10ms windows at or above SILENCE_RMS."""
    return [i for i in range(0, len(pcm), WINDOW_BYTES) if audioop.rms(pcm[i : i + WINDOW_BYTES], 2) >= SILENCE_RMS]


def trim_silence(pcm: bytes) -> bytes:
    """Drop leading and trailing quiet windows, so an opener's playout is all cover."""
    loud = loud_windows(pcm)
    return pcm[loud[0] : loud[-1] + WINDOW_BYTES] if loud else pcm


@dataclass
class BargeIn:
    """One barge-in, timed on the event-loop clock (seconds).

    The bridge plays queued caller audio in real time, filling any stall with silence, so the opener
    plays from ``opener_at`` for ``opener_ms``, and the reply's first audible sample plays at the later
    of the opener ending and its arrival. The gap is the difference, as the script reads it off its
    replayed track.
    """

    decision_ms: float
    p_interrupt: float
    forced: bool = False
    """Fired by forced_interruption_every_turns, whatever the decision model said."""
    opener: str = ""
    opener_at: float = 0.0
    """The opener went out: queued, then moved to its first frame on the wire by ``opener_on_wire``."""
    opener_ms: float = 0.0
    heard: str = ""
    """What the reply was written from: everything heard when the opener started."""
    content: str = ""
    audible_at: float | None = None
    """First audible reply chunk arrived from TTS."""

    def play_opener(self, phrase: str, pcm: bytes, now: float) -> None:
        self.opener, self.opener_at, self.opener_ms = phrase, now, pcm_ms(len(pcm))

    def opener_on_wire(self, at: float) -> None:
        """Time the opener from its first frame on the wire rather than from when it was queued.

        Anything synchronous between queuing it and the event loop's next turn delays that frame.
        """
        self.opener_at = at

    def audible(self, chunk: bytes, now: float) -> bytes:
        """Record a reply chunk's arrival; returns it with any leading silence of the reply dropped.

        Causal, as a real-time player could do it: only audio that has already arrived is inspected.
        A reply ready within MIN_GAP_MS of the opener ending is padded out to it; a later one is not.
        """
        if self.audible_at is not None:
            return chunk
        loud = loud_windows(chunk)
        if not loud:
            return b""
        opener_end = self.opener_at + self.opener_ms / 1000.0
        self.audible_at = now
        pad = round(max(0.0, opener_end + MIN_GAP_MS / 1000.0 - max(now, opener_end)) * TTS_SAMPLE_RATE) * 2
        return b"\x00" * pad + chunk[loud[0] :]

    def record(self) -> BargeInRecord:
        opener_end = self.opener_at + self.opener_ms / 1000.0
        gap_ms = None if self.audible_at is None else max(MIN_GAP_MS, (self.audible_at - opener_end) * 1000.0)
        return BargeInRecord(
            heard=self.heard,
            opener=self.opener,
            content=self.content,
            p_interrupt=self.p_interrupt,
            forced=self.forced,
            decision_ms=self.decision_ms,
            gap_ms=gap_ms,
        )
