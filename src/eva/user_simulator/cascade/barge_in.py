"""Live timing of the cascade caller's barge-ins, measured as in scripts/measure_barge_in_latency.py."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import audioop
except ImportError:  # Python 3.13+
    import audioop_lts as audioop

from eva.models.results import BargeInLatency
from eva.user_simulator.cascade.tts import TTS_SAMPLE_RATE

SILENCE_RMS = 200
"""PCM16 RMS per window below which caller audio counts as silence (opener trimming and the gap)."""
WINDOW_BYTES = TTS_SAMPLE_RATE * 2 // 100
"""10ms of PCM16 mono."""


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
    """One barge-in's timing, on the event-loop clock (seconds).

    The bridge plays queued caller audio in real time, filling any stall with silence, so the opener
    plays from ``opener_at`` for ``opener_ms``, and the reply's first audible sample plays at the later
    of the opener ending and its arrival. The gap is the difference, as the script reads it off its
    replayed track.
    """

    heard_at: float
    """The transcript update the firing decision was asked about became readable."""
    decision_ms: float
    p_interrupt: float
    forced: bool = False
    """Fired by forced_interruption_every_turns, whatever the decision model said."""
    segment_start: float = 0.0
    trigger_words: int = 0
    """Where that update's last word sits: its STT segment, and the partial's word count."""
    word_end_at: float | None = None
    """That last word stopped playing to the caller, from STT word timestamps; None if they never came."""
    opener: str = ""
    opener_at: float = 0.0
    """The opener went out: queued, then moved to its first frame on the wire by ``opener_on_wire``."""
    opener_ms: float = 0.0
    heard: str = ""
    """What the reply was written from: everything heard when the opener started."""
    content: str = ""
    first_token_at: float | None = None
    """The reply LLM's first text token arrived."""
    segment_sent_at: float | None = None
    """The reply's first segment was written and sent to TTS."""
    first_chunk_at: float | None = None
    audible_at: float | None = None
    """First audible reply chunk arrived from TTS."""

    def play_opener(self, phrase: str, pcm: bytes, now: float) -> None:
        self.opener, self.opener_at, self.opener_ms = phrase, now, pcm_ms(len(pcm))

    def opener_on_wire(self, at: float) -> None:
        """Time the opener from its first frame on the wire rather than from when it was queued.

        Anything synchronous between queuing it and the event loop's next turn delays that frame.
        """
        self.opener_at = at

    def first_token(self, now: float) -> None:
        if self.first_token_at is None:
            self.first_token_at = now

    def segment_sent(self, now: float) -> None:
        if self.segment_sent_at is None:
            self.segment_sent_at = now

    def audible(self, chunk: bytes, now: float) -> bytes:
        """Record a reply chunk's arrival; returns it with any leading silence of the reply dropped.

        Causal, as a real-time player could do it: only audio that has already arrived is inspected.
        """
        if self.first_chunk_at is None:
            self.first_chunk_at = now
        if self.audible_at is not None:
            return chunk
        loud = loud_windows(chunk)
        if not loud:
            return b""
        self.audible_at = now
        return chunk[loud[0] :]

    def latency(self) -> BargeInLatency:
        def ms(start: float, end: float | None) -> float | None:
            return None if end is None else (end - start) * 1000.0

        opener_end = self.opener_at + self.opener_ms / 1000.0
        content_plays_at = None if self.audible_at is None else max(opener_end, self.audible_at)
        spoken_at = self.word_end_at
        return BargeInLatency(
            heard=self.heard,
            opener=self.opener,
            content=self.content,
            p_interrupt=self.p_interrupt,
            forced=self.forced,
            opener_ms=self.opener_ms,
            stt_lag_ms=None if spoken_at is None else ms(spoken_at, self.heard_at),
            decision_ms=self.decision_ms,
            reaction_ms=None if spoken_at is None else ms(spoken_at, self.opener_at),
            reply_first_token_ms=ms(self.opener_at, self.first_token_at),
            reply_text_ms=ms(self.opener_at, self.segment_sent_at),
            tts_first_audio_ms=None if self.segment_sent_at is None else ms(self.segment_sent_at, self.first_chunk_at),
            content_ready_ms=ms(self.opener_at, self.audible_at),
            gap_ms=ms(opener_end, content_plays_at),
            total_ms=None if spoken_at is None or content_plays_at is None else ms(spoken_at, content_plays_at),
        )
