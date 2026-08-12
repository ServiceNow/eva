"""Audio buffer processor tuned for EVA's continuous bot-to-bot streams.

pipecat 1.x's ``AudioBufferProcessor`` inserts wall-clock silence into a
recording buffer whenever a frame is *processed* more than 200ms after the
previous one (``_fill_buffer_silence_gap``). That behavior assumes a gap in
processing time means a real silent period (a muted microphone, an idle pause).

That assumption does not hold for EVA. The user simulator sends a *continuous*
20ms stream — real caller audio plus its own silence frames — so the audio is
never actually gapped. When the assistant's event loop stalls under high
concurrency, user frames simply arrive late and bunched; pipecat then sees a
>200ms wall-clock gap and fabricates silence *between* frames whose audio is
all present. The result is a hard-cut silence run in the middle of speech — the
"chop" heard in ``audio_user.wav`` — even though the conversation (and the
real-time STT stream) was unaffected. This regressed when pipecat went
0.0.104 -> 1.x.

We keep the byte-count track synchronization (``_sync_buffer_to_position``,
which aligns the user and bot channels and predates 1.x) but disable the
wall-clock silence fabrication, restoring contiguous recording of the incoming
stream.
"""

from __future__ import annotations

from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class ContiguousAudioBufferProcessor(AudioBufferProcessor):
    """AudioBufferProcessor that records frames contiguously.

    Disables pipecat's wall-clock ``_fill_buffer_silence_gap`` because EVA's
    input stream is already continuous; any processing-time gap is event-loop
    jitter, not a real silent period, so filling it fabricates the choppy
    silence artifact. Track alignment still happens via ``_sync_buffer_to_position``.

    Emits a log line on init and periodically reports how much silence it
    suppressed, so a run can confirm the override was actually active.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._suppressed_silence_bytes = 0
        logger.info("ContiguousAudioBufferProcessor active — pipecat wall-clock silence fill disabled")

    def _fill_buffer_silence_gap(self, buffer, last_update_time, now, frame_bytes) -> None:
        # No-op: do not fabricate silence from processing-time gaps. Track what
        # would have been inserted so the effect is measurable in the logs.
        if last_update_time is None or self._sample_rate == 0:
            return
        gap = (now - last_update_time) - frame_bytes / (self._sample_rate * 2)
        if gap > 0.2:
            would_insert = int(gap * self._sample_rate * 2)
            self._suppressed_silence_bytes += would_insert
            logger.debug(
                f"Suppressed {gap * 1000:.0f}ms wall-clock silence fill "
                f"(cumulative {self._suppressed_silence_bytes / (self._sample_rate * 2):.1f}s)"
            )
