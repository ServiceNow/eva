"""Turn-end fallback: nudge the assistant to reprompt when a user turn is never detected.

The assistant normally responds only when the pipeline detects a completed user turn.
When VAD / turn detection silently drops a real user utterance (observed in the wild: the
caller speaks but no turn-start/turn-end fires, so nothing is ever processed), the call
would otherwise hang until the provider's inactivity backstop ends it ~2 minutes later.

This module provides a shared timer that arms once the assistant stops speaking and, after
``turn_end_fallback_time`` seconds of no detected user turn, fires a nudge asking the caller
to repeat. It is hosted on the pipeline spine (``UserObserver``) so it works for both the
cascade and audio-LLM pipelines, and it invokes a pipeline-specific ``process_turn_fallback``
to actually inject the nudge (the two pipelines drive the model differently).
"""

import asyncio

from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from eva.utils.logging import get_logger

logger = get_logger(__name__)


async def emit_emulated_user_turn_boundary(processor) -> None:
    """Assert a user-turn start/stop boundary so the nudge is its own assistant turn.

    The turn-end fallback is the VAD's backup: when it fires it decides the user's turn has
    ended (we waited long enough). Pipecat's turn tracker only closes a turn when the user
    starts speaking or an inter-turn timeout elapses; neither happens between back-to-back
    nudges, so their TTS collapses into one tracked turn/segment. Emitting an emulated
    user-turn boundary marks that decision explicitly.

    Pushed from the model processor (``agent_processor`` / ``audio_llm_processor``), which sits
    downstream of the user aggregator, so the aggregator does not re-flush a buffered transcript
    as a duplicate real turn. The bot is idle when the fallback fires, so this does not interrupt
    in-flight speech.
    """
    await processor.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await processor.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


# Give up after this many consecutive nudges without the user coming back, then let the
# provider's inactivity backstop end the call (the pre-fallback behavior).
MAX_CONSECUTIVE_FALLBACK_NUDGES = 3


def build_fallback_nudge(timeout_seconds: int | None, partial: str, has_audio: bool = False) -> tuple[str, str]:
    """Build the (marker, llm_content) pair for a turn-end fallback nudge.

    The marker is a plain transcript entry (recorded with ``message_type="turn_fallback"``
    so downstream metrics can identify and zero it). The llm_content is the richer prompt
    actually sent to the model.

    Three cases, in priority order:
      1. ``partial`` text present (cascade) — embed the partial transcript.
      2. no text but ``has_audio`` (audio-LLM) — the partial context is the buffered audio
         itself, forwarded separately; no transcript to embed.
      3. neither — nothing was captured, so ask the caller to repeat.

    Args:
        timeout_seconds: The configured ``EVA_TURN_END_FALLBACK_TIME`` value.
        partial: Best-effort partial user transcription; empty string if none.
        has_audio: True when partial user *audio* was captured (audio-LLM) even without a
            transcript, so the nudge should acknowledge-and-answer rather than ask to repeat.

    Returns:
        ``(marker, llm_content)``.
    """
    if partial:
        marker = f"[TURN-END FALLBACK after {timeout_seconds}s] partial user speech: {partial!r}"
        nudge = (
            f"[Turn-end detection timed out after {timeout_seconds}s, so the caller's message may "
            f"be incomplete or imperfectly transcribed: {partial!r}. Begin your reply by briefly "
            "acknowledging you may not have heard perfectly, then answer or address what you did hear, "
            "continuing in the same language and tone as the conversation so far. Do not mention timeouts, "
            "transcription, or this system note.]"
        )
    elif has_audio:
        marker = f"[TURN-END FALLBACK after {timeout_seconds}s] partial user audio (no transcript)"
        nudge = (
            f"[Turn-end detection timed out after {timeout_seconds}s, so the caller's audio may be "
            "incomplete or imperfectly heard. Begin your reply by briefly acknowledging you may not "
            "have heard perfectly, then answer or address what you did hear, continuing in the same "
            "language and tone as the conversation so far. Do not mention timeouts or this system note.]"
        )
    else:
        marker = f"[TURN-END FALLBACK after {timeout_seconds}s] no user speech captured"
        nudge = (
            f"[Turn-end detection timed out after {timeout_seconds}s and no user speech was "
            "captured. Briefly acknowledge you may not have heard them and ask them to repeat, continuing in the same language "
            "and tone as the conversation so far. Do not mention timeouts or this system note.]"
        )
    return marker, nudge


class TurnEndFallbackTimer:
    """A single-shot timer that fires a nudge after a window of undetected user silence.

    Owns the timer task and the consecutive-nudge give-up counter. Arm/cancel are driven by
    the host processor from pipeline frames: arm when the assistant stops speaking, cancel
    when the assistant or the user starts speaking, re-arm when the user stops speaking
    without a completed turn. When the window elapses, ``on_fire`` is awaited.

    No-op when ``fallback_time`` is ``None`` (feature disabled), preserving the old behavior
    of waiting for the provider's inactivity timeout.
    """

    def __init__(
        self,
        fallback_time: int | None,
        on_fire,
        max_consecutive_nudges: int = MAX_CONSECUTIVE_FALLBACK_NUDGES,
    ) -> None:
        """Initialize the timer.

        Args:
            fallback_time: Seconds of undetected user silence after the assistant stops
                speaking before firing. ``None`` disables the timer.
            on_fire: Async callable ``on_fire(timeout_seconds)`` invoked when the window
                elapses. Should perform the pipeline-specific nudge.
            max_consecutive_nudges: Give up after this many consecutive nudges without a
                real user turn resetting the counter.
        """
        self._fallback_time = fallback_time
        self._on_fire = on_fire
        self._max_consecutive_nudges = max_consecutive_nudges
        self._timer_task: asyncio.Task | None = None
        self._consecutive_nudges = 0

    @property
    def enabled(self) -> bool:
        return self._fallback_time is not None

    def arm(self) -> None:
        """Start a fresh timer. No-op when disabled. Cancels any previously armed timer."""
        if self._fallback_time is None:
            return
        self.cancel()
        self._timer_task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        """Cancel a pending timer, if any."""
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = None

    def reset_counter(self) -> None:
        """Reset the consecutive-nudge counter (call when a real user turn lands)."""
        self._consecutive_nudges = 0

    def note_nudge(self) -> bool:
        """Record that a nudge is about to fire; return False if the give-up limit is hit.

        Increments the consecutive-nudge counter. Returns True if the nudge should proceed,
        False if the limit has been exceeded (caller should skip and leave the call to the
        inactivity backstop).
        """
        self._consecutive_nudges += 1
        if self._consecutive_nudges > self._max_consecutive_nudges:
            logger.warning(
                f"Turn-end fallback exhausted after {self._max_consecutive_nudges} consecutive "
                "nudges without a user turn; leaving the call to the inactivity backstop"
            )
            return False
        return True

    @property
    def consecutive_nudges(self) -> int:
        return self._consecutive_nudges

    @property
    def max_consecutive_nudges(self) -> int:
        return self._max_consecutive_nudges

    async def _run(self) -> None:
        try:
            await asyncio.sleep(self._fallback_time)
        except asyncio.CancelledError:
            return
        await self._on_fire(self._fallback_time)
