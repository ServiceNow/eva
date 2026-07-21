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

from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Give up after this many consecutive nudges without the user coming back, then let the
# provider's inactivity backstop end the call (the pre-fallback behavior).
MAX_CONSECUTIVE_FALLBACK_NUDGES = 3


def build_fallback_nudge(timeout_seconds: int | None, partial: str) -> tuple[str, str]:
    """Build the (marker, llm_content) pair for a turn-end fallback nudge.

    The marker is a plain transcript entry (recorded with ``message_type="turn_fallback"``
    so downstream metrics can identify and zero it). The llm_content is the richer prompt
    actually sent to the model.

    Args:
        timeout_seconds: The configured ``EVA_TURN_END_FALLBACK_TIME`` value.
        partial: Best-effort partial user transcription captured since the assistant last
            spoke; empty string if none.

    Returns:
        ``(marker, llm_content)``.
    """
    if partial:
        marker = f"turn-end not detected within {timeout_seconds}s; partial user speech: {partial!r}"
        nudge = (
            f"[Turn-end detection timed out after {timeout_seconds}s. The user may not have "
            f"finished speaking and this transcription may be incomplete: {partial!r}. "
            "If it is clear enough, respond to it; otherwise briefly ask the caller to finish "
            "or repeat, continuing in the same language and tone as the conversation so far. "
            "Do not mention this system note.]"
        )
    else:
        marker = f"no user speech detected within {timeout_seconds}s"
        nudge = (
            f"[Turn-end detection timed out after {timeout_seconds}s with no user speech "
            "captured. Briefly ask the caller to repeat what they said, continuing in the "
            "same language and tone as the conversation so far. Do not mention this system note.]"
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
