"""Tick scheduler: virtual clock, playout queue, and turn-state machine."""

from __future__ import annotations

from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.constants import (
    BYTES_PER_TICK,
    SILENCE_BYTE,
    WAIT_TO_RESPOND_OTHER_MS,
    WAIT_TO_RESPOND_SELF_MS,
    ms_to_ticks,
)
from eva.user_simulator.cascade.tick_result import TickResult

_NEVER_SPOKE = 10**9


class TickScheduler:
    """Advances the virtual clock and decides who holds the floor.

    Caller turn boundaries are authored: an utterance is queued whole and drains
    one tick at a time from a known start tick. The assistant's are detected from
    consecutive silent ticks. Both land in the same synchronous step, so their
    relative order can never come out ambiguous.
    """

    def __init__(self, adapter: Adapter, *, bytes_per_tick: int = BYTES_PER_TICK) -> None:
        self._adapter = adapter
        self._bytes_per_tick = bytes_per_tick
        self._playout = bytearray()
        self.tick = 0
        self._ticks_since_assistant_speech = _NEVER_SPOKE
        self._ticks_since_caller_speech = _NEVER_SPOKE
        self._assistant_has_spoken = False

    def enqueue_utterance(self, audio: bytes) -> None:
        """Queue caller audio to drain one tick at a time starting next tick."""
        self._playout.extend(audio)

    @property
    def caller_is_speaking(self) -> bool:
        """Whether caller audio is still queued for playout."""
        return bool(self._playout)

    def may_take_turn(self) -> bool:
        """Whether both silence thresholds are satisfied (tau: streaming.py:2590-2606).

        Gated on the assistant having spoken at least once: the assistant opens
        the call with a greeting, and without this the caller would talk over it
        on tick 0, since neither silence counter has anything to measure yet.
        """
        if not self._assistant_has_spoken:
            return False
        return self._ticks_since_assistant_speech > ms_to_ticks(
            WAIT_TO_RESPOND_OTHER_MS
        ) and self._ticks_since_caller_speech > ms_to_ticks(WAIT_TO_RESPOND_SELF_MS)

    async def run_tick(self) -> TickResult:
        """Exchange one tick with the adapter and advance the turn-state machine."""
        outgoing = self._next_chunk()
        result = await self._adapter.run_tick(self.tick, outgoing)

        self._ticks_since_caller_speech = 0 if outgoing else self._ticks_since_caller_speech + 1
        self._ticks_since_assistant_speech = (
            0 if result.has_assistant_speech else self._ticks_since_assistant_speech + 1
        )
        self._assistant_has_spoken = self._assistant_has_spoken or result.has_assistant_speech

        self.tick += 1
        return result

    def _next_chunk(self) -> bytes | None:
        """Pull one tick of queued caller audio, or None when the caller is silent."""
        if not self._playout:
            return None
        chunk = bytes(self._playout[: self._bytes_per_tick])
        del self._playout[: self._bytes_per_tick]
        if len(chunk) < self._bytes_per_tick:
            chunk += SILENCE_BYTE * (self._bytes_per_tick - len(chunk))
        return chunk
