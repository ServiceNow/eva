"""Tick scheduler: virtual clock, playout queue, and turn-state machine."""

from __future__ import annotations

from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.constants import (
    ASSISTANT_UNRESPONSIVE_MS,
    BYTES_PER_TICK,
    WAIT_TO_RESPOND_OTHER_MS,
    WAIT_TO_RESPOND_SELF_MS,
    ms_to_ticks,
)
from eva.user_simulator.cascade.tick_result import TickResult, split_tick_audio

_NEVER_SPOKE = 10**9


class TickScheduler:
    """Advances the virtual clock and decides who holds the floor.

    Caller turn boundaries are authored: an utterance is queued whole and drains
    one tick at a time from a known start tick. The assistant's are detected from
    consecutive silent ticks. Both land in the same synchronous step, so their
    relative order can never come out ambiguous. This guarantee assumes a single
    caller drives `run_tick` sequentially; concurrent calls are unsupported.
    """

    def __init__(self, adapter: Adapter, *, bytes_per_tick: int = BYTES_PER_TICK) -> None:
        self._adapter = adapter
        self._bytes_per_tick = bytes_per_tick
        self._playout = bytearray()
        self._tick = 0
        self._ticks_since_assistant_speech = _NEVER_SPOKE
        self._ticks_since_caller_speech = _NEVER_SPOKE
        self._assistant_has_spoken = False
        self._awaiting_reply = False
        self._caller_spoke_this_tick = False

    @property
    def tick(self) -> int:
        """Current tick index; advances only after a successful `run_tick`."""
        return self._tick

    def enqueue_utterance(self, audio: bytes) -> None:
        """Append audio to drain from the next tick onward.

        Appends concatenate into one continuous stream with no gap between them.
        Queuing a logically separate utterance while one is still draining is the
        caller's responsibility to avoid.
        """
        self._playout.extend(audio)

    @property
    def caller_is_speaking(self) -> bool:
        """Whether caller audio is still queued for playout."""
        return bool(self._playout)

    @property
    def caller_spoke_this_tick(self) -> bool:
        """Whether real caller audio went out on the most recent tick.

        Distinct from `caller_is_speaking`, which is true as soon as an utterance is
        queued. This flips exactly on the ticks audio enters and leaves the wire, so
        it dates the authored turn boundary rather than estimating it from silence.
        """
        return self._caller_spoke_this_tick

    def may_take_turn(self) -> bool:
        """Whether both silence thresholds are satisfied (tau: streaming.py:2590-2606).

        Gated on the assistant having spoken at least once: the assistant opens
        the call with a greeting, and without this the caller would talk over it
        on tick 0, since neither silence counter has anything to measure yet.

        Also gated on the assistant having replied since the caller's last turn:
        the silence thresholds alone are satisfied a fixed time after the caller
        stops talking regardless of whether a reply ever arrived, which lets the
        caller repeat itself into a slow assistant. That gate releases only after
        ASSISTANT_UNRESPONSIVE_MS, so an assistant that stops answering entirely
        cannot strand the caller before it reaches its end_call turn.
        """
        if not self._assistant_has_spoken:
            return False
        if self._awaiting_reply and self._ticks_since_assistant_speech <= ms_to_ticks(ASSISTANT_UNRESPONSIVE_MS):
            return False
        return self._ticks_since_assistant_speech > ms_to_ticks(
            WAIT_TO_RESPOND_OTHER_MS
        ) and self._ticks_since_caller_speech > ms_to_ticks(WAIT_TO_RESPOND_SELF_MS)

    async def run_tick(self) -> TickResult:
        """Exchange one tick with the adapter and advance the turn-state machine.

        The playout queue is only drained after the adapter call succeeds, so a
        raised exception leaves the queue and tick count exactly as they were.
        """
        outgoing, consumed = self._peek_chunk()
        result = await self._adapter.run_tick(self._tick, outgoing)
        del self._playout[:consumed]

        self._caller_spoke_this_tick = outgoing is not None
        self._ticks_since_caller_speech = 0 if outgoing else self._ticks_since_caller_speech + 1
        self._ticks_since_assistant_speech = (
            0 if result.has_assistant_speech else self._ticks_since_assistant_speech + 1
        )
        self._assistant_has_spoken = self._assistant_has_spoken or result.has_assistant_speech
        if result.has_assistant_speech:
            self._awaiting_reply = False
        if outgoing:
            self._awaiting_reply = True

        self._tick += 1
        return result

    def _peek_chunk(self) -> tuple[bytes | None, int]:
        """Preview one tick of queued caller audio without consuming it.

        Returns the padded chunk (or None when silent) and how many raw bytes
        of `_playout` it was drawn from, for the caller to commit after success.
        """
        if not self._playout:
            return None, 0
        raw = bytes(self._playout[: self._bytes_per_tick])
        chunk, _ = split_tick_audio(raw, self._bytes_per_tick)
        return chunk, len(raw)
