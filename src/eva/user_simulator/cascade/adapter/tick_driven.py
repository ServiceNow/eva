"""Tick-driven adapter: the caller owns the simulation clock.

Talks to the same assistant server over the same Twilio WebSocket as the
real-time adapter, and sends a full tick of audio — real or silence — on every
tick just as it does. The difference is timing: outbound frames carry no pacing
sleeps, because the server is not pacing its own output either
(``paced_output=False``), and there is no minimum tick duration.

Because the provider's VAD advances on audio received rather than on wall time,
the conversation advances only when the caller ticks it. While the caller is
generating a turn no tick runs, so no audio flows and the assistant is frozen —
that is what makes caller compute time invisible. The freeze comes from *not
ticking*, not from emitting nothing within a tick: a tick that sent nothing would
starve the VAD of the trailing silence that ends the caller's turn, and the
assistant would never reply at all.

Inbound audio is released strictly one tick at a time however much arrives at
once, which the real-time adapter already does; the difference is that here the
release cadence *is* the simulation clock rather than an approximation of it.

Implemented as a subclass of :class:`RealtimeWSAdapter` rather than a peer: the
handshake, the receive loop, the speech-event frames and — critically — the
mulaw/PCM resamplers all carry per-instance filter state that must not be
duplicated. Only the timing is overridden.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

from eva.user_simulator.cascade.adapter.realtime_ws import FRAMES_PER_TICK, RealtimeWSAdapter
from eva.user_simulator.cascade.constants import BYTES_PER_TICK, SILENCE_BYTE, TICK_DURATION_MS
from eva.user_simulator.cascade.tick_result import TickResult, played_audio_ms, split_tick_audio
from eva.utils.logging import get_logger

logger = get_logger(__name__)

MAX_INACTIVE_SECONDS = 40.0
"""Fail loudly if the provider goes quiet this long (tau: DEFAULT_AUDIO_NATIVE_MAX_INACTIVE_SECONDS)."""

QUIET_TICK_GRACE_S = TICK_DURATION_MS / 1000
"""How long a tick waits for a full tick of assistant audio before calling it silence.

Not pacing: it never *delays* audio that has already arrived, so a provider
generating faster than real time is still drained as fast as it produces, and
caller compute still costs the conversation nothing. It is only the bound on how
long "the assistant has said nothing yet" takes to establish. Without it the loop
spins through its whole tick budget in milliseconds and every conversation ends at
tick zero with nothing said."""


class TickDrivenAdapter(RealtimeWSAdapter):
    """Exchanges one tick of audio with an unpaced assistant server."""

    def __init__(
        self,
        *,
        websocket,
        conversation_id: str,
        bytes_per_tick: int = BYTES_PER_TICK,
        perturbator=None,
    ) -> None:
        super().__init__(
            websocket=websocket,
            conversation_id=conversation_id,
            bytes_per_tick=bytes_per_tick,
            perturbator=perturbator,
        )
        self._ticks_released = 0
        self._last_inbound_monotonic = time.monotonic()
        self._audio_arrived = asyncio.Event()

    @property
    def played_ms(self) -> int:
        """Assistant audio released into the conversation so far, in simulated ms."""
        return played_audio_ms(ticks_released=self._ticks_released)

    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None, *, barge_in: bool = False) -> TickResult:
        """Send this tick's caller audio unpaced and release exactly one tick of assistant audio.

        When ``barge_in`` is set, first tell the assistant to discard the audio it
        generated past the position the caller has actually heard.
        """
        if self._error is not None:
            raise RuntimeError("TickDrivenAdapter receive loop failed") from self._error

        interruption_start: int | None = None
        if barge_in:
            interruption_start = self.played_ms
            await self._ws.send(
                json.dumps(
                    {
                        "event": "truncate",
                        "conversation_id": self._conversation_id,
                        "audio_end_ms": interruption_start,
                    }
                )
            )
            # Everything already buffered is audio the caller never heard.
            self._inbound.clear()

        is_speaking = bool(outgoing_audio)
        if is_speaking and not self._caller_speaking:
            await self._send_speech_event("user_speech_start")
        elif not is_speaking and self._caller_speaking:
            await self._send_speech_event("user_speech_stop")
        self._caller_speaking = is_speaking

        # Every tick puts a full tick on the wire, silence included, exactly as the
        # real-time adapter does. The provider's VAD advances on audio *received*, so
        # a tick that emits nothing never ends the caller's turn and the assistant
        # never replies. The freeze this plan is built on comes from the caller not
        # calling `run_tick` while it thinks — not from emitting nothing inside one.
        outgoing = self._apply_perturbation(outgoing_audio)
        await self._send_unpaced(outgoing or SILENCE_BYTE * self._bytes_per_tick)

        await self._await_tick_of_audio()
        raw = bytes(self._inbound[: self._bytes_per_tick])
        del self._inbound[: len(raw)]
        chunk, _ = split_tick_audio(raw, self._bytes_per_tick)
        if raw:
            self._ticks_released += 1

        stalled = self._provider_has_stalled(bool(raw))

        if self._error is not None:
            raise RuntimeError("TickDrivenAdapter receive loop failed") from self._error

        return TickResult(
            tick_number=tick_number,
            assistant_audio=chunk,
            assistant_audio_raw_bytes=len(raw),
            wall_clock_ms=int(time.time() * 1000),
            interruption_audio_start_ms=interruption_start,
            provider_stalled=stalled,
        )

    def _on_inbound_audio(self) -> None:
        """Wake a tick that is waiting on assistant audio."""
        self._audio_arrived.set()

    async def _await_tick_of_audio(self) -> None:
        """Wait until a whole tick of assistant audio is buffered, or the grace expires.

        Returns as soon as the buffer is full, so nothing already generated is held
        back. The grace only bounds the silent case.
        """
        deadline = time.monotonic() + QUIET_TICK_GRACE_S
        while len(self._inbound) < self._bytes_per_tick:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._audio_arrived.clear()
            if len(self._inbound) >= self._bytes_per_tick:
                return
            try:
                await asyncio.wait_for(self._audio_arrived.wait(), timeout=remaining)
            except TimeoutError:
                return

    async def _send_unpaced(self, pcm: bytes) -> None:
        """Split one tick of PCM16 into wire frames and send them with no sleeps."""
        mulaw = self._pcm16k_to_mulaw8k(pcm)
        if not mulaw:
            return
        frame_size = len(mulaw) // FRAMES_PER_TICK or len(mulaw)
        for index in range(0, len(mulaw), frame_size):
            payload = base64.b64encode(mulaw[index : index + frame_size]).decode()
            await self._ws.send(
                json.dumps(
                    {
                        "event": "media",
                        "conversation_id": self._conversation_id,
                        "media": {"payload": payload},
                    }
                )
            )

    def _provider_has_stalled(self, received: bool) -> bool:
        """Whether the provider has produced nothing for too long.

        Wall clock is the right measure here and only here: this is a liveness
        check on a real network peer, not a measurement of conversation time.

        Reported, not raised. Raising aborted the tick loop from underneath the
        simulator, so the record ended on the generic "error" reason with no way to
        tell a stalled provider from a bug in the caller, and the terminal state the
        runner reads was never written. The caller ends the conversation instead.
        """
        now = time.monotonic()
        if received:
            self._last_inbound_monotonic = now
            return False
        return now - self._last_inbound_monotonic > MAX_INACTIVE_SECONDS
