"""Tick-driven adapter: the caller owns the simulation clock.

Talks to the same assistant server over the same Twilio WebSocket as the
real-time adapter, with two differences. Outbound audio is not paced, because
the server is not pacing its own output either (``paced_output=False``); and
nothing at all is emitted on a tick where the caller is silent.

Because the provider's VAD advances on audio received rather than wall time,
not sending audio freezes the assistant. That is what makes caller compute time
invisible here — and why nothing may be emitted on a stalled tick.

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
from eva.user_simulator.cascade.constants import BYTES_PER_TICK
from eva.user_simulator.cascade.tick_result import TickResult, played_audio_ms, split_tick_audio
from eva.utils.logging import get_logger

logger = get_logger(__name__)

MAX_INACTIVE_SECONDS = 40.0
"""Fail loudly if the provider goes quiet this long (tau: DEFAULT_AUDIO_NATIVE_MAX_INACTIVE_SECONDS)."""


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

        if outgoing_audio:
            # Perturbation only ever rides on real caller audio here: mixing ambient
            # noise into a stalled tick would emit frames and unfreeze the assistant.
            await self._send_unpaced(self._apply_perturbation(outgoing_audio) or outgoing_audio)

        await asyncio.sleep(0)
        raw = bytes(self._inbound[: self._bytes_per_tick])
        del self._inbound[: len(raw)]
        chunk, _ = split_tick_audio(raw, self._bytes_per_tick)
        if raw:
            self._ticks_released += 1

        self._check_provider_alive(bool(raw))

        if self._error is not None:
            raise RuntimeError("TickDrivenAdapter receive loop failed") from self._error

        return TickResult(
            tick_number=tick_number,
            assistant_audio=chunk,
            assistant_audio_raw_bytes=len(raw),
            wall_clock_ms=int(time.time() * 1000),
            interruption_audio_start_ms=interruption_start,
        )

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

    def _check_provider_alive(self, received: bool) -> None:
        """Raise if the provider has produced nothing for too long.

        Wall clock is the right measure here and only here: this is a liveness
        check on a real network peer, not a measurement of conversation time.
        """
        now = time.monotonic()
        if received:
            self._last_inbound_monotonic = now
        elif now - self._last_inbound_monotonic > MAX_INACTIVE_SECONDS:
            raise RuntimeError(f"No assistant audio for {MAX_INACTIVE_SECONDS}s; provider appears stalled")
