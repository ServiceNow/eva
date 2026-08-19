"""Real-time Twilio WebSocket adapter for unmodified assistant servers."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time

try:
    import audioop
except ImportError:  # pragma: no cover - Python 3.13+
    import audioop_lts as audioop

from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.constants import BYTES_PER_TICK, CALLER_SAMPLE_RATE, SILENCE_BYTE, TICK_DURATION_MS
from eva.user_simulator.cascade.tick_result import TickResult, split_tick_audio
from eva.utils.logging import get_logger

logger = get_logger(__name__)

WIRE_SAMPLE_RATE = 8000
WIRE_FRAME_MS = 20
FRAMES_PER_TICK = TICK_DURATION_MS // WIRE_FRAME_MS
_PCM_WIDTH = 2


class RealtimeWSAdapter(Adapter):
    """Exchanges tick-sized audio with an assistant server over the Twilio WS protocol.

    Outbound audio is paced at the real 20ms cadence the assistant expects
    (docs/assistant_server_contract.md section 3), and every tick sends a full
    tick's worth of frames — real audio or synthesized silence — so the assistant's
    STT/VAD always sees an unbroken stream, the way a real phone line would; gaps
    with no frames at all are what caused turn detection to misfire. A background
    task continuously drains inbound frames into an adapter-owned buffer; `run_tick`
    releases exactly one tick's worth per call, so a provider that generates faster
    than real time cannot run ahead of the simulation clock. `run_tick` also enforces
    a minimum tick duration as a safety net for ticks that send quickly.

    Each direction resamples a continuous stream, so each carries its own
    `audioop.ratecv` filter state across calls; the stateless helpers in
    `audio_utils` cannot express that and are not reused here for that reason.
    """

    def __init__(
        self,
        *,
        websocket,
        conversation_id: str,
        bytes_per_tick: int = BYTES_PER_TICK,
        perturbator=None,
    ) -> None:
        self._ws = websocket
        self._conversation_id = conversation_id
        self._bytes_per_tick = bytes_per_tick
        self._perturbator = perturbator
        self._inbound = bytearray()
        self._receive_task: asyncio.Task | None = None
        self._inbound_resample_state = None
        self._outbound_resample_state = None
        self._caller_speaking = False
        self._error: BaseException | None = None

    async def start(self) -> None:
        """Send the connect/start handshake and begin buffering inbound audio."""
        for event in ("connected", "start"):
            await self._ws.send(json.dumps({"event": event, "conversation_id": self._conversation_id}))
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None, *, barge_in: bool = False) -> TickResult:
        """Send one tick of caller audio at wire cadence and collect one tick of assistant audio.

        ``barge_in`` is accepted and ignored: this assistant paces its own output, so
        it has generated nothing past what the caller already heard to discard.
        """
        tick_start = asyncio.get_event_loop().time()
        if self._error is not None:
            raise RuntimeError("RealtimeWSAdapter receive loop failed") from self._error

        is_speaking = bool(outgoing_audio)
        if is_speaking and not self._caller_speaking:
            await self._send_speech_event("user_speech_start")
        elif not is_speaking and self._caller_speaking:
            await self._send_speech_event("user_speech_stop")
        self._caller_speaking = is_speaking

        outgoing = self._apply_perturbation(outgoing_audio)
        await self._send_tick_audio(outgoing or SILENCE_BYTE * self._bytes_per_tick)

        raw = bytes(self._inbound[: self._bytes_per_tick])
        del self._inbound[: len(raw)]
        chunk, _ = split_tick_audio(raw, self._bytes_per_tick)

        if self._error is not None:
            raise RuntimeError("RealtimeWSAdapter receive loop failed") from self._error

        result = TickResult(
            tick_number=tick_number,
            assistant_audio=chunk,
            assistant_audio_raw_bytes=len(raw),
            wall_clock_ms=int(time.time() * 1000),
        )

        remaining = TICK_DURATION_MS / 1000 - (asyncio.get_event_loop().time() - tick_start)
        if remaining > 0:
            await asyncio.sleep(remaining)

        return result

    def _apply_perturbation(self, outgoing_audio: bytes | None) -> bytes | None:
        """Mix ambient noise into this tick's outgoing audio.

        Real-time path: the mic is always open, so noise is emitted even when the
        caller is silent — it replaces the silence this adapter already sends every
        tick rather than adding frames. Never call this for a tick that emits nothing
        on the tick-driven path (Plan 3): audio sent during a stall would advance the
        assistant's VAD and break the freeze.
        """
        if self._perturbator is None:
            return outgoing_audio
        if outgoing_audio:
            return self._perturbator.apply(outgoing_audio)
        if getattr(self._perturbator, "has_ambient_noise", False):
            return self._perturbator.get_ambient_chunk(self._bytes_per_tick)
        return outgoing_audio

    async def stop(self) -> None:
        """Send stop, cancel the receive loop, and close the socket. Safe to call twice."""
        if self._receive_task is not None:
            task = self._receive_task
            self._receive_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except Exception:
                pass
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"event": "stop", "conversation_id": self._conversation_id}))
        with contextlib.suppress(Exception):
            await self._ws.close()

    async def _send_speech_event(self, event: str) -> None:
        """Emit a user_speech_start/stop event matching the existing bridge's payload shape."""
        await self._ws.send(
            json.dumps(
                {
                    "event": event,
                    "conversation_id": self._conversation_id,
                    "timestamp_ms": str(int(round(time.time() * 1000))),
                }
            )
        )

    async def _send_tick_audio(self, pcm: bytes) -> None:
        """Split one tick of PCM16 into 20ms mulaw frames and send them at real-time pace."""
        mulaw = self._pcm16k_to_mulaw8k(pcm)
        if not mulaw:
            return
        frame_size = len(mulaw) // FRAMES_PER_TICK
        frames = [mulaw[index : index + frame_size] for index in range(0, len(mulaw), frame_size)]
        interval = WIRE_FRAME_MS / 1000
        start_time = asyncio.get_event_loop().time()
        for frame_index, frame in enumerate(frames):
            payload = base64.b64encode(frame).decode()
            await self._ws.send(
                json.dumps(
                    {
                        "event": "media",
                        "conversation_id": self._conversation_id,
                        "media": {"payload": payload},
                    }
                )
            )
            if frame_index == len(frames) - 1:
                break
            deadline = start_time + (frame_index + 1) * interval
            sleep_for = deadline - asyncio.get_event_loop().time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    async def _receive_loop(self) -> None:
        """Continuously buffer inbound assistant audio as PCM16 at the caller sample rate."""
        while True:
            try:
                raw = await self._ws.recv()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("RealtimeWSAdapter receive loop failed")
                self._error = exc
                return
            self._ingest(raw)

    def _ingest(self, raw: str) -> None:
        """Decode one inbound frame, keeping only media payloads."""
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        if message.get("event") != "media":
            return
        payload = message.get("media", {}).get("payload", "")
        if payload:
            self._inbound.extend(self._mulaw8k_to_pcm16k(base64.b64decode(payload)))

    def _mulaw8k_to_pcm16k(self, mulaw: bytes) -> bytes:
        """Convert 8kHz mulaw from the wire to PCM16 at the caller sample rate."""
        pcm_8k = audioop.ulaw2lin(mulaw, _PCM_WIDTH)
        pcm_16k, self._inbound_resample_state = audioop.ratecv(
            pcm_8k, _PCM_WIDTH, 1, WIRE_SAMPLE_RATE, CALLER_SAMPLE_RATE, self._inbound_resample_state
        )
        return pcm_16k

    def _pcm16k_to_mulaw8k(self, pcm: bytes) -> bytes:
        """Convert caller PCM16 to the 8kHz mulaw the assistant expects."""
        pcm_8k, self._outbound_resample_state = audioop.ratecv(
            pcm, _PCM_WIDTH, 1, CALLER_SAMPLE_RATE, WIRE_SAMPLE_RATE, self._outbound_resample_state
        )
        return audioop.lin2ulaw(pcm_8k, _PCM_WIDTH)
