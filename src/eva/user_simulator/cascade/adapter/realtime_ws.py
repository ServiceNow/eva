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
from eva.user_simulator.cascade.constants import BYTES_PER_TICK, CALLER_SAMPLE_RATE, TICK_DURATION_MS
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
    (docs/assistant_server_contract.md section 3). Inbound audio is buffered and
    released exactly one tick at a time, so a provider that generates faster than
    real time cannot run ahead of the simulation clock.

    Each direction resamples a continuous stream, so each carries its own
    `audioop.ratecv` filter state across calls; the stateless helpers in
    `audio_utils` cannot express that and are not reused here for that reason.
    """

    def __init__(self, *, websocket, conversation_id: str, bytes_per_tick: int = BYTES_PER_TICK) -> None:
        self._ws = websocket
        self._conversation_id = conversation_id
        self._bytes_per_tick = bytes_per_tick
        self._inbound = bytearray()
        self._pending_recv: asyncio.Task | None = None
        self._inbound_resample_state = None
        self._outbound_resample_state = None

    async def start(self) -> None:
        """Send the connect/start handshake."""
        for event in ("connected", "start"):
            await self._ws.send(json.dumps({"event": event, "conversation_id": self._conversation_id}))

    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None) -> TickResult:
        """Send one tick of caller audio at wire cadence and collect one tick of assistant audio."""
        if outgoing_audio:
            await self._send_tick_audio(outgoing_audio)

        await self._drain_pending()
        raw = bytes(self._inbound[: self._bytes_per_tick])
        del self._inbound[: len(raw)]
        chunk, _ = split_tick_audio(raw, self._bytes_per_tick)

        return TickResult(
            tick_number=tick_number,
            assistant_audio=chunk,
            assistant_audio_raw_bytes=len(raw),
            wall_clock_ms=int(time.time() * 1000),
        )

    async def stop(self) -> None:
        """Send stop, cancel any in-flight receive, and close the socket. Safe to call twice."""
        if self._pending_recv is not None:
            self._pending_recv.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pending_recv
            self._pending_recv = None
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"event": "stop", "conversation_id": self._conversation_id}))
        with contextlib.suppress(Exception):
            await self._ws.close()

    async def _send_tick_audio(self, pcm: bytes) -> None:
        """Split one tick of PCM16 into 20ms mulaw frames and send them at real-time pace."""
        mulaw = self._pcm16k_to_mulaw8k(pcm)
        frame_size = len(mulaw) // FRAMES_PER_TICK or len(mulaw)
        interval = WIRE_FRAME_MS / 1000
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
            await asyncio.sleep(interval)

    async def _drain_pending(self) -> None:
        """Pull any inbound frames that have already arrived, without blocking on new ones.

        A single ``recv()`` call is always in flight, tracked as ``_pending_recv``. Each
        tick, we yield control once so an already-arrived frame (queued before this call,
        as in tests that never start a background loop) can complete that task, then keep
        harvesting completed tasks and re-arming a fresh one until none are ready. If the
        pending task is still waiting on the wire, we leave it in place for the next tick
        to pick up rather than cancelling it — a websocket has only one live recv() at a
        time. This makes the drain path identical whether `start()` launched anything or
        not, so there is no test-only branch in production code.
        """
        if self._pending_recv is None:
            self._pending_recv = asyncio.ensure_future(self._ws.recv())

        await asyncio.sleep(0)
        while self._pending_recv is not None and self._pending_recv.done():
            task = self._pending_recv
            self._pending_recv = None
            try:
                raw = task.result()
            except Exception:
                return
            self._ingest(raw)
            self._pending_recv = asyncio.ensure_future(self._ws.recv())
            await asyncio.sleep(0)

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
