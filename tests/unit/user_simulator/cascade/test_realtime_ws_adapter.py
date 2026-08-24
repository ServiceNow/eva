import asyncio
import base64
import json

import pytest

try:
    import audioop
except ImportError:  # pragma: no cover - Python 3.13+
    import audioop_lts as audioop

from eva.user_simulator.cascade.adapter.realtime_ws import FRAMES_PER_TICK, RealtimeWSAdapter

BYTES_PER_TICK = 6400
_SETTLE_ROUNDS = 300


class FakeWebSocket:
    """Collects sent frames and replays queued inbound frames.

    recv() completes without ever suspending when a frame is already queued —
    unlike real `websockets.recv()`. Use SuspendingFakeWebSocket to model that.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.inbound: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        return await self.inbound.get()

    async def close(self) -> None:
        self.closed = True


class SuspendingFakeWebSocket(FakeWebSocket):
    """A recv() that always suspends at least once before returning, like the real thing."""

    async def recv(self) -> str:
        await asyncio.sleep(0)
        return await self.inbound.get()


class RaisingFakeWebSocket(FakeWebSocket):
    """A recv() that raises once a configured number of successful frames have been read."""

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self._fail_after = fail_after
        self._count = 0

    async def recv(self) -> str:
        await asyncio.sleep(0)
        if self._count >= self._fail_after:
            raise ConnectionError("simulated disconnect")
        self._count += 1
        return await self.inbound.get()


def _media_frame(mulaw: bytes) -> str:
    payload = base64.b64encode(mulaw).decode()
    return json.dumps({"event": "media", "media": {"payload": payload}})


async def _settle() -> None:
    """Give a background receive task many event-loop turns to drain queued frames."""
    for _ in range(_SETTLE_ROUNDS):
        await asyncio.sleep(0)


async def test_tick_with_no_inbound_audio_yields_padded_silence():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    result = await adapter.run_tick(0, None)

    assert result.assistant_audio_raw_bytes == 0
    assert result.has_assistant_speech is False
    assert len(result.assistant_audio) == BYTES_PER_TICK

    await adapter.stop()


async def test_receive_loop_drains_frames_from_a_suspending_websocket():
    """Regression test: a suspending recv() must still be drained by the background loop."""
    ws = SuspendingFakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await ws.inbound.put(_media_frame(b"\xff" * 160))
    await adapter.start()
    await _settle()

    result = await adapter.run_tick(0, None)

    assert result.has_assistant_speech is True
    assert result.assistant_audio_raw_bytes > 0

    await adapter.stop()


async def test_burst_of_frames_ingested_then_released_one_tick_at_a_time():
    ws = SuspendingFakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    frame_count = 20  # 400ms of wire audio delivered as one burst (two ticks' worth).
    for _ in range(frame_count):
        await ws.inbound.put(_media_frame(b"\xff" * 160))
    await adapter.start()
    await _settle()

    first = await adapter.run_tick(0, None)
    second = await adapter.run_tick(1, None)
    third = await adapter.run_tick(2, None)

    assert len(first.assistant_audio) == BYTES_PER_TICK
    assert len(second.assistant_audio) == BYTES_PER_TICK
    assert first.has_assistant_speech is True
    assert second.has_assistant_speech is True
    total_raw = first.assistant_audio_raw_bytes + second.assistant_audio_raw_bytes + third.assistant_audio_raw_bytes
    assert abs(total_raw - frame_count * 640) <= 2

    await adapter.stop()


async def test_receive_error_surfaces_from_run_tick():
    ws = RaisingFakeWebSocket(fail_after=0)
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    await _settle()

    with pytest.raises(RuntimeError):
        await adapter.run_tick(0, None)

    await adapter.stop()


async def test_inbound_mulaw_is_converted_and_reported_as_speech():
    ws = SuspendingFakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    # 160 mulaw bytes @8kHz == 20ms, resampled to roughly 640 PCM16 bytes @16kHz.
    await ws.inbound.put(_media_frame(b"\xff" * 160))
    await adapter.start()
    await _settle()

    result = await adapter.run_tick(0, None)

    assert result.assistant_audio_raw_bytes > 0
    assert result.has_assistant_speech is True
    assert len(result.assistant_audio) == BYTES_PER_TICK

    await adapter.stop()


async def test_outgoing_caller_audio_is_sent_as_twilio_media_frames():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK)

    media_frames = [json.loads(m) for m in ws.sent if json.loads(m).get("event") == "media"]
    # One tick (200ms) is ten 20ms wire frames.
    assert len(media_frames) == 10

    await adapter.stop()


async def test_silent_tick_emits_a_full_tick_of_silence_frames():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    await adapter.run_tick(0, None)

    media_frames = [json.loads(m) for m in ws.sent if json.loads(m).get("event") == "media"]
    assert len(media_frames) == 10
    for frame in media_frames:
        mulaw = base64.b64decode(frame["media"]["payload"])
        pcm = audioop.ulaw2lin(mulaw, 2)
        assert audioop.max(pcm, 2) == 0

    await adapter.stop()


async def test_mixed_speak_silent_sequence_has_no_gaps_in_outbound_stream():
    """Regression test: a stall between speech ticks must still send silence, not nothing."""
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    outgoing_sequence = [b"\x00" * BYTES_PER_TICK, None, None, b"\x00" * BYTES_PER_TICK]
    for tick_number, outgoing in enumerate(outgoing_sequence):
        await adapter.run_tick(tick_number, outgoing)

    media_frames = [json.loads(m) for m in ws.sent if json.loads(m).get("event") == "media"]
    assert len(media_frames) == len(outgoing_sequence) * 10

    await adapter.stop()


async def test_overflow_audio_carries_into_the_next_tick():
    ws = SuspendingFakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    # 400ms of assistant audio arrives at once: 3200 mulaw bytes -> ~12800 PCM bytes,
    # more than one tick's worth, so it must span both ticks with real audio in each.
    await ws.inbound.put(_media_frame(b"\xff" * 3200))
    await adapter.start()
    await _settle()

    first = await adapter.run_tick(0, None)
    second = await adapter.run_tick(1, None)

    assert len(first.assistant_audio) == BYTES_PER_TICK
    assert len(second.assistant_audio) == BYTES_PER_TICK
    assert first.assistant_audio_raw_bytes == BYTES_PER_TICK
    assert second.assistant_audio_raw_bytes > 0
    assert first.has_assistant_speech is True
    assert second.has_assistant_speech is True

    await adapter.stop()


async def test_per_frame_resampling_does_not_accumulate_drift():
    ws = SuspendingFakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    frame_count = 50
    for _ in range(frame_count):
        await ws.inbound.put(_media_frame(b"\xff" * 160))
    await adapter.start()
    await _settle()

    ideal_bytes = frame_count * 640
    ticks_needed = ideal_bytes // BYTES_PER_TICK + 2
    total_raw_bytes = 0
    for tick_number in range(ticks_needed):
        result = await adapter.run_tick(tick_number, None)
        total_raw_bytes += result.assistant_audio_raw_bytes

    # 1 sample (2 bytes) of PCM16 warm-up loss is expected for the whole
    # stream; per-frame loss must not accumulate beyond that.
    assert abs(total_raw_bytes - ideal_bytes) <= 2

    await adapter.stop()


async def test_user_speech_start_emitted_once_on_silence_to_audio_transition():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    await adapter.run_tick(0, None)
    await adapter.run_tick(1, b"\x00" * BYTES_PER_TICK)
    await adapter.run_tick(2, b"\x00" * BYTES_PER_TICK)

    events = [json.loads(m) for m in ws.sent]
    starts = [e for e in events if e.get("event") == "user_speech_start"]
    assert len(starts) == 1
    assert isinstance(starts[0]["timestamp_ms"], str)

    await adapter.stop()


async def test_silent_tick_paces_to_approximately_one_tick_duration():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    start = asyncio.get_event_loop().time()
    await adapter.run_tick(0, None)
    elapsed = asyncio.get_event_loop().time() - start

    assert 0.15 < elapsed < 0.4

    await adapter.stop()


async def test_speaking_tick_paces_to_approximately_one_tick_duration_not_double():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    start = asyncio.get_event_loop().time()
    await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK)
    elapsed = asyncio.get_event_loop().time() - start

    assert 0.15 < elapsed < 0.4

    await adapter.stop()


async def test_user_speech_stop_emitted_once_on_audio_to_silence_transition():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK)
    await adapter.run_tick(1, None)
    await adapter.run_tick(2, None)

    events = [json.loads(m) for m in ws.sent]
    stops = [e for e in events if e.get("event") == "user_speech_stop"]
    assert len(stops) == 1
    assert isinstance(stops[0]["timestamp_ms"], str)

    await adapter.stop()


class StubPerturbator:
    """Stands in for AudioPerturbator with a constant, recognizable noise floor."""

    has_ambient_noise = True

    def get_ambient_chunk(self, size: int) -> bytes:
        return b"\x11" * size

    def apply(self, audio: bytes) -> bytes:
        return b"\x22" * len(audio)


def _media_payloads(ws) -> list[bytes]:
    import base64

    return [
        base64.b64decode(json.loads(m)["media"]["payload"]) for m in ws.sent if json.loads(m).get("event") == "media"
    ]


async def test_ambient_noise_replaces_silence_when_the_caller_is_not_speaking():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(
        websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK, perturbator=StubPerturbator()
    )

    await adapter.run_tick(0, None)

    payloads = _media_payloads(ws)
    assert len(payloads) == FRAMES_PER_TICK
    # Silence would encode to a constant mulaw byte; ambient noise must not.
    assert set(b"".join(payloads)) != {0xFF}


async def test_ambient_noise_is_mixed_into_caller_speech():
    ws = FakeWebSocket()
    perturbator = StubPerturbator()
    adapter = RealtimeWSAdapter(
        websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK, perturbator=perturbator
    )

    await adapter.run_tick(0, b"\x01" * BYTES_PER_TICK)

    assert len(_media_payloads(ws)) == FRAMES_PER_TICK


async def test_silence_is_still_sent_every_tick_without_a_perturbator():
    # Plan 1: a tick that sends no frames at all makes the assistant's turn detection misfire.
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)

    await adapter.run_tick(0, None)

    assert len(_media_payloads(ws)) == FRAMES_PER_TICK
