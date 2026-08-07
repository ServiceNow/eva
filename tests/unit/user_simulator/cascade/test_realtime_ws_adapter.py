import asyncio
import base64
import json

from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter

BYTES_PER_TICK = 6400


class FakeWebSocket:
    """Collects sent frames and replays queued inbound frames."""

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


def _media_frame(mulaw: bytes) -> str:
    payload = base64.b64encode(mulaw).decode()
    return json.dumps({"event": "media", "media": {"payload": payload}})


async def test_tick_with_no_inbound_audio_yields_padded_silence():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)

    result = await adapter.run_tick(0, None)

    assert result.assistant_audio_raw_bytes == 0
    assert result.has_assistant_speech is False
    assert len(result.assistant_audio) == BYTES_PER_TICK


async def test_inbound_mulaw_is_converted_and_reported_as_speech():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    # 160 mulaw bytes @8kHz == 20ms == 640 PCM16 bytes @16kHz.
    await ws.inbound.put(_media_frame(b"\xff" * 160))

    result = await adapter.run_tick(0, None)

    assert result.assistant_audio_raw_bytes == 640
    assert result.has_assistant_speech is True
    assert len(result.assistant_audio) == BYTES_PER_TICK


async def test_outgoing_caller_audio_is_sent_as_twilio_media_frames():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)

    await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK)

    media_frames = [json.loads(m) for m in ws.sent if json.loads(m).get("event") == "media"]
    # One tick (200ms) is ten 20ms wire frames.
    assert len(media_frames) == 10


async def test_overflow_audio_carries_into_the_next_tick():
    ws = FakeWebSocket()
    adapter = RealtimeWSAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    # 400ms of assistant audio arrives at once: 3200 mulaw bytes -> 12800 PCM bytes.
    await ws.inbound.put(_media_frame(b"\xff" * 3200))

    first = await adapter.run_tick(0, None)
    second = await adapter.run_tick(1, None)

    assert first.assistant_audio_raw_bytes == BYTES_PER_TICK
    assert second.assistant_audio_raw_bytes == BYTES_PER_TICK
