import asyncio
import json
import time

from eva.user_simulator.cascade.adapter.tick_driven import TickDrivenAdapter
from tests.unit.user_simulator.cascade.test_realtime_ws_adapter import (
    BYTES_PER_TICK,
    FakeWebSocket,
    _media_frame,
    _settle,
)


def _media(ws: FakeWebSocket) -> list[dict]:
    return [json.loads(m) for m in ws.sent if json.loads(m).get("event") == "media"]


async def test_outbound_audio_is_sent_without_pacing_sleeps():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    started = time.monotonic()
    await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK)

    # The real-time adapter would spend ~200ms pacing this.
    assert time.monotonic() - started < 0.05
    assert len(_media(ws)) == 10
    await adapter.stop()


async def test_burst_of_provider_audio_releases_one_tick_at_a_time():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    # 1 second of assistant audio arrives at once: 8000 mulaw bytes -> 32000 PCM bytes.
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()

    results = [await adapter.run_tick(tick, None) for tick in range(5)]

    # The resampler's filter state leaves the last tick a couple of samples short;
    # what matters is that no tick releases more than one tick's worth.
    raw = [r.assistant_audio_raw_bytes for r in results]
    assert raw[:4] == [BYTES_PER_TICK] * 4
    assert 0 < raw[4] <= BYTES_PER_TICK
    # Short ticks are silence-padded, so every released chunk is exactly tick-sized.
    assert all(len(r.assistant_audio) == BYTES_PER_TICK for r in results)
    await adapter.stop()


async def test_nothing_is_emitted_on_a_silent_tick():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    await adapter.run_tick(0, None)

    # Audio sent during a stall would advance the provider's VAD and break the freeze.
    assert _media(ws) == []
    await adapter.stop()


async def test_played_position_tracks_released_ticks():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()

    for tick in range(3):
        await adapter.run_tick(tick, None)

    assert adapter.played_ms == 600
    await adapter.stop()


async def test_a_tick_with_no_assistant_audio_does_not_advance_the_played_position():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    for tick in range(3):
        await adapter.run_tick(tick, None)

    assert adapter.played_ms == 0
    await adapter.stop()


async def test_run_tick_does_not_wait_out_the_tick_duration():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    started = time.monotonic()
    await asyncio.gather(*(adapter.run_tick(tick, None) for tick in range(3)))

    # The real-time adapter enforces a 200ms floor per tick; this one must not.
    assert time.monotonic() - started < 0.05
    await adapter.stop()


async def test_barge_in_reports_the_played_position_not_the_received_position():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    # 1s of audio arrives at once but only 3 ticks (600ms) get released.
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()
    for tick in range(3):
        await adapter.run_tick(tick, None)

    result = await adapter.run_tick(3, b"\x00" * BYTES_PER_TICK, barge_in=True)

    assert result.interruption_audio_start_ms == 600
    truncate = [json.loads(m) for m in ws.sent if json.loads(m).get("event") == "truncate"]
    assert truncate and truncate[0]["audio_end_ms"] == 600
    await adapter.stop()


async def test_barge_in_discards_audio_the_caller_never_heard():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()

    result = await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK, barge_in=True)

    # The buffered second of assistant audio is audio the caller cut off.
    assert result.assistant_audio_raw_bytes == 0
    assert adapter.played_ms == 0
    await adapter.stop()
