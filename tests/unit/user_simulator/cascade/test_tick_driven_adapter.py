import asyncio
import json
import time

from eva.user_simulator.cascade.adapter.tick_driven import (
    MAX_INACTIVE_SECONDS,
    QUIET_TICK_GRACE_S,
    TickDrivenAdapter,
)
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
    # Assistant audio already buffered, so the tick has no reason to wait for any.
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()

    started = time.monotonic()
    await adapter.run_tick(0, b"\x00" * BYTES_PER_TICK)

    # The real-time adapter would spend ~200ms pacing the outbound frames.
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


async def test_a_silent_tick_still_puts_a_full_tick_of_silence_on_the_wire():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    await adapter.run_tick(0, None)

    # The provider's VAD ends the caller's turn on received silence. A tick that
    # emitted nothing would starve it and the assistant would never reply.
    assert len(_media(ws)) == 10
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


async def test_ticks_with_audio_already_buffered_do_not_wait_out_the_tick_duration():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    # 1s of audio arrives at once; releasing it must not take 1s of wall time.
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()

    started = time.monotonic()
    for tick in range(4):
        await adapter.run_tick(tick, None)

    # The real-time adapter enforces a 200ms floor per tick; this one must not.
    assert time.monotonic() - started < 0.05
    await adapter.stop()


async def test_a_tick_waits_out_the_grace_before_calling_the_assistant_silent():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    started = time.monotonic()
    result = await adapter.run_tick(0, None)

    # Without this bound the tick loop spins through its whole budget instantly and
    # the call ends at tick zero having heard nothing.
    assert result.assistant_audio_raw_bytes == 0
    assert time.monotonic() - started >= QUIET_TICK_GRACE_S
    await adapter.stop()


async def test_a_tick_returns_as_soon_as_audio_arrives_mid_grace():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()

    async def _deliver_late() -> None:
        await asyncio.sleep(QUIET_TICK_GRACE_S / 4)
        await ws.inbound.put(_media_frame(b"\xff" * 8000))

    asyncio.create_task(_deliver_late())
    started = time.monotonic()
    result = await adapter.run_tick(0, None)

    assert result.assistant_audio_raw_bytes == BYTES_PER_TICK
    assert time.monotonic() - started < QUIET_TICK_GRACE_S
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


async def test_a_stalled_provider_is_reported_not_raised():
    # Raising aborted the tick loop from under the simulator, so the record ended on the
    # generic "error" reason and the terminal state the runner reads was never written.
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    adapter._last_inbound_monotonic = time.monotonic() - (MAX_INACTIVE_SECONDS + 1)

    result = await adapter.run_tick(0, None)

    assert result.provider_stalled is True
    await adapter.stop()


async def test_a_live_provider_is_never_reported_as_stalled():
    ws = FakeWebSocket()
    adapter = TickDrivenAdapter(websocket=ws, conversation_id="c1", bytes_per_tick=BYTES_PER_TICK)
    await adapter.start()
    adapter._last_inbound_monotonic = time.monotonic() - (MAX_INACTIVE_SECONDS + 1)
    await ws.inbound.put(_media_frame(b"\xff" * 8000))
    await _settle()

    result = await adapter.run_tick(0, None)

    # Audio arrived this tick, so the liveness clock resets rather than firing.
    assert result.provider_stalled is False
    await adapter.stop()
