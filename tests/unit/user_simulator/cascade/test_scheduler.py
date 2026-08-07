from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.scheduler import TickScheduler
from eva.user_simulator.cascade.tick_result import TickResult

BYTES_PER_TICK = 8


class FakeAdapter(Adapter):
    """Replays a scripted sequence of assistant speech/silence and records what it was sent."""

    def __init__(self, speech_ticks: list[bool]) -> None:
        self.speech_ticks = speech_ticks
        self.sent: list[bytes | None] = []

    async def start(self) -> None:
        pass

    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None) -> TickResult:
        self.sent.append(outgoing_audio)
        speaking = self.speech_ticks[tick_number] if tick_number < len(self.speech_ticks) else False
        return TickResult(
            tick_number=tick_number,
            assistant_audio=(b"\x01" if speaking else b"\x00") * BYTES_PER_TICK,
            assistant_audio_raw_bytes=BYTES_PER_TICK if speaking else 0,
            wall_clock_ms=tick_number,
        )

    async def stop(self) -> None:
        pass


def _scheduler(speech_ticks: list[bool]) -> TickScheduler:
    return TickScheduler(FakeAdapter(speech_ticks), bytes_per_tick=BYTES_PER_TICK)


async def test_caller_may_open_the_conversation_once_the_assistant_falls_silent():
    # Assistant greets for 3 ticks, then goes quiet.
    scheduler = _scheduler([True, True, True])

    for _ in range(3):
        await scheduler.run_tick()
    assert scheduler.may_take_turn() is False

    # WAIT_TO_RESPOND_OTHER_MS is 5 ticks and the comparison is strict.
    for _ in range(5):
        await scheduler.run_tick()
    assert scheduler.may_take_turn() is False

    await scheduler.run_tick()
    assert scheduler.may_take_turn() is True


async def test_assistant_speech_resets_the_silence_counter():
    scheduler = _scheduler([False] * 10 + [True])

    for _ in range(11):
        await scheduler.run_tick()

    assert scheduler.may_take_turn() is False


async def test_caller_must_also_wait_out_its_own_silence_threshold():
    scheduler = _scheduler([True])  # assistant greets on tick 0
    await scheduler.run_tick()
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)

    await scheduler.run_tick()  # caller speaks this tick
    assert scheduler.may_take_turn() is False

    # Assistant silence is satisfied almost immediately, self-silence needs 25 ticks.
    for _ in range(25):
        await scheduler.run_tick()
    assert scheduler.may_take_turn() is False

    await scheduler.run_tick()
    assert scheduler.may_take_turn() is True


async def test_caller_does_not_speak_before_the_assistant_has_greeted():
    # The assistant sends the opening message; the caller must not race it.
    scheduler = _scheduler([])

    for _ in range(50):
        await scheduler.run_tick()

    assert scheduler.may_take_turn() is False


async def test_queued_utterance_drains_one_tick_at_a_time():
    adapter = FakeAdapter([])
    scheduler = TickScheduler(adapter, bytes_per_tick=BYTES_PER_TICK)
    scheduler.enqueue_utterance(b"\x02" * (BYTES_PER_TICK * 3))

    for _ in range(4):
        await scheduler.run_tick()

    assert adapter.sent[0] == b"\x02" * BYTES_PER_TICK
    assert adapter.sent[1] == b"\x02" * BYTES_PER_TICK
    assert adapter.sent[2] == b"\x02" * BYTES_PER_TICK
    assert adapter.sent[3] is None
    assert scheduler.caller_is_speaking is False


async def test_partial_final_chunk_is_padded_to_a_whole_tick():
    adapter = FakeAdapter([])
    scheduler = TickScheduler(adapter, bytes_per_tick=BYTES_PER_TICK)
    scheduler.enqueue_utterance(b"\x02" * (BYTES_PER_TICK + 3))

    await scheduler.run_tick()
    await scheduler.run_tick()

    assert adapter.sent[1] == b"\x02" * 3 + b"\x00" * (BYTES_PER_TICK - 3)
    assert scheduler.caller_is_speaking is False


async def test_caller_is_speaking_while_audio_remains_queued():
    scheduler = _scheduler([])
    scheduler.enqueue_utterance(b"\x02" * (BYTES_PER_TICK * 2))

    assert scheduler.caller_is_speaking is True
    await scheduler.run_tick()
    assert scheduler.caller_is_speaking is True
    await scheduler.run_tick()
    assert scheduler.caller_is_speaking is False
