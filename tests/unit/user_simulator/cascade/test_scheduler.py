from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.constants import ASSISTANT_UNRESPONSIVE_MS, ms_to_ticks
from eva.user_simulator.cascade.scheduler import TickScheduler
from eva.user_simulator.cascade.tick_result import TickResult

BYTES_PER_TICK = 8


class FakeAdapter(Adapter):
    """Replays a scripted sequence of assistant speech/silence and records what it was sent."""

    def __init__(self, speech_ticks: list[bool]) -> None:
        self.speech_ticks = speech_ticks
        self.sent: list[bytes | None] = []
        self.received_ticks: list[int] = []

    async def start(self) -> None:
        pass

    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None) -> TickResult:
        self.sent.append(outgoing_audio)
        self.received_ticks.append(tick_number)
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
    scheduler = _scheduler([True, False, True] + [False] * 30)  # greet, caller turn, quick reply, then silence
    await scheduler.run_tick()  # tick 0: assistant greets
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)

    await scheduler.run_tick()  # tick 1: caller speaks this tick
    assert scheduler.may_take_turn() is False

    await scheduler.run_tick()  # tick 2: assistant replies, clearing the awaiting-reply gate
    assert scheduler.may_take_turn() is False

    # Assistant silence is satisfied almost immediately, self-silence needs 25 ticks.
    for _ in range(24):
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


async def test_caller_cannot_take_a_second_turn_while_awaiting_a_reply():
    scheduler = _scheduler([True])  # assistant greets on tick 0, then never speaks again
    await scheduler.run_tick()
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)
    await scheduler.run_tick()  # caller takes its turn

    for _ in range(100):
        await scheduler.run_tick()
        assert scheduler.may_take_turn() is False


async def test_caller_stops_waiting_once_the_assistant_goes_unresponsive():
    # The assistant that never answers a goodbye would otherwise hold the caller
    # in awaiting-reply forever, so it could never reach its end_call turn.
    scheduler = _scheduler([True])
    await scheduler.run_tick()
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)
    await scheduler.run_tick()

    for _ in range(ms_to_ticks(ASSISTANT_UNRESPONSIVE_MS) - 5):
        await scheduler.run_tick()
    assert scheduler.may_take_turn() is False

    for _ in range(10):
        await scheduler.run_tick()
    assert scheduler.may_take_turn() is True


async def test_unresponsive_threshold_clears_the_longest_observed_real_gap():
    # Longest legitimate assistant gap measured across live runs was 220 ticks;
    # firing inside that would make the caller talk over a merely slow assistant.
    assert ms_to_ticks(ASSISTANT_UNRESPONSIVE_MS) > 220


async def test_caller_may_take_a_second_turn_once_the_assistant_replies():
    scheduler = _scheduler([True, False, True] + [False] * 30)
    await scheduler.run_tick()  # tick 0: assistant greets
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)
    await scheduler.run_tick()  # tick 1: caller takes its turn
    await scheduler.run_tick()  # tick 2: assistant replies

    for _ in range(25):
        await scheduler.run_tick()

    assert scheduler.may_take_turn() is True


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


async def test_simultaneous_speech_resets_both_counters_and_blocks_the_turn():
    scheduler = _scheduler([True])
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)

    await scheduler.run_tick()  # both sides speak on tick 0

    assert scheduler.may_take_turn() is False


async def test_tick_number_reaches_the_adapter_and_increments():
    adapter = FakeAdapter([])
    scheduler = TickScheduler(adapter, bytes_per_tick=BYTES_PER_TICK)

    for _ in range(3):
        await scheduler.run_tick()

    assert adapter.received_ticks == [0, 1, 2]
    assert scheduler.tick == 3


async def test_consecutive_enqueues_drain_contiguously_with_no_silence_gap():
    adapter = FakeAdapter([])
    scheduler = TickScheduler(adapter, bytes_per_tick=BYTES_PER_TICK)
    scheduler.enqueue_utterance(b"\x02" * BYTES_PER_TICK)
    scheduler.enqueue_utterance(b"\x03" * BYTES_PER_TICK)

    await scheduler.run_tick()
    await scheduler.run_tick()

    assert adapter.sent[0] == b"\x02" * BYTES_PER_TICK
    assert adapter.sent[1] == b"\x03" * BYTES_PER_TICK


class RaisingAdapter(Adapter):
    """Raises on a chosen tick to exercise the peek-then-commit failure path."""

    def __init__(self, fail_on_tick: int) -> None:
        self.fail_on_tick = fail_on_tick

    async def start(self) -> None:
        pass

    async def run_tick(self, tick_number: int, outgoing_audio: bytes | None) -> TickResult:
        if tick_number == self.fail_on_tick:
            raise RuntimeError("adapter failure")
        return TickResult(
            tick_number=tick_number,
            assistant_audio=b"\x00" * BYTES_PER_TICK,
            assistant_audio_raw_bytes=0,
            wall_clock_ms=tick_number,
        )

    async def stop(self) -> None:
        pass


async def test_failed_adapter_call_leaves_queue_and_tick_unadvanced():
    adapter = RaisingAdapter(fail_on_tick=0)
    scheduler = TickScheduler(adapter, bytes_per_tick=BYTES_PER_TICK)
    utterance = b"\x02" * BYTES_PER_TICK
    scheduler.enqueue_utterance(utterance)

    try:
        await scheduler.run_tick()
    except RuntimeError:
        pass

    assert scheduler.tick == 0
    assert bytes(scheduler._playout) == utterance
