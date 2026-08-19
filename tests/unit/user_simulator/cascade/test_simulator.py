from eva.user_simulator.cascade.simulator import CascadeUserSimulator, extract_turn, parse_turn_response


def test_parse_turn_response_returns_the_spoken_line_unchanged():
    assert parse_turn_response("I need to reset my password.") == "I need to reset my password."


def test_parse_turn_response_returns_empty_for_a_toolcall_only_turn():
    # The model hangs up by calling end_call and says nothing; content is empty.
    assert parse_turn_response("") == ""


def test_extract_turn_reads_a_plain_string_as_no_hangup():
    # LiteLLMClient returns a bare str when the model made no tool call.
    assert extract_turn("Still here.") == ("Still here.", False)


def test_extract_turn_detects_the_end_call_tool():
    class _Fn:
        name = "end_call"

    class _Call:
        function = _Fn()

    class _Message:
        content = ""
        tool_calls = [_Call()]

    assert extract_turn(_Message()) == ("", True)


def test_extract_turn_ignores_an_unrelated_tool_call():
    class _Fn:
        name = "something_else"

    class _Call:
        function = _Fn()

    class _Message:
        content = "Go on."
        tool_calls = [_Call()]

    assert extract_turn(_Message()) == ("Go on.", False)


def test_outbound_perturbation_reaches_the_adapter():
    # background_noise / connection_degradation are applied per tick by RealtimeWSAdapter,
    # so the cascade simulator no longer warns that it drops them.
    assert not hasattr(CascadeUserSimulator, "_warn_unsupported_perturbation")


def _make_bare_simulator() -> CascadeUserSimulator:
    """Build a CascadeUserSimulator without running __init__, for pure _messages() testing."""
    sim = object.__new__(CascadeUserSimulator)
    sim._build_prompt = lambda: "SYSTEM PROMPT"
    sim._history = []
    return sim


def test_messages_flips_roles_so_the_user_simulator_llm_sees_its_own_lines_as_assistant():
    sim = _make_bare_simulator()
    sim._history = [
        {"role": "assistant", "content": "What is your email?"},
        {"role": "user", "content": "It's jane@example.com."},
    ]

    messages = sim._messages()

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What is your email?"}
    assert messages[2] == {"role": "assistant", "content": "It's jane@example.com."}


class _FakeEventLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, name, data):
        self.events.append((name, data))


class _FakeScheduler:
    tick = 7


def _simulator_with_buffer(committed: str = "", in_flight: str = ""):
    """Build a bare simulator with just the attributes _collect_heard_text touches."""
    from eva.user_simulator.cascade.stt import TranscriptBuffer

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    buffer = TranscriptBuffer()
    buffer.committed, buffer.in_flight = committed, in_flight
    sim._stt = type("_Stt", (), {"buffer": buffer})()
    sim._ticks_awaiting_transcript = 0
    sim._missed_transcripts = 0
    sim.event_logger = _FakeEventLogger()
    return sim


def test_committed_transcript_is_taken_immediately():
    sim = _simulator_with_buffer(committed="I can help with that.")

    assert sim._collect_heard_text(_FakeScheduler()) == ("I can help with that.", False)


def test_empty_buffer_waits_rather_than_generating_a_turn():
    # Finalization is not instant; the first empty read means "not ready yet".
    sim = _simulator_with_buffer()

    assert sim._collect_heard_text(_FakeScheduler()) == ("", True)


def test_wait_expires_into_the_in_flight_partial():
    from eva.user_simulator.cascade.constants import TRANSCRIPT_WAIT_MS, ms_to_ticks

    sim = _simulator_with_buffer(in_flight="Please confirm your username")
    for _ in range(ms_to_ticks(TRANSCRIPT_WAIT_MS)):
        assert sim._collect_heard_text(_FakeScheduler()) == ("", True)

    assert sim._collect_heard_text(_FakeScheduler()) == ("Please confirm your username", False)
    assert sim._stt.buffer.in_flight == ""


def test_the_wait_counter_resets_after_a_successful_read():
    sim = _simulator_with_buffer()
    for _ in range(3):
        sim._collect_heard_text(_FakeScheduler())
    sim._stt.buffer.committed = "Thanks, Marcus."

    sim._collect_heard_text(_FakeScheduler())

    assert sim._ticks_awaiting_transcript == 0


def _boundary_simulator():
    """Bare simulator exposing only what _log_audio_boundaries touches."""
    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim.event_logger = _FakeAudioEventLogger()
    return sim


class _FakeAudioEventLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def log_audio_start(self, role, timestamp=None):
        self.calls.append(("audio_start", role, timestamp))

    def log_audio_end(self, role, timestamp=None):
        self.calls.append(("audio_end", role, timestamp))


class _Sched:
    def __init__(self, spoke: bool) -> None:
        self.caller_spoke_this_tick = spoke


def _tick(assistant_speech: bool, ms: int = 2000):
    from eva.user_simulator.cascade.tick_result import TickResult

    return TickResult(
        tick_number=0,
        assistant_audio=b"\x00" * 8,
        assistant_audio_raw_bytes=8 if assistant_speech else 0,
        wall_clock_ms=ms,
    )


def test_caller_audio_start_is_logged_on_the_first_tick_of_playout():
    sim = _boundary_simulator()

    sim._log_audio_boundaries(_Sched(True), _tick(False), False, False)

    assert sim.event_logger.calls == [("audio_start", "simulated_user", 2.0)]


def test_caller_audio_end_is_logged_when_playout_stops():
    sim = _boundary_simulator()

    sim._log_audio_boundaries(_Sched(False), _tick(False), False, True)

    assert sim.event_logger.calls == [("audio_end", "simulated_user", 2.0)]


def test_no_event_while_the_caller_keeps_speaking():
    sim = _boundary_simulator()

    sim._log_audio_boundaries(_Sched(True), _tick(False), False, True)

    assert sim.event_logger.calls == []


def test_assistant_boundaries_are_logged_too():
    # The metrics processor expects both roles, not just the user.
    sim = _boundary_simulator()

    sim._log_audio_boundaries(_Sched(False), _tick(True), False, False)
    sim._log_audio_boundaries(_Sched(False), _tick(False), True, False)

    assert sim.event_logger.calls == [
        ("audio_start", "assistant", 2.0),
        ("audio_end", "assistant", 2.0),
    ]


def test_timestamp_is_unix_seconds_not_milliseconds():
    # log_audio_* store the value as audio_timestamp, which metrics read as seconds.
    sim = _boundary_simulator()

    sim._log_audio_boundaries(_Sched(True), _tick(False, ms=1786127928923), False, False)

    assert sim.event_logger.calls[0][2] == 1786127928.923


def test_hearing_nothing_keeps_waiting_instead_of_speaking_into_the_void():
    # An assistant that never replies is an inactivity timeout, not a cue to talk again.
    from eva.user_simulator.cascade.constants import TRANSCRIPT_WAIT_MS, ms_to_ticks

    sim = _simulator_with_buffer()
    for _ in range(ms_to_ticks(TRANSCRIPT_WAIT_MS) + 3):
        assert sim._collect_heard_text(_FakeScheduler()) == ("", True)


class _SilenceScheduler:
    tick = 0
    assistant_has_spoken = True


def test_inactivity_ends_the_call_after_the_shared_two_minute_threshold():
    from eva.user_simulator.cascade.constants import INACTIVITY_TIMEOUT_MS, ms_to_ticks

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._ticks_assistant_silent = 0
    silent = _tick(False)
    for _ in range(ms_to_ticks(INACTIVITY_TIMEOUT_MS)):
        assert sim._assistant_is_inactive(_SilenceScheduler(), silent) is False

    assert sim._assistant_is_inactive(_SilenceScheduler(), silent) is True


def test_assistant_speech_resets_the_inactivity_counter():
    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._ticks_assistant_silent = 500

    assert sim._assistant_is_inactive(_SilenceScheduler(), _tick(True)) is False
    assert sim._ticks_assistant_silent == 0


def test_inactivity_does_not_fire_before_the_assistant_ever_speaks():
    # The assistant opens the call; waiting for its greeting is not inactivity.
    from eva.user_simulator.cascade.constants import INACTIVITY_TIMEOUT_MS, ms_to_ticks

    class _NeverSpoke:
        tick = 0
        assistant_has_spoken = False

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._ticks_assistant_silent = 0
    for _ in range(ms_to_ticks(INACTIVITY_TIMEOUT_MS) + 5):
        assert sim._assistant_is_inactive(_NeverSpoke(), _tick(False)) is False


def test_tick_counters_exist_without_manual_setup():
    # The unit tests build bare instances, so a counter initialised only inside __init__
    # would still pass them and then AttributeError on the first live tick.
    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)

    assert sim._ticks_assistant_silent == 0
    assert sim._ticks_awaiting_transcript == 0


class RecordingScheduler:
    """Captures what the simulator queues for playout."""

    def __init__(self) -> None:
        self.queued: list[bytes] = []
        self.backchannels: list[bytes] = []
        self.tick = 0

    def enqueue_utterance(self, audio: bytes) -> None:
        self.queued.append(audio)

    def enqueue_backchannel(self, audio: bytes) -> None:
        self.backchannels.append(audio)


class StubCache:
    """Phrase cache stand-in that always picks the first phrase."""

    def choose(self, phrases):
        return phrases[0]

    def get(self, phrase):
        return b"CACHED"


async def test_backchannel_queues_cached_audio_without_synthesis():
    from eva.user_simulator.cascade import simulator as module

    scheduler = RecordingScheduler()
    played = module.play_backchannel(scheduler, StubCache(), ["uh-huh", "mm-hmm"])

    assert played == "uh-huh"
    # Queued as a backchannel so it does not consume the caller's turn.
    assert scheduler.backchannels == [b"CACHED"]
    assert scheduler.queued == []


def test_verdict_with_no_action_queues_nothing():
    from eva.user_simulator.cascade.decisions import ListenerVerdict

    verdict = ListenerVerdict(should_interrupt=False, should_backchannel=False)

    assert verdict.should_interrupt is False
    assert verdict.should_backchannel is False


def test_interrupt_kept_when_slip_is_within_budget():
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(slip_ms=800, assistant_still_speaking=True) is False


def test_interrupt_dropped_when_slip_exceeds_budget():
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(slip_ms=2000, assistant_still_speaking=True) is True


def test_interrupt_dropped_when_the_assistant_already_stopped():
    # No longer an interruption — it would land as an ordinary reply.
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(slip_ms=100, assistant_still_speaking=False) is True


def test_correction_fires_once_the_delay_has_elapsed():
    from eva.user_simulator.cascade.simulator import should_fire_self_correction

    assert should_fire_self_correction(ticks_since_assistant_started=6, assistant_speaking=True) is True


def test_correction_does_not_fire_before_the_delay():
    from eva.user_simulator.cascade.simulator import should_fire_self_correction

    assert should_fire_self_correction(ticks_since_assistant_started=2, assistant_speaking=True) is False


def test_correction_is_abandoned_if_the_assistant_never_replied():
    from eva.user_simulator.cascade.simulator import should_fire_self_correction

    assert should_fire_self_correction(ticks_since_assistant_started=6, assistant_speaking=False) is False


def test_extract_correction_reads_a_plain_line():
    from eva.user_simulator.cascade.simulator import extract_correction

    assert extract_correction("Actually, wait — I said Thursday, I meant Friday.") == (
        "Actually, wait — I said Thursday, I meant Friday."
    )


def test_extract_correction_strips_a_code_fence():
    from eva.user_simulator.cascade.simulator import extract_correction

    assert extract_correction("```\nI meant Friday.\n```") == "I meant Friday."


def test_extract_correction_is_empty_for_an_empty_reply():
    from eva.user_simulator.cascade.simulator import extract_correction

    assert extract_correction("") == ""


def test_extract_correction_reads_through_a_message_object():
    from eva.user_simulator.cascade.simulator import extract_correction

    class _Message:
        content = "I meant Friday."
        tool_calls: list = []

    assert extract_correction(_Message()) == "I meant Friday."


def test_extract_correction_rejects_a_refusal_style_non_answer():
    # The prompt allows the model to decline by returning NONE.
    from eva.user_simulator.cascade.simulator import extract_correction

    assert extract_correction("NONE") == ""


def _correcting_simulator(reply: str, *, rate_roll: float = 0.0):
    """Bare simulator wired for _maybe_arm_self_correction only."""
    from eva.models.config import CascadeSimulatorConfig

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._config = CascadeSimulatorConfig(enable_self_correction=True)
    sim._rng = type("_Rng", (), {"random": staticmethod(lambda: rate_roll)})()
    sim._build_prompt = lambda: "SYSTEM PROMPT"
    sim._history = []
    sim._voice_id = "voice-f"
    sim.event_logger = _FakeEventLogger()
    sim._armed_correction = b""
    sim._armed_correction_text = ""

    class _Llm:
        async def complete(self, messages, tools=None):
            return reply, {}

    class _Tts:
        async def synthesize(self, text, *, voice_id):
            return text.encode()

    sim._llm, sim._tts = _Llm(), _Tts()
    return sim


async def test_the_slip_is_spoken_and_the_original_line_is_armed_as_the_correction():
    # Wrong-then-right: the goal-consistent line must be what lands last.
    sim = _correcting_simulator("Book me Thursday.")

    spoken = await sim._maybe_arm_self_correction("Book me Friday.")

    assert spoken == "Book me Thursday."
    assert sim._armed_correction_text == "Book me Friday."


async def test_no_correction_is_armed_when_the_rate_gate_declines():
    sim = _correcting_simulator("Book me Thursday.", rate_roll=0.99)

    spoken = await sim._maybe_arm_self_correction("Book me Friday.")

    assert spoken == "Book me Friday."
    assert sim._armed_correction == b""


async def test_a_none_reply_leaves_the_turn_unchanged():
    sim = _correcting_simulator("NONE")

    spoken = await sim._maybe_arm_self_correction("Thanks, goodbye.")

    assert spoken == "Thanks, goodbye."
    assert sim._armed_correction == b""


async def test_a_failed_correction_call_degrades_to_an_ordinary_turn():
    sim = _correcting_simulator("unused")

    class _Failing:
        async def complete(self, messages, tools=None):
            raise RuntimeError("provider down")

    sim._llm = _Failing()

    assert await sim._maybe_arm_self_correction("Book me Friday.") == "Book me Friday."


async def test_self_correction_is_skipped_when_the_behavior_is_disabled():
    from eva.models.config import CascadeSimulatorConfig

    sim = _correcting_simulator("Book me Thursday.")
    sim._config = CascadeSimulatorConfig()

    assert await sim._maybe_arm_self_correction("Book me Friday.") == "Book me Friday."


async def test_relevance_gate_allows_a_still_relevant_candidate():
    from eva.user_simulator.cascade.simulator import candidate_is_relevant
    from tests.unit.user_simulator.cascade.test_decisions import FakeLLM

    llm = FakeLLM(["YES"])

    assert await candidate_is_relevant(llm, candidate="I wanted Friday.", heard="Booking Thursday now") is True


async def test_relevance_gate_rejects_a_stale_candidate():
    from eva.user_simulator.cascade.simulator import candidate_is_relevant
    from tests.unit.user_simulator.cascade.test_decisions import FakeLLM

    llm = FakeLLM(["NO"])

    assert await candidate_is_relevant(llm, candidate="I wanted Friday.", heard="What is your name?") is False


async def test_relevance_gate_fails_closed():
    from eva.user_simulator.cascade.simulator import candidate_is_relevant
    from tests.unit.user_simulator.cascade.test_decisions import FakeLLM

    llm = FakeLLM(error=RuntimeError("down"))

    assert await candidate_is_relevant(llm, candidate="x", heard="y") is False


def test_slip_is_measured_on_the_wall_clock_not_the_tick_counter():
    # The tick counter cannot advance during _play_interruption's await: run_tick is
    # only pumped by _run, so a tick-delta slip is structurally always zero.
    from eva.user_simulator.cascade.simulator import interrupt_slip_ms

    assert interrupt_slip_ms(elapsed_s=1.0) == 1000
    assert interrupt_slip_ms(elapsed_s=0.0) == 0
    assert interrupt_slip_ms(elapsed_s=2.4) == 2400


def test_slip_never_reports_negative_for_a_clock_hiccup():
    from eva.user_simulator.cascade.simulator import interrupt_slip_ms

    assert interrupt_slip_ms(elapsed_s=-0.5) == 0


def test_self_correction_rng_differs_per_conversation():
    # Seeding every conversation with 0 made the 15% gate unreachable: Random(0)
    # first drops below 0.15 on draw 26, and conversations run ~7 turns.
    from eva.user_simulator.cascade.simulator import correction_rng

    a = correction_rng("record-1")
    b = correction_rng("record-2")

    assert [a.random() for _ in range(5)] != [b.random() for _ in range(5)]


def test_self_correction_rng_is_reproducible_for_the_same_conversation():
    from eva.user_simulator.cascade.simulator import correction_rng

    first = [correction_rng("record-7").random() for _ in range(3)]
    again = [correction_rng("record-7").random() for _ in range(3)]

    assert first == again


def test_self_correction_gate_actually_opens_within_a_normal_conversation():
    # Across a realistic spread of records, the 15% rate must be reachable.
    from eva.user_simulator.cascade.constants import SELF_CORRECTION_RATE
    from eva.user_simulator.cascade.simulator import correction_rng

    turns_per_conversation = 7
    fired = 0
    for index in range(60):
        rng = correction_rng(f"record-{index}")
        if any(rng.random() < SELF_CORRECTION_RATE for _ in range(turns_per_conversation)):
            fired += 1

    assert fired > 20, f"only {fired}/60 conversations could ever self-correct"


class _EndCallMessage:
    """LLM reply that hangs up via the tool and says nothing."""

    content = ""

    class _Fn:
        name = "end_call"

    class _Call:
        function = None

    def __init__(self) -> None:
        call = self._Call()
        call.function = self._Fn()
        self.tool_calls = [call]


def _interrupting_simulator(message):
    """Bare simulator wired for _play_interruption only."""
    from eva.models.config import CascadeSimulatorConfig
    from eva.user_simulator.cascade.stt import TranscriptBuffer

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._config = CascadeSimulatorConfig(enable_interruptions=True)
    sim._history = []
    sim._voice_id = "voice-f"
    sim._build_prompt = lambda: "SYSTEM PROMPT"
    sim.event_logger = _FakeEventLogger()
    sim._phrase_cache = StubCache()
    sim._record_audio = lambda *a, **k: None
    sim._on_user_speaks = lambda *a, **k: None
    sim._on_assistant_speaks = lambda *a, **k: None
    sim.ended = []
    sim._on_conversation_end = sim.ended.append
    buffer = TranscriptBuffer()
    buffer.committed = "Your account is unlocked."
    sim._stt = type("_Stt", (), {"buffer": buffer})()

    class _Llm:
        async def complete(self, messages, tools=None):
            return message, {}

    class _Tts:
        async def stream(self, text, *, voice_id):
            yield text.encode()

    sim._llm, sim._tts = _Llm(), _Tts()
    return sim


class _InterruptScheduler:
    tick = 40
    assistant_is_speaking = True

    def __init__(self) -> None:
        self.queued: list[bytes] = []

    def enqueue_utterance(self, audio: bytes) -> None:
        self.queued.append(audio)


async def test_a_hangup_during_an_interruption_ends_the_call():
    # Discarding end_call here stranded the caller: it emitted nothing and the
    # conversation ran on to the inactivity timeout, looping the assistant.
    sim = _interrupting_simulator(_EndCallMessage())

    hung_up = await sim._play_interruption(_InterruptScheduler())

    assert hung_up is True
    assert sim.ended == ["goodbye"]


async def test_a_hangup_is_never_dropped_as_stale():
    sim = _interrupting_simulator(_EndCallMessage())

    class _StoppedScheduler(_InterruptScheduler):
        assistant_is_speaking = False  # would normally drop the interruption

    assert await sim._play_interruption(_StoppedScheduler()) is True
    assert sim.ended == ["goodbye"]


async def test_an_ordinary_interruption_does_not_end_the_call():
    sim = _interrupting_simulator("It says my account is locked out.")

    assert await sim._play_interruption(_InterruptScheduler()) is False
    assert sim.ended == []
