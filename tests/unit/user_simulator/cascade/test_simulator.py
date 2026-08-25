from eva.user_simulator.cascade.phrases import load_phrases
from eva.user_simulator.cascade.simulator import CascadeUserSimulator, extract_turn, parse_turn_response


def _trace_sink():
    """DecisionLog that accumulates in memory and never writes."""
    from pathlib import Path

    from eva.user_simulator.cascade.decision_log import DecisionLog

    return DecisionLog(Path("unused-decision-trace.jsonl"))


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
    sim._decision_log = _trace_sink()
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
    sim._decision_log = _trace_sink()
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
    sim._decision_log = _trace_sink()
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
    sim._decision_log = _trace_sink()
    sim._ticks_assistant_silent = 0
    silent = _tick(False)
    for _ in range(ms_to_ticks(INACTIVITY_TIMEOUT_MS)):
        assert sim._assistant_is_inactive(_SilenceScheduler(), silent) is False

    assert sim._assistant_is_inactive(_SilenceScheduler(), silent) is True


def test_assistant_speech_resets_the_inactivity_counter():
    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._decision_log = _trace_sink()
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
    sim._decision_log = _trace_sink()
    sim._ticks_assistant_silent = 0
    for _ in range(ms_to_ticks(INACTIVITY_TIMEOUT_MS) + 5):
        assert sim._assistant_is_inactive(_NeverSpoke(), _tick(False)) is False


def test_tick_counters_exist_without_manual_setup():
    # The unit tests build bare instances, so a counter initialised only inside __init__
    # would still pass them and then AttributeError on the first live tick.
    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._decision_log = _trace_sink()

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


def test_a_barge_in_is_kept_while_the_assistant_is_mid_utterance():
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(assistant_still_speaking=True, same_assistant_turn=True) is False


def test_a_slow_barge_in_is_no_longer_dropped_for_being_slow():
    # Wall-clock slip is not staleness: the assistant is still talking, so the line still lands.
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(assistant_still_speaking=True, same_assistant_turn=True) is False


def test_a_barge_in_is_dropped_once_the_assistant_stopped():
    # No longer an interruption — it would land as an ordinary reply.
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(assistant_still_speaking=False, same_assistant_turn=True) is True


def test_a_barge_in_is_dropped_when_the_assistant_moved_to_a_later_turn():
    # "Still speaking" is satisfied by a *different* turn, which would land the line as a
    # non-sequitur against speech the caller never reacted to.
    from eva.user_simulator.cascade.simulator import should_drop_interrupt

    assert should_drop_interrupt(assistant_still_speaking=True, same_assistant_turn=False) is True


def test_extract_optional_line_reads_a_plain_line():
    from eva.user_simulator.cascade.simulator import extract_optional_line

    assert extract_optional_line("I wanted Friday, not Thursday.") == "I wanted Friday, not Thursday."


def test_extract_optional_line_strips_a_code_fence():
    from eva.user_simulator.cascade.simulator import extract_optional_line

    assert extract_optional_line("```\nI meant Friday.\n```") == "I meant Friday."


def test_extract_optional_line_is_empty_for_an_empty_reply():
    from eva.user_simulator.cascade.simulator import extract_optional_line

    assert extract_optional_line("") == ""


def test_extract_optional_line_reads_through_a_message_object():
    from eva.user_simulator.cascade.simulator import extract_optional_line

    class _Message:
        content = "I meant Friday."

    assert extract_optional_line(_Message()) == "I meant Friday."


def test_extract_optional_line_rejects_a_refusal_style_non_answer():
    # The prompt allows the model to decline by returning NONE.
    from eva.user_simulator.cascade.simulator import extract_optional_line

    assert extract_optional_line("NONE") == ""


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


def _interrupting_simulator(message, framework="elevenlabs"):
    """Bare simulator wired for _play_interruption only."""
    from eva.models.config import CascadeSimulatorConfig
    from eva.user_simulator.cascade.stt import TranscriptBuffer

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._decision_log = _trace_sink()
    sim._framework = framework
    sim._config = CascadeSimulatorConfig(enable_interruptions=True)
    sim._phrases = load_phrases("en")
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
        self.barge_in_armed = False

    def arm_barge_in(self) -> None:
        self.barge_in_armed = True

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


async def test_only_one_interruption_fires_per_assistant_turn():
    # The eligibility flag is cleared on firing, so a second check in the same
    # assistant turn is offered allow_interrupt=False and cannot barge in again.
    from eva.models.config import CascadeSimulatorConfig
    from eva.user_simulator.cascade.decisions import ListenerVerdict

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._decision_log = _trace_sink()
    sim._config = CascadeSimulatorConfig(enable_interruptions=True)
    sim._phrase_cache = StubCache()
    sim._may_interrupt_this_turn = True
    sim.plays = 0
    offered = []

    class _Decisions:
        async def evaluate(self, text, *, allow_interrupt, allow_backchannel):
            offered.append(allow_interrupt)
            return ListenerVerdict(should_interrupt=allow_interrupt, should_backchannel=False)

    async def _play(scheduler):
        sim.plays += 1
        return False

    sim._decisions = _Decisions()
    sim._play_interruption = _play
    # The transcript has to grow between the two checks, or the unchanged-transcript
    # gate skips the second one before eligibility is ever consulted.
    from eva.user_simulator.cascade.stt import TranscriptBuffer

    buffer = TranscriptBuffer()
    sim._stt = type("_Stt", (), {"buffer": buffer})()

    buffer.committed = "Let me pull up"
    await sim._run_checks(_InterruptScheduler())
    buffer.committed = "Let me pull up your account."
    await sim._run_checks(_InterruptScheduler())

    assert offered == [True, False]
    assert sim.plays == 1


def test_a_short_pause_inside_one_assistant_turn_is_not_a_new_turn():
    # A single quiet tick used to re-arm the interruption cap mid-utterance.
    from eva.user_simulator.cascade.simulator import is_new_assistant_turn

    assert is_new_assistant_turn(ticks_silent_before=1) is False
    assert is_new_assistant_turn(ticks_silent_before=4) is False


def test_a_sustained_gap_starts_a_new_assistant_turn():
    from eva.user_simulator.cascade.constants import WAIT_TO_RESPOND_OTHER_MS, ms_to_ticks
    from eva.user_simulator.cascade.simulator import is_new_assistant_turn

    assert is_new_assistant_turn(ticks_silent_before=ms_to_ticks(WAIT_TO_RESPOND_OTHER_MS)) is True


def test_inactivity_measures_contiguous_silence_not_cumulative():
    # The reset was unreachable in the live loop, so scattered quiet ticks accumulated
    # and killed healthy calls once they happened to total two minutes.
    from eva.user_simulator.cascade.constants import INACTIVITY_TIMEOUT_MS, ms_to_ticks

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._decision_log = _trace_sink()
    sim._ticks_assistant_silent = 0
    limit = ms_to_ticks(INACTIVITY_TIMEOUT_MS)

    # Almost time out, then the assistant speaks once, then go quiet again.
    for _ in range(limit):
        assert sim._assistant_is_inactive(_SilenceScheduler(), _tick(False)) is False
    assert sim._assistant_is_inactive(_SilenceScheduler(), _tick(True)) is False
    for _ in range(limit):
        assert sim._assistant_is_inactive(_SilenceScheduler(), _tick(False)) is False

    assert sim._assistant_is_inactive(_SilenceScheduler(), _tick(False)) is True


async def test_a_dropped_interruption_emits_no_audio_at_all():
    # Emitting the opener up front meant a stale drop left it orphaned on the wire.
    sim = _interrupting_simulator("Active Directory.")

    class _StaleScheduler(_InterruptScheduler):
        assistant_is_speaking = False  # forces should_drop_interrupt

    scheduler = _StaleScheduler()
    assert await sim._play_interruption(scheduler) is False
    assert scheduler.queued == []


async def test_a_hangup_during_an_interruption_emits_no_opener():
    sim = _interrupting_simulator(_EndCallMessage())
    scheduler = _InterruptScheduler()

    assert await sim._play_interruption(scheduler) is True
    assert scheduler.queued == []
    assert sim.ended == ["goodbye"]


async def test_a_kept_interruption_speaks_the_opener_then_the_content():
    sim = _interrupting_simulator("Active Directory.")
    scheduler = _InterruptScheduler()

    assert await sim._play_interruption(scheduler) is False

    assert scheduler.queued[0] == b"CACHED"
    assert b"Active Directory." in b"".join(scheduler.queued[1:])
    # A tick-driven transport needs to know this audio cuts the assistant off.
    assert scheduler.barge_in_armed is True


def test_openai_realtime_uses_the_tick_driven_adapter():
    from eva.user_simulator.cascade.adapter.tick_driven import TickDrivenAdapter
    from eva.user_simulator.cascade.simulator import adapter_class_for_framework

    assert adapter_class_for_framework("openai_realtime") is TickDrivenAdapter


def test_pipecat_stays_on_the_real_time_adapter():
    # Pipecat owns its own clock; freezing it is not possible.
    from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
    from eva.user_simulator.cascade.simulator import adapter_class_for_framework

    assert adapter_class_for_framework("pipecat") is RealtimeWSAdapter


def test_elevenlabs_stays_on_the_real_time_adapter():
    from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
    from eva.user_simulator.cascade.simulator import adapter_class_for_framework

    assert adapter_class_for_framework("elevenlabs") is RealtimeWSAdapter


def test_unported_frameworks_default_to_the_real_time_adapter():
    from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
    from eva.user_simulator.cascade.simulator import adapter_class_for_framework

    assert adapter_class_for_framework("gemini_live") is RealtimeWSAdapter


async def test_the_incomplete_marker_never_reaches_the_conversation_history():
    sim = _interrupting_simulator("Active Directory.")
    sim._stt.buffer.apply_partial("and I still nee")

    await sim._play_interruption(_InterruptScheduler())

    heard = [m["content"] for m in sim._history if m["role"] == "assistant"]
    assert heard == ["Your account is unlocked. and I still nee"]


async def test_a_slow_barge_in_survives_on_every_transport():
    # Slip is now reported, not enforced: the assistant is mid-utterance, so the line lands.
    from eva.user_simulator.cascade import simulator as module

    for framework in ("openai_realtime", "pipecat"):
        sim = _interrupting_simulator("Active Directory.", framework=framework)
        scheduler = _InterruptScheduler()

        original = module.interrupt_slip_ms
        module.interrupt_slip_ms = lambda *, elapsed_s: 2400
        try:
            assert await sim._play_interruption(scheduler) is False
        finally:
            module.interrupt_slip_ms = original

        assert scheduler.queued != [], framework


async def test_a_barge_in_is_abandoned_when_a_new_assistant_turn_started_meanwhile():
    # The assistant finished the utterance we reacted to and began another one while the
    # line was being generated; firing now would answer speech the caller never heard.
    sim = _interrupting_simulator("Active Directory.", framework="pipecat")
    scheduler = _InterruptScheduler()

    original_complete = sim._llm.complete

    async def _complete_then_new_turn(messages, tools=None):
        sim._assistant_turn_index += 1
        return await original_complete(messages, tools)

    sim._llm.complete = _complete_then_new_turn

    assert await sim._play_interruption(scheduler) is False
    assert scheduler.queued == []


def test_provider_stall_is_a_distinct_terminal_reason_from_inactivity():
    # inactivity_timeout is a legitimate end the metrics treat as definitive when the
    # user spoke last; a stalled peer is an invalid record the runner should retry. The
    # two must not share a reason, or a dead provider scores as a finished conversation.
    from eva.user_simulator.cascade.tick_result import TickResult

    stalled = TickResult(
        tick_number=9,
        assistant_audio=b"\x00" * 8,
        assistant_audio_raw_bytes=0,
        wall_clock_ms=0,
        provider_stalled=True,
    )

    assert stalled.provider_stalled is True
    assert (
        TickResult(tick_number=9, assistant_audio=b"", assistant_audio_raw_bytes=0, wall_clock_ms=0).provider_stalled
        is False
    )


class _CountingDecisions:
    """ListenerDecisions stand-in that records every evaluate() it is asked to run."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def evaluate(self, heard, *, allow_interrupt, allow_backchannel):
        from eva.user_simulator.cascade.decisions import ListenerVerdict

        self.calls.append(heard)
        self.last_allow_backchannel = allow_backchannel
        return ListenerVerdict(should_interrupt=False, should_backchannel=False)


def _checking_simulator():
    from eva.models.config import CascadeSimulatorConfig
    from eva.user_simulator.cascade.stt import TranscriptBuffer

    sim = CascadeUserSimulator.__new__(CascadeUserSimulator)
    sim._decision_log = _trace_sink()
    sim._config = CascadeSimulatorConfig(enable_interruptions=True, enable_backchannel=True)
    sim._phrases = load_phrases("en")
    sim._decisions = _CountingDecisions()
    sim._phrase_cache = StubCache()
    sim._may_interrupt_this_turn = True
    sim._last_checked_text = ""
    sim.event_logger = _FakeEventLogger()
    sim._record_audio = lambda *a, **k: None
    buffer = TranscriptBuffer()
    sim._stt = type("_Stt", (), {"buffer": buffer})()
    return sim, buffer


async def test_an_unchanged_transcript_does_not_re_ask_the_judges():
    # The checks fire on a timer but read only the transcript, so a tick where it has
    # not moved re-asks a question already answered — ~101 checks x 2 calls per call.
    sim, buffer = _checking_simulator()
    buffer.committed = "Let me pull up your account."

    await sim._run_checks(_FakeScheduler())
    await sim._run_checks(_FakeScheduler())
    await sim._run_checks(_FakeScheduler())

    assert sim._decisions.calls == ["Let me pull up your account."]


async def test_a_grown_transcript_is_judged_again():
    sim, buffer = _checking_simulator()
    buffer.committed = "Let me pull up"

    await sim._run_checks(_FakeScheduler())
    buffer.committed = "Let me pull up your account."
    await sim._run_checks(_FakeScheduler())

    assert len(sim._decisions.calls) == 2


async def test_an_empty_transcript_is_never_judged():
    sim, _buffer = _checking_simulator()

    await sim._run_checks(_FakeScheduler())

    assert sim._decisions.calls == []


async def test_backchannelling_survives_a_barge_in_in_the_same_turn():
    # Interrupting and later humming along are not mutually exclusive for a real
    # listener; the per-turn cap is about not talking over the assistant twice, not
    # about going silent for the rest of the turn.
    sim, buffer = _checking_simulator()
    sim._may_interrupt_this_turn = False  # this turn has already barged in
    buffer.committed = "Let me pull up your account."

    await sim._run_checks(_FakeScheduler())

    assert sim._decisions.last_allow_backchannel is True


async def test_backchannelling_is_off_only_when_the_config_disables_it():
    from eva.models.config import CascadeSimulatorConfig

    sim, buffer = _checking_simulator()
    sim._config = CascadeSimulatorConfig(enable_interruptions=True, enable_backchannel=False)
    buffer.committed = "Let me pull up your account."

    await sim._run_checks(_FakeScheduler())

    assert sim._decisions.last_allow_backchannel is False


async def test_a_transcript_already_judged_in_an_earlier_turn_is_judged_again():
    # take_committed() runs between turns, so current_text() can return a string this
    # gate has already seen. Without clearing on a turn boundary that later turn's
    # check would be skipped as "unchanged" and the barge-in never considered.
    sim, buffer = _checking_simulator()
    buffer.committed = "Anything else I can help with?"
    await sim._run_checks(_FakeScheduler())
    assert len(sim._decisions.calls) == 1

    # The ordinary turn path consumes the transcript, then the assistant says the same
    # thing again a turn later — a real pattern for closing questions.
    buffer.take_committed()
    sim._last_checked_text = ""  # what the new-turn branch in _run does
    buffer.committed = "Anything else I can help with?"

    await sim._run_checks(_FakeScheduler())

    assert len(sim._decisions.calls) == 2


async def test_a_skipped_check_writes_no_listener_check_row(tmp_path):
    # The trace exists to separate "the model said NO" from "the check never ran". A row
    # for a check that was gated out before reaching the judges would blur exactly that.
    from eva.user_simulator.cascade.decision_log import DecisionLog

    sim, buffer = _checking_simulator()
    sim._decision_log = DecisionLog(tmp_path / "trace.jsonl")
    buffer.committed = "Let me pull up your account."

    await sim._run_checks(_FakeScheduler())
    await sim._run_checks(_FakeScheduler())
    await sim._run_checks(_FakeScheduler())

    assert sim._decisions.calls == ["Let me pull up your account."]
    assert sim._decision_log._counts.get("listener_check") == 1


async def test_the_tick_trace_records_whether_the_transcript_moved(tmp_path):
    # A check tick with no listener_check row is otherwise ambiguous between "nothing
    # new was heard" and "the check ran and failed".
    from eva.user_simulator.cascade.decision_log import DecisionLog

    sim, buffer = _checking_simulator()
    rows = []
    sim._decision_log = DecisionLog(tmp_path / "trace.jsonl")
    sim._decision_log.log = lambda kind, **fields: rows.append((kind, fields))
    sim._ticks_assistant_silent = 0
    sim._ticks_since_assistant_started = 3
    sim._candidate_audio = b""
    buffer.committed = "Let me pull up your account."

    sim._log_tick_state(_TickTraceScheduler(), _tick(True), is_check_tick=True)
    sim._last_checked_text = buffer.current_text()
    sim._log_tick_state(_TickTraceScheduler(), _tick(True), is_check_tick=True)

    assert rows[0][1]["transcript_moved"] is True
    assert rows[1][1]["transcript_moved"] is False


class _TickTraceScheduler:
    tick = 12
    caller_is_speaking = False
    caller_spoke_this_tick = False
