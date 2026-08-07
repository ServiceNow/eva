import logging

from eva.models.config import PerturbationConfig
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


def test_warn_unsupported_perturbation_fires_for_background_noise(caplog):
    with caplog.at_level(logging.WARNING):
        CascadeUserSimulator._warn_unsupported_perturbation(PerturbationConfig(background_noise="road_noise"))
    assert any("background_noise" in record.message for record in caplog.records)


def test_warn_unsupported_perturbation_is_silent_for_a_default_config(caplog):
    with caplog.at_level(logging.WARNING):
        CascadeUserSimulator._warn_unsupported_perturbation(PerturbationConfig())
    assert caplog.records == []


def test_warn_unsupported_perturbation_is_silent_for_none(caplog):
    with caplog.at_level(logging.WARNING):
        CascadeUserSimulator._warn_unsupported_perturbation(None)
    assert caplog.records == []


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
        self.tick = 0

    def enqueue_utterance(self, audio: bytes) -> None:
        self.queued.append(audio)


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
    assert scheduler.queued == [b"CACHED"]


def test_verdict_with_no_action_queues_nothing():
    from eva.user_simulator.cascade.decisions import ListenerVerdict

    verdict = ListenerVerdict(should_interrupt=False, should_backchannel=False)

    assert verdict.should_interrupt is False
    assert verdict.should_backchannel is False
