"""Tests for the Krisp server's heard text, transcript placement and tool gateway."""

import asyncio

import httpx
import pytest

from eva.assistant import krisp_server
from eva.assistant.krisp_server import KrispAssistantServer, KrispCallInvalidError, _ToolGateway, heard_text
from eva.utils import krisp_agents


def _msg(text, **heard):
    message = {"content": [{"type": "text", "text": text}]}
    if heard:
        message["heard"] = heard
    return message


def test_uninterrupted_reply_is_whole():
    assert heard_text(_msg("All set.")) == "All set."


def test_interrupted_reply_is_cut_at_a_word_and_marked():
    assert heard_text(_msg("Your ticket is open now.", interrupted=True, chars=17)) == "Your ticket is [interrupted]"


def test_reply_interrupted_before_any_word_is_dropped():
    assert heard_text(_msg("Hello there.", interrupted=True, chars=0)) is None


def test_reply_interrupted_after_its_last_char_is_whole_and_marked():
    assert heard_text(_msg("Done.", interrupted=True, chars=5)) == "Done. [interrupted]"


def test_reply_cut_by_a_speech_error_is_cut_but_not_marked_as_a_barge_in():
    cut = _msg("Your ticket is open now.", interrupted=True, chars=17, cause="tts_error")
    assert heard_text(cut) == "Your ticket is"
    barge_in = _msg("Your ticket is open now.", interrupted=True, chars=17, cause="interruption")
    assert heard_text(barge_in) == "Your ticket is [interrupted]"


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


class _Audit:
    def __init__(self, tool_times=(), result_times=()):
        self.transcript = [{"message_type": "tool_call", "timestamp": str(t)} for t in tool_times]
        self.transcript += [{"message_type": "tool_response", "timestamp": str(t)} for t in result_times]
        self.calls = []

    def append_user_input(self, text, timestamp_ms):
        self.calls.append(("user", text, int(timestamp_ms)))

    def append_assistant_output(self, text, timestamp_ms):
        self.calls.append(("assistant", text, int(timestamp_ms)))


def _server(replies, caller_starts, tool_times=(), result_times=()):
    srv = object.__new__(KrispAssistantServer)
    srv._session_id = "s1"
    srv._session_start_ms = 1000
    srv._replies = replies
    srv._caller_starts = caller_starts
    srv.audit_log = _Audit(tool_times, result_times)
    srv._fw_log = _Recorder()
    return srv


def _entry(ts, user=None, agent=None):
    entry = {"timestamp": ts}
    if user:
        entry["user_message"] = {"content": [{"type": "text", "text": user}]}
    if agent:
        entry["agent_message"] = {"content": [{"type": "text", "text": agent}]}
    return entry


def test_transcript_puts_each_reply_at_its_audio_and_caller_words_at_speech_start():
    srv = _server(
        replies=[
            {"start_ms": 1200, "end_ms": 2500, "interrupted": False},
            {"start_ms": 6100, "end_ms": 8000, "interrupted": False},
        ],
        caller_starts=[3000],
        tool_times=[5600],
    )
    srv._write_transcript(
        [_entry(2600, agent="Hello! How can I help?"), _entry(6000, user="Reset my VPN.", agent="Done.")]
    )
    assert srv.audit_log.calls == [
        ("assistant", "Hello! How can I help?", 1200),
        ("user", "Reset my VPN.", 3000),
        ("assistant", "Done.", 6100),
    ]
    starts = [c for c in srv._fw_log.calls if c[0] == "turn_start"]
    responses = [c[1][0] for c in srv._fw_log.calls if c[0] == "llm_response"]
    assert len(starts) == 2 and responses == ["Hello! How can I help?", "Done."]


def test_caller_words_never_land_after_the_tool_they_triggered():
    srv = _server(replies=[], caller_starts=[], tool_times=[4000])
    srv._write_transcript([_entry(5000, user="Cancel it.", agent="Cancelled.")])
    assert srv.audit_log.calls[0] == ("user", "Cancel it.", 3999)


def test_a_caller_utterance_split_in_two_stays_before_the_reply_it_prompted():
    srv = _server(
        replies=[{"start_ms": 672858, "end_ms": 676817, "interrupted": False}],
        caller_starts=[665108, 686081],
    )
    srv._write_transcript(
        [
            _entry(666650, user="I already did a reboot earlier."),
            _entry(672889, user="It still connects, then disconnects.", agent="Try a different network."),
        ]
    )
    assert srv.audit_log.calls == [
        ("user", "I already did a reboot earlier.", 665108),
        ("user", "It still connects, then disconnects.", 665109),
        ("assistant", "Try a different network.", 672858),
    ]


def test_audio_burst_without_history_text_still_logs_a_turn():
    srv = _server(replies=[{"start_ms": 1200, "end_ms": 1500, "interrupted": True}], caller_starts=[])
    srv._write_transcript([])
    assert [c[0] for c in srv._fw_log.calls] == ["turn_start", "turn_end"]
    assert srv._fw_log.calls[1][2] == {"was_interrupted": True, "timestamp_ms": 1500}


class _Owner:
    def __init__(self):
        self.calls = []

    async def execute_tool(self, name, args):
        self.calls.append((name, args))
        return {"status": "success"}


@pytest.fixture
def gateway():
    return _ToolGateway(0, "b" * 40)


async def _post(gateway, headers, body=None):
    transport = httpx.ASGITransport(app=gateway._app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        return await client.post("/tools/lookup", json=body or {"id": "1"}, headers=headers)


def test_gateway_routes_by_session_header(gateway):
    owner = _Owner()
    gateway.bind("s1", owner)
    headers = {"Authorization": "Bearer " + "b" * 40, krisp_agents.SESSION_ID_HEADER: "s1"}
    resp = asyncio.run(_post(gateway, headers))
    assert resp.status_code == 200 and owner.calls == [("lookup", {"id": "1"})]


def test_gateway_refuses_a_wrong_secret(gateway):
    owner = _Owner()
    gateway.bind("s1", owner)
    headers = {"Authorization": "Bearer nope", krisp_agents.SESSION_ID_HEADER: "s1"}
    assert asyncio.run(_post(gateway, headers)).status_code == 401
    assert owner.calls == []


def test_gateway_refuses_a_non_ascii_secret_with_401(gateway):
    headers = {"Authorization": "Bearer café".encode("latin-1"), krisp_agents.SESSION_ID_HEADER: "s1"}
    assert asyncio.run(_post(gateway, headers)).status_code == 401


def test_gateway_refuses_an_unknown_session(gateway):
    headers = {"Authorization": "Bearer " + "b" * 40, krisp_agents.SESSION_ID_HEADER: "other"}
    assert asyncio.run(_post(gateway, headers)).status_code == 404


def test_clock_tool_answers_with_the_scenario_date():
    srv = object.__new__(KrispAssistantServer)
    srv.current_date_time = "2026-03-10 09:15 EST"
    result = asyncio.run(srv.execute_tool(krisp_agents.CLOCK_TOOL_NAME, {}))
    assert result == {"status": "success", "current_date_time": "2026-03-10 09:15 EST"}


def test_reply_lands_after_the_tools_and_the_lead_in_before_them():
    srv = _server(
        replies=[
            {"start_ms": 5000, "end_ms": 5400, "interrupted": False},
            {"start_ms": 7000, "end_ms": 9000, "interrupted": False},
        ],
        caller_starts=[3000],
        tool_times=[5600, 5700],
    )
    entry = _entry(7500, user="Go ahead and change me.", agent="You're changed to SK703.")
    entry["fillers"] = [{"text": "One moment.", "timestamp_ms": 4900}]
    srv._write_transcript([entry])
    assert srv.audit_log.calls == [
        ("user", "Go ahead and change me.", 3000),
        ("assistant", "One moment.", 5000),
        ("assistant", "You're changed to SK703.", 7000),
    ]
    responses = [c[1][0] for c in srv._fw_log.calls if c[0] == "llm_response"]
    assert responses == ["One moment.", "You're changed to SK703."]


def test_a_reply_is_logged_after_the_tool_result_it_reports():
    # The reply's audio is already playing when the last tool returns.
    srv = _server(
        replies=[{"start_ms": 5000, "end_ms": 9000, "interrupted": False}],
        caller_starts=[3000],
        tool_times=[5600],
        result_times=[5610],
    )
    srv._write_transcript([_entry(7500, user="Book the window seat.", agent="Your window seat is booked.")])
    reply_ts = srv.audit_log.calls[-1][2]
    assert srv.audit_log.calls[-1][1] == "Your window seat is booked." and reply_ts > 5610


def test_reply_skips_unlabelled_audio_that_played_before_its_tools():
    srv = _server(
        replies=[
            {"start_ms": 5000, "end_ms": 5400, "interrupted": False},
            {"start_ms": 7000, "end_ms": 9000, "interrupted": False},
        ],
        caller_starts=[3000],
        tool_times=[5600],
    )
    srv._write_transcript([_entry(7500, user="Cancel it.", agent="It's cancelled.")])
    assert srv.audit_log.calls[-1] == ("assistant", "It's cancelled.", 7000)


def test_a_reply_spoken_straight_through_a_fast_tool_stays_in_its_turn():
    # 2026-10-30 airline 1.1.2: lead-in, 11 ms tool and answer played as one burst.
    srv = _server(
        replies=[{"start_ms": 16164, "end_ms": 34438, "interrupted": False}],
        caller_starts=[10000],
        tool_times=[16175],
    )
    entry = _entry(18221, user="Move it to the 25th.", agent="The best option is SK703, one hundred fifteen dollars.")
    entry["fillers"] = [{"text": "I'll look for flights that fit.", "timestamp_ms": 16100}]
    srv._write_transcript([entry])
    responses = [c[1][0] for c in srv._fw_log.calls if c[0] == "llm_response"]
    assert responses == ["I'll look for flights that fit. The best option is SK703, one hundred fifteen dollars."]
    reply_ts = next(ts for kind, text, ts in srv.audit_log.calls if text.startswith("The best option"))
    assert reply_ts > 16175


def test_a_lead_in_never_precedes_the_caller_words_that_prompted_it():
    srv = _server(replies=[], caller_starts=[], tool_times=[588098])
    entry = _entry(588200, user="Last name Rodriguez.", agent="Found it.")
    entry["fillers"] = [{"text": "Let me check that.", "timestamp_ms": 587944}]
    srv._write_transcript([entry])
    order = [text for _, text, _ in sorted(srv.audit_log.calls, key=lambda c: c[2])]
    assert order == ["Last name Rodriguez.", "Let me check that.", "Found it."]
    assert srv.audit_log.calls[0] == ("user", "Last name Rodriguez.", 587943)  # before the lead-in, not at the tool


def test_a_lead_in_whose_audio_lands_after_a_fast_tool_is_still_heard_and_logged_first():
    srv = _server(
        replies=[{"start_ms": 554453, "end_ms": 565128, "interrupted": False}],
        caller_starts=[547329],
        tool_times=[554331],
    )
    entry = _entry(554500, user="It's VK3FFW.", agent="I'm not finding it.")
    entry["fillers"] = [{"text": "Let me pull that up.", "timestamp_ms": 554215}]
    srv._write_transcript([entry])
    responses = [c[1][0] for c in srv._fw_log.calls if c[0] == "llm_response"]
    assert responses == ["Let me pull that up. I'm not finding it."]
    stamps = {text: ts for _, text, ts in srv.audit_log.calls}
    assert stamps["Let me pull that up."] < 554331 < stamps["I'm not finding it."]


def test_the_gateway_answers_on_a_second_event_loop():
    # 2026-10-30: second domain's tool calls hung on a server bound to the first loop.
    gw = _ToolGateway(0, "b" * 40)
    owner = _Owner()
    gw.bind("s1", owner)
    headers = {"Authorization": "Bearer " + "b" * 40, krisp_agents.SESSION_ID_HEADER: "s1"}

    async def call() -> int:
        await gw.ensure_started()
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.post(f"http://127.0.0.1:{gw.port}/tools/lookup", json={}, headers=headers)
        return resp.status_code

    try:
        assert asyncio.run(call()) == 200
        assert asyncio.run(call()) == 200
    finally:
        gw.close()
    assert len(owner.calls) == 2


def test_a_gateway_whose_port_is_taken_fails_loudly_and_retries_later():
    import socket

    from eva.assistant import krisp_server

    blocker = socket.socket()
    blocker.bind(("0.0.0.0", 0))
    blocker.listen()
    gw = _ToolGateway(blocker.getsockname()[1], "b" * 40)
    try:
        with pytest.raises(OSError):
            asyncio.run(gw.ensure_started())
        assert gw._server is None
    finally:
        blocker.close()
        gw.close()
    assert krisp_server._bind(0).getsockname()[1] > 0


def test_waiting_for_a_server_whose_task_died_raises_instead_of_hanging():
    from eva.assistant.krisp_server import _wait_started

    class _Never:
        started = False

    async def scenario():
        async def die():
            raise OSError("address in use")

        task = asyncio.create_task(die())
        await asyncio.wait_for(_wait_started(_Never(), task, "server"), timeout=2)

    with pytest.raises(RuntimeError, match="stopped before it started"):
        asyncio.run(scenario())


def test_a_gateway_that_dies_after_binding_is_not_reused(monkeypatch):
    async def dies(self, sockets=None):
        raise RuntimeError("startup failed")

    gw = _ToolGateway(0, "b" * 40)
    monkeypatch.setattr("uvicorn.Server.serve", dies)
    with pytest.raises(RuntimeError, match="stopped before it started"):
        asyncio.run(gw.ensure_started())
    assert gw._server is None and gw._socket is None


def test_a_history_failure_still_stops_the_server(monkeypatch):
    class _Server:
        should_exit = False

    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._record_t0, srv._session_id, srv._gateway = True, None, "s1", None
    srv._params, srv.failure, srv.port = {}, None, 0
    srv._server, srv._server_task = _Server(), None
    stopped = srv._server

    def boom(*a, **k):
        raise TimeoutError("history timed out")

    monkeypatch.setattr("eva.utils.krisp_agents.fetch_session_history", boom)
    with pytest.raises(KrispCallInvalidError) as raised:
        asyncio.run(srv._shutdown())
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert stopped.should_exit and srv._server is None
    assert "transcript not written" in srv.failure


def test_a_lead_in_starting_late_after_its_tool_keeps_its_own_audio_and_the_reply_its_own():
    srv = _server(
        replies=[
            {"start_ms": 6200, "end_ms": 6800, "interrupted": False},
            {"start_ms": 8000, "end_ms": 9000, "interrupted": False},
        ],
        caller_starts=[3000],
        tool_times=[5600],
    )
    entry = _entry(9100, user="Cancel it.", agent="It's cancelled.")
    entry["fillers"] = [{"text": "One moment.", "timestamp_ms": 5590}]
    srv._write_transcript([entry])
    responses = [(c[2]["timestamp_ms"], c[1][0]) for c in srv._fw_log.calls if c[0] == "llm_response"]
    assert responses == [(6200, "One moment."), (8000, "It's cancelled.")]


def test_a_reply_never_lands_on_the_next_turns_audio():
    # Lead-in and answer form one burst; the next burst is a new turn.
    srv = _server(
        replies=[
            {"start_ms": 5800, "end_ms": 7000, "interrupted": False},
            {"start_ms": 8600, "end_ms": 9500, "interrupted": False},
        ],
        caller_starts=[3000, 7600],
        tool_times=[5600],
    )
    entry = _entry(7100, user="Cancel it.", agent="It's cancelled.")
    entry["fillers"] = [{"text": "One moment.", "timestamp_ms": 5590}]
    srv._write_transcript([entry])
    responses = [(c[2]["timestamp_ms"], c[1][0]) for c in srv._fw_log.calls if c[0] == "llm_response"]
    assert responses == [(5800, "One moment. It's cancelled.")]


def test_a_call_whose_tools_never_reached_the_gateway_fails_for_a_redial(monkeypatch):
    class _Server:
        should_exit = False

    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._record_t0, srv._session_id, srv._gateway = True, None, "s1", None
    srv._params, srv.failure, srv.port = {}, None, 0
    srv._server, srv._server_task = _Server(), None
    summary = '{"proxy_outcome": "transient_error", "upstream_status": null, "body": {"error": "connection refused"}}'
    history = [{"tool_calls": [{"tool_name": "get_reservation", "result_summary": summary}]}]
    monkeypatch.setattr("eva.utils.krisp_agents.fetch_session_history", lambda *a, **k: history)
    monkeypatch.setattr(KrispAssistantServer, "_write_transcript", lambda self, h: None)
    with pytest.raises(KrispCallInvalidError, match="tool gateway unreachable"):
        asyncio.run(srv._shutdown())
    assert srv.failure == "tool gateway unreachable: get_reservation: connection refused" and srv._server is None


def test_the_session_stays_bound_to_the_gateway_until_history_is_read(monkeypatch):
    class _Server:
        should_exit = False

    gw = _ToolGateway(0, "b" * 40)
    srv = object.__new__(KrispAssistantServer)
    gw.bind("s1", srv)
    srv._running, srv._record_t0, srv._session_id, srv._gateway = True, None, "s1", gw
    srv._params, srv.failure, srv.port = {}, None, 0
    srv._server, srv._server_task = _Server(), None
    srv.user_audio_buffer, srv.assistant_audio_buffer, srv._audio_buffer = bytearray(), bytearray(), bytearray()
    bound_during_fetch = []

    def fetch(*a, **k):
        bound_during_fetch.append("s1" in gw._sessions)
        return []

    monkeypatch.setattr("eva.utils.krisp_agents.fetch_session_history", fetch)
    monkeypatch.setattr(KrispAssistantServer, "_write_transcript", lambda self, h: None)
    asyncio.run(srv._shutdown())
    assert bound_during_fetch == [True] and "s1" not in gw._sessions


def test_a_failed_shutdown_saves_no_outputs_so_the_record_is_never_scored(monkeypatch):
    srv = object.__new__(KrispAssistantServer)
    saved = []

    async def fail():
        raise KrispCallInvalidError("tool gateway unreachable")

    async def save():
        saved.append(True)

    monkeypatch.setattr(srv, "_shutdown", fail, raising=False)
    monkeypatch.setattr(srv, "save_outputs", save, raising=False)
    with pytest.raises(KrispCallInvalidError):
        asyncio.run(srv.stop())
    assert saved == []


def test_shutdown_mixes_the_recording_so_eva_never_runs_its_slow_mix(monkeypatch):
    import struct

    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._record_t0, srv._session_id, srv._gateway = True, None, None, None
    srv._server, srv._server_task, srv.port, srv.failure = None, None, 0, None
    srv.user_audio_buffer = bytearray(struct.pack("<3h", 30000, -30000, 100))
    srv.assistant_audio_buffer = bytearray(struct.pack("<2h", 10000, -10000))
    srv._audio_buffer = bytearray()

    def slow_mix(*a):
        raise AssertionError("EVA's per-sample mix ran on the event loop")

    monkeypatch.setattr("eva.assistant.base_server.pcm16_mix", slow_mix)
    asyncio.run(srv._shutdown())
    srv._ensure_mixed_audio()
    assert bytes(srv._audio_buffer) == struct.pack("<3h", 32767, -32768, 100)


def test_mix_pcm16_matches_eva_mix_byte_for_byte():
    import os

    from eva.utils.audio_utils import pcm16_mix

    user, assistant = os.urandom(4000), os.urandom(3000)
    assert krisp_server._mix_pcm16(user, assistant) == pcm16_mix(user, assistant)


def test_response_latency_is_measured_from_the_simulators_speech_stop(monkeypatch):
    import base64
    import json

    monkeypatch.setattr(krisp_server, "_now_ms", lambda: 10_000)
    stop_ms = 9_200
    loud = base64.b64encode(b"\x00" * 160).decode()
    caller_messages = [
        json.dumps({"event": "user_speech_stop", "timestamp_ms": str(stop_ms)}),
        json.dumps({"event": "media", "media": {"payload": loud}}),
    ]
    caller_done = asyncio.Event()

    class _Simulator:
        closed = False

        async def close(self):
            _Simulator.closed = True

        async def receive_text(self):
            if caller_messages:
                return caller_messages.pop(0)
            caller_done.set()
            await asyncio.sleep(5)

        async def send_text(self, _):
            pass

    class _Platform:
        close_code, close_reason = 1000, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            await caller_done.wait()
            yield json.dumps({"event": "media", "media": {"payload": loud}})

    class _Metrics:
        latencies: list[float] = []

        def write_latency(self, stage, seconds, model):
            self.latencies.append(seconds)

    monkeypatch.setattr(krisp_server.websockets, "connect", lambda *a, **k: _Platform())
    session = {"session_id": "s1", "session_token": "t", "media_stream_url": "wss://media.example.com/ws"}
    monkeypatch.setattr(krisp_agents, "mint_voice_session", lambda *a, **k: session)
    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._agent_id, srv._params, srv._model = True, "a1", {}, "m"
    srv.conversation_id, srv._session_id, srv._record_t0, srv._session_start_ms = "c1", None, None, None
    srv._replies, srv._caller_starts, srv.failure = [], [], None
    srv.current_date_time = "2026-03-10 09:15 EST"
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv._gateway, srv._metrics_log = _ToolGateway(0, "b" * 40), _Metrics()

    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=5))
    assert srv._metrics_log.latencies == [0.8]
    assert _Simulator.closed


@pytest.mark.parametrize(("params", "sent"), [({}, "2026-03-10 09:15 EST"), ({"call_date": "clock"}, None)])
def test_the_session_start_carries_the_scenario_date_only_in_session_mode(monkeypatch, params, sent):
    calls = []

    def fake_mint(*args):
        calls.append(args)
        raise RuntimeError("stop after the exchange")

    class _Simulator:
        async def close(self, code=None):
            pass

    monkeypatch.setattr(krisp_agents, "mint_voice_session", fake_mint)
    srv = object.__new__(KrispAssistantServer)
    srv._agent_id, srv._params, srv.conversation_id = "a1", params, "c1"
    srv.current_date_time, srv.failure = "2026-03-10 09:15 EST", None
    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=5))
    assert calls == [("a1", params, sent)]
    assert "could not open a session" in srv.failure


@pytest.mark.parametrize("preamble", ["on", "off"])
def test_the_agent_check_compares_lead_ins_with_the_declared_preamble(monkeypatch, preamble):
    seen = {}
    monkeypatch.setattr(krisp_agents, "fetch_agent", lambda *a: {"id": "a1"})
    monkeypatch.setattr(krisp_agents, "system_prompt", lambda *a: "p")
    monkeypatch.setattr(krisp_agents, "eva_tools", lambda *a: [])
    monkeypatch.setattr(krisp_agents, "check_agent_matches_config", lambda agent, **kw: seen.update(kw))
    srv = object.__new__(KrispAssistantServer)
    srv._agent_id, srv._params, srv._model = "a1", {"preamble": preamble}, "m"
    srv.agent, srv.language, srv.initial_message = None, "en", "hi"
    srv.pipeline_config = type("P", (), {"pre_tool_speech": "off"})()
    srv._check_agent()
    assert seen["preamble"] == preamble


def test_a_call_that_never_opened_is_marked_unscoreable(monkeypatch):
    from eva.assistant.base_server import RecordInvalidError

    class _Server:
        should_exit = False

    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._record_t0, srv._session_id, srv._gateway = True, None, None, None
    srv._params, srv.failure, srv.port = {}, "could not open a session: HTTP 503", 0
    srv._server, srv._server_task = _Server(), None
    with pytest.raises(RecordInvalidError, match="could not open a session"):
        asyncio.run(srv._shutdown())


def test_turns_are_counted_before_the_transcript_is_written(monkeypatch):
    from eva.assistant.base_server import AbstractAssistantServer

    monkeypatch.setattr(
        AbstractAssistantServer, "get_conversation_stats", lambda self: {"num_turns": 0, "num_tool_calls": 2}
    )
    srv = object.__new__(KrispAssistantServer)
    srv._caller_starts = [1000, 4000, 9000]
    assert srv.get_conversation_stats() == {"num_turns": 3, "num_tool_calls": 2}


def test_agent_audio_cleared_before_the_caller_heard_it_is_not_recorded(monkeypatch):
    import base64
    import json

    loud = base64.b64encode(b"\x00" * 480).decode()
    gate = asyncio.Event()

    class _Simulator:
        async def close(self, code=None):
            pass

        async def receive_text(self):
            await asyncio.sleep(5)

        async def send_text(self, _):
            await gate.wait()

    class _Platform:
        close_code, close_reason = 1000, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            yield json.dumps({"event": "media", "media": {"payload": loud}})
            await asyncio.sleep(0.05)
            yield json.dumps({"event": "clear"})
            gate.set()
            await asyncio.sleep(0.05)

    monkeypatch.setattr(krisp_server.websockets, "connect", lambda *a, **k: _Platform())
    session = {"session_id": "s1", "session_token": "t", "media_stream_url": "wss://media.example.com/ws"}
    monkeypatch.setattr(krisp_agents, "mint_voice_session", lambda *a, **k: session)
    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._agent_id, srv._params, srv._model = True, "a1", {"call_date": "clock"}, "m"
    srv.conversation_id, srv._session_id, srv._record_t0, srv._session_start_ms = "c1", None, None, None
    srv._replies, srv._caller_starts, srv.failure = [], [], None
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv._gateway, srv._metrics_log = _ToolGateway(0, "b" * 40), None

    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=5))
    one_chunk = len(krisp_server._to_pcm16k(b"\x00" * 160, None)[0])
    assert len(srv.assistant_audio_buffer) == one_chunk  # the two cleared chunks were never heard


def test_a_stall_inside_speech_does_not_split_the_recording(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(krisp_server.time, "monotonic", lambda: now["t"])
    srv = object.__new__(KrispAssistantServer)
    srv._record_t0 = None
    track, frame = bytearray(), b"\x01\x00" * 320
    srv._record(track, frame, at_start=True)
    now["t"] = 1.0  # a frame that arrived 1 s late, mid-speech
    srv._record(track, frame, at_start=False)
    assert len(track) == 2 * len(frame)
    now["t"] = 1.5  # the next turn is placed at its real time
    srv._record(track, frame, at_start=True)
    assert len(track) == int(1.5 * 16000) * 2 + len(frame)


def test_reply_end_counts_silence_in_20_ms_chunks_whatever_the_packet_size(monkeypatch):
    import base64
    import json

    loud = base64.b64encode(b"\x00" * 160).decode()
    long_silence = base64.b64encode(b"\xff" * 160 * 20).decode()  # 400 ms in one message

    class _Simulator:
        async def close(self, code=None):
            pass

        async def receive_text(self):
            await asyncio.sleep(5)

        async def send_text(self, _):
            pass

    class _Platform:
        close_code, close_reason = 1000, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            for payload in (loud, long_silence, loud):
                yield json.dumps({"event": "media", "media": {"payload": payload}})
            await asyncio.sleep(0.05)

    monkeypatch.setattr(krisp_server.websockets, "connect", lambda *a, **k: _Platform())
    session = {"session_id": "s1", "session_token": "t", "media_stream_url": "wss://media.example.com/ws"}
    monkeypatch.setattr(krisp_agents, "mint_voice_session", lambda *a, **k: session)
    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._agent_id, srv._params, srv._model = True, "a1", {"call_date": "clock"}, "m"
    srv.conversation_id, srv._session_id, srv._record_t0, srv._session_start_ms = "c1", None, None, None
    srv._replies, srv._caller_starts, srv.failure = [], [], None
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv._gateway, srv._metrics_log = _ToolGateway(0, "b" * 40), None

    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=5))
    assert len(srv._replies) == 2  # 400 ms of silence ends the first reply even in one packet


def test_caller_audio_recorded_packet_by_packet_matches_one_continuous_conversion(monkeypatch):
    import base64
    import json
    import math

    audioop = krisp_server.audioop
    tone = b"".join(
        int(8000 * math.sin(2 * math.pi * 440 * i / 8000)).to_bytes(2, "little", signed=True) for i in range(8000)
    )
    mulaw = audioop.lin2ulaw(tone, 2)
    packets = [mulaw[i : i + 160] for i in range(0, len(mulaw), 160)]
    inbox = [json.dumps({"event": "media", "media": {"payload": base64.b64encode(p).decode()}}) for p in packets]

    class _Simulator:
        async def close(self, code=None):
            pass

        async def receive_text(self):
            if inbox:
                return inbox.pop(0)
            await asyncio.sleep(5)

        async def send_text(self, _):
            pass

    class _Platform:
        close_code, close_reason = 1000, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            while inbox:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            return
            yield

    monkeypatch.setattr(krisp_server.websockets, "connect", lambda *a, **k: _Platform())
    session = {"session_id": "s1", "session_token": "t", "media_stream_url": "wss://media.example.com/ws"}
    monkeypatch.setattr(krisp_agents, "mint_voice_session", lambda *a, **k: session)
    srv = object.__new__(KrispAssistantServer)
    srv._running, srv._agent_id, srv._params, srv._model = True, "a1", {"call_date": "clock"}, "m"
    srv.conversation_id, srv._session_id, srv._record_t0, srv._session_start_ms = "c1", None, None, None
    srv._replies, srv._caller_starts, srv.failure = [], [], None
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv._gateway, srv._metrics_log = _ToolGateway(0, "b" * 40), None

    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=5))
    whole, _ = audioop.ratecv(audioop.ulaw2lin(mulaw, 2), 2, 1, 8000, 16000, None)
    assert bytes(srv.user_audio_buffer) == whole  # no sample lost or reset at any of the 50 packet joins


def test_agent_audio_recorded_packet_by_packet_matches_one_continuous_conversion(monkeypatch):
    import base64
    import json
    import math

    audioop = krisp_server.audioop
    tone = b"".join(
        int(8000 * math.sin(2 * math.pi * 440 * i / 8000)).to_bytes(2, "little", signed=True) for i in range(1600)
    )
    mulaw = audioop.lin2ulaw(tone, 2)
    whole, _ = audioop.ratecv(audioop.ulaw2lin(mulaw, 2), 2, 1, 8000, 16000, None)
    monkeypatch.setattr(krisp_server, "CHUNK_DURATION_S", 0.0)
    srv = object.__new__(KrispAssistantServer)
    srv._pad = lambda buffer: None  # isolate the resampler from wall-time padding

    class _Simulator:
        async def close(self, code=None):
            pass

        async def receive_text(self):
            await asyncio.sleep(5)

        async def send_text(self, _):
            pass

    class _Platform:
        close_code, close_reason = 1000, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            yield json.dumps({"event": "media", "media": {"payload": base64.b64encode(mulaw).decode()}})
            for _ in range(500):
                if len(srv.assistant_audio_buffer) >= len(whole):
                    break
                await asyncio.sleep(0.01)

    monkeypatch.setattr(krisp_server.websockets, "connect", lambda *a, **k: _Platform())
    session = {"session_id": "s1", "session_token": "t", "media_stream_url": "wss://media.example.com/ws"}
    monkeypatch.setattr(krisp_agents, "mint_voice_session", lambda *a, **k: session)
    srv._running, srv._agent_id, srv._params, srv._model = True, "a1", {"call_date": "clock"}, "m"
    srv.conversation_id, srv._session_id, srv._record_t0, srv._session_start_ms = "c1", None, None, None
    srv._replies, srv._caller_starts, srv.failure = [], [], None
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv._gateway, srv._metrics_log = _ToolGateway(0, "b" * 40), None

    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=10))
    assert bytes(srv.assistant_audio_buffer) == whole  # ten 20 ms chunks joined without a lost sample


def test_agent_goodbye_queued_when_the_platform_hangs_up_is_still_played_and_recorded(monkeypatch):
    import base64
    import json

    audioop = krisp_server.audioop
    goodbye = audioop.lin2ulaw(b"\x10\x27" * 1600, 2)  # 200 ms, queued as ten 20 ms chunks
    whole, _ = audioop.ratecv(audioop.ulaw2lin(goodbye, 2), 2, 1, 8000, 16000, None)
    sent = []

    class _Simulator:
        async def close(self, code=None):
            pass

        async def receive_text(self):
            await asyncio.sleep(5)

        async def send_text(self, message):
            sent.append(message)

    class _Platform:
        close_code, close_reason = 1000, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, _):
            pass

        async def __aiter__(self):
            yield json.dumps({"event": "media", "media": {"payload": base64.b64encode(goodbye).decode()}})

    monkeypatch.setattr(krisp_server.websockets, "connect", lambda *a, **k: _Platform())
    session = {"session_id": "s1", "session_token": "t", "media_stream_url": "wss://media.example.com/ws"}
    monkeypatch.setattr(krisp_agents, "mint_voice_session", lambda *a, **k: session)
    srv = object.__new__(KrispAssistantServer)
    srv._pad = lambda buffer: None
    srv._running, srv._agent_id, srv._params, srv._model = True, "a1", {"call_date": "clock"}, "m"
    srv.conversation_id, srv._session_id, srv._record_t0, srv._session_start_ms = "c1", None, None, None
    srv._replies, srv._caller_starts, srv.failure = [], [], None
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv._gateway, srv._metrics_log = _ToolGateway(0, "b" * 40), None

    asyncio.run(asyncio.wait_for(srv._handle_session(_Simulator()), timeout=10))
    assert len(sent) == 10
    assert bytes(srv.assistant_audio_buffer) == whole
