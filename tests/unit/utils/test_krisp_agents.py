"""Tests for the Krisp agent config check, prompt and media-stream URL."""

import json

import pytest

from eva.models.agents import AgentConfig
from eva.utils import krisp_agents

PROMPT = "prompt"
GREETING = "Hello! How can I help you today?"


def _agent(**overrides) -> dict:
    config = {
        "system_prompt": PROMPT,
        "greeting": {"en": GREETING},
        "llm_config": {"model": "gpt-5.5"},
        "languages_config": {"supported": ["en"]},
    }
    config.update(overrides)
    lookup = {"name": "lookup", "config_json": {"filler": {"enabled": True}}}
    clock = {"name": krisp_agents.CLOCK_TOOL_NAME, "config_json": {"filler": {"enabled": False}}}
    return {"id": "a1", "config": config, "tools": [lookup, clock]}


def _check(agent: dict, **overrides) -> None:
    kwargs = {
        "model": "gpt-5.5",
        "language": "en",
        "system_prompt": PROMPT,
        "initial_message": GREETING,
        "tool_names": ["lookup", krisp_agents.CLOCK_TOOL_NAME],
        "preamble": "on",
    }
    kwargs.update(overrides)
    krisp_agents.check_agent_matches_config(agent, **kwargs)


def test_matching_agent_passes():
    _check(_agent())


def test_string_greeting_is_accepted():
    _check(_agent(greeting=GREETING))


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"system_prompt": "edited"}, "system_prompt"),
        ({"greeting": {"en": "Hi"}}, "greeting"),
        ({"llm_config": {"model": "gpt-5.4"}}, "llm_config.model"),
        ({"languages_config": {"supported": ["fr"]}}, "languages_config"),
    ],
)
def test_each_mismatch_is_named(overrides, problem):
    with pytest.raises(ValueError, match=problem):
        _check(_agent(**overrides))


def test_missing_tool_is_a_mismatch():
    agent = _agent()
    agent["tools"] = agent["tools"][:1]
    with pytest.raises(ValueError, match="tools"):
        _check(agent)


def test_prompt_is_evas_with_only_the_date_line_changed():
    agent = AgentConfig.from_yaml("configs/agents/itsm_agent.yaml")
    prompt = krisp_agents.system_prompt(agent, "off")
    assert f"Today is {krisp_agents.CLOCK_DATETIME}." in prompt
    assert agent.instructions.strip()[:200] in prompt
    assert "## Responsiveness" not in prompt
    assert "## Responsiveness" in krisp_agents.system_prompt(agent, "auto")


def test_eva_tools_end_with_the_clock():
    agent = AgentConfig.from_yaml("configs/agents/airline_agent.yaml")
    names = [t["function"]["name"] for t in krisp_agents.eva_tools(agent)]
    assert names[-1] == krisp_agents.CLOCK_TOOL_NAME
    assert names[:-1] == [t["function"]["name"] for t in agent.build_tools_for_agent()]


def test_session_mode_keeps_evas_date_line_for_the_platform_to_fill():
    agent = AgentConfig.from_yaml("configs/agents/itsm_agent.yaml")
    prompt = krisp_agents.system_prompt(agent, "off", "session")
    assert prompt.count("Today is {{vars.datetime}}.") == 1
    assert krisp_agents.CLOCK_TOOL_NAME not in prompt


def test_session_mode_drops_the_clock_tool():
    agent = AgentConfig.from_yaml("configs/agents/airline_agent.yaml")
    names = [t["function"]["name"] for t in krisp_agents.eva_tools(agent, "session")]
    assert names == [t["function"]["name"] for t in agent.build_tools_for_agent()]


def test_call_date_mode_defaults_to_session_and_refuses_unknown_values():
    assert krisp_agents.call_date_mode({}) == "session"
    assert krisp_agents.call_date_mode({"call_date": "clock"}) == "clock"
    with pytest.raises(ValueError, match="call_date"):
        krisp_agents.call_date_mode({"call_date": "variables"})


def _mint(monkeypatch, reply):
    sent = []

    def fake_request(method, path, params, body=None, retries=6):
        sent.append(body)
        return reply

    monkeypatch.setattr(krisp_agents, "request", fake_request)
    return sent


_REPLY = {"session_id": "s1", "session_token": "t", "ws_url": "wss://voice.example.com/x"}


def test_session_start_sends_the_scenario_date(monkeypatch):
    sent = _mint(monkeypatch, {**_REPLY, "accepted_variables": ["datetime"]})
    krisp_agents.mint_voice_session("a1", {}, "2026-03-09 10:00 EST")
    assert sent == [{"agent_id": "a1", "modality": "voice", "variables": {"datetime": "2026-03-09 10:00 EST"}}]


def test_session_start_stops_when_the_platform_ignores_the_date(monkeypatch):
    _mint(monkeypatch, _REPLY)
    with pytest.raises(RuntimeError, match="did not accept the call date"):
        krisp_agents.mint_voice_session("a1", {}, "2026-03-09 10:00 EST")


def test_clock_mode_sends_no_variables(monkeypatch):
    sent = _mint(monkeypatch, _REPLY)
    krisp_agents.mint_voice_session("a1", {})
    assert sent == [{"agent_id": "a1", "modality": "voice"}]


def test_media_stream_url_uses_the_exchange_host():
    assert krisp_agents.media_stream_url("wss://media.example.com/start") == "wss://media.example.com/ws"


def test_request_retries_a_rate_limit(monkeypatch):
    import io
    import urllib.error

    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {"Retry-After": "0"}, io.BytesIO(b"{}"))
        return _Resp(b'{"data": {"ok": true}}')

    monkeypatch.setattr(krisp_agents.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(krisp_agents.time, "sleep", lambda s: None)
    assert krisp_agents.request("GET", "/x", {"api_url": "https://h", "api_key": "k"}) == {"ok": True}
    assert len(calls) == 2


def test_request_does_not_retry_a_server_error(monkeypatch):
    import io
    import urllib.error

    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(krisp_agents.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(krisp_agents.time, "sleep", lambda s: None)
    with pytest.raises(krisp_agents.KrispApiError):
        krisp_agents.request("GET", "/sessions/s", {"api_url": "https://h", "api_key": "k"}, {})
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, 0.0), ("2", 2.0), ("soon", 0.0), ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0)],
)
def test_retry_after_accepts_seconds_and_dates(header, expected):
    assert krisp_agents._retry_after_s(header) == expected


def test_retry_after_date_in_the_future_waits_until_then(monkeypatch):
    monkeypatch.setattr(krisp_agents.time, "time", lambda: 1445412470.0)
    assert krisp_agents._retry_after_s("Wed, 21 Oct 2015 07:28:00 GMT") == 10.0


def _fetch_with(monkeypatch, responses):
    calls = []

    def fake_request(method, path, params, body=None):
        calls.append(path)
        r = responses[min(len(calls), len(responses)) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(krisp_agents, "request", fake_request)
    monkeypatch.setattr(krisp_agents.time, "sleep", lambda s: None)
    return calls


def test_history_fetch_raises_at_once_on_an_auth_error(monkeypatch):
    calls = _fetch_with(monkeypatch, [krisp_agents.KrispApiError(401, "HTTP 401")])
    with pytest.raises(krisp_agents.KrispApiError):
        krisp_agents.fetch_session_history("s1", {})
    assert len(calls) == 1


def test_history_fetch_waits_for_a_session_still_flushing(monkeypatch):
    done = {"history": [{"sequence": 1}], "ended_at": "t"}
    calls = _fetch_with(monkeypatch, [krisp_agents.KrispApiError(404, "HTTP 404"), done])
    assert krisp_agents.fetch_session_history("s1", {}) == [{"sequence": 1}]
    assert len(calls) == 2


def test_a_call_with_no_history_is_an_error_not_an_empty_transcript(monkeypatch):
    _fetch_with(monkeypatch, [{"history": []}])
    with pytest.raises(RuntimeError, match="no history"):
        krisp_agents.fetch_session_history("s1", {}, attempts=2)


def _call(name, outcome, upstream, body):
    summary = {"proxy_outcome": outcome, "upstream_status": upstream, "body": body}
    return {"tool_name": name, "result_summary": json.dumps(summary)}


def test_tool_calls_that_never_reached_the_gateway_are_named():
    history = [
        {"tool_calls": [_call("get_reservation", "transient_error", None, {"error": "connection refused"})]},
        {"tool_calls": [_call("rebook_flight", "transient_error", 502, "<html>Bad gateway</html>")]},
        {"tool_calls": [_call("get_flight", "success", 200, {"ok": True})]},
        {"tool_calls": [_call("cancel", "error", 200, {"status": "error", "message": "not cancellable"})]},
        {"user_message": "hi"},
    ]
    assert krisp_agents.unreached_tool_calls(history) == [
        "get_reservation: connection refused",
        "rebook_flight: <html>Bad gateway</html>",
    ]


def test_a_regional_language_matches_the_base_code_the_platform_stores():
    agent = {
        "id": "a1",
        "config": {
            "system_prompt": "p",
            "greeting": "Bonjour",
            "llm_config": {"model": "m"},
            "languages_config": {"supported": ["fr"]},
        },
        "tools": [{"name": "t1"}],
    }
    krisp_agents.check_agent_matches_config(
        agent,
        model="m",
        language="fr-CA",
        system_prompt="p",
        initial_message="Bonjour",
        tool_names=["t1"],
        preamble="off",
    )


def test_lead_ins_that_differ_from_the_declared_preamble_are_refused():
    with pytest.raises(ValueError, match=r"lead-ins are on on tools \['lookup'\], but s2s_params.preamble is 'off'"):
        _check(_agent(), preamble="off")


def test_the_clock_tool_is_exempt_from_the_preamble_check():
    agent = _agent()
    agent["tools"][0]["config_json"]["filler"]["enabled"] = False
    agent["tools"][1]["config_json"]["filler"]["enabled"] = True
    _check(agent, preamble="off")


def test_preamble_defaults_to_on_and_refuses_unknown_values():
    assert krisp_agents.preamble_mode({}) == "on"
    assert krisp_agents.preamble_mode({"preamble": "off"}) == "off"
    with pytest.raises(ValueError, match="s2s_params.preamble"):
        krisp_agents.preamble_mode({"preamble": "auto"})


def test_a_greeting_stored_under_the_base_language_matches_a_regional_run():
    assert krisp_agents._greeting({"greeting": {"fr": "Bonjour"}}, "fr-CA") == "Bonjour"
    assert krisp_agents._greeting({"greeting": {"fr-CA": "Salut", "fr": "Bonjour"}}, "fr-CA") == "Salut"
