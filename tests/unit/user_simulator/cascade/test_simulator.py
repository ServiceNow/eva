import json
import logging

import pytest

from eva.models.config import PerturbationConfig
from eva.user_simulator.cascade.simulator import CascadeUserSimulator, extract_turn, parse_turn_response


def test_parse_turn_response_reads_a_clean_json_object():
    assert parse_turn_response('{"utterance": "Hi there."}') == "Hi there."


def test_parse_turn_response_tolerates_markdown_fences():
    assert parse_turn_response('```json\n{"utterance": "Bye."}\n```') == "Bye."


def test_parse_turn_response_falls_back_to_raw_text_when_not_json():
    assert parse_turn_response("I need to reset my password.") == "I need to reset my password."


def test_parse_turn_response_returns_empty_for_a_toolcall_only_turn():
    # The model hangs up by calling end_call and says nothing; content is empty.
    assert parse_turn_response("") == ""


def test_parse_turn_response_rejects_a_non_string_utterance():
    with pytest.raises(ValueError, match="utterance"):
        parse_turn_response(json.dumps({"utterance": 42}))


def test_extract_turn_reads_a_plain_string_as_no_hangup():
    # LiteLLMClient returns a bare str when the model made no tool call.
    assert extract_turn('{"utterance": "Still here."}') == ("Still here.", False)


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
        content = '{"utterance": "Go on."}'
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


def test_messages_flips_roles_so_the_caller_llm_sees_its_own_lines_as_assistant():
    sim = _make_bare_simulator()
    sim._history = [
        {"role": "assistant", "content": "What is your email?"},
        {"role": "user", "content": "It's jane@example.com."},
    ]

    messages = sim._messages()

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What is your email?"}
    assert messages[2] == {"role": "assistant", "content": "It's jane@example.com."}


def test_messages_appends_a_trailing_role_reminder():
    sim = _make_bare_simulator()

    messages = sim._messages()

    assert messages[-1]["role"] == "system"
    assert "CUSTOMER" in messages[-1]["content"]
    assert "Do NOT respond as the customer service agent" in messages[-1]["content"]
