"""Tests for the Cartesia managed agent check against the EVA run config."""

import pytest

from eva.utils.cartesia_agents import check_agent_matches_config


def _agent(**overrides) -> dict:
    config = {
        "instructions": "{{system_prompt}}",
        "prompt_optimization": False,
        "initial_message": "{{initial_message}}",
        "model": {"id": "gpt-5.5", "fallback_strategy": None},
        "language": {"primary": "en"},
        "turn": {"inactivity_check_in_secs": None},
        "tools": [{"name": "get_reservation", "pre_tool_speech": "auto"}],
    }
    config.update(overrides)
    return {"id": "agent_x", "config": config}


def _check(agent, language="en", pre_tool_speech="auto", turn_end_fallback_time=None):
    check_agent_matches_config(agent, "gpt-5.5", language, pre_tool_speech, turn_end_fallback_time, ["get_reservation"])


def test_matching_agent_passes():
    _check(_agent())
    _check(_agent(turn={"inactivity_check_in_secs": 10}), turn_end_fallback_time=10)
    _check(_agent(language={"primary": "fr"}), language="fr-CA")  # Cartesia has no regional variants


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"model": {"id": "gpt-5.5", "fallback_strategy": {"type": "auto"}}}, "fallback_strategy"),
        ({"model": {"id": "gpt-5.4", "fallback_strategy": None}}, "s2s_params.model"),
        ({"language": {"primary": "fr"}}, "EVA_LANGUAGE"),
        ({"prompt_optimization": True}, "prompt_optimization"),
        ({"instructions": "Be nice. {{system_prompt}}"}, "instructions"),
        ({"turn": {"inactivity_check_in_secs": 15}}, "inactivity_check_in_secs"),
        ({"tools": []}, "do not match"),
    ],
)
def test_mismatched_agent_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _check(_agent(**overrides))


def test_pre_tool_speech_follows_eva_setting():
    with pytest.raises(ValueError, match="pre_tool_speech"):
        _check(_agent(), pre_tool_speech="off")
