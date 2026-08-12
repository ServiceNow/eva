from pathlib import Path

from eva.models.config import ElevenLabsSimulatorConfig, OpenAIRealtimeSimulatorConfig
from eva.user_simulator.elevenlabs import ElevenLabsUserSimulator
from eva.user_simulator.factory import create_user_simulator
from eva.user_simulator.openai_realtime import OpenAIRealtimeUserSimulator


def _kwargs(tmp_path: Path) -> dict:
    return {
        "current_date_time": "2026-06-05T12:00:00",
        "persona_config": {"user_persona_id": 1},
        "goal": {
            "high_level_user_goal": "Reset my password.",
            "decision_tree": {
                "must_have_criteria": ["Reset the password."],
                "escalation_behavior": "Escalate if blocked.",
                "nice_to_have_criteria": [],
                "negotiation_behavior": "Accept a valid resolution.",
                "resolution_condition": "The password is reset.",
                "failure_condition": "The password cannot be reset.",
                "edge_cases": [],
            },
            "information_required": ["employee ID"],
            "starting_utterance": "I need to reset my password.",
        },
        "server_url": "ws://localhost:9999/ws",
        "output_dir": tmp_path,
        "agent_id": "agent_itsm",
    }


def test_factory_keeps_elevenlabs_as_default(tmp_path):
    simulator = create_user_simulator(ElevenLabsSimulatorConfig(), **_kwargs(tmp_path))

    assert isinstance(simulator, ElevenLabsUserSimulator)
    assert simulator.provider == "elevenlabs"


def test_factory_selects_openai_realtime(tmp_path):
    config = OpenAIRealtimeSimulatorConfig()
    simulator = create_user_simulator(config, **_kwargs(tmp_path))

    assert isinstance(simulator, OpenAIRealtimeUserSimulator)
    assert simulator.caller_model == "gpt-realtime-1.5"
