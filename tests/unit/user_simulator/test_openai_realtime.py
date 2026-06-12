from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.models.config import PerturbationConfig, UserSimulatorConfig
from eva.user_simulator.openai_realtime import OpenAIRealtimeUserSimulator


def _simulator(tmp_path: Path, *, persona_id: int = 1, **config_overrides) -> OpenAIRealtimeUserSimulator:
    return OpenAIRealtimeUserSimulator(
        current_date_time="2026-06-05T12:00:00",
        persona_config={"user_persona_id": persona_id},
        goal={
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
        server_url="ws://localhost:9999/ws",
        output_dir=tmp_path,
        agent_id="agent_itsm",
        simulator_config=UserSimulatorConfig(provider="openai_realtime", **config_overrides),
    )


def test_session_config_matches_parity_profile(tmp_path):
    simulator = _simulator(tmp_path)
    config = simulator._build_session_config()

    assert simulator.caller_model == "gpt-realtime-1.5"
    assert config["audio"]["output"]["voice"] == "marin"
    assert config["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert config["audio"]["input"]["turn_detection"]["create_response"] is False
    assert config["audio"]["input"]["turn_detection"]["interrupt_response"] is False
    assert config["parallel_tool_calls"] is False
    assert config["tools"][0]["name"] == "end_call"
    assert "Reset the password." in config["instructions"]


def test_model_and_persona_voices_are_configurable(tmp_path):
    female = _simulator(tmp_path / "female", model="gpt-realtime-2", female_voice="coral")
    male = _simulator(tmp_path / "male", persona_id=2, male_voice="verse")

    assert female.caller_model == "gpt-realtime-2"
    assert female.caller_voice == "coral"
    assert male.caller_voice == "verse"


def test_elevenlabs_specific_accent_variants_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="ElevenLabs-specific"):
        OpenAIRealtimeUserSimulator(
            current_date_time="2026-06-05T12:00:00",
            persona_config={"user_persona_id": 1},
            goal=_simulator(tmp_path).goal,
            server_url="ws://localhost:9999/ws",
            output_dir=tmp_path,
            agent_id="agent_itsm",
            perturbation_config=PerturbationConfig(accent="french"),
            simulator_config=UserSimulatorConfig(provider="openai_realtime"),
        )


def test_behavior_variant_uses_shared_prompt(tmp_path):
    simulator = OpenAIRealtimeUserSimulator(
        current_date_time="2026-06-05T12:00:00",
        persona_config={"user_persona_id": 1},
        goal=_simulator(tmp_path).goal,
        server_url="ws://localhost:9999/ws",
        output_dir=tmp_path,
        agent_id="agent_itsm",
        perturbation_config=PerturbationConfig(behavior="aggressive_impatient"),
        simulator_config=UserSimulatorConfig(provider="openai_realtime"),
    )

    prompt = simulator._build_prompt()

    assert "You are impatient and easily frustrated" in prompt


@pytest.mark.asyncio
async def test_end_call_writes_neutral_artifact(tmp_path):
    simulator = _simulator(tmp_path)
    conn = MagicMock()

    await simulator._handle_caller_event(
        SimpleNamespace(type="response.output_audio_transcript.done", transcript="Goodbye.")
    )
    await simulator._handle_caller_event(
        SimpleNamespace(type="response.function_call_arguments.done", name="end_call", arguments="{}")
    )
    assert not simulator._conversation_done.is_set()
    await simulator._finish_caller_response(conn)
    simulator.event_logger.log_connection_state("session_ended", {"reason": simulator._end_reason})
    simulator.event_logger.save()

    path = tmp_path / "user_simulator_events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["type"] for event in events] == ["user_speech", "tool_call", "connection_state"]
    assert events[0]["provider"] == "openai_realtime"
    assert events[0]["data"]["source"] == "simulated_user"
    assert events[-1]["data"]["details"]["reason"] == "goodbye"


@pytest.mark.asyncio
async def test_unexpected_background_task_completion_is_an_error(tmp_path):
    simulator = _simulator(tmp_path)
    completion_task = asyncio.create_task(asyncio.sleep(60))
    forward_task = asyncio.create_task(asyncio.sleep(0))
    listener_task = asyncio.create_task(asyncio.sleep(60))

    with pytest.raises(RuntimeError, match="audio forwarder stopped unexpectedly"):
        await simulator._wait_for_session_completion(completion_task, forward_task, listener_task)

    completion_task.cancel()
    listener_task.cancel()
    await asyncio.gather(completion_task, listener_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_response_error_retries_pending_turn_after_response_done(tmp_path):
    simulator = _simulator(tmp_path)
    conn = MagicMock()
    conn.response.create = AsyncMock()
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.assistant_audio_ended_at = None

    await simulator._handle_caller_event(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(code="conversation_already_has_active_response"),
        )
    )
    await simulator._finish_caller_response(conn)
    await simulator._caller_response_task

    conn.response.create.assert_awaited_once()
    assert simulator._caller_response_pending is False


@pytest.mark.asyncio
async def test_streaming_resampler_preserves_state_across_deltas(tmp_path, monkeypatch):
    simulator = _simulator(tmp_path)
    simulator._audio_interface = MagicMock()
    states = []

    def ratecv(data, width, channels, input_rate, output_rate, state):
        states.append(state)
        return b"converted", f"state-{len(states)}"

    monkeypatch.setattr("eva.user_simulator.openai_realtime.audioop.ratecv", ratecv)
    delta = "AAAAAA=="

    await simulator._handle_caller_event(SimpleNamespace(type="response.output_audio.delta", delta=delta))
    await simulator._handle_caller_event(SimpleNamespace(type="response.output_audio.delta", delta=delta))
    await simulator._handle_caller_event(SimpleNamespace(type="response.output_audio.done"))

    assert states == [None, "state-1"]
    assert simulator._resampler_state is None


@pytest.mark.asyncio
async def test_response_requests_are_coalesced(tmp_path):
    simulator = _simulator(tmp_path)
    conn = MagicMock()
    conn.response.create = AsyncMock()
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = False
    simulator._audio_interface.assistant_audio_ended_at = asyncio.get_running_loop().time() - 3
    simulator._assistant_transcript_ready = True

    simulator._schedule_caller_response(conn, trigger="first")
    first_task = simulator._caller_response_task
    simulator._schedule_caller_response(conn, trigger="duplicate")
    await first_task

    conn.response.create.assert_awaited_once()
    assert any(event.get("type") == "caller_response_coalesced" for event in simulator.event_logger.get_events())


@pytest.mark.asyncio
async def test_response_create_failure_ends_conversation_and_resets_state(tmp_path):
    simulator = _simulator(tmp_path)
    conn = MagicMock()
    conn.response.create = AsyncMock(side_effect=RuntimeError("connection closed"))
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = False
    simulator._audio_interface.assistant_audio_ended_at = asyncio.get_running_loop().time() - 3
    simulator._assistant_transcript_ready = True

    await simulator._create_caller_response_when_ready(conn, "vad_speech_stopped")

    assert simulator._caller_response_active is False
    assert simulator._conversation_done.is_set()
    assert simulator._end_reason == "error"
    errors = simulator.event_logger.get_events("error")
    assert errors[-1]["data"]["details"]["error"] == "connection closed"


@pytest.mark.asyncio
async def test_timeout_is_terminal_and_cannot_be_overwritten_by_late_goodbye(tmp_path):
    simulator = _simulator(tmp_path)
    simulator.timeout = 0.001

    await simulator._wait_for_conversation_end()
    simulator._on_conversation_end("goodbye")

    assert simulator._conversation_done.is_set()
    assert simulator._end_reason == "timeout"


@pytest.mark.asyncio
async def test_failed_background_task_does_not_interrupt_cleanup(tmp_path):
    simulator = _simulator(tmp_path)

    async def fail():
        raise RuntimeError("transport failed")

    task = asyncio.create_task(fail())
    await asyncio.sleep(0)

    await simulator._cancel_background_task(task)


@pytest.mark.asyncio
async def test_missing_openai_key_fails_before_connecting(tmp_path, monkeypatch):
    simulator = _simulator(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await simulator.run_conversation()
