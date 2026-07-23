"""Tests for the Nova Sonic speech-to-speech user simulator.

Fully mocked — no AWS credentials and no experimental SDK required. The simulator
imports ``aws-sdk-bedrock-runtime`` lazily, so these exercise the event protocol
(``_start_session``) and the response-handling logic (``_handle_event``) directly.
"""

import base64
import json

from eva.models.config import BedrockS2SSimulatorConfig
from eva.user_simulator.bedrock_s2s import BedrockS2SUserSimulator
from eva.user_simulator.factory import create_user_simulator


async def _noop_sleep(_seconds):
    return None


class _FakeBridge:
    def __init__(self, caller_playing: bool = False):
        self.outputs: list[bytes] = []
        self._caller_playing = caller_playing
        self.utterance_complete_calls = 0

    def output(self, pcm: bytes) -> None:
        self.outputs.append(pcm)

    def notify_user_utterance_complete(self) -> None:
        self.utterance_complete_calls += 1

    def is_caller_playing(self) -> bool:
        return self._caller_playing


class _FakeInputStream:
    def __init__(self):
        self.sent: list[bytes] = []

    async def send(self, chunk) -> None:
        self.sent.append(chunk)

    async def close(self) -> None:
        pass


class _FakeStream:
    def __init__(self):
        self.input_stream = _FakeInputStream()


_GOAL = {
    "high_level_user_goal": "Reset my password.",
    "decision_tree": {
        "must_have_criteria": ["Reset the password."],
        "nice_to_have_criteria": [],
        "negotiation_behavior": ["Accept a valid resolution."],
        "resolution_condition": "The password is reset.",
        "failure_condition": "The password cannot be reset.",
        "escalation_behavior": "Escalate if blocked.",
        "edge_cases": [],
    },
    "information_required": {"employee_id": "12345"},
    "starting_utterance": "I need to reset my password.",
}


def _make_sim(tmp_path, persona_id: int = 1):
    cfg = BedrockS2SSimulatorConfig()
    return create_user_simulator(
        cfg,
        current_date_time="now",
        persona_config={"user_persona": "frustrated user", "user_persona_id": persona_id, "name": "Ruth Callahan"},
        goal=_GOAL,
        server_url="ws://localhost:1/ws",
        output_dir=tmp_path,
        agent_id="agent_itsm",
        timeout=10,
        perturbation_config=None,
        language="en",
    )


def _sent_events(stream: _FakeStream) -> list[dict]:
    """Decode the JSON `event` payloads pushed to the input stream."""
    return [json.loads(raw.decode("utf-8"))["event"] for raw in stream.input_stream.sent]


def test_config_and_factory_dispatch(tmp_path):
    cfg = BedrockS2SSimulatorConfig(model_id="amazon.nova-sonic-v1:0")
    assert cfg.provider == "bedrock_s2s"
    assert cfg.model_id == "amazon.nova-sonic-v1:0"
    sim = _make_sim(tmp_path)
    assert isinstance(sim, BedrockS2SUserSimulator)
    assert sim.provider == "bedrock_s2s"


def test_caller_voice_follows_persona_gender(tmp_path):
    assert _make_sim(tmp_path, persona_id=1).caller_voice == "tiffany"  # female
    assert _make_sim(tmp_path, persona_id=2).caller_voice == "matthew"  # male


async def test_start_session_sends_system_prompt_and_tool(tmp_path):
    sim = _make_sim(tmp_path)
    sim._stream = _FakeStream()
    sim._encode_chunk = lambda payload: payload  # identity: keep raw JSON bytes

    await sim._start_session()
    events = _sent_events(sim._stream)

    # Ordered init sequence.
    assert "sessionStart" in events[0]
    assert "promptStart" in events[1]
    assert [k for e in events for k in e] == [
        "sessionStart",
        "promptStart",
        "contentStart",  # SYSTEM text
        "textInput",
        "contentEnd",
        "contentStart",  # AUDIO input block
    ]

    prompt_start = events[1]["promptStart"]
    assert prompt_start["audioOutputConfiguration"]["voiceId"] == "tiffany"
    assert prompt_start["audioOutputConfiguration"]["sampleRateHertz"] == 16000
    tools = prompt_start["toolConfiguration"]["tools"]
    assert tools[0]["toolSpec"]["name"] == "end_call"

    # System prompt is delivered as a SYSTEM text block carrying the persona+goal template.
    sys_start = events[2]["contentStart"]
    assert sys_start["type"] == "TEXT" and sys_start["role"] == "SYSTEM"
    assert "password" in events[3]["textInput"]["content"].lower()

    audio_start = events[5]["contentStart"]
    assert audio_start["type"] == "AUDIO" and audio_start["role"] == "USER"
    assert audio_start["audioInputConfiguration"]["sampleRateHertz"] == 16000


async def test_role_mapping_audio_and_end_call(tmp_path, monkeypatch):
    monkeypatch.setattr("eva.user_simulator.bedrock_s2s.asyncio.sleep", _noop_sleep)
    sim = _make_sim(tmp_path)
    sim._audio_interface = _FakeBridge()

    spoken = {"user": [], "assistant": []}
    sim._on_user_speaks = lambda t: spoken["user"].append(t)
    sim._on_assistant_speaks = lambda t: spoken["assistant"].append(t)

    caller_pcm = b"\x01\x02" * 80

    async def feed(event: dict) -> None:
        await sim._handle_event({"event": event})

    # 1. Nova's ASR of the agent's speech (role USER) -> assistant_speech.
    await feed({"contentStart": {"type": "TEXT", "role": "USER"}})
    await feed({"textOutput": {"role": "USER", "content": "Hi, this is IT support, how can I help?"}})
    # 2. Caller text: SPECULATIVE preview is ignored, only FINAL is recorded as user_speech.
    await feed(
        {
            "contentStart": {
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
            }
        }
    )
    await feed({"textOutput": {"role": "ASSISTANT", "content": "please reset my (preview)"}})
    await feed(
        {"contentStart": {"type": "TEXT", "role": "ASSISTANT", "additionalModelFields": '{"generationStage":"FINAL"}'}}
    )
    await feed({"textOutput": {"role": "ASSISTANT", "content": "Please reset my password, employee ID 12345."}})
    # 3. Caller audio streams back and plays over the bridge.
    await feed({"contentStart": {"type": "AUDIO", "role": "ASSISTANT"}})
    await feed({"audioOutput": {"content": base64.b64encode(caller_pcm).decode("ascii")}})
    assert sim._caller_response_active is True
    # 4. end_call tool, then the caller-audio turn ends.
    await feed({"toolUse": {"toolName": "end_call", "content": "{}"}})
    await feed({"contentEnd": {"type": "AUDIO", "stopReason": "END_TURN"}})

    assert spoken["assistant"] == ["Hi, this is IT support, how can I help?"]
    # Caller text buffered across SPECULATIVE->FINAL and flushed once at turn end.
    assert spoken["user"] == ["Please reset my password, employee ID 12345."]
    assert sim._audio_interface.outputs == [caller_pcm]
    # Turn end must signal the bridge, or is_caller_playing() wedges and starves Nova.
    assert sim._audio_interface.utterance_complete_calls == 1
    assert sim._end_reason == "goodbye"
    assert sim._conversation_done.is_set()
    assert sim._caller_response_active is False  # released after the turn drained


def test_half_duplex_suppresses_agent_audio_while_caller_speaks(tmp_path):
    mulaw = b"\xff" * 160  # μ-law silence, valid frame

    # Caller is speaking -> agent audio is dropped (not queued to Nova).
    sim = _make_sim(tmp_path)
    sim._audio_interface = _FakeBridge(caller_playing=True)
    sim._on_agent_audio(mulaw)
    assert sim._agent_audio_queue.empty()

    # Caller idle -> agent audio is decoded, resampled, and queued.
    sim2 = _make_sim(tmp_path)
    sim2._audio_interface = _FakeBridge(caller_playing=False)
    sim2._on_agent_audio(mulaw)
    assert sim2._agent_audio_queue.qsize() == 1
    assert isinstance(sim2._agent_audio_queue.get_nowait(), bytes)
