"""Tests for the Cartesia managed server's session_create, transcript pieces, segments and audio mix."""

import asyncio
import base64
import json
import time
from unittest.mock import patch

import numpy as np

from eva.assistant.cartesia_managed_server import CartesiaManagedAssistantServer, _mix_pcm16, _SpeechSegment
from eva.utils.audio_utils import pcm16_mix


def _bare_server() -> CartesiaManagedAssistantServer:
    srv = object.__new__(CartesiaManagedAssistantServer)
    srv._system_prompt = "you are a helpful agent"
    srv.initial_message = "Hello, how can I help you today?"
    return srv


def test_session_create_streams_pcm_and_sends_prompt_and_greeting():
    assert _bare_server()._session_create() == {
        "type": "session_create",
        "audio": {"input_format": "pcm_16000", "continuous_input": True},
        "dynamic_variables": {
            "system_prompt": "you are a helpful agent",
            "initial_message": "Hello, how can I help you today?",
        },
    }


def test_speech_pieces_split_at_messages_and_tool_calls(monkeypatch):
    segment = _SpeechSegment(1000)
    segment.add_text("One moment.")
    monkeypatch.setattr("eva.assistant.cartesia_managed_server._now_ms", lambda: 2500)
    segment.add_text("Checking now")
    segment.end_piece()  # tool call
    monkeypatch.setattr("eva.assistant.cartesia_managed_server._now_ms", lambda: 4000)
    segment.add_text(" and you are all set.")
    segment.end_piece()
    assert segment.take_pieces() == [("One moment.", 1000), ("Checking now", 2500), ("and you are all set.", 4000)]


def test_mix_matches_eva_mix_with_clipping_and_unequal_lengths():
    rng = np.random.default_rng(0)
    a = rng.integers(-32768, 32767, 1000, dtype=np.int16).tobytes()
    b = rng.integers(-32768, 32767, 700, dtype=np.int16).tobytes()
    assert _mix_pcm16(a, b) == pcm16_mix(a, b)
    assert _mix_pcm16(b, a) == pcm16_mix(b, a)


class _Recorder:
    """Records method calls, standing in for the audit and framework logs."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


class _Simulator:
    """Sends (seconds from session start, message) pairs from the user simulator, then stays silent."""

    def __init__(self, script):
        self.script = list(script)
        self.started = None

    async def receive_text(self):
        self.started = self.started or time.monotonic()
        if not self.script:
            await asyncio.Future()
        at, message = self.script[0]
        await asyncio.sleep(max(0.0, self.started + at - time.monotonic()))
        self.script.pop(0)
        return json.dumps(message)

    async def send_text(self, text):
        pass


class _Agent:
    """Replays (delay seconds, event) pairs as the agent WebSocket, then closes shortly after."""

    close_code, close_reason = 1000, ""

    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def send(self, text):
        pass

    async def recv(self):
        return json.dumps({"type": "session_ready", "call_id": "call_test"})

    async def __aiter__(self):
        for delay, event in self.events:
            await asyncio.sleep(delay)
            yield json.dumps(event)
        await asyncio.sleep(0.1)


def _audio(seconds: float) -> dict:
    pcm = np.full(int(seconds * 16000), 1000, dtype=np.int16).tobytes()
    return {"type": "audio_output", "audio": base64.b64encode(pcm).decode()}


def _text(text: str) -> dict:
    return {"type": "turn_output_text_delta", "turn": 2, "role": "assistant", "text": text}


def _ended(text: str, interrupted: bool = False, tool_calls: tuple = ()) -> dict:
    calls = [{"name": name, "arguments": {}} for name in tool_calls]
    return {
        "type": "turn_ended",
        "turn": 2,
        "role": "assistant",
        "text": text,
        "interrupted": interrupted,
        "tool_calls": calls,
    }


async def _session(events, script=()) -> list[tuple]:
    """Runs one session against the replayed agent and simulator; returns the framework log calls."""
    srv = _bare_server()
    srv.conversation_id, srv._running, srv._model = "conv_test", True, "managed-test"
    srv._base_url, srv._cartesia_agent_id, srv._cartesia_version, srv._api_key = (
        "ws://agent",
        "agent_test",
        "test",
        "key",
    )
    srv.user_audio_buffer, srv.assistant_audio_buffer = bytearray(), bytearray()
    srv.audit_log, srv._fw_log, srv._metrics_log = _Recorder(), _Recorder(), _Recorder()
    with patch("eva.assistant.cartesia_managed_server.websockets.connect", return_value=_Agent(events)):
        await srv._handle_session(_Simulator(script))
    return srv._fw_log.calls


def _responses(calls) -> list[str]:
    return [args[0] for name, args, _ in calls if name == "llm_response"]


def _turn_starts(calls) -> int:
    return sum(name == "turn_start" for name, _, _ in calls)


async def test_each_spoken_agent_turn_is_one_response():
    # A pause inside a turn does not split it; speech before a tool call and the reply after it are separate turns.
    calls = await _session(
        [
            (0, _audio(0.1)),
            (0, _text("One")),
            (0.6, _audio(0.1)),
            (0, _text(" moment.")),
            (0, _ended("", tool_calls=("lookup",))),
            (0, _ended("One moment.")),
            (0.2, _audio(0.1)),
            (0, _text("You are all set.")),
            (0, _ended("You are all set.")),
        ]
    )
    assert _responses(calls) == ["One moment.", "You are all set."]
    assert _turn_starts(calls) == 1


async def test_caller_speech_starts_a_new_turn_and_interruptions_are_marked():
    calls = await _session(
        [
            (0, _audio(1.0)),
            (0, _text("Let me explain the")),
            (0.1, {"type": "audio_output_clear"}),
            (0, _ended("Let me explain the", interrupted=True)),
            (0.3, _audio(0.1)),
            (0, _text("Sure.")),
            (0, _ended("Sure.")),
        ],
        script=[(0.05, {"event": "user_speech_start", "timestamp_ms": "1"})],
    )
    assert _responses(calls) == ["Let me explain the", "Sure."]
    assert _turn_starts(calls) == 2
    assert [kwargs["was_interrupted"] for name, _, kwargs in calls if name == "turn_end"] == [True, False]
