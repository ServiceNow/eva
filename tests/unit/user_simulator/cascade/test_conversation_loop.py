import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eva.models.config import CascadeSimulatorConfig
from eva.user_simulator.cascade import simulator as module
from eva.user_simulator.cascade.constants import BYTES_PER_TICK
from eva.user_simulator.cascade.stt import TranscriptBuffer
from eva.user_simulator.cascade.tick_result import TickResult


def make_conversation(monkeypatch, tmp_path, *, speech, partial=False, stall_at=None):
    buffer = TranscriptBuffer()
    adapter = SimpleNamespace(tick=-1, sent=[], start=AsyncMock(), stop=AsyncMock())

    async def run_tick(tick_number, outgoing_audio, **kwargs):
        adapter.tick = tick_number
        adapter.sent.append((tick_number, outgoing_audio))
        speaking = tick_number in speech
        return TickResult(
            tick_number=tick_number,
            assistant_audio=(b"\x01\x00" if speaking else b"\x00\x00") * (BYTES_PER_TICK // 2),
            assistant_audio_raw_bytes=BYTES_PER_TICK if speaking else 0,
            wall_clock_ms=tick_number * 200,
            provider_stalled=tick_number == stall_at,
        )

    async def feed(pcm, *, commit=False):
        text = speech.get(adapter.tick)
        if text:
            buffer.apply_partial(text)
        if commit and not partial and buffer.in_flight:
            buffer.commit(buffer.in_flight)

    adapter.run_tick = run_tick
    stt = SimpleNamespace(buffer=buffer, start=AsyncMock(), stop=AsyncMock(), feed=feed)
    llm = SimpleNamespace(calls=[])

    async def complete(*, messages, tools=None):
        llm.calls.append((adapter.tick, messages, tools))
        if len(llm.calls) == 1:
            return "Please unlock my account.", None
        return SimpleNamespace(
            content="", tool_calls=[SimpleNamespace(function=SimpleNamespace(name="end_call"))]
        ), None

    llm.complete = complete
    tts = SimpleNamespace(
        voice_for_persona=lambda persona: "voice",
        synthesize=AsyncMock(return_value=b"\x02\x00" * (BYTES_PER_TICK // 2)),
    )
    monkeypatch.setattr(module, "LiveKitStreamingSTT", lambda *a, **kw: stt)
    monkeypatch.setattr(module, "LiteLLMClient", lambda **kw: llm)
    monkeypatch.setattr(module, "CartesiaTTS", lambda *a, **kw: tts)
    monkeypatch.setattr(module, "adapter_class_for_framework", lambda framework: lambda **kw: adapter)
    monkeypatch.setattr(module.websockets, "connect", AsyncMock(return_value=object()))
    sim = module.CascadeUserSimulator(
        current_date_time="2026-09-16T12:00:00",
        persona_config={},
        goal={},
        server_url="ws://test",
        output_dir=tmp_path,
        agent_id="agent_itsm",
        timeout=20,
        simulator_config=CascadeSimulatorConfig(),
    )
    sim._build_prompt = lambda: "Help the caller unlock their account."
    return sim, adapter, stt, llm


@pytest.mark.parametrize("partial", [False, True])
async def test_normal_conversation_waits_for_replies_and_hangs_up(monkeypatch, tmp_path, partial):
    sim, adapter, stt, llm = make_conversation(
        monkeypatch, tmp_path, speech={3: "How can I help?", 42: "Your account is unlocked."}, partial=partial
    )

    assert await sim.run_conversation() == "goodbye"

    assert len(llm.calls) == 2
    assert llm.calls[0][0] > 3
    assert llm.calls[1][0] > 42
    assert llm.calls[0][1][-1] == {"role": "user", "content": "How can I help?"}
    assert llm.calls[1][1][-1] == {"role": "user", "content": "Your account is unlocked."}
    assert llm.calls[0][2][0]["function"]["name"] == "end_call"
    sent_ticks = [tick for tick, audio in adapter.sent if audio is not None]
    assert sent_ticks == [llm.calls[0][0] + 1]
    assert (tmp_path / "audio_user_clean.wav").exists()
    events = [json.loads(line) for line in (tmp_path / "user_simulator_events.jsonl").read_text().splitlines()]
    assert sum(event.get("type") == "caller_turn" for event in events) == 1
    assert any(event.get("event_type") == "audio_start" for event in events)
    assert any(event.get("event_type") == "audio_end" for event in events)
    assert any(event.get("type") == "transcript_partial_fallback" for event in events) is partial
    assert events[-1]["data"]["details"]["reason"] == "goodbye"
    stt.stop.assert_awaited_once()
    adapter.stop.assert_awaited_once()


async def test_loop_resets_inactivity_on_each_assistant_speech_tick(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "INACTIVITY_TIMEOUT_MS", 1600)
    sim, adapter, _, llm = make_conversation(monkeypatch, tmp_path, speech={1: "", 7: "", 13: ""})

    assert await sim.run_conversation() == "inactivity_timeout"

    assert adapter.tick == 22
    assert llm.calls == []


async def test_provider_stall_preserves_partial_artifacts_and_signals_failure(monkeypatch, tmp_path):
    sim, adapter, stt, llm = make_conversation(monkeypatch, tmp_path, speech={3: "How can I help?"}, stall_at=20)
    endings = []
    sim.on_conversation_ending = endings.append

    assert await sim.run_conversation() == "provider_stalled"

    assert endings == ["provider_stalled"]
    assert len(llm.calls) == 1
    assert (tmp_path / "audio_user_clean.wav").exists()
    events = [json.loads(line) for line in (tmp_path / "user_simulator_events.jsonl").read_text().splitlines()]
    assert events[-1]["data"]["details"]["reason"] == "provider_stalled"
    assert not any(event.get("type") == "error" for event in events)
    assert (tmp_path / "user_simulator_decisions.jsonl").exists()
    stt.stop.assert_awaited_once()
    adapter.stop.assert_awaited_once()
