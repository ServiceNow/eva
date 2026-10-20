from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.cascade import simulator as cascade_simulator
from eva.user_simulator.cascade.simulator import CascadeUserSimulator, extract_turn, next_segment
from eva.user_simulator.cascade.stt import ScribeStreamingSTT
from eva.user_simulator.cascade.tts import CartesiaStreamingTTS

GOAL = {
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
}


@pytest.fixture(autouse=True)
def _no_router(monkeypatch):
    monkeypatch.setattr(cascade_simulator, "LiteLLMClient", MagicMock())
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("CARTESIA_API_KEY", "test-key")


def _simulator(tmp_path: Path, *, persona_id: int = 1, **kwargs) -> CascadeUserSimulator:
    return CascadeUserSimulator(
        current_date_time="2026-06-05T12:00:00",
        persona_config={"user_persona_id": persona_id},
        goal=GOAL,
        server_url="ws://localhost:9999/ws",
        output_dir=tmp_path,
        agent_id="agent_itsm",
        simulator_config=CascadeSimulatorConfig(),
        **kwargs,
    )


def _tool_message(content: str, *tool_names: str) -> SimpleNamespace:
    calls = [SimpleNamespace(function=SimpleNamespace(name=name)) for name in tool_names]
    return SimpleNamespace(content=content, tool_calls=calls)


async def _chunks(*chunks: bytes):
    for chunk in chunks:
        yield chunk


def test_config_rejects_non_streaming_stt():
    with pytest.raises(ValidationError, match="streaming STT"):
        CascadeSimulatorConfig(stt_model="scribe_v2")


def test_accent_variants_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="ElevenLabs-specific"):
        _simulator(tmp_path, perturbation_config=PerturbationConfig(accent="french"))


def test_persona_selects_voice(tmp_path):
    config = CascadeSimulatorConfig()

    assert _simulator(tmp_path / "f").caller_voice == config.female_voice
    assert _simulator(tmp_path / "m", persona_id=2).caller_voice == config.male_voice


def test_regional_language_uses_base_code(tmp_path):
    simulator = _simulator(tmp_path, language="fr-CA")

    assert simulator._stt.language == "fr"
    assert simulator._tts.language == "fr"


def test_extract_turn_handles_text_and_tool_calls():
    assert extract_turn("```\nHi there.\n```") == ("Hi there.", False)
    assert extract_turn(_tool_message("Bye.", "end_call")) == ("Bye.", True)
    assert extract_turn(_tool_message("", "other")) == ("", False)


def test_messages_flip_roles_into_caller_frame(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._history = [
        {"role": "assistant", "content": "How can I help?"},
        {"role": "user", "content": "Reset my password."},
    ]

    messages = simulator._messages()

    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant"]


def test_stt_accumulates_committed_segments_and_partials():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt.expect_commit()

    stt._on_committed({"text": "Hello,"})
    stt._on_partial({"text": "how can"})

    assert stt.heard_text() == "Hello, how can"
    assert stt.take() == "Hello,"
    assert stt.partial == "how can"
    assert stt.take(include_partial=True) == "how can"
    assert stt.heard_text() == ""


@pytest.mark.asyncio
async def test_stt_send_raises_after_provider_error():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._on_error({"error": "auth failed"})

    with pytest.raises(RuntimeError, match="auth failed"):
        await stt.send(b"\xff")


@pytest.mark.asyncio
async def test_stt_waits_for_every_pending_commit():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt.expect_commit()
    stt.expect_commit()

    stt._on_committed({"text": "One moment."})
    assert not stt.committed_event.is_set()
    stt._on_committed({"text": "I found it."})

    assert stt.committed_event.is_set()
    assert stt.take() == "One moment. I found it."


@pytest.mark.asyncio
async def test_stt_drops_late_final_after_partial_fallback():
    dropped: list[str] = []
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x", on_dropped_final=dropped.append)
    stt.expect_commit()
    stt._on_partial({"text": "your ticket is"})
    assert stt.take(include_partial=True) == "your ticket is"

    stt._on_committed({"text": "your ticket is open."})
    stt.expect_commit()
    stt._on_committed({"text": "Anything else?"})

    assert stt.take() == "Anything else?"
    assert dropped == ["your ticket is open."]


@pytest.mark.asyncio
async def test_stt_fails_on_final_with_no_commit_outstanding():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._on_committed({"text": "Where did this come from?"})

    assert stt.take() == ""
    with pytest.raises(RuntimeError, match="no commit outstanding"):
        await stt.send(b"\xff")


@pytest.mark.asyncio
async def test_commit_is_queued_behind_the_next_bridge_chunk(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._stt = MagicMock(send=AsyncMock(), commit=AsyncMock())
    order: list[str] = []
    simulator._stt.send.side_effect = lambda chunk: order.append(f"send:{chunk.decode()}")
    simulator._stt.commit.side_effect = lambda: order.append("commit")

    simulator._on_assistant_audio(b"speech")
    simulator._request_commit()
    simulator._on_assistant_audio(b"tail")
    simulator._on_assistant_audio(b"silence")
    forwarder = asyncio.create_task(simulator._forward_assistant_audio())
    while simulator._assistant_audio_queue.qsize():
        await asyncio.sleep(0)
    forwarder.cancel()

    assert order == ["send:speech", "send:tail", "commit", "send:silence"]


@pytest.mark.asyncio
async def test_error_signals_assistant_and_ends_with_session_ended(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._stt.start = AsyncMock(side_effect=RuntimeError("boom"))
    simulator._stt.stop = AsyncMock()
    simulator._tts.close = AsyncMock()
    simulator._save_clean_user_audio = MagicMock()
    signaled: list[str | None] = []
    simulator.on_conversation_ending = signaled.append

    assert await simulator.run_conversation() == "error"

    assert signaled == ["error"]
    events = [json.loads(line) for line in (tmp_path / "user_simulator_events.jsonl").read_text().splitlines()]
    assert events[-1]["type"] == "connection_state"
    assert events[-1]["data"]["details"]["reason"] == "error"


@pytest.mark.asyncio
async def test_collect_heard_text_falls_back_to_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_simulator, "TRANSCRIPT_WAIT_SECONDS", 0.01)
    simulator = _simulator(tmp_path)
    simulator._stt._on_partial({"text": "your ticket is"})

    heard = await simulator._collect_heard_text(asyncio.get_running_loop().time())

    assert heard == "your ticket is"


def _llm_stream(*deltas: str, message=None):
    """A complete_stream stand-in: text deltas, then the assembled final message."""

    async def stream(**_):
        for delta in deltas:
            yield ("delta", delta)
        yield ("final", (message if message is not None else "".join(deltas), {}))

    return stream


def test_next_segment_speaks_first_clause_then_sentences():
    assert next_segment("I have my ID, and", first=True, final=False) == ("I have my ID,", " and")
    # Too few words for an early clause; later segments only end at sentences.
    assert next_segment("Yes, my ID", first=True, final=False) == ("", "Yes, my ID")
    assert next_segment("It is 42, thanks. More", first=False, final=False) == ("It is 42, thanks.", " More")
    # A break needs following whitespace, so codes and decimals never split.
    assert next_segment("Code MRC-07 or 4.5 now", first=True, final=False) == ("", "Code MRC-07 or 4.5 now")
    assert next_segment("trailing words", first=False, final=True) == ("trailing words", "")


@pytest.mark.asyncio
async def test_take_turn_speaks_whole_line_in_one_request(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._llm.complete_stream = _llm_stream("I have it here, ", "my ID is 42. ", "Thanks.")
    rendered: list[str] = []

    def stream(text, voice_id):
        rendered.append(text)
        return _chunks(b"\x01\x02", b"\x03\x04")

    simulator._tts.stream = stream
    simulator._audio_interface = MagicMock()

    hung_up = await simulator._take_turn("What is your employee ID?")

    assert hung_up is False
    assert rendered == ["I have it here, my ID is 42. Thanks."]
    outputs = [call.args[0] for call in simulator._audio_interface.output.call_args_list]
    assert outputs == [b"\x01\x02", b"\x03\x04", b"\x00\x00"]
    simulator._audio_interface.notify_user_utterance_complete.assert_called_once()
    assert simulator._history == [
        {"role": "assistant", "content": "What is your employee ID?"},
        {"role": "user", "content": "I have it here, my ID is 42. Thanks."},
    ]


@pytest.mark.asyncio
async def test_take_turn_asks_to_repeat_untranscribed_speech(tmp_path):
    simulator = _simulator(tmp_path)
    sent: list[list[dict]] = []

    async def stream(messages, **_):
        sent.append(messages)
        yield ("final", ("Sorry, could you repeat that?", {}))

    simulator._llm.complete_stream = stream
    simulator._tts.stream = lambda text, voice_id: _chunks(b"\x01\x02")
    simulator._audio_interface = MagicMock()
    simulator._on_assistant_speaks = MagicMock()

    assert await simulator._take_turn(None) is False

    assert sent[0][-1] == {"role": "system", "content": cascade_simulator.INAUDIBLE_NOTE}
    simulator._on_assistant_speaks.assert_not_called()  # nothing made up lands in the transcript
    assert simulator._history == [
        {"role": "assistant", "content": cascade_simulator.INAUDIBLE},
        {"role": "user", "content": "Sorry, could you repeat that?"},
    ]


@pytest.mark.asyncio
async def test_end_call_without_speech_skips_tts(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._llm.complete_stream = _llm_stream(message=_tool_message("", "end_call"))
    simulator._tts.stream = MagicMock()
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_caller_playing.return_value = False

    assert await simulator._take_turn("Goodbye.") is True
    simulator._tts.stream.assert_not_called()
    simulator._audio_interface.output.assert_not_called()


@pytest.mark.asyncio
async def test_end_call_hangs_up_after_playback(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._llm.complete_stream = _llm_stream("Thanks, bye.", message=_tool_message("Thanks, bye.", "end_call"))
    simulator._tts.stream = MagicMock(return_value=_chunks(b"\x01\x02"))
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_caller_playing.return_value = False

    hung_up = await simulator._take_turn("Your password is reset.")

    assert hung_up is True
    assert simulator._end_reason == "goodbye"


def _sse(*events: dict) -> httpx.MockTransport:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.MockTransport(lambda request: httpx.Response(200, text=body))


@pytest.mark.asyncio
async def test_tts_streams_sample_aligned_chunks_and_raises_on_error():
    tts = CartesiaStreamingTTS("sonic-3.5", api_key="x")
    odd = base64.b64encode(b"\x01\x02\x03").decode()
    tail = base64.b64encode(b"\x04").decode()
    tts._client = httpx.AsyncClient(transport=_sse({"type": "chunk", "data": odd}, {"type": "chunk", "data": tail}))
    assert [chunk async for chunk in tts.stream("Hi", voice_id="v")] == [b"\x01\x02", b"\x03\x04"]
    # Normal turns speak at the voice's natural speed; CALLER_SPEED is reserved for barge-ins.
    assert "generation_config" not in tts._body("Hi", "v")

    tts._client = httpx.AsyncClient(transport=_sse({"type": "error", "error": "bad voice"}))
    with pytest.raises(RuntimeError, match="bad voice"):
        [chunk async for chunk in tts.stream("Hi", voice_id="v")]


@pytest.mark.asyncio
async def test_stt_ignores_errors_from_its_own_close():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._connection = MagicMock(close=AsyncMock(side_effect=lambda: stt._on_error({"error": "sent 1000 (OK)"})))
    await stt.stop()
    assert stt.error is None


@pytest.mark.asyncio
async def test_stt_sends_bridge_mulaw_as_16k_pcm16():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._connection = MagicMock(send=AsyncMock())
    for _ in range(50):
        await stt.send(b"\xff" * 160)  # 20ms of 8kHz mu-law silence each, 1s total
    sent = sum(len(base64.b64decode(c.args[0]["audio_base_64"])) for c in stt._connection.send.call_args_list)
    # 1s of 16kHz PCM16 is 32000 bytes; the resampler carries a sample of state between chunks.
    assert abs(sent - 32000) <= 4
