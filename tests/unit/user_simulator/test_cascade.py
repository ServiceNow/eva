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
from eva.user_simulator.cascade.barge_in import MIN_GAP_MS, BargeIn, trim_silence
from eva.user_simulator.cascade.interruptor import DecisionsInterruptor
from eva.user_simulator.cascade.simulator import CascadeUserSimulator, extract_turn, next_segment
from eva.user_simulator.cascade.stt import OpenAIStreamingSTT, ScribeStreamingSTT
from eva.user_simulator.cascade.tts import TTS_SAMPLE_RATE, CartesiaStreamingTTS

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

SILENT_PCM = bytes(640)
"""20ms of PCM16 silence."""
LOUD_PCM = b"\x00\x40" * 320
"""20ms of PCM16 well above the silence threshold."""


@pytest.fixture(autouse=True)
def _no_router(monkeypatch):
    monkeypatch.setattr(cascade_simulator, "LiteLLMClient", MagicMock(side_effect=lambda **_: MagicMock()))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cascade_simulator, "DecisionsInterruptor", MagicMock())


def _simulator(tmp_path: Path, *, persona_id: int = 1, **kwargs) -> CascadeUserSimulator:
    return CascadeUserSimulator(
        current_date_time="2026-06-05T12:00:00",
        persona_config={"user_persona_id": persona_id},
        goal=GOAL,
        server_url="ws://localhost:9999/ws",
        output_dir=tmp_path,
        agent_id="agent_itsm",
        simulator_config=CascadeSimulatorConfig(model="gpt-test", stt="elevenlabs"),
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
        CascadeSimulatorConfig(model="gpt-test", stt="elevenlabs", stt_model="scribe_v2")


def test_accent_variants_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="ElevenLabs-specific"):
        _simulator(tmp_path, perturbation_config=PerturbationConfig(accent="french"))


def test_persona_selects_voice(tmp_path):
    config = CascadeSimulatorConfig(model="gpt-test")

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


def test_system_prompt_is_built_once(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._build_prompt = MagicMock(return_value="prompt")

    simulator._messages()
    simulator._messages()

    simulator._build_prompt.assert_called_once()


def test_stt_accumulates_committed_segments_and_partials():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt.expect_commit()

    stt._on_committed({"text": "Hello,"})
    stt._on_partial({"text": "how can"})

    assert stt.heard_text() == "Hello, how can"
    assert stt.take() == "Hello,"
    assert stt.partial == "how can"
    assert stt.heard_text() == "how can"


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


def test_stt_keeps_final_with_no_commit_outstanding():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._on_committed({"text": "Scribe committed this itself."})

    assert stt.error is None
    assert stt.take() == "Scribe committed this itself."


@pytest.mark.asyncio
async def test_stt_pads_a_commit_that_is_too_short_for_scribe():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._connection = AsyncMock()
    await stt.send(b"\xff" * 80)  # 10ms of audio

    await stt.commit()

    padded = sum(len(base64.b64decode(c.args[0]["audio_base_64"])) for c in stt._connection.send.await_args_list)
    assert padded >= 0.3 * 16000 * 2
    stt._connection.commit.assert_awaited_once()


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
async def test_collect_heard_text_fails_when_the_final_never_arrives(tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_simulator, "TRANSCRIPT_WAIT_SECONDS", 0.01)
    simulator = _simulator(tmp_path)
    simulator._stt.expect_commit()
    simulator._stt._on_partial({"text": "your ticket is"})

    with pytest.raises(RuntimeError, match="No final transcript"):
        await simulator._collect_heard_text(asyncio.get_running_loop().time())


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
    # Never right after a spelled-out letter or digit, which would be said last in its own request.
    assert next_segment("The code is Z K three F F W, not V", first=True, final=False) == (
        "",
        "The code is Z K three F F W, not V",
    )


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
async def test_keep_listening_stays_silent_until_the_agent_finishes(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._llm.complete_stream = _llm_stream(message=_tool_message("", "keep_listening"))
    simulator._tts.stream = MagicMock()
    simulator._audio_interface = MagicMock()

    assert await simulator._take_turn("Let me check that.") is False

    simulator._tts.stream.assert_not_called()
    simulator._audio_interface.output.assert_not_called()
    assert simulator._history == [{"role": "assistant", "content": "Let me check that."}]


@pytest.mark.asyncio
async def test_keep_listening_is_withheld_after_repeated_waits(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._kept_listening = cascade_simulator.MAX_KEEP_LISTENING_IN_A_ROW
    offered: list[str] = []

    async def stream(*, tools, **_):
        offered.extend(t["function"]["name"] for t in tools)
        yield ("final", ("Okay.", {}))

    simulator._llm.complete_stream = stream
    simulator._tts.stream = MagicMock(return_value=_chunks())
    simulator._audio_interface = MagicMock()

    await simulator._take_turn("Anything else?")

    assert offered == ["end_call"]
    assert simulator._kept_listening == 0


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


def _trigger() -> BargeIn:
    return BargeIn(decision_ms=0.0, p_interrupt=0.0)


@pytest.mark.asyncio
async def test_interrupt_decision_plays_opener_then_reply_once(tmp_path):
    simulator = _simulator(tmp_path)

    async def decide(goal: str, history: list, heard: str) -> tuple[str, float]:
        await asyncio.sleep(0)  # both requests are in flight before either answers
        return "interrupt", 0.9

    simulator._interruptor.decide = decide
    simulator._openers = [("Sorry\u2014", b"\x07\x07")]
    simulator._barge_in_llm.complete_stream = _llm_stream("It's 4471.")
    simulator._tts.stream = MagicMock(return_value=_chunks(SILENT_PCM, LOUD_PCM))
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = True
    simulator._audio_interface.is_caller_playing.return_value = False
    simulator._stt.partial = "Can I get your employee"

    # Two overlapping decisions both say interrupt; only the first barges in.
    await asyncio.gather(
        simulator._decide("goal", "Can I get", _trigger()), simulator._decide("goal", "Can I get your", _trigger())
    )

    # The reply's leading silence is dropped, so it follows the opener with no padding.
    outputs = [call.args[0] for call in simulator._audio_interface.output.call_args_list]
    assert [outputs[0], outputs[1].lstrip(b"\x00"), outputs[2]] == [b"\x07\x07", LOUD_PCM.lstrip(b"\x00"), b"\x00\x00"]
    assert len(outputs[1]) > len(LOUD_PCM)  # held off the opener by the minimum gap
    assert simulator._history[-2:] == [
        {"role": "assistant", "content": "Can I get your employee"},
        {"role": "user", "content": "Sorry— It's 4471."},  # the opener is part of what the caller said
    ]
    assert simulator._turn_active is False
    assert simulator._stt._stale_commits == 1  # the cut-off segment's final will be dropped
    [barge_in] = simulator.barge_ins
    assert (barge_in.heard, barge_in.content, barge_in.p_interrupt) == ("Can I get your employee", "It's 4471.", 0.9)
    assert barge_in.gap_ms is not None


@pytest.mark.asyncio
async def test_barge_in_opener_is_timed_from_its_first_frame_on_the_wire(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._openers = [("Sorry—", LOUD_PCM)]
    simulator._barge_in_llm.complete_stream = _llm_stream("It's 4471.")
    simulator._tts.stream = MagicMock(return_value=_chunks(LOUD_PCM))
    bridge = MagicMock(assistant_audio_starts=0, user_audio_starts=3, user_audio_started_at=None)
    bridge.is_assistant_playing.return_value = True

    def output(pcm: bytes) -> None:
        if bridge.user_audio_starts == 3:  # the opener starts an utterance, after work that held up the loop
            bridge.user_audio_starts, bridge.user_audio_started_at = 4, 100.7

    bridge.output.side_effect = output
    simulator._audio_interface = bridge
    barge_in = _trigger()

    await simulator._interrupt(barge_in)

    assert barge_in.opener_at == pytest.approx(100.7)


@pytest.mark.asyncio
async def test_barge_in_reply_streams_each_segment_to_tts_in_order(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._barge_in_llm.complete_stream = _llm_stream("My ID is 4471, ", "and my name ", "is Sam. Thanks.")
    rendered: list[str] = []
    audio = {"My ID is 4471,": b"\x00\x10" * 320, "and my name is Sam.": b"\x00\x20" * 320, "Thanks.": LOUD_PCM}

    def stream(text, voice_id):
        rendered.append(text)
        return _chunks(audio[text])

    simulator._tts.stream = stream
    simulator._audio_interface = MagicMock()
    barge_in = _trigger()
    barge_in.play_opener("Sorry\u2014", b"", now=0.0)

    message, _, spoke = await simulator._speak_barge_in([], barge_in)

    # The first clause goes out before the sentence it belongs to is finished.
    assert rendered == ["My ID is 4471,", "and my name is Sam.", "Thanks."]
    outputs = [call.args[0] for call in simulator._audio_interface.output.call_args_list]
    assert outputs == [audio[text] for text in rendered]
    assert spoke and barge_in.content == "My ID is 4471, and my name is Sam. Thanks."


def test_barge_in_times_the_gap_after_the_opener():
    barge_in = _trigger()
    barge_in.decision_ms = 150.0
    barge_in.play_opener("Sorry\u2014", bytes(TTS_SAMPLE_RATE // 5 * 2), now=10.2)  # 200ms opener

    assert barge_in.audible(SILENT_PCM, now=10.55) == b""
    assert barge_in.audible(SILENT_PCM + LOUD_PCM, now=10.6) == LOUD_PCM
    latency = barge_in.latency()

    assert latency.decision_ms == 150.0
    assert latency.gap_ms == pytest.approx(200.0)  # opener ended at 10.4, reply audible at 10.6


def test_barge_in_reply_ready_before_opener_ends_is_padded_to_the_min_gap():
    barge_in = _trigger()
    barge_in.play_opener("Sorry\u2014", bytes(TTS_SAMPLE_RATE * 2), now=0.1)  # 1s opener

    assert barge_in.audible(LOUD_PCM, now=0.5) == bytes(round(MIN_GAP_MS / 1000 * TTS_SAMPLE_RATE) * 2) + LOUD_PCM
    assert barge_in.latency().gap_ms == pytest.approx(MIN_GAP_MS)


def test_barge_in_reply_just_after_opener_is_padded_only_the_rest_of_the_way():
    barge_in = _trigger()
    barge_in.play_opener("Sorry\u2014", bytes(TTS_SAMPLE_RATE * 2), now=0.0)  # 1s opener

    assert barge_in.audible(LOUD_PCM, now=1.05) == bytes(round(0.08 * TTS_SAMPLE_RATE) * 2) + LOUD_PCM
    assert barge_in.latency().gap_ms == pytest.approx(MIN_GAP_MS)


def test_trim_silence_keeps_only_loud_windows():
    assert trim_silence(SILENT_PCM + LOUD_PCM + SILENT_PCM) == LOUD_PCM


def test_stt_take_with_partial_drops_the_finals_it_already_returned():
    stt = ScribeStreamingSTT("scribe_v2_realtime", api_key="x")
    stt._on_committed({"text": "Hello,"})
    stt._on_partial({"text": "can I get your"})
    stt.expect_commit()

    assert stt.take(include_partial=True) == "Hello, can I get your"
    assert stt.committed_event.is_set()
    stt._on_committed({"text": "can I get your ID?"})  # the barge-in's segment
    stt.expect_commit()
    stt._on_committed({"text": "Thanks."})

    assert stt.take() == "Thanks."


@pytest.mark.asyncio
async def test_no_interrupt_while_caller_is_speaking(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._interruptor.decide = AsyncMock(return_value=("interrupt", 0.9))
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = True
    simulator._audio_interface.is_caller_playing.return_value = True

    await simulator._decide("goal", "Can I get your", _trigger())

    simulator._audio_interface.output.assert_not_called()


def test_forced_interruptions_hit_every_nth_agent_turn_after_enough_updates(tmp_path):
    simulator = _simulator(tmp_path)
    assert not simulator._forces_interruption(4, 99)  # off by default
    simulator.simulator_config = CascadeSimulatorConfig(
        model="gpt-test", forced_interruption_every_turns=5, forced_interruption_after_updates=15
    )

    forced = [turn for turn in range(15) if simulator._forces_interruption(turn, 15)]
    assert forced == [4, 9, 14]
    assert not simulator._forces_interruption(4, 14)
    assert simulator._forces_interruption(4, 16)  # a later update still fires if the 15th could not


@pytest.mark.asyncio
async def test_forced_decision_barges_in_over_a_keep_listening_verdict(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._interruptor.decide = AsyncMock(return_value=("keep_listening", 0.1))
    simulator._openers = [("Sorry—", b"\x07\x07")]
    simulator._barge_in_llm.complete_stream = _llm_stream("It's 4471.")
    simulator._tts.stream = MagicMock(return_value=_chunks(LOUD_PCM))
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = True
    simulator._audio_interface.is_caller_playing.return_value = False
    simulator._stt.partial = "Can I get your employee"
    trigger = _trigger()
    trigger.forced = True

    await simulator._decide("goal", "Can I get your", trigger)

    simulator._interruptor.decide.assert_awaited_once()  # the real decision is still made
    [barge_in] = simulator.barge_ins
    assert (barge_in.forced, barge_in.p_interrupt) == (True, 0.1)
    assert simulator._caller_turns == 1


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


@pytest.mark.asyncio
async def test_assistant_line_spoken_during_caller_turn_gets_its_own_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_simulator, "TURN_SETTLE_SECONDS", 0)
    monkeypatch.setattr(cascade_simulator, "POLL_SECONDS", 0)
    simulator = _simulator(tmp_path)
    # The caller's own speech clears the end time, as the real bridge does on user audio start.
    bridge = MagicMock(assistant_audio_starts=0, assistant_audio_ended_at=None)
    bridge.is_assistant_playing.return_value = False
    bridge.is_caller_playing.return_value = False
    simulator._audio_interface = bridge
    simulator._request_commit = MagicMock()
    simulator._collect_heard_text = AsyncMock(side_effect=["What is your ID?", "One moment."])
    simulator.event_logger = MagicMock()
    heard: list[str] = []

    async def take_turn(text):
        heard.append(text)
        if len(heard) == 1:
            bridge.assistant_audio_starts += 1  # "One moment." starts and ends while the caller replies
            return False
        return True

    simulator._take_turn = take_turn

    async def first_pause():
        # Seed the loop as if the assistant just finished its first line.
        bridge.is_assistant_playing.return_value = True
        await asyncio.sleep(0)
        bridge.is_assistant_playing.return_value = False

    seeder = asyncio.create_task(first_pause())
    await asyncio.wait_for(simulator._run_turns(), timeout=1)
    await seeder

    assert heard == ["What is your ID?", "One moment."]
    simulator.event_logger.log_event.assert_called_with("assistant_spoke_during_caller_turn", {})


@pytest.mark.asyncio
async def test_repeated_untranscribed_agent_lines_fail_the_call(tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_simulator, "TURN_SETTLE_SECONDS", 0)
    monkeypatch.setattr(cascade_simulator, "POLL_SECONDS", 0)
    simulator = _simulator(tmp_path)
    bridge = MagicMock(assistant_audio_starts=0, assistant_audio_ended_at=None)
    bridge.is_caller_playing.return_value = False
    simulator._audio_interface = bridge
    simulator._request_commit = MagicMock()
    simulator.event_logger = MagicMock()
    simulator._collect_heard_text = AsyncMock(return_value="")
    turns: list[str | None] = []

    async def take_turn(text):
        turns.append(text)
        bridge.assistant_audio_starts += 1  # the agent repeats itself, still untranscribed
        return False

    simulator._take_turn = take_turn
    bridge.is_assistant_playing.side_effect = [True, *[False] * 1000]

    with pytest.raises(RuntimeError, match="2 agent lines in a row"):
        await asyncio.wait_for(simulator._run_turns(), timeout=1)

    assert turns == [None]  # asked to repeat once, then gave up


@pytest.mark.asyncio
async def test_decision_input_carries_goal_history_and_the_line_in_progress(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        choice = {"choice": "keep_listening", "probabilities": [{"value": "interrupt", "probability": 0.2}]}
        return httpx.Response(200, json={"answers": [choice]})

    interruptor = DecisionsInterruptor("gpt-test-decider")
    interruptor._client = httpx.AsyncClient(base_url="https://api.test", transport=httpx.MockTransport(answer))
    history = [{"role": "assistant", "content": "How can I help?"}, {"role": "user", "content": "Reset my password."}]

    assert await interruptor.decide("GOAL: reset", history, "Sure, can I get your") == ("keep_listening", 0.2)

    [body] = requests
    assert body["model"] == "gpt-test-decider"
    assert body["input"] == (
        "<user_goal>\nGOAL: reset\n</user_goal>\n\n"
        "<conversation>\nAgent: How can I help?\nCaller: Reset my password.\n</conversation>\n\n"
        "Agent (speaking now): Sure, can I get your"
    )


@pytest.mark.asyncio
async def test_talked_over_tail_of_the_interrupted_line_gets_no_second_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(cascade_simulator, "TURN_SETTLE_SECONDS", 0)
    monkeypatch.setattr(cascade_simulator, "POLL_SECONDS", 0)
    simulator = _simulator(tmp_path)
    bridge = MagicMock(assistant_audio_starts=0, assistant_audio_ended_at=None)
    bridge.is_assistant_playing.return_value = True
    bridge.is_caller_playing.return_value = False
    simulator._audio_interface = bridge
    simulator.event_logger = MagicMock()
    simulator._take_turn = AsyncMock(return_value=False)

    async def barge_in():
        # The agent's line runs on under the barge-in, then stops before the caller finishes.
        simulator._turn_active = True
        await asyncio.sleep(0.01)
        simulator._stt.partial = "and what"
        bridge.is_assistant_playing.return_value = False
        await asyncio.sleep(0.01)
        simulator._turn_active = False
        simulator._record_talked_over_tail(starts=0)
        await asyncio.sleep(0.01)
        simulator._conversation_done.set()

    simulator._history = [{"role": "user", "content": "Sorry— It's 4471."}]
    await asyncio.gather(simulator._run_turns(), barge_in())

    simulator._take_turn.assert_not_called()  # before, an empty final here read as an inaudible line
    simulator.event_logger.log_event.assert_called_with("barge_in_tail", {"text": "and what"})
    assert simulator._stt.heard_text() == ""
    # Heard, so it is in history after the barge-in, but not answered.
    assert simulator._history[-1] == {"role": "assistant", "content": "and what"}


def test_agent_speech_after_a_barge_in_is_kept_when_it_is_a_new_line(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._audio_interface = MagicMock(assistant_audio_starts=1)
    simulator._audio_interface.is_assistant_playing.return_value = False
    simulator.event_logger = MagicMock()
    simulator._stt.partial = "Yes, your identity is confirmed."

    simulator._record_talked_over_tail(starts=0)  # the agent answered the barge-in within it

    assert simulator._stt.heard_text() == "Yes, your identity is confirmed."
    assert simulator._tails_recorded == 0
    assert simulator._history == []  # left for a regular turn, which replies to it


@pytest.mark.asyncio
async def test_no_decision_is_requested_for_the_first_updates_of_an_agent_turn(tmp_path):
    simulator = _simulator(tmp_path)
    simulator.simulator_config = CascadeSimulatorConfig(
        model="gpt-test", interruption_first_turn=0, interruption_min_updates=2
    )
    simulator._can_interrupt = MagicMock(return_value=True)
    asked: list[str] = []

    async def decide(goal, heard, barge_in):
        asked.append(heard)

    simulator._decide = decide
    watcher = asyncio.create_task(simulator._watch_for_interruptions())
    for partial in ["Can", "Can I", "Can I get"]:
        simulator._stt.partial = partial
        simulator._stt.updated.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    simulator._caller_turns += 1  # a new agent turn starts the count over
    simulator._stt.partial = "Thanks"
    simulator._stt.updated.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    watcher.cancel()

    assert asked == ["Can I get"]


@pytest.mark.asyncio
async def test_no_decision_or_barge_in_after_the_caller_hangs_up(tmp_path):
    simulator = _simulator(tmp_path)
    simulator._llm.complete_stream = _llm_stream("Thanks, bye.", message=_tool_message("Thanks, bye.", "end_call"))
    simulator._tts.stream = MagicMock(return_value=_chunks(LOUD_PCM))
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = True  # the agent says goodbye back
    simulator._audio_interface.is_caller_playing.return_value = False
    simulator._wait_for_caller_playback = AsyncMock()
    simulator._on_conversation_end = MagicMock()

    assert await simulator._take_turn("Anything else?") is True
    assert simulator._can_interrupt() is False


@pytest.mark.asyncio
async def test_barge_in_llm_is_warmed_once_as_barge_ins_become_possible(tmp_path, monkeypatch):
    simulator = _simulator(tmp_path)
    simulator.simulator_config = CascadeSimulatorConfig(
        model="gpt-test", interruption_first_turn=0, interruption_min_updates=1
    )
    simulator._can_interrupt = MagicMock(return_value=True)
    simulator._decide = AsyncMock()
    acompletion = AsyncMock()
    monkeypatch.setattr(cascade_simulator.router, "get", lambda: SimpleNamespace(acompletion=acompletion))

    watcher = asyncio.create_task(simulator._watch_for_interruptions())
    for partial in ["Can", "Can I", "Can I get"]:
        simulator._stt.partial = partial
        simulator._stt.updated.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    watcher.cancel()

    acompletion.assert_awaited_once()
    assert acompletion.await_args.kwargs["max_tokens"] == 1


async def _updates(simulator, partials: list[str]) -> None:
    for partial in partials:
        simulator._stt.partial = partial
        simulator._stt.updated.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_no_decision_is_requested_before_the_first_interruptible_agent_turn(tmp_path):
    simulator = _simulator(tmp_path)
    simulator.simulator_config = CascadeSimulatorConfig(
        model="gpt-test", interruption_first_turn=2, interruption_min_updates=0
    )
    simulator._can_interrupt = MagicMock(return_value=True)
    asked: list[str] = []

    async def decide(goal, heard, barge_in):
        asked.append(heard)

    simulator._decide = decide
    watcher = asyncio.create_task(simulator._watch_for_interruptions())
    for turn, partial in enumerate(["Hello", "Sure", "Your flight"]):
        simulator._caller_turns = turn
        await _updates(simulator, [partial])
    watcher.cancel()

    assert asked == ["Your flight"]


@pytest.mark.asyncio
async def test_decisions_pause_for_the_cooldown_after_a_barge_in_and_stop_at_the_cap(tmp_path):
    simulator = _simulator(tmp_path)
    simulator.simulator_config = CascadeSimulatorConfig(
        model="gpt-test",
        interruption_first_turn=0,
        interruption_min_updates=0,
        interruption_cooldown_updates=2,
        interruption_max_per_call=2,
    )
    simulator._can_interrupt = MagicMock(return_value=True)
    asked: list[str] = []

    async def decide(goal, heard, barge_in):
        asked.append(heard)
        simulator._barge_ins += 1  # every decision here fires a barge-in

    simulator._decide = decide
    watcher = asyncio.create_task(simulator._watch_for_interruptions())
    await _updates(simulator, ["a", "a b", "a b c", "a b c d", "a b c d e", "a b c d e f", "a b c d e f g"])
    watcher.cancel()

    # "a" fires; two updates of cooldown; "a b c d" fires and reaches the cap.
    assert asked == ["a", "a b c d"]


def test_config_defaults_to_openai_stt():
    config = CascadeSimulatorConfig(model="gpt-test")

    assert (config.stt, config.stt_model) == ("openai", "gpt-live-transcribe")
    assert CascadeSimulatorConfig(model="gpt-test", stt="elevenlabs").stt_model == "scribe_v2_realtime"


def _delta(item: str, text: str) -> dict:
    return {"type": "conversation.item.input_audio_transcription.delta", "item_id": item, "delta": text}


def _acked(item: str) -> dict:
    return {"type": "input_audio_buffer.committed", "item_id": item}


def _completed(item: str, text: str) -> dict:
    return {"type": "conversation.item.input_audio_transcription.completed", "item_id": item, "transcript": text}


@pytest.mark.asyncio
async def test_openai_stt_accumulates_deltas_and_waits_for_every_commit():
    stt = OpenAIStreamingSTT("gpt-live-transcribe", api_key="x")
    stt.expect_commit()
    stt.expect_commit()
    stt._on_event(_acked("a"))
    stt._on_event(_acked("b"))

    stt._on_event(_delta("a", " One"))
    stt._on_event(_delta("a", " moment."))
    assert stt.heard_text() == "One moment."
    stt._on_event(_completed("a", "One moment."))
    stt._on_event(_delta("b", " I found"))
    assert not stt.committed_event.is_set()
    assert stt.heard_text() == "One moment. I found"
    stt._on_event(_completed("b", "I found it."))

    assert stt.committed_event.is_set()
    assert stt.take() == "One moment. I found it."


@pytest.mark.asyncio
async def test_openai_stt_drops_late_deltas_and_final_of_a_taken_segment():
    stt = OpenAIStreamingSTT("gpt-live-transcribe", api_key="x")
    stt._on_event(_delta("a", " Can I get"))
    stt.expect_commit()

    assert stt.take(include_partial=True) == "Can I get"
    stt._on_event(_acked("a"))
    stt._on_event(_delta("a", " your"))
    stt._on_event(_completed("a", "Can I get your"))

    assert stt.heard_text() == ""


@pytest.mark.asyncio
async def test_openai_stt_pads_a_commit_that_is_too_short():
    stt = OpenAIStreamingSTT("gpt-live-transcribe", api_key="x")
    stt._ws = AsyncMock()
    await stt.send(b"\xff" * 80)  # 10ms of audio

    await stt.commit()

    events = [json.loads(c.args[0]) for c in stt._ws.send.await_args_list]
    appended = sum(len(base64.b64decode(e["audio"])) for e in events if e["type"] == "input_audio_buffer.append")
    assert appended >= 0.1 * 24000 * 2
    assert events[-1] == {"type": "input_audio_buffer.commit"}


@pytest.mark.asyncio
async def test_interrupt_is_rejected_when_the_decision_is_too_slow(tmp_path):
    simulator = _simulator(tmp_path)
    delay = 0.3

    async def decide(goal: str, history: list, heard: str) -> tuple[str, float]:
        await asyncio.sleep(delay)
        return "interrupt", 0.9

    simulator._interruptor.decide = decide
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = True
    simulator._audio_interface.is_caller_playing.return_value = False
    simulator._interrupt = AsyncMock()
    simulator.simulator_config = CascadeSimulatorConfig(model="gpt-test", interruption_max_decision_ms=150)

    await simulator._decide("goal", "Can I get your", _trigger())  # a >=300ms decision > 150ms
    simulator._interrupt.assert_not_called()

    delay = 0.0
    await simulator._decide("goal", "Can I get your", _trigger())
    simulator._interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_stt_matches_out_of_order_finals_by_item():
    stt = OpenAIStreamingSTT("gpt-live-transcribe", api_key="x")
    stt._on_event(_delta("a", " Can I get"))
    stt.expect_commit()
    stt._on_event(_acked("a"))
    assert stt.take(include_partial=True) == "Can I get"
    stt.expect_commit()
    stt._on_event(_acked("b"))

    stt._on_event(_completed("b", "your name?"))  # the fresh segment's final arrives first
    stt._on_event(_completed("a", "Can I get"))

    assert stt.committed_event.is_set()
    assert stt.take() == "your name?"


def test_config_rejects_an_elevenlabs_model_on_openai():
    with pytest.raises(ValidationError, match="ElevenLabs model"):
        CascadeSimulatorConfig(model="gpt-test", stt_model="scribe_v2_realtime")


@pytest.mark.asyncio
async def test_forced_interrupt_fires_even_when_the_decision_is_slow(tmp_path):
    simulator = _simulator(tmp_path)

    async def decide(goal: str, history: list, heard: str) -> tuple[str, float]:
        await asyncio.sleep(0.3)
        return "continue", 0.1

    simulator._interruptor.decide = decide
    simulator._audio_interface = MagicMock()
    simulator._audio_interface.is_assistant_playing.return_value = True
    simulator._audio_interface.is_caller_playing.return_value = False
    simulator._interrupt = AsyncMock()
    simulator.simulator_config = CascadeSimulatorConfig(model="gpt-test", interruption_max_decision_ms=150)
    trigger = _trigger()
    trigger.forced = True

    await simulator._decide("goal", "Can I get your", trigger)
    simulator._interrupt.assert_awaited_once()
