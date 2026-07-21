"""Tests for the turn-end fallback timer and its wiring into UserObserver.

The fallback nudges the assistant to reprompt when a user turn is never detected within
EVA_TURN_END_FALLBACK_TIME seconds after the assistant stops speaking. The timer is hosted on
the pipeline spine (UserObserver) and fires a pipeline-specific process_turn_fallback.

These tests drive the timer/observer directly (no real Pipecat pipeline) using a tiny window so
they run in well under a second.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from eva.assistant.pipeline.agent_processor import UserObserver
from eva.assistant.pipeline.fallback import (
    MAX_CONSECUTIVE_FALLBACK_NUDGES,
    TurnEndFallbackTimer,
    build_fallback_nudge,
)

FALLBACK_TIME = 0.02  # 20ms window keeps timer tests fast
SETTLE = FALLBACK_TIME + 0.03  # wait past the window for the timer to fire


# --------------------------------------------------------------------------------------------
# build_fallback_nudge
# --------------------------------------------------------------------------------------------


def test_build_fallback_nudge_no_partial():
    marker, nudge = build_fallback_nudge(4, "")
    assert marker == "no user speech detected within 4s"
    assert "no user speech captured" in nudge
    assert "Do not mention this system note" in nudge


def test_build_fallback_nudge_with_partial():
    marker, nudge = build_fallback_nudge(4, "ten o'clock")
    assert "partial user speech" in marker
    assert "ten o'clock" in marker
    assert "ten o'clock" in nudge


# --------------------------------------------------------------------------------------------
# TurnEndFallbackTimer
# --------------------------------------------------------------------------------------------


async def test_timer_disabled_is_noop():
    fired = []
    timer = TurnEndFallbackTimer(None, lambda t: fired.append(t))
    assert not timer.enabled
    timer.arm()
    await asyncio.sleep(SETTLE)
    assert fired == []


async def test_timer_fires_on_silence():
    fired = []

    async def on_fire(t):
        fired.append(t)

    timer = TurnEndFallbackTimer(FALLBACK_TIME, on_fire)
    timer.arm()
    await asyncio.sleep(SETTLE)
    assert fired == [FALLBACK_TIME]


async def test_timer_cancel_prevents_fire():
    fired = []

    async def on_fire(t):
        fired.append(t)

    timer = TurnEndFallbackTimer(FALLBACK_TIME, on_fire)
    timer.arm()
    timer.cancel()
    await asyncio.sleep(SETTLE)
    assert fired == []


async def test_timer_note_nudge_give_up():
    timer = TurnEndFallbackTimer(FALLBACK_TIME, AsyncMock())
    # First MAX nudges proceed; the one after the limit is refused.
    for _ in range(MAX_CONSECUTIVE_FALLBACK_NUDGES):
        assert timer.note_nudge() is True
    assert timer.note_nudge() is False
    assert timer.consecutive_nudges == MAX_CONSECUTIVE_FALLBACK_NUDGES + 1


async def test_timer_reset_counter():
    timer = TurnEndFallbackTimer(FALLBACK_TIME, AsyncMock())
    timer.note_nudge()
    timer.note_nudge()
    timer.reset_counter()
    assert timer.consecutive_nudges == 0


# --------------------------------------------------------------------------------------------
# UserObserver wiring
# --------------------------------------------------------------------------------------------


def _make_observer(fallback_time=FALLBACK_TIME):
    """Build a UserObserver with a mock fallback processor, bypassing pipecat push_frame."""
    proc = MagicMock()
    proc.query_in_flight = False
    proc.process_turn_fallback = AsyncMock()
    obs = UserObserver(turn_end_fallback_time=fallback_time, fallback_processor=proc)
    # Stub the FrameProcessor plumbing so process_frame can run outside a real pipeline.
    obs.push_frame = AsyncMock()

    async def _noop_super(frame, direction):
        return None

    # super().process_frame is called first in UserObserver.process_frame; patch the bound
    # FrameProcessor method on the instance's class chain is awkward, so monkeypatch via a flag.
    return obs, proc


async def _send(obs, frame):
    # UserObserver.process_frame calls super().process_frame first; that requires pipecat setup.
    # We bypass it by calling the frame-handling logic through the public method with the
    # FrameProcessor base stubbed.
    await obs.process_frame(frame, FrameDirection.DOWNSTREAM)


async def test_observer_fires_nudge_on_silence(monkeypatch):
    obs, proc = _make_observer()
    # Bypass FrameProcessor.process_frame (needs a running pipeline).
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_awaited_once()
    # partial is empty (no transcription seen)
    args = proc.process_turn_fallback.await_args
    assert args.args[0] == FALLBACK_TIME
    assert args.args[1] == ""


async def test_observer_user_started_speaking_cancels(monkeypatch):
    """The bug-1 regression: a UserStartedSpeaking within the window suppresses the nudge."""
    obs, proc = _make_observer()
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())  # arm
    await _send(obs, UserStartedSpeakingFrame())  # should cancel
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_not_awaited()


async def test_observer_bot_started_speaking_cancels(monkeypatch):
    obs, proc = _make_observer()
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())  # arm
    await _send(obs, BotStartedSpeakingFrame())  # cancel (assistant speaking again)
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_not_awaited()


async def test_observer_rearms_on_user_stopped(monkeypatch):
    """User stops without a completed turn (query not in flight) -> re-arm and eventually fire."""
    obs, proc = _make_observer()
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())  # arm
    await _send(obs, UserStartedSpeakingFrame())  # cancel
    await _send(obs, UserStoppedSpeakingFrame())  # re-arm (query not in flight)
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_awaited_once()


async def test_observer_no_rearm_when_query_in_flight(monkeypatch):
    """Race guard: if a real turn is being processed, UserStopped must NOT re-arm a nudge."""
    obs, proc = _make_observer()
    proc.query_in_flight = True  # a real completed turn is being processed
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())  # arm
    await _send(obs, UserStartedSpeakingFrame())  # cancel
    await _send(obs, UserStoppedSpeakingFrame())  # would re-arm, but query in flight -> skip
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_not_awaited()


async def test_observer_skips_nudge_when_query_in_flight(monkeypatch):
    """If a query starts after arming, the fire handler skips (no stacked nudge)."""
    obs, proc = _make_observer()
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())  # arm
    proc.query_in_flight = True  # real turn started processing before the window elapsed
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_not_awaited()


async def test_observer_forwards_partial_transcript(monkeypatch):
    obs, proc = _make_observer()
    monkeypatch.setattr(
        "pipecat.processors.frame_processor.FrameProcessor.process_frame",
        AsyncMock(),
    )
    await _send(obs, BotStoppedSpeakingFrame())  # arm
    await _send(obs, TranscriptionFrame("ten o'clock", "user", "2026-01-01T00:00:00Z"))
    await asyncio.sleep(SETTLE)
    proc.process_turn_fallback.assert_awaited_once()
    args = proc.process_turn_fallback.await_args
    assert args.args[1] == "ten o'clock"


# --------------------------------------------------------------------------------------------
# AudioLLMProcessor.process_turn_fallback (the audio-LLM extension)
# --------------------------------------------------------------------------------------------


async def test_audio_llm_process_turn_fallback_injects_text_nudge():
    """The audio-LLM nudge logs a turn_fallback marker and drives a text query to TTS.

    Exercises the real AudioLLMProcessor.process_turn_fallback with a mocked agentic system,
    since the example/7 end-to-end path needs a remote model endpoint not available in tests.
    """
    from eva.assistant.agentic.audit_log import AuditLog
    from eva.assistant.pipeline.audio_llm_processor import AudioLLMProcessor

    proc = object.__new__(AudioLLMProcessor)
    proc._current_query_task = None
    proc._interrupted = asyncio.Event()
    proc.fallback_timer = None
    proc.audit_log = AuditLog()
    proc.on_assistant_response = None
    proc.push_frame = AsyncMock()

    async def fake_process_query(text, log_user_input=True):
        assert log_user_input is False  # marker already logged separately
        yield "Sorry, I didn't catch that. Could you repeat?"

    proc.agentic_system = MagicMock()
    proc.agentic_system.process_query = fake_process_query

    await proc.process_turn_fallback(4, "")

    # A turn_fallback marker was recorded with the richer nudge as llm_content.
    fb = [e for e in proc.audit_log.transcript if e.get("message_type") == "turn_fallback"]
    assert len(fb) == 1
    assert fb[0]["value"] == "no user speech detected within 4s"
    assert "repeat" in fb[0]["llm_content"].lower()
    # The response was pushed toward TTS (LLMMessageFrame + TTSSpeakFrame).
    assert proc.push_frame.await_count >= 1
