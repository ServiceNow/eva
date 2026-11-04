"""Unit tests for turn_config factory functions."""

from unittest.mock import MagicMock, patch

import pytest
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)

from eva.assistant.pipeline.turn_config import (
    create_turn_start_strategy,
    create_turn_stop_strategy,
    create_vad_analyzer,
)

# ---------------------------------------------------------------------------
# create_vad_analyzer
# ---------------------------------------------------------------------------


class TestCreateVadAnalyzer:
    """Tests for create_vad_analyzer factory."""

    def test_silero_no_params(self):
        """'silero' with empty params creates SileroVADAnalyzer with params=None."""
        mock_analyzer = MagicMock()
        with patch("eva.assistant.pipeline.turn_config.SileroVADAnalyzer", return_value=mock_analyzer) as mock_cls:
            result = create_vad_analyzer("silero", {})

        mock_cls.assert_called_once_with(params=None)
        assert result is mock_analyzer

    def test_silero_with_params(self):
        """'silero' with params passes VADParams constructed from them."""
        from pipecat.audio.vad.vad_analyzer import VADParams

        mock_analyzer = MagicMock()
        with patch("eva.assistant.pipeline.turn_config.SileroVADAnalyzer", return_value=mock_analyzer) as mock_cls:
            result = create_vad_analyzer("silero", {"stop_secs": 0.8, "confidence": 0.7})

        call_args = mock_cls.call_args
        passed_params = call_args.kwargs["params"]
        assert isinstance(passed_params, VADParams)
        assert passed_params.stop_secs == 0.8
        assert passed_params.confidence == 0.7
        assert result is mock_analyzer

    def test_silero_case_insensitive(self):
        """Silero type is matched case-insensitively."""
        mock_analyzer = MagicMock()
        with patch("eva.assistant.pipeline.turn_config.SileroVADAnalyzer", return_value=mock_analyzer):
            for variant in ("SILERO", "Silero", "SiLeRo"):
                result = create_vad_analyzer(variant, {})
                assert result is mock_analyzer

    def test_unsupported_vad_type_raises(self):
        """Unknown VAD type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported VAD type: webrtc"):
            create_vad_analyzer("webrtc", {})

    def test_none_vad_type_returns_none(self):
        """'none' vad_type returns None without loading any model."""
        result = create_vad_analyzer("none", {})
        assert result is None

    def test_none_vad_type_case_insensitive(self):
        """'none' is matched case-insensitively."""
        assert create_vad_analyzer("None", {}) is None
        assert create_vad_analyzer("NONE", {}) is None

    def test_unsupported_vad_type_error_lists_supported(self):
        """ValueError message lists supported types including 'none'."""
        with pytest.raises(ValueError, match="silero"):
            create_vad_analyzer("unknown", {})
        with pytest.raises(ValueError, match="none"):
            create_vad_analyzer("unknown", {})


# ---------------------------------------------------------------------------
# create_turn_start_strategy
# ---------------------------------------------------------------------------


class TestCreateTurnStartStrategy:
    """Tests for create_turn_start_strategy factory."""

    def test_vad_strategy(self):
        """'vad' returns VADUserTurnStartStrategy."""
        result = create_turn_start_strategy("vad", {})
        assert isinstance(result, VADUserTurnStartStrategy)

    def test_transcription_strategy(self):
        """'transcription' returns TranscriptionUserTurnStartStrategy."""
        result = create_turn_start_strategy("transcription", {})
        assert isinstance(result, TranscriptionUserTurnStartStrategy)

    def test_external_strategy(self):
        """'external' returns ExternalUserTurnStartStrategy."""
        result = create_turn_start_strategy("external", {})
        assert isinstance(result, ExternalUserTurnStartStrategy)

    def test_case_insensitive(self):
        """Strategy types are matched case-insensitively."""
        assert isinstance(create_turn_start_strategy("VAD", {}), VADUserTurnStartStrategy)
        assert isinstance(create_turn_start_strategy("Vad", {}), VADUserTurnStartStrategy)
        assert isinstance(create_turn_start_strategy("TRANSCRIPTION", {}), TranscriptionUserTurnStartStrategy)
        assert isinstance(create_turn_start_strategy("External", {}), ExternalUserTurnStartStrategy)

    def test_unsupported_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported turn start strategy: magic"):
            create_turn_start_strategy("magic", {})

    def test_unsupported_strategy_error_lists_supported(self):
        """ValueError message lists the supported strategies."""
        with pytest.raises(ValueError, match="vad.*transcription.*external"):
            create_turn_start_strategy("unknown", {})


# ---------------------------------------------------------------------------
# create_turn_stop_strategy
# ---------------------------------------------------------------------------


class TestCreateTurnStopStrategy:
    """Tests for create_turn_stop_strategy factory."""

    def test_speech_timeout_strategy(self):
        """'speech_timeout' returns SpeechTimeoutUserTurnStopStrategy."""
        result = create_turn_stop_strategy("speech_timeout", {})
        assert isinstance(result, SpeechTimeoutUserTurnStopStrategy)

    def test_speech_timeout_with_params(self):
        """speech_timeout strategy passes through strategy_params."""
        result = create_turn_stop_strategy("speech_timeout", {"user_speech_timeout": 1.2})
        assert isinstance(result, SpeechTimeoutUserTurnStopStrategy)
        assert result._user_speech_timeout == 1.2

    def test_turn_analyzer_strategy(self):
        """'turn_analyzer' returns TurnAnalyzerUserTurnStopStrategy."""
        mock_analyzer = MagicMock()
        with patch(
            "eva.assistant.pipeline.turn_config.LocalSmartTurnAnalyzerV3",
            return_value=mock_analyzer,
        ):
            result = create_turn_stop_strategy("turn_analyzer", {})

        assert isinstance(result, TurnAnalyzerUserTurnStopStrategy)

    def test_turn_analyzer_without_stop_secs_uses_default_smart_params(self):
        """When smart_turn_stop_secs is None, SmartTurnParams is not passed explicitly."""
        mock_analyzer = MagicMock()
        with (
            patch(
                "eva.assistant.pipeline.turn_config.LocalSmartTurnAnalyzerV3",
                return_value=mock_analyzer,
            ) as mock_cls,
            patch("eva.assistant.pipeline.turn_config.SmartTurnParams") as mock_smart_params,
        ):
            create_turn_stop_strategy("turn_analyzer", {}, smart_turn_stop_secs=None)

        mock_smart_params.assert_not_called()
        mock_cls.assert_called_once_with(params=None)

    def test_turn_analyzer_with_stop_secs(self):
        """When smart_turn_stop_secs is provided, SmartTurnParams uses it."""
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

        mock_analyzer = MagicMock()
        with patch(
            "eva.assistant.pipeline.turn_config.LocalSmartTurnAnalyzerV3",
            return_value=mock_analyzer,
        ) as mock_cls:
            create_turn_stop_strategy("turn_analyzer", {}, smart_turn_stop_secs=0.8)

        call_args = mock_cls.call_args
        passed_params = call_args.kwargs["params"]
        assert isinstance(passed_params, SmartTurnParams)
        assert passed_params.stop_secs == 0.8

    def test_turn_analyzer_smart_turn_stop_secs_via_strategy_params(self):
        """smart_turn_stop_secs in strategy_params takes precedence over the function argument."""
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

        mock_analyzer = MagicMock()
        with patch(
            "eva.assistant.pipeline.turn_config.LocalSmartTurnAnalyzerV3",
            return_value=mock_analyzer,
        ) as mock_cls:
            create_turn_stop_strategy(
                "turn_analyzer",
                {"smart_turn_stop_secs": 1.5},
                smart_turn_stop_secs=0.8,
            )

        call_args = mock_cls.call_args
        passed_params = call_args.kwargs["params"]
        assert isinstance(passed_params, SmartTurnParams)
        assert passed_params.stop_secs == 1.5

    def test_external_stop_strategy(self):
        """'external' returns ExternalUserTurnStopStrategy."""
        result = create_turn_stop_strategy("external", {})
        assert isinstance(result, ExternalUserTurnStopStrategy)

    def test_case_insensitive(self):
        """Strategy types are matched case-insensitively."""
        assert isinstance(create_turn_stop_strategy("SPEECH_TIMEOUT", {}), SpeechTimeoutUserTurnStopStrategy)
        assert isinstance(create_turn_stop_strategy("External", {}), ExternalUserTurnStopStrategy)

        mock_analyzer = MagicMock()
        with patch("eva.assistant.pipeline.turn_config.LocalSmartTurnAnalyzerV3", return_value=mock_analyzer):
            assert isinstance(create_turn_stop_strategy("TURN_ANALYZER", {}), TurnAnalyzerUserTurnStopStrategy)

    def test_unsupported_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported turn stop strategy: magic"):
            create_turn_stop_strategy("magic", {})

    def test_unsupported_strategy_error_lists_supported(self):
        """ValueError message lists supported strategies."""
        with pytest.raises(ValueError, match="speech_timeout.*turn_analyzer.*krisp_viva_turn.*external"):
            create_turn_stop_strategy("unknown", {})


# ---------------------------------------------------------------------------
# create_turn_stop_strategy — krisp_viva_turn
# ---------------------------------------------------------------------------


class TestCreateTurnStopStrategyKrispViva:
    """Tests for the 'krisp_viva_turn' branch of create_turn_stop_strategy.

    KrispVivaTurn depends on the proprietary ``krisp_audio`` SDK, which is not a
    project dependency. The factory imports ``pipecat.audio.turn.krisp_viva_turn``
    lazily, so these tests inject a fake module into ``sys.modules``.
    """

    @staticmethod
    def _install_fake_krisp(monkeypatch):
        """Inject a fake pipecat.audio.turn.krisp_viva_turn module and return it."""
        import sys

        fake_module = MagicMock()
        fake_module.KrispVivaTurn.return_value = MagicMock()
        monkeypatch.setitem(sys.modules, "pipecat.audio.turn.krisp_viva_turn", fake_module)
        return fake_module

    def test_krisp_viva_turn_strategy(self, monkeypatch):
        """'krisp_viva_turn' returns a TurnAnalyzerUserTurnStopStrategy with a KrispVivaTurn."""
        fake_module = self._install_fake_krisp(monkeypatch)

        result = create_turn_stop_strategy("krisp_viva_turn", {})

        assert isinstance(result, TurnAnalyzerUserTurnStopStrategy)
        # No tuning params given -> KrispTurnParams left at its defaults (params=None).
        fake_module.KrispVivaTurn.assert_called_once_with(params=None)
        fake_module.KrispTurnParams.assert_not_called()

    def test_krisp_viva_turn_case_insensitive(self, monkeypatch):
        """Strategy type is matched case-insensitively."""
        self._install_fake_krisp(monkeypatch)
        assert isinstance(create_turn_stop_strategy("KRISP_VIVA_TURN", {}), TurnAnalyzerUserTurnStopStrategy)

    def test_krisp_viva_turn_tuning_params(self, monkeypatch):
        """Tuning params threshold / frame_duration_ms are forwarded to KrispTurnParams."""
        fake_module = self._install_fake_krisp(monkeypatch)

        create_turn_stop_strategy("krisp_viva_turn", {"threshold": 0.7, "frame_duration_ms": 30})

        fake_module.KrispTurnParams.assert_called_once_with(threshold=0.7, frame_duration_ms=30)
        # The constructed params object is passed to the analyzer.
        assert fake_module.KrispVivaTurn.call_args.kwargs["params"] is fake_module.KrispTurnParams.return_value

    def test_krisp_viva_turn_constructor_params(self, monkeypatch):
        """model_path / api_key / sample_rate are forwarded to the KrispVivaTurn constructor."""
        fake_module = self._install_fake_krisp(monkeypatch)

        create_turn_stop_strategy(
            "krisp_viva_turn",
            {"model_path": "/models/turn.kef", "api_key": "secret", "sample_rate": 16000},
        )

        call_kwargs = fake_module.KrispVivaTurn.call_args.kwargs
        assert call_kwargs["model_path"] == "/models/turn.kef"
        assert call_kwargs["api_key"] == "secret"
        assert call_kwargs["sample_rate"] == 16000
        # None of these should leak into the strategy as tuning params.
        fake_module.KrispTurnParams.assert_not_called()


# ---------------------------------------------------------------------------
# Phantom ink-2 turn.start must not open a turn (issue #561)
# ---------------------------------------------------------------------------


class TestInk2PhantomTurnStart:
    """Drive a real UserTurnController with the strategies used for ink-2 (cartesia) runs.

    ink-2 emits ProposedUserStartedSpeakingFrame on turn.start, InterimTranscriptionFrame /
    TranscriptionFrame only when there is text, and ProposedUserStoppedSpeakingFrame on turn.end.
    A turn.start produced by trailing audio of the previous utterance has no words, so it must not
    open a user turn: opening one broadcasts an interruption that cancels the in-flight LLM query.
    """

    @staticmethod
    async def _controller(start_strategy):
        from pipecat.clocks.system_clock import SystemClock
        from pipecat.processors.frame_processor import FrameProcessorSetup
        from pipecat.turns.user_turn_controller import UserTurnController
        from pipecat.turns.user_turn_strategies import UserTurnStrategies
        from pipecat.utils.asyncio.task_manager import TaskManager

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=[start_strategy], stop=[ExternalUserTurnStopStrategy()])
        )
        await controller.setup(
            FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=MagicMock())
        )

        started = []

        @controller.event_handler("on_user_turn_started")
        async def _on_started(controller, strategy, params):
            started.append(params)

        return controller, started

    @staticmethod
    def _strategy():
        return create_turn_start_strategy("transcription", {})

    async def test_turn_start_without_words_does_not_open_a_turn(self):
        from pipecat.frames.frames import ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame

        controller, started = await self._controller(self._strategy())
        try:
            await controller.process_frame(ProposedUserStartedSpeakingFrame())
            await controller.process_frame(ProposedUserStoppedSpeakingFrame())  # turn.end with ''
        finally:
            await controller.cleanup()

        assert started == []

    async def test_first_interim_opens_turn_with_interruption(self):
        from pipecat.frames.frames import InterimTranscriptionFrame, ProposedUserStartedSpeakingFrame

        controller, started = await self._controller(self._strategy())
        try:
            await controller.process_frame(ProposedUserStartedSpeakingFrame())
            assert started == []  # still waiting for a word
            await controller.process_frame(InterimTranscriptionFrame("hello", "u", "2026-01-01T00:00:00Z"))
        finally:
            await controller.cleanup()

        assert len(started) == 1
        assert started[0].enable_interruptions is True  # a real barge-in still interrupts


class TestInk2FinalSurvivesTurnStartInterruption:
    """Regression for the final-frame stall found while fixing eva-internal#561.

    ink-2 can deliver a late interim, the final TranscriptionFrame and the
    ProposedUserStoppedSpeakingFrame back-to-back. The interim opens the user turn, the
    aggregator broadcasts an InterruptionFrame, and pipecat flushes the aggregator's queued
    data/control frames — the final and the stop proposal — so the turn never closes.
    UserObserver (just ahead of the aggregator) must make those two frames survive the flush.
    """

    @staticmethod
    async def _run(frames):
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.tests.utils import run_test
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        from eva.assistant.pipeline.agent_processor import UserObserver

        pair = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[create_turn_start_strategy("transcription", {})],
                    stop=[ExternalUserTurnStopStrategy()],
                )
            ),
        )
        events = []

        @pair.user().event_handler("on_user_turn_stopped")
        async def _on_stopped(aggregator, strategy, message):
            events.append(message.content)

        await run_test(
            Pipeline([UserObserver(uninterruptible_finals=True), pair.user()]),
            frames_to_send=frames,
            ignore_start=True,
            start_timeout=15.0,
        )
        return events

    async def test_back_to_back_interim_final_stop_closes_the_turn(self):
        from pipecat.frames.frames import (
            InterimTranscriptionFrame,
            ProposedUserStartedSpeakingFrame,
            ProposedUserStoppedSpeakingFrame,
            TranscriptionFrame,
        )
        from pipecat.tests.utils import SleepFrame

        now = "2026-01-01T00:00:00Z"
        events = await self._run(
            [
                ProposedUserStartedSpeakingFrame(),
                InterimTranscriptionFrame("Yes", "user", now),
                TranscriptionFrame("Yes, that's correct.", "user", now, finalized=True),
                ProposedUserStoppedSpeakingFrame(),
                # Shorter than pipecat's 5s stop watchdog, so only the real stop path can close it.
                SleepFrame(1.5),
            ]
        )
        # Exactly one stop, before EndFrame teardown could flush the aggregation.
        assert events == ["Yes, that's correct."]

    async def test_stop_proposal_before_any_text_still_closes_turn_on_late_final(self):
        # ink-2 can send turn.start, turn.end (no words yet), then the final. The stop proposal
        # lands while no turn is open; the late final opens the turn and must still close it.
        from pipecat.frames.frames import (
            ProposedUserStartedSpeakingFrame,
            ProposedUserStoppedSpeakingFrame,
            TranscriptionFrame,
        )
        from pipecat.tests.utils import SleepFrame

        now = "2026-01-01T00:00:00Z"
        events = await self._run(
            [
                ProposedUserStartedSpeakingFrame(),
                ProposedUserStoppedSpeakingFrame(),
                TranscriptionFrame("Yes, that's correct.", "user", now, finalized=True),
                SleepFrame(1.5),  # shorter than pipecat's 5s stop watchdog
            ]
        )
        assert events == ["Yes, that's correct."]


class TestUninterruptibleFinalsScope:
    async def _pushed(self, observer, frame):
        from pipecat.processors.frame_processor import FrameDirection

        pushed = []

        async def _capture(f, direction=None):
            pushed.append(f)

        observer.push_frame = _capture
        await observer.process_frame(frame, FrameDirection.DOWNSTREAM)
        return pushed[-1]

    async def test_default_leaves_final_interruptible(self):
        from pipecat.frames.frames import TranscriptionFrame, UninterruptibleFrame

        from eva.assistant.pipeline.agent_processor import UserObserver

        out = await self._pushed(UserObserver(), TranscriptionFrame("hi", "u", "2026-01-01T00:00:00Z"))
        assert not isinstance(out, UninterruptibleFrame)

    async def test_enabled_makes_final_uninterruptible(self):
        from pipecat.frames.frames import TranscriptionFrame, UninterruptibleFrame

        from eva.assistant.pipeline.agent_processor import UserObserver

        out = await self._pushed(
            UserObserver(uninterruptible_finals=True), TranscriptionFrame("hi", "u", "2026-01-01T00:00:00Z")
        )
        assert isinstance(out, UninterruptibleFrame)
