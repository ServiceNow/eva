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
