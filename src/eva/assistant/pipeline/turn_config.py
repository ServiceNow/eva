"""Factory functions for creating turn strategies and VAD analyzers from configuration.

This module provides functions to instantiate Pipecat turn strategies and VAD analyzers
based on configuration settings from environment variables or config files.
"""

from typing import Any

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class EmptyProposalStopStrategy(ExternalUserTurnStopStrategy):
    """External turn stop that also closes a turn whose committed transcript is empty.

    A self-endpointing STT (Cartesia ink-2) can open a turn from trailing audio and then commit it
    with no words. It skips the TranscriptionFrame for an empty commit but still proposes the turn
    stop. The stock strategy waits for text, so it never closes the turn, and pipecat's 5s
    user_turn_stop_timeout watchdog closes it instead. That delays re-running a query the phantom
    turn cancelled by about 5s.

    Only use this with an STT that pushes the final TranscriptionFrame before the proposed stop,
    as ink-2 does. Otherwise a real turn would be closed before its text arrives.
    """

    async def _handle_user_stopped_speaking(self, *, announced_elsewhere: bool):
        had_text = bool(self._text)
        await super()._handle_user_stopped_speaking(announced_elsewhere=announced_elsewhere)
        # `had_text` keeps this from firing twice: with text, the parent already triggered the stop.
        if not had_text and not announced_elsewhere and self._turn_open:
            logger.debug(f"{self}: closing a turn committed with no text")
            await self._trigger_user_turn_stopped()


def create_vad_analyzer(vad_type: str, vad_params: dict[str, Any]) -> VADAnalyzer | None:
    """Create a VAD analyzer from configuration.

    Args:
        vad_type: VAD analyzer type ('silero', 'none')
        vad_params: VAD parameters (confidence, start_secs, stop_secs, min_volume)

    Returns:
        VAD analyzer instance, or None if vad_type is 'none'

    Raises:
        ValueError: If vad_type is not supported
    """
    vad_type_lower = vad_type.lower()

    if vad_type_lower == "none":
        return None
    elif vad_type_lower == "silero":
        # Create VADParams, respecting existing defaults if no params specified
        params = VADParams(**vad_params) if vad_params else None
        return SileroVADAnalyzer(params=params)
    else:
        raise ValueError(f"Unsupported VAD type: {vad_type}. Supported types: 'silero', 'none'")


def create_turn_start_strategy(
    strategy_type: str,
    strategy_params: dict[str, Any],
) -> BaseUserTurnStartStrategy:
    """Create a user turn start strategy from configuration.

    Args:
        strategy_type: Strategy type ('vad', 'transcription', 'external')
        strategy_params: Strategy-specific parameters

    Returns:
        Turn start strategy instance

    Raises:
        ValueError: If strategy_type is not supported
    """
    strategy_type_lower = strategy_type.lower()

    if strategy_type_lower == "vad":
        # VADUserTurnStartStrategy has no required parameters
        return VADUserTurnStartStrategy(**strategy_params)
    elif strategy_type_lower == "transcription":
        # TranscriptionUserTurnStartStrategy has no required parameters
        return TranscriptionUserTurnStartStrategy(**strategy_params)
    elif strategy_type_lower == "external":
        # ExternalUserTurnStartStrategy has no required parameters
        return ExternalUserTurnStartStrategy(**strategy_params)
    else:
        raise ValueError(
            f"Unsupported turn start strategy: {strategy_type}. Supported types: 'vad', 'transcription', 'external'"
        )


def create_turn_stop_strategy(
    strategy_type: str,
    strategy_params: dict[str, Any],
    smart_turn_stop_secs: float | None = None,
    close_empty_turns: bool = False,
) -> BaseUserTurnStopStrategy:
    """Create a user turn stop strategy from configuration.

    Args:
        strategy_type: Strategy type ('speech_timeout', 'turn_analyzer', 'krisp_viva_turn', 'external')
        strategy_params: Strategy-specific parameters
        smart_turn_stop_secs: stop_secs for SmartTurnParams (used with turn_analyzer strategy)
        close_empty_turns: for 'external' only, close a turn the STT committed with no text instead
            of waiting for pipecat's user_turn_stop_timeout. Set for ink-2 (see EmptyProposalStopStrategy).

    Returns:
        Turn stop strategy instance

    Raises:
        ValueError: If strategy_type is not supported
    """
    strategy_type_lower = strategy_type.lower()

    if strategy_type_lower == "speech_timeout":
        # SpeechTimeoutUserTurnStopStrategy accepts user_speech_timeout parameter
        return SpeechTimeoutUserTurnStopStrategy(**strategy_params)
    elif strategy_type_lower == "turn_analyzer":
        # TurnAnalyzerUserTurnStopStrategy requires a turn_analyzer instance
        # smart_turn_stop_secs can be passed via strategy_params (takes precedence) or the explicit argument
        params = dict(strategy_params)
        stop_secs = params.pop("smart_turn_stop_secs", smart_turn_stop_secs)
        smart_params = SmartTurnParams(stop_secs=stop_secs) if stop_secs is not None else None
        turn_analyzer = LocalSmartTurnAnalyzerV3(params=smart_params)
        return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer, **params)
    elif strategy_type_lower == "krisp_viva_turn":
        # KrispVivaTurn uses Krisp's VIVA SDK turn detection v3 (Tt) API, processing audio
        # frame-by-frame in real time with an external VAD flag. It depends on the proprietary
        # krisp_audio SDK plus a .kef model file (KRISP_VIVA_TURN_MODEL_PATH) and license key
        # (KRISP_VIVA_API_KEY), so import lazily to keep the SDK optional.
        from pipecat.audio.turn.krisp_viva_turn import KrispTurnParams, KrispVivaTurn

        params = dict(strategy_params)
        # Krisp analyzer tuning params; fall back to KrispTurnParams defaults when absent.
        krisp_kwargs = {k: params.pop(k) for k in ("threshold", "frame_duration_ms") if k in params}
        krisp_params = KrispTurnParams(**krisp_kwargs) if krisp_kwargs else None
        # Constructor kwargs; model_path/api_key fall back to their env vars when omitted.
        analyzer_kwargs = {k: params.pop(k) for k in ("model_path", "api_key", "sample_rate") if k in params}
        krisp_analyzer = KrispVivaTurn(params=krisp_params, **analyzer_kwargs)
        return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=krisp_analyzer, **params)
    elif strategy_type_lower == "external":
        # ExternalUserTurnStopStrategy has no required parameters
        if close_empty_turns:
            return EmptyProposalStopStrategy(**strategy_params)
        return ExternalUserTurnStopStrategy(**strategy_params)
    else:
        raise ValueError(
            f"Unsupported turn stop strategy: {strategy_type}. "
            f"Supported types: 'speech_timeout', 'turn_analyzer', 'krisp_viva_turn', 'external'"
        )
