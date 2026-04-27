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
) -> BaseUserTurnStopStrategy:
    """Create a user turn stop strategy from configuration.

    Args:
        strategy_type: Strategy type ('speech_timeout', 'turn_analyzer', 'external')
        strategy_params: Strategy-specific parameters
        smart_turn_stop_secs: stop_secs for SmartTurnParams (used with turn_analyzer strategy)

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
        # If smart_turn_stop_secs is provided, use it; otherwise let SmartTurnParams use its default
        smart_params = SmartTurnParams(stop_secs=smart_turn_stop_secs) if smart_turn_stop_secs is not None else None
        turn_analyzer = LocalSmartTurnAnalyzerV3(params=smart_params)
        return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer, **strategy_params)
    elif strategy_type_lower == "external":
        # ExternalUserTurnStopStrategy has no required parameters
        return ExternalUserTurnStopStrategy(**strategy_params)
    else:
        raise ValueError(
            f"Unsupported turn stop strategy: {strategy_type}. "
            f"Supported types: 'speech_timeout', 'turn_analyzer', 'external'"
        )
