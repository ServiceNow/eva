"""Factory for simulated caller providers."""

from __future__ import annotations

from typing import Any

from eva.models.config import (
    BedrockS2SSimulatorConfig,
    ElevenLabsSimulatorConfig,
    OpenAIRealtimeSimulatorConfig,
    UserSimulatorConfig,
)
from eva.user_simulator.base import AbstractUserSimulator


def create_user_simulator(
    simulator_config: UserSimulatorConfig,
    **kwargs: Any,
) -> AbstractUserSimulator:
    """Create the configured simulated caller without importing unused providers."""
    if isinstance(simulator_config, ElevenLabsSimulatorConfig):
        from eva.user_simulator.elevenlabs import ElevenLabsUserSimulator

        return ElevenLabsUserSimulator(**kwargs)
    if isinstance(simulator_config, OpenAIRealtimeSimulatorConfig):
        from eva.user_simulator.openai_realtime import OpenAIRealtimeUserSimulator

        return OpenAIRealtimeUserSimulator(simulator_config=simulator_config, **kwargs)
    if isinstance(simulator_config, BedrockS2SSimulatorConfig):
        try:
            from eva.user_simulator.bedrock_s2s import BedrockS2SUserSimulator
        except ImportError as exc:  # experimental SDK requires Python >=3.12
            raise RuntimeError(
                "The 'bedrock_s2s' caller requires the optional dependency 'aws-sdk-bedrock-runtime' "
                "(Python >=3.12). Install it with: pip install 'eva[bedrock-s2s]'."
            ) from exc

        return BedrockS2SUserSimulator(simulator_config=simulator_config, **kwargs)
    raise ValueError(f"Unknown user simulator provider: {simulator_config.provider!r}")
