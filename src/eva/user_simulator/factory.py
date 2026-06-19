"""Factory for simulated caller providers."""

from __future__ import annotations

from typing import Any

from eva.models.config import ElevenLabsSimulatorConfig, OpenAIRealtimeSimulatorConfig, UserSimulatorConfig
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
    raise ValueError(f"Unknown user simulator provider: {simulator_config.provider!r}")
