"""Factory for simulated caller providers."""

from __future__ import annotations

from typing import Any

from eva.models.config import UserSimulatorConfig
from eva.user_simulator.base import AbstractUserSimulator


def create_user_simulator(
    simulator_config: UserSimulatorConfig,
    **kwargs: Any,
) -> AbstractUserSimulator:
    """Create the configured simulated caller without importing unused providers."""
    if simulator_config.provider == "elevenlabs":
        from eva.user_simulator.client import ElevenLabsUserSimulator

        return ElevenLabsUserSimulator(**kwargs)
    if simulator_config.provider == "openai_realtime":
        from eva.user_simulator.openai_realtime import OpenAIRealtimeUserSimulator

        return OpenAIRealtimeUserSimulator(simulator_config=simulator_config, **kwargs)
    raise ValueError(f"Unknown user simulator provider: {simulator_config.provider!r}")
