"""Factory for simulated caller providers."""

from __future__ import annotations

from typing import Any

from eva.models.config import (
    CascadeSimulatorConfig,
    ElevenLabsSimulatorConfig,
    OpenAIRealtimeSimulatorConfig,
    UserSimulatorConfig,
)
from eva.user_simulator.base import AbstractUserSimulator


def create_user_simulator(
    simulator_config: UserSimulatorConfig,
    **kwargs: Any,
) -> AbstractUserSimulator:
    """Create the configured simulated caller without importing unused providers.

    ``framework`` names the assistant framework and is consumed here rather than
    forwarded: only the cascade caller varies its transport by framework, and the
    other two providers would raise on the unexpected keyword.
    """
    framework = kwargs.pop("framework", "pipecat")
    if isinstance(simulator_config, ElevenLabsSimulatorConfig):
        from eva.user_simulator.elevenlabs import ElevenLabsUserSimulator

        return ElevenLabsUserSimulator(**kwargs)
    if isinstance(simulator_config, OpenAIRealtimeSimulatorConfig):
        from eva.user_simulator.openai_realtime import OpenAIRealtimeUserSimulator

        return OpenAIRealtimeUserSimulator(simulator_config=simulator_config, **kwargs)
    if isinstance(simulator_config, CascadeSimulatorConfig):
        from eva.user_simulator.cascade.simulator import CascadeUserSimulator

        return CascadeUserSimulator(simulator_config=simulator_config, framework=framework, **kwargs)
    raise ValueError(f"Unknown user simulator provider: {simulator_config.provider!r}")
