"""Pipeline components for the Pipecat-based assistant."""

from eva.assistant.pipeline.agent_processor import (
    BenchmarkAgentProcessor,
    UserAudioCollector,
    UserObserver,
)
from eva.assistant.pipeline.frames import (
    LLMMessageFrame,
    SpokenMessageFrame,
    TurnTimestampFrame,
    UserContextFrame,
    UserMessageFrame,
    VADBufferFrame,
)
from eva.assistant.pipeline.observers import BenchmarkLogObserver
from eva.assistant.pipeline.services import create_audio_llm_client, create_stt_service, create_tts_service

__all__ = [
    # Frame types
    "TurnTimestampFrame",
    "SpokenMessageFrame",
    "VADBufferFrame",
    "UserMessageFrame",
    "LLMMessageFrame",
    "UserContextFrame",
    # Processors
    "BenchmarkAgentProcessor",
    "UserObserver",
    "UserAudioCollector",
    # Services
    "create_audio_llm_client",
    "create_stt_service",
    "create_tts_service",
    # Observers
    "BenchmarkLogObserver",
]
