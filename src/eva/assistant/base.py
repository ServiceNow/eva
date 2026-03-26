"""Abstract base class for assistant servers in the EVA benchmark framework.

Any voice framework (Pipecat, RoomKit, etc.) must implement this interface
so that the orchestrator can run conversations framework-agnostically.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from eva.models.agents import AgentConfig
from eva.models.config import AudioLLMConfig, PipelineConfig, SpeechToSpeechConfig


class AssistantServerBase(ABC):
    """Contract between ConversationWorker and any voice framework.

    Implementations must:
    - Listen for WebSocket connections on the assigned port using the
      Twilio-style JSON protocol (events: connected, start, media, stop)
      with 8 kHz mu-law audio encoding.
    - Write all required output files to ``output_dir`` when ``stop()``
      is called.
    - Expose conversation statistics via ``get_conversation_stats()``.
    """

    @abstractmethod
    def __init__(
        self,
        current_date_time: str,
        pipeline_config: PipelineConfig | SpeechToSpeechConfig | AudioLLMConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
    ) -> None:
        """Initialize the assistant server.

        Args:
            current_date_time: Current date/time string from the evaluation record.
            pipeline_config: Model pipeline configuration (STT+LLM+TTS, S2S, or Audio-LLM).
            agent: Agent configuration loaded from YAML.
            agent_config_path: Path to the agent YAML configuration file.
            scenario_db_path: Path to the scenario database JSON file.
            output_dir: Directory where output files must be written.
            port: WebSocket port to listen on.
            conversation_id: Unique identifier for this conversation.
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the WebSocket server.

        Must block until the server is accepting connections on the assigned port.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server and write all output files to ``output_dir``.

        Required output files:
            - ``audit_log.json`` — structured conversation log
            - ``transcript.jsonl`` — ``{timestamp, role, content}`` per line
            - ``initial_scenario_db.json`` — scenario DB snapshot before conversation
            - ``final_scenario_db.json`` — scenario DB snapshot after conversation
            - ``audio_mixed.wav`` — mixed user+assistant audio (16-bit PCM)
            - ``audio_user.wav`` — user-only audio track
            - ``audio_assistant.wav`` — assistant-only audio track
            - ``framework_logs.jsonl`` — JSONL with event types:
              ``tts_text``, ``llm_response``, ``turn_start``, ``turn_end``
            - ``response_latencies.json`` — ``{latencies, mean, max, count}``

        Optional (framework-specific):
            - ``pipecat_metrics.jsonl`` or equivalent latency metrics
        """
        ...

    @abstractmethod
    def get_conversation_stats(self) -> dict[str, Any]:
        """Return conversation statistics.

        Must include at minimum:
            - ``num_turns``: number of user turns
            - ``num_tool_calls``: total tool invocations
            - ``tools_called``: list of tool names invoked
        """
        ...
