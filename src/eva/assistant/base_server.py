"""Abstract base class for assistant server implementations.

All framework-specific assistant servers (Pipecat, OpenAI Realtime, Gemini Live, etc.)
must inherit from AbstractAssistantServer and implement the required interface.

See docs/assistant_server_contract.md for the full specification.
"""

import json
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.models.config import AudioLLMConfig, PipelineConfig, SpeechToSpeechConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

INITIAL_MESSAGE = "Hello! How can I help you today?"
SAMPLE_RATE = 24000


class AbstractAssistantServer(ABC):
    """Base class for all assistant server implementations.

    Each implementation must:
    1. Expose a WebSocket endpoint at ws://localhost:{port}/ws with Twilio frame format
    2. Bridge audio between the user simulator and the framework's native format
    3. Execute tool calls via the local ToolExecutor
    4. Produce all required output files (audit_log.json, framework_logs.jsonl, audio, etc.)
    5. Populate the AuditLog with conversation events
    """

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
    ):
        """Initialize the assistant server.

        Args:
            current_date_time: Current date/time string from the evaluation record
            pipeline_config: Configuration for the model/pipeline
            agent: Single agent configuration to use
            agent_config_path: Path to agent YAML configuration
            scenario_db_path: Path to scenario database JSON
            output_dir: Directory for output files
            port: Port to listen on
            conversation_id: Unique ID for this conversation
        """
        self.current_date_time = current_date_time
        self.pipeline_config = pipeline_config
        self.agent: AgentConfig = agent
        self.agent_config_path = agent_config_path
        self.scenario_db_path = scenario_db_path
        self.output_dir = Path(output_dir)
        self.port = port
        self.conversation_id = conversation_id

        # Core components - all implementations must use these
        self.audit_log = AuditLog()
        self.tool_handler = ToolExecutor(
            tool_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            tool_module_path=self.agent.tool_module_path,
            current_date_time=self.current_date_time,
        )

        # Audio buffers for recording
        self._audio_buffer = bytearray()
        self.user_audio_buffer = bytearray()
        self.assistant_audio_buffer = bytearray()
        self._audio_sample_rate: int = SAMPLE_RATE  # Subclasses can override

    @abstractmethod
    async def start(self) -> None:
        """Start the server.

        Must be non-blocking (return after the server is ready to accept connections).
        Must expose a WebSocket endpoint at ws://localhost:{port}/ws using FastAPI+uvicorn
        with TwilioFrameSerializer for compatibility with the user simulator.

        The implementation must:
        1. Create a FastAPI app with /ws and / WebSocket endpoints
        2. Start a uvicorn server on the configured port
        3. Return once the server is accepting connections
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server and save all outputs.

        Must:
        1. Gracefully shut down the server
        2. Call save_outputs() to persist all data
        """
        ...

    def get_conversation_stats(self) -> dict[str, Any]:
        """Get statistics about the conversation.

        Returns dict with: num_turns, num_tool_calls, tools_called, etc.
        """
        return self.audit_log.get_stats()

    def get_initial_scenario_db(self) -> dict[str, Any]:
        """Get initial (pristine) scenario database state."""
        return self.tool_handler.original_db

    def get_final_scenario_db(self) -> dict[str, Any]:
        """Get final (mutated) scenario database state."""
        return self.tool_handler.db

    # ── Shared tool execution ─────────────────────────────────────────

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool call and record it in the audit log.

        Logs the call and response as separate timestamped entries so latency
        between them is preserved.  Use this whenever the server handles tool
        calls directly (s2s/realtime events, or any custom cascade that
        doesn't delegate to AgenticSystem).

        Note: AgenticSystem has its own tool execution + logging loop
        (``append_tool_call``), so Pipecat cascade pipelines that use
        AgenticSystem should *not* also call this method.
        """
        self.audit_log.append_realtime_tool_call(tool_name, arguments)
        result = await self.tool_handler.execute(tool_name, arguments)
        self.audit_log.append_tool_response(tool_name, result)
        return result

    # ── Shared output helpers ──────────────────────────────────────────

    async def save_outputs(self) -> None:
        """Save all required output files. Called by stop().

        Subclasses can override to add framework-specific outputs,
        but must call super().save_outputs().
        """
        # Save audit log
        self.audit_log.save(self.output_dir / "audit_log.json")

        # Save simplified transcript
        transcript_path = self.output_dir / "transcript.jsonl"
        self.audit_log.save_transcript_jsonl(transcript_path)

        # Save audio recordings
        self._save_audio()

        # Save scenario database states (REQUIRED for deterministic metrics)
        self._save_scenario_dbs()

        logger.info(f"Outputs saved to {self.output_dir}")

    def _save_audio(self) -> None:
        """Save accumulated audio buffers to WAV files.

        If _audio_buffer (mixed) is empty but user and assistant buffers are
        available, compute mixed audio automatically via sample-wise addition.

        NOTE: user_audio_buffer and assistant_audio_buffer must be time-aligned
        (same total length in samples) before this method is called.  S2s/realtime
        servers are responsible for calling ``sync_buffer_to_position`` during
        streaming so the two tracks stay aligned.  A length mismatch produces a
        usable but temporally skewed mixed recording.
        """
        # Auto-compute mixed audio from user + assistant tracks when not populated
        if not self._audio_buffer and self.user_audio_buffer and self.assistant_audio_buffer:
            diff_bytes = abs(len(self.user_audio_buffer) - len(self.assistant_audio_buffer))
            diff_ms = diff_bytes / (2 * self._audio_sample_rate) * 1000  # 16-bit PCM → 2 bytes/sample
            if diff_ms > 500:
                logger.warning(
                    f"Audio buffer length mismatch: user={len(self.user_audio_buffer)} "
                    f"assistant={len(self.assistant_audio_buffer)} "
                    f"diff={diff_ms:.0f}ms — mixed recording may be temporally skewed"
                )
            from eva.assistant.audio_bridge import pcm16_mix

            self._audio_buffer = bytearray(pcm16_mix(bytes(self.user_audio_buffer), bytes(self.assistant_audio_buffer)))
        elif not self._audio_buffer and self.user_audio_buffer:
            self._audio_buffer = bytearray(self.user_audio_buffer)
        elif not self._audio_buffer and self.assistant_audio_buffer:
            self._audio_buffer = bytearray(self.assistant_audio_buffer)

        if self._audio_buffer:
            self._save_wav_file(
                bytes(self._audio_buffer),
                self.output_dir / "audio_mixed.wav",
                self._audio_sample_rate,
                1,
            )
        if self.user_audio_buffer:
            self._save_wav_file(
                bytes(self.user_audio_buffer),
                self.output_dir / "audio_user.wav",
                self._audio_sample_rate,
                1,
            )
        if self.assistant_audio_buffer:
            self._save_wav_file(
                bytes(self.assistant_audio_buffer),
                self.output_dir / "audio_assistant.wav",
                self._audio_sample_rate,
                1,
            )

    def _save_wav_file(self, audio_data: bytes, file_path: Path, sample_rate: int, num_channels: int) -> None:
        """Save raw 16-bit PCM audio data to a WAV file."""
        try:
            with wave.open(str(file_path), "wb") as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            logger.debug(f"Audio saved to {file_path} ({len(audio_data)} bytes)")
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")

    def _save_scenario_dbs(self) -> None:
        """Save initial and final scenario database states."""
        try:
            initial_db_path = self.output_dir / "initial_scenario_db.json"
            with open(initial_db_path, "w") as f:
                json.dump(self.get_initial_scenario_db(), f, indent=2, sort_keys=True, default=str)

            final_db_path = self.output_dir / "final_scenario_db.json"
            with open(final_db_path, "w") as f:
                json.dump(self.get_final_scenario_db(), f, indent=2, sort_keys=True, default=str)

            logger.info(f"Saved scenario database states to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error saving scenario database states: {e}", exc_info=True)
            raise
