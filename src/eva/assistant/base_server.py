"""Abstract base class for assistant server implementations.

All framework-specific assistant servers (Pipecat, OpenAI Realtime, Gemini Live, etc.)
must inherit from AbstractAssistantServer and implement the required interface.

See docs/assistant_server_contract.md for the full specification.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.models.config import AudioLLMConfig, PipelineConfig, SpeechToSpeechConfig
from eva.utils.audio_utils import save_pcm_as_wav
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

    async def stop(self) -> asyncio.Task | None:
        """Stop the server: shut down framework, extract audio, save outputs.

        Concrete template method — subclasses implement _shutdown() instead of stop().

        Sequence:
        1. _shutdown(): framework-specific teardown (server stop, task cancellation)
        2. Auto-compute mixed audio from tracks if not already populated
        3. Extract and clear audio buffers so the caller can release its concurrency
           slot while audio hits disk
        4. save_outputs(): persist audit log, transcript, scenario DBs
        5. Return a deferred asyncio.Task for audio disk writes

        Returns:
            asyncio.Task that completes when audio files are written, or None if
            no audio was recorded.
        """
        await self._shutdown()

        # Auto-compute mixed audio from tracks if not already populated (S2S servers
        # populate user/assistant tracks but not the mixed buffer directly).
        if not self._audio_buffer:
            if self.user_audio_buffer and self.assistant_audio_buffer:
                diff_bytes = abs(len(self.user_audio_buffer) - len(self.assistant_audio_buffer))
                diff_ms = diff_bytes / (2 * self._audio_sample_rate) * 1000
                if diff_ms > 500:
                    logger.warning(
                        f"Audio buffer length mismatch: user={len(self.user_audio_buffer)} "
                        f"assistant={len(self.assistant_audio_buffer)} "
                        f"diff={diff_ms:.0f}ms — mixed recording may be temporally skewed"
                    )
                from eva.assistant.audio_bridge import pcm16_mix  # lazy: avoids circular import at module load

                self._audio_buffer = bytearray(
                    pcm16_mix(bytes(self.user_audio_buffer), bytes(self.assistant_audio_buffer))
                )
            elif self.user_audio_buffer:
                self._audio_buffer = bytearray(self.user_audio_buffer)
            elif self.assistant_audio_buffer:
                self._audio_buffer = bytearray(self.assistant_audio_buffer)

        # Extract bytes and clear in-memory buffers so the caller can release its
        # concurrency slot while audio writes happen in a background thread.
        mixed_audio = bytes(self._audio_buffer)
        user_audio = bytes(self.user_audio_buffer)
        assistant_audio = bytes(self.assistant_audio_buffer)
        sample_rate = self._audio_sample_rate
        self._audio_buffer.clear()
        self.user_audio_buffer.clear()
        self.assistant_audio_buffer.clear()

        await self.save_outputs()

        if mixed_audio or user_audio or assistant_audio:
            return asyncio.create_task(
                asyncio.to_thread(self._save_audio_deferred, mixed_audio, user_audio, assistant_audio, sample_rate)
            )
        return None

    @abstractmethod
    async def _shutdown(self) -> None:
        """Framework-specific shutdown: stop server, cancel tasks, etc.

        Called by stop() before audio buffer extraction. Implementations should:
        1. Check / set the running flag
        2. Stop the WebSocket server (set should_exit, await server task)
        3. Cancel any pending framework tasks (pipeline, session, etc.)
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

        Note: audio files are NOT saved here — they are written by the deferred
        asyncio.Task returned by stop() so the concurrency slot is freed first.
        """
        # Save audit log
        self.audit_log.save(self.output_dir / "audit_log.json")

        # Save transcript (subclasses can override _save_transcript for custom logic)
        self._save_transcript()

        # Save scenario database states (REQUIRED for deterministic metrics)
        self._save_scenario_dbs()

        logger.info(f"Outputs saved to {self.output_dir}")

    def _save_transcript(self) -> None:
        """Save transcript.jsonl from the audit log.

        Subclasses can override to customize transcript handling (e.g. conditional
        overwrite logic for S2S vs pipeline modes).
        """
        self.audit_log.save_transcript_jsonl(self.output_dir / "transcript.jsonl")

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
            save_pcm_as_wav(
                bytes(self._audio_buffer),
                self.output_dir / "audio_mixed.wav",
                self._audio_sample_rate,
                1,
            )
        if self.user_audio_buffer:
            save_pcm_as_wav(
                bytes(self.user_audio_buffer),
                self.output_dir / "audio_user.wav",
                self._audio_sample_rate,
                1,
            )
        if self.assistant_audio_buffer:
            save_pcm_as_wav(
                bytes(self.assistant_audio_buffer),
                self.output_dir / "audio_assistant.wav",
                self._audio_sample_rate,
                1,
            )

    def _save_audio_deferred(
        self,
        mixed_audio: bytes,
        user_audio: bytes,
        assistant_audio: bytes,
        sample_rate: int,
    ) -> None:
        """Write pre-extracted audio bytes to WAV files off the event loop."""
        if mixed_audio:
            save_pcm_as_wav(mixed_audio, self.output_dir / "audio_mixed.wav", sample_rate, 1)
        if user_audio:
            save_pcm_as_wav(user_audio, self.output_dir / "audio_user.wav", sample_rate, 1)
        if assistant_audio:
            save_pcm_as_wav(assistant_audio, self.output_dir / "audio_assistant.wav", sample_rate, 1)
        if mixed_audio or user_audio or assistant_audio:
            logger.info(f"Saved audio files to {self.output_dir} ({len(mixed_audio)} bytes mixed)")

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
