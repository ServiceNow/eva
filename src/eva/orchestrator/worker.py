"""Conversation worker for running individual conversations."""

import asyncio
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from eva.assistant.base_server import AbstractAssistantServer
from eva.models.agents import AgentConfig
from eva.models.config import RunConfig
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult, ErrorDetails, LatencyStats
from eva.user_simulator.client import UserSimulator
from eva.utils.error_handler import create_error_details
from eva.utils.hash_utils import get_dict_hash
from eva.utils.logging import add_record_log_file, current_record_id, get_logger, remove_record_log_file

logger = get_logger(__name__)


def _get_server_class(framework: str) -> type[AbstractAssistantServer]:
    """Return the server class for the given framework name.

    Uses lazy imports to avoid importing heavy dependencies (pipecat, openai, etc.)
    unless the framework is actually selected.
    """
    if framework == "pipecat":
        from eva.assistant.pipecat_server import PipecatAssistantServer

        return PipecatAssistantServer
    elif framework == "openai_realtime":
        from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer

        return OpenAIRealtimeAssistantServer
    elif framework == "gemini_live":
        from eva.assistant.gemini_live_server import GeminiLiveAssistantServer

        return GeminiLiveAssistantServer
    else:
        raise ValueError(
            f"Unknown framework: {framework!r}. Supported: pipecat, openai_realtime, gemini_live, elevenlabs, deepgram"
        )


def _percentile(sorted_data: list[float], p: float) -> float:
    """Calculate the p-th percentile using the nearest-rank method.

    The nearest-rank percentile is the smallest value in the sorted dataset
    such that at least p% of the data falls at or below it.

    Args:
        sorted_data: Pre-sorted list of values (ascending).
        p: Percentile in (0, 100].

    Returns:
        The percentile value.
    """
    n = len(sorted_data)
    rank = math.ceil(p / 100.0 * n)
    return sorted_data[rank - 1]


class ConversationWorker:
    """Runs a single conversation between assistant and user simulator.

    Each worker manages:
    - Starting the assistant server on an assigned port
    - Connecting the user simulator
    - Running the conversation until completion or timeout
    - Collecting outputs (audio, transcripts, logs)
    """

    def __init__(
        self,
        config: RunConfig,
        record: EvaluationRecord,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_base_path: str,
        output_dir: Path,
        port: int,
        output_id: str,
    ):
        """Initialize the conversation worker.

        Args:
            config: Run configuration
            record: Evaluation record to run
            agent: Single agent configuration to use
            agent_config_path: Path to agent YAML configuration
            scenario_base_path: Base path for scenario files (will append record ID)
            output_dir: Output directory for this record
            port: WebSocket server port to use
            output_id: Output identifier (may include trial suffix like "1.2.1/trial_0")
        """
        self.config = config
        self.record = record
        self.agent = agent
        self.agent_config_path = agent_config_path
        self.scenario_db_path = f"{scenario_base_path}/{record.id}.json"
        self.output_dir = output_dir
        self.port = port
        self.output_id = output_id

        # Will be set during run
        self._assistant_server = None
        self._user_simulator = None
        self._conversation_stats: dict[str, Any] = {}
        self._log_file_handler = None

    async def run(self) -> ConversationResult:
        """Execute one complete conversation.

        Returns:
            ConversationResult with details about the conversation
        """
        started_at = datetime.now()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tag this asyncio task so per-record FileHandlers only capture
        # logs emitted by *this* worker (not other concurrent workers).
        # Use output_id (not record.id) to differentiate trials of the same record.
        current_record_id.set(self.output_id)

        # Add file handler to capture all logs for this record/trial
        log_file_path = self.output_dir / "logs.log"
        self._log_file_handler = add_record_log_file(self.output_id, str(log_file_path))

        logger.info(f"Starting conversation for record {self.record.id} on port {self.port}")

        conversation_ended_reason: str | None = None
        error: str | None = None
        error_details: ErrorDetails | None = None

        try:
            # 1. Start assistant server
            await self._start_assistant()
            logger.debug(f"Assistant server started on port {self.port}")

            # 2. Connect user simulator
            await self._start_user_simulator()
            logger.debug("User simulator connected")

            # 3. Run conversation until completion or timeout
            try:
                conversation_ended_reason = await asyncio.wait_for(
                    self._run_conversation(),
                    timeout=self.config.conversation_timeout_seconds,
                )
                logger.info(f"Conversation {self.record.id} ended: {conversation_ended_reason}")
            except TimeoutError:
                conversation_ended_reason = "timeout"
                logger.warning(f"Conversation {self.record.id} timed out")
            except asyncio.CancelledError:
                conversation_ended_reason = "cancelled"
                logger.info(f"Conversation {self.record.id} was cancelled")

        except asyncio.CancelledError:
            conversation_ended_reason = "cancelled"
            logger.info(f"Conversation {self.record.id} was cancelled during setup")

        except Exception as e:
            error = str(e)
            conversation_ended_reason = "error"
            logger.error(f"Conversation {self.record.id} error: {e}", exc_info=True)

            # Create structured error details using centralized error handler
            error_details = create_error_details(
                error=e,
                retry_count=0,
                retry_succeeded=False,
            )
        finally:
            await self._cleanup()
            # Remove the log file handler after cleanup is complete
            if self._log_file_handler:
                remove_record_log_file(self._log_file_handler)
                self._log_file_handler = None

        # If the conversation errored, return a failed result immediately. DB hashes or latency stats cannot be computed if the run did not complete.
        if error is not None:
            now = datetime.now()
            return ConversationResult(
                record_id=self.record.id,
                completed=False,
                error=error,
                error_details=error_details,
                started_at=started_at,
                ended_at=now,
                duration_seconds=(now - started_at).total_seconds(),
                output_dir=str(self.output_dir),
                conversation_ended_reason="error",
            )

        ended_at = datetime.now()

        # Compute scenario database hashes (REQUIRED for deterministic metrics)
        initial_db_path = self.output_dir / "initial_scenario_db.json"
        final_db_path = self.output_dir / "final_scenario_db.json"

        if not initial_db_path.exists():
            raise FileNotFoundError(
                f"Initial scenario database not found at {initial_db_path}. "
                "This is required for deterministic task completion metrics."
            )
        if not final_db_path.exists():
            raise FileNotFoundError(
                f"Final scenario database not found at {final_db_path}. "
                "This is required for deterministic task completion metrics."
            )

        with open(initial_db_path) as f:
            initial_db = json.load(f)
        with open(final_db_path) as f:
            final_db = json.load(f)

        initial_scenario_db_hash = get_dict_hash(initial_db)
        final_scenario_db_hash = get_dict_hash(final_db)

        logger.info(
            f"Computed scenario DB hashes - Initial: {initial_scenario_db_hash[:8]}..., "
            f"Final: {final_scenario_db_hash[:8]}..."
        )

        # Calculate latency statistics
        llm_latency = self._calculate_llm_latency()
        stt_latency = self._calculate_stt_latency()
        tts_latency = self._calculate_tts_latency()
        model_response_latency = self._calculate_model_response_latency()

        return ConversationResult(
            record_id=self.record.id,
            completed=error is None and conversation_ended_reason != "error",
            error=error,
            error_details=error_details,
            llm_latency=llm_latency,
            stt_latency=stt_latency,
            tts_latency=tts_latency,
            model_response_latency=model_response_latency,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=(ended_at - started_at).total_seconds(),
            output_dir=str(self.output_dir),
            audio_assistant_path=str(self.output_dir / "audio_assistant.wav"),
            audio_user_path=str(self.output_dir / "audio_user.wav"),
            audio_mixed_path=str(self.output_dir / "audio_mixed.wav"),
            transcript_path=str(self.output_dir / "transcript.jsonl"),
            audit_log_path=str(self.output_dir / "audit_log.json"),
            conversation_log_path=str(self.output_dir / "logs.log"),
            pipecat_logs_path=self._resolve_framework_logs_path(),
            elevenlabs_logs_path=str(self.output_dir / "elevenlabs_events.jsonl"),
            num_turns=self._conversation_stats.get("num_turns", 0),
            num_tool_calls=self._conversation_stats.get("num_tool_calls", 0),
            tools_called=self._conversation_stats.get("tools_called", []),
            conversation_ended_reason=conversation_ended_reason,
            initial_scenario_db_hash=initial_scenario_db_hash,
            final_scenario_db_hash=final_scenario_db_hash,
        )

    async def _start_assistant(self) -> None:
        """Start the assistant server using the configured framework."""
        server_cls = _get_server_class(self.config.framework)
        self._assistant_server = server_cls(
            current_date_time=self.record.current_date_time,
            pipeline_config=self.config.model,
            agent=self.agent,
            agent_config_path=self.agent_config_path,
            scenario_db_path=self.scenario_db_path,
            output_dir=self.output_dir,
            port=self.port,
            conversation_id=self.record.id,
        )

        await self._assistant_server.start()

    async def _start_user_simulator(self) -> None:
        """Start the user simulator."""
        self._user_simulator = UserSimulator(
            current_date_time=self.record.current_date_time,
            persona_config=self.record.user_config,
            goal=self.record.user_goal,
            server_url=f"ws://localhost:{self.port}/ws",
            output_dir=self.output_dir,
            user_simulator_context=self.agent.user_simulator_context,
            perturbation_config=self.config.perturbation,
        )

    async def _run_conversation(self) -> str:
        """Run the conversation until completion.

        Returns:
            Reason the conversation ended
        """
        if self._user_simulator is None:
            raise RuntimeError("User simulator not initialized")

        ended_reason = await self._user_simulator.run_conversation()

        # Collect stats from assistant
        if self._assistant_server:
            self._conversation_stats = self._assistant_server.get_conversation_stats()

        return ended_reason

    def _resolve_framework_logs_path(self) -> str:
        """Resolve the framework/pipecat logs path, preferring framework_logs.jsonl."""
        framework_path = self.output_dir / "framework_logs.jsonl"
        pipecat_path = self.output_dir / "pipecat_logs.jsonl"
        if framework_path.exists():
            return str(framework_path)
        return str(pipecat_path)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._assistant_server:
            try:
                await self._assistant_server.stop()
            except Exception as e:
                logger.warning(f"Error stopping assistant server: {e}")
            self._assistant_server = None

        if self._user_simulator:
            self._user_simulator = None

    def _calculate_stt_latency(self) -> LatencyStats | None:
        """Calculate STT latency statistics from pipecat_metrics.jsonl.

        Accepts both Pipecat-native ProcessingMetricsData entries (written by
        MetricsFileObserver) and LatencyMetric entries with stage="stt" (written
        by MetricsLogWriter for non-Pipecat cascade frameworks).
        """
        metrics_path = self.output_dir / "pipecat_metrics.jsonl"
        if not metrics_path.exists():
            return None

        try:
            latencies = []
            with open(metrics_path) as f:
                for line in f:
                    metric = json.loads(line)
                    value_sec = metric.get("value")
                    if not value_sec or not (0 < value_sec < 30):
                        continue
                    if metric.get("type") == "ProcessingMetricsData" and "STTService" in metric.get("processor", ""):
                        latencies.append(value_sec * 1000)
                    elif metric.get("type") == "LatencyMetric" and metric.get("stage") == "stt":
                        latencies.append(value_sec * 1000)

            if not latencies:
                return None

            latencies.sort()
            n = len(latencies)
            return LatencyStats(
                mean_ms=sum(latencies) / n,
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                p99_ms=_percentile(latencies, 99),
                total_calls=n,
            )

        except Exception as e:
            logger.warning(f"Failed to calculate STT latency: {e}")
            return None

    def _calculate_tts_latency(self) -> LatencyStats | None:
        """Calculate TTS latency statistics from pipecat_metrics.jsonl.

        Accepts both Pipecat-native TTFBMetricsData entries and LatencyMetric
        entries with stage="tts".
        """
        metrics_path = self.output_dir / "pipecat_metrics.jsonl"
        if not metrics_path.exists():
            return None

        try:
            latencies = []
            with open(metrics_path) as f:
                for line in f:
                    metric = json.loads(line)
                    value_sec = metric.get("value")
                    if not value_sec or not (0 < value_sec < 10):
                        continue
                    if metric.get("type") == "TTFBMetricsData" and "TTSService" in metric.get("processor", ""):
                        latencies.append(value_sec * 1000)
                    elif metric.get("type") == "LatencyMetric" and metric.get("stage") == "tts":
                        latencies.append(value_sec * 1000)

            if not latencies:
                return None

            latencies.sort()
            n = len(latencies)
            return LatencyStats(
                mean_ms=sum(latencies) / n,
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                p99_ms=_percentile(latencies, 99),
                total_calls=n,
            )

        except Exception as e:
            logger.warning(f"Failed to calculate TTS latency: {e}")
            return None

    def _calculate_model_response_latency(self) -> LatencyStats | None:
        """Calculate model response latency for s2s/realtime frameworks.

        Reads LatencyMetric entries with stage="model_response" from
        pipecat_metrics.jsonl. These measure time from user speech end to
        first audio chunk from the model — the end-to-end latency users perceive
        for s2s models where STT and TTS are not separate stages.
        """
        metrics_path = self.output_dir / "pipecat_metrics.jsonl"
        if not metrics_path.exists():
            return None

        try:
            latencies = []
            with open(metrics_path) as f:
                for line in f:
                    metric = json.loads(line)
                    if metric.get("type") == "LatencyMetric" and metric.get("stage") == "model_response":
                        value_sec = metric.get("value")
                        if value_sec and 0 < value_sec < 30:
                            latencies.append(value_sec * 1000)

            if not latencies:
                return None

            latencies.sort()
            n = len(latencies)
            return LatencyStats(
                mean_ms=sum(latencies) / n,
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                p99_ms=_percentile(latencies, 99),
                total_calls=n,
            )

        except Exception as e:
            logger.warning(f"Failed to calculate model response latency: {e}")
            return None

    def _calculate_llm_latency(self) -> LatencyStats | None:
        """Calculate LLM latency statistics from audit log.

        LLM latency = time from LLM call start to response completion

        Returns:
            LatencyStats if audit log exists with latency data, None otherwise
        """
        audit_log_path = self.output_dir / "audit_log.json"
        if not audit_log_path.exists():
            return None

        try:
            # Load audit log
            with open(audit_log_path) as f:
                audit_log = json.load(f)

            # Extract latency_ms from all LLM calls
            latencies = []
            for llm_call in audit_log.get("llm_prompts", []):
                latency_ms = llm_call.get("latency_ms")
                if latency_ms is not None and latency_ms > 0:
                    # Sanity check: 0-60 seconds (60000 ms)
                    if 0 < latency_ms < 60000:
                        latencies.append(latency_ms)

            if not latencies:
                return None

            # Calculate statistics
            latencies.sort()
            n = len(latencies)

            return LatencyStats(
                mean_ms=sum(latencies) / n,
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                p99_ms=_percentile(latencies, 99),
                total_calls=n,
            )

        except Exception as e:
            logger.warning(f"Failed to calculate LLM latency: {e}")
            return None
