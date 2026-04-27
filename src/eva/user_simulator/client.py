"""User simulator client using ElevenLabs Conversational AI.

This module creates a simulated user that connects to the assistant server
using ElevenLabs Conversational AI as the user simulation engine.
"""

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path

import httpx
import yaml
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    ConversationInitiationData,
)

from eva.models.config import PerturbationConfig
from eva.user_simulator.audio_interface import ELEVENLABS_OUTPUT_RATE, BotToBotAudioInterface
from eva.user_simulator.event_logger import ElevenLabsEventLogger
from eva.user_simulator.perturbation import AudioPerturbator
from eva.utils.audio_utils import save_pcm_as_wav
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

_BEHAVIORS_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "user_behaviors.yaml"
_PERSONA_GENDER = {1: "F", 2: "M"}


@lru_cache(maxsize=1)
def _load_behavior_prompts() -> dict:
    with open(_BEHAVIORS_PATH) as f:
        return yaml.safe_load(f)


class UserSimulator:
    """ElevenLabs-based user simulator that connects to the assistant.

    Uses ElevenLabs Conversational AI to simulate a real user:
    - Generates natural speech based on persona and goal
    - Responds to assistant speech in real-time
    - Detects conversation end conditions (goodbye, transfer, etc.)
    """

    def __init__(
        self,
        current_date_time: str,
        persona_config: dict,
        goal: dict,
        server_url: str,
        output_dir: Path,
        agent_id: str,
        timeout: int = 600,
        perturbation_config: PerturbationConfig | None = None,
    ):
        """Initialize the user simulator.

        Args:
            current_date_time: Current date/time string from the evaluation record
            persona_config: User persona configuration (includes ElevenLabs agent_id)
            goal: Description of what the user wants to accomplish
            server_url: WebSocket URL of the assistant server
            output_dir: Directory for output files
            timeout: Conversation timeout in seconds
            agent_id: Agent identifier used to select the domain-specific simulator prompt
            perturbation_config: Optional perturbation to apply to user audio
        """
        self.persona_config = persona_config
        self.goal = goal
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.current_date_time = current_date_time
        self.agent_id = agent_id
        self._perturbation_config = perturbation_config
        self._perturbator = (
            AudioPerturbator(perturbation_config)
            if perturbation_config is not None
            and (perturbation_config.background_noise is not None or perturbation_config.connection_degradation)
            else None
        )

        # State
        self._conversation = None
        self._audio_interface: BotToBotAudioInterface | None = None
        self._end_reason: str = "unknown"
        self._conversation_done = asyncio.Event()

        # Event logger
        self.event_logger = ElevenLabsEventLogger(self.output_dir / "elevenlabs_events.jsonl")

        # Audio recording buffers
        self._user_audio_chunks: list[bytes] = []
        self._assistant_audio_chunks: list[bytes] = []
        self._user_clean_audio_chunks: list[bytes] = []

        # Keep-alive inactivity detection
        self._consecutive_keepalive_count = 0
        self._max_consecutive_keepalives = 12  # End call after this many pings without activity (2 minutes)

    def _on_conversation_end(self, reason: str = "goodbye") -> None:
        """Signal conversation completion.

        Thread-safe - can be called from any thread/callback.
        Only the first call takes effect (Event.set() is idempotent).

        Args:
            reason: Why conversation ended (goodbye/transfer/error)
        """
        if not self._conversation_done.is_set():
            self._end_reason = reason
            self._conversation_done.set()
            logger.info(f"Conversation end signaled: {reason}")

    async def run_conversation(self) -> str:
        """Run the conversation until completion.

        Returns:
            Reason the conversation ended:
            - "goodbye": Natural conversation end
            - "transfer": Assistant initiated transfer
            - "timeout": Conversation timed out
            - "error": Error occurred
        """
        # Check for ElevenLabs API key
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not set")
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        try:
            return await self._run_elevenlabs_conversation(api_key)
        except Exception as e:
            logger.error(f"ElevenLabs conversation error: {e}", exc_info=True)
            self._end_reason = "error"
            self.event_logger.log_error(str(e))
            return self._end_reason
        finally:
            # Save event log
            self.event_logger.save()

    async def _run_elevenlabs_conversation(self, api_key: str) -> str:
        """Run conversation using ElevenLabs Conversational AI."""
        # Generate conversation ID
        conversation_id = self.output_dir.name

        # Create audio interface
        self._audio_interface = BotToBotAudioInterface(
            websocket_uri=self.server_url,
            conversation_id=conversation_id,
            record_callback=self._record_audio,
            event_logger=self.event_logger,
            conversation_done_callback=self._on_conversation_end,
            perturbator=self._perturbator,
        )

        # Start the audio interface WebSocket connection
        await self._audio_interface.start_async()
        self.event_logger.log_connection_state("connected", {"server_url": self.server_url})

        try:
            # Create ElevenLabs client with custom httpx client (no SSL verification for local testing)
            http_client = httpx.Client(verify=False, timeout=30.0)
            client = ElevenLabs(
                api_key=api_key,
                timeout=30.0,
                httpx_client=http_client,
            )

            # TODO: test and improve behavior prompts to more closely match desired user behavior
            behavior_prompts = _load_behavior_prompts()
            if self._perturbation_config and self._perturbation_config.behavior:
                behavior_key = self._perturbation_config.behavior.value
                user_persona = behavior_prompts[behavior_key]
            else:
                user_persona = behavior_prompts["default"]

            # Derive domain from agent_id (e.g. "agent_airline" → "airline")
            domain = self.agent_id.removeprefix("agent_")
            prompt = PromptManager().get_prompt(
                f"user_simulator.system_prompt_{domain}",
                high_level_user_goal=self.goal["high_level_user_goal"],
                must_have_criteria=self.goal["decision_tree"]["must_have_criteria"],
                escalation_behavior=self.goal["decision_tree"]["escalation_behavior"],
                nice_to_have_criteria=self.goal["decision_tree"]["nice_to_have_criteria"],
                negotiation_behavior=self.goal["decision_tree"]["negotiation_behavior"],
                resolution_condition=self.goal["decision_tree"]["resolution_condition"],
                failure_condition=self.goal["decision_tree"]["failure_condition"],
                edge_cases=self.goal["decision_tree"]["edge_cases"],
                information_required=self.goal["information_required"],
                user_persona=user_persona,
                starting_utterance=self.goal["starting_utterance"],
                current_date_time=self.current_date_time,
            )

            # Create conversation config with dynamic variables
            config = ConversationInitiationData(dynamic_variables={"prompt": prompt})

            # ElevenLabs user simulator agent ID
            persona_id = self.persona_config["user_persona_id"]
            gender = _PERSONA_GENDER[persona_id]
            if self._perturbation_config and self._perturbation_config.accent:
                key = self._perturbation_config.accent.value.upper()
                env_var = f"EVA_{key}_ACCENT_USER_{gender}"
            elif self._perturbation_config and self._perturbation_config.behavior:
                key = self._perturbation_config.behavior.value.upper()
                env_var = f"EVA_{key}_USER_{gender}"
            else:
                env_var = f"EVA_DEFAULT_USER_{gender}"
            ELEVENLABS_USER_AGENT_ID = os.getenv(env_var)
            logger.info(f"Using agent ID from env var: {env_var}")

            # Create the conversation
            if not ELEVENLABS_USER_AGENT_ID:
                raise ValueError(f"Missing ElevenLabs agent ID environment variable: {env_var}")

            self._client = client

            self._conversation = Conversation(
                client,
                ELEVENLABS_USER_AGENT_ID,
                config=config,
                requires_auth=True,
                audio_interface=self._audio_interface,
                callback_agent_response=self._on_user_speaks,
                callback_agent_response_correction=self._on_user_response_correction,
                callback_user_transcript=self._on_assistant_speaks,
            )

            # Start the conversation session (blocking call, run in executor)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._conversation.start_session)
            logger.info("ElevenLabs conversation started successfully")
            self.event_logger.log_connection_state("session_started")

            # Start keep-alive task to prevent ElevenLabs timeout
            keep_alive_task = asyncio.create_task(self._keep_alive_task())

            # Wait for conversation to end or timeout
            try:
                await asyncio.wait_for(self._conversation_done.wait(), timeout=self.timeout)
                logger.info(f"Conversation ended: {self._end_reason}")
            except TimeoutError:
                logger.info(f"Conversation timed out after {self.timeout}s")
                self._end_reason = "timeout"
                self.event_logger.log_event("timeout", {"duration": self.timeout})
            finally:
                # Cancel keep-alive task when conversation ends
                keep_alive_task.cancel()
                try:
                    await keep_alive_task
                except asyncio.CancelledError:
                    pass

            # End the session
            logger.info("Ending ElevenLabs session...")
            self._conversation.end_session()

            # Post-hoc check: detect end_call tool via ElevenLabs Conversations API
            # The conversation may still be "in-progress" immediately after end_session(),
            # so we poll with backoff until the transcript is available.
            conversation_id = getattr(self._conversation, "_conversation_id", None)
            if conversation_id:
                try:
                    end_call_found = await self._check_end_call_via_api(conversation_id)
                    if end_call_found:
                        self._end_reason = "goodbye"
                except Exception as e:
                    logger.warning(f"Failed to check conversation history for end_call: {e}")

                try:
                    await self._fetch_elevenlabs_audio(conversation_id)
                except Exception as e:
                    logger.warning(f"Failed to fetch ElevenLabs server audio: {e}")

            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

        except Exception as e:
            logger.error(f"Error in ElevenLabs conversation: {e}", exc_info=True)
            self._end_reason = "error"
            raise
        finally:
            # Save response latencies from audio interface before cleanup
            if self._audio_interface:
                latencies = self._audio_interface.get_latencies()
                if latencies:
                    latency_file = self.output_dir / "response_latencies.json"
                    with open(latency_file, "w") as f:
                        json.dump(
                            {
                                "latencies": latencies,
                                "mean": sum(latencies) / len(latencies),
                                "max": max(latencies),
                                "count": len(latencies),
                            },
                            f,
                            indent=2,
                        )
                    logger.info(f"Saved {len(latencies)} response latencies to {latency_file}")

            if self._user_clean_audio_chunks:
                clean_audio_path = self.output_dir / "audio_user_clean.wav"
                save_pcm_as_wav(
                    b"".join(self._user_clean_audio_chunks),
                    clean_audio_path,
                    sample_rate=ELEVENLABS_OUTPUT_RATE,
                    num_channels=1,
                )
                logger.info(f"Saved clean user audio to {clean_audio_path}")

            # Grace period: keep the WebSocket open so the assistant pipeline
            # (Pipecat STT) can finish processing the last user utterance.
            # Observed delay from "User audio END" to "UserStoppedSpeaking"
            logger.info("Waiting 4s for assistant STT to finish last utterance...")
            await asyncio.sleep(4.0)
            await self._audio_interface.stop_async()

        return self._end_reason

    async def _check_end_call_via_api(self, conversation_id: str) -> bool:
        """Check ElevenLabs Conversations API for end_call tool invocation.

        Polls with exponential backoff since the transcript may not be available
        immediately after end_session() (conversation status may be "in-progress").

        Args:
            conversation_id: The ElevenLabs conversation ID to check.

        Returns:
            True if end_call was found in the transcript, False otherwise.
        """
        max_attempts = 5
        delay = 2.0  # initial delay in seconds

        for attempt in range(max_attempts):
            await asyncio.sleep(delay)
            conv_details = self._client.conversational_ai.conversations.get(conversation_id)

            # Dump full response to file for debugging/analysis
            details_path = self.output_dir / "elevenlabs_conversation_details.json"
            try:
                with open(details_path, "w") as f:
                    json.dump(conv_details.model_dump(), f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to write conversation details to {details_path}: {e}")

            if conv_details.transcript:
                for turn in conv_details.transcript:
                    if turn.tool_results:
                        for tool_result in turn.tool_results:
                            if hasattr(tool_result, "tool_name") and tool_result.tool_name == "end_call":
                                logger.info("end_call tool detected via ElevenLabs API")
                                return True
                # Transcript populated but no end_call found
                logger.info("Conversation transcript available but no end_call tool found")
                return False

            # Transcript still empty, retry with backoff
            logger.debug(
                f"Conversation transcript not yet available (attempt {attempt + 1}/{max_attempts}, "
                f"status={conv_details.status})"
            )
            delay = min(delay * 2, 10.0)

        logger.warning(f"Conversation transcript still empty after {max_attempts} attempts")
        return False

    async def _fetch_elevenlabs_audio(self, conversation_id: str) -> None:
        max_attempts = 5
        delay = 2.0

        for attempt in range(max_attempts):
            try:
                audio_iter = self._client.conversational_ai.conversations.audio.get(conversation_id)
                audio_path = self.output_dir / "elevenlabs_audio_recording.mp3"
                with open(audio_path, "wb") as f:
                    f.writelines(audio_iter)
                logger.info(f"Saved ElevenLabs server-side audio to {audio_path}")
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.debug(f"Audio not yet available (attempt {attempt + 1}/{max_attempts}): {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
                else:
                    logger.warning(f"Failed to fetch ElevenLabs server audio after {max_attempts} attempts: {e}")

    def _reset_keepalive_counter(self) -> None:
        """Reset the consecutive keep-alive counter on user/agent activity."""
        self._consecutive_keepalive_count = 0

    async def _keep_alive_task(self) -> None:
        """Periodically ping ElevenLabs to prevent session timeout.

        Sends register_user_activity() every 10 seconds to keep the session alive.
        This prevents the ElevenLabs conversation from timing out during long LLM processing.

        If 12 consecutive keep-alives are sent without any user or agent activity,
        the conversation is ended to prevent stuck sessions.
        """
        try:
            while not self._conversation_done.is_set():
                await asyncio.sleep(10)  # Ping every 10 seconds

                if self._conversation and not self._conversation_done.is_set():
                    try:
                        # Send keep-alive ping to ElevenLabs (synchronous method, run in executor)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self._conversation.register_user_activity)

                        # Reset counter if assistant is actively speaking (audio streaming)
                        # _on_assistant_speaks transcript callback doesn't fire reliably
                        # during long utterances, but audio activity tracking is reliable
                        if self._audio_interface and self._audio_interface._assistant_audio_active:
                            self._reset_keepalive_counter()
                            logger.info("🔊 Assistant still speaking - resetting inactivity counter")
                        else:
                            self._consecutive_keepalive_count += 1
                            logger.info(
                                f"📡 Sent keep-alive ping to ElevenLabs "
                                f"({self._consecutive_keepalive_count}/{self._max_consecutive_keepalives})"
                            )

                        # End conversation if too many consecutive keep-alives without activity
                        if self._consecutive_keepalive_count >= self._max_consecutive_keepalives:
                            logger.warning(
                                f"Ending conversation: {self._max_consecutive_keepalives} consecutive "
                                "keep-alives without user/agent activity"
                            )
                            self._on_conversation_end("inactivity_timeout")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to send keep-alive ping: {e}")
        except asyncio.CancelledError:
            logger.info("Keep-alive task cancelled")
            raise

    def _on_user_speaks(self, response: str) -> None:
        """Callback when ElevenLabs (simulated user) generates a response.

        Args:
            response: The text that the simulated user said
        """
        self._reset_keepalive_counter()
        logger.info(f"🎭 User (ElevenLabs): {response}")

        self.event_logger.log_event(
            "user_speech",
            {
                "text": response,
                "source": "elevenlabs_agent",
            },
        )

    def _on_user_response_correction(self, original: str, corrected: str) -> None:
        """Callback when ElevenLabs corrects a user response.

        Args:
            original: Original response
            corrected: Corrected response
        """
        logger.debug(f"User response corrected: {original} -> {corrected}")

        self.event_logger.log_event(
            "user_speech_correction",
            {
                "original": original,
                "corrected": corrected,
            },
        )

    def _on_assistant_speaks(self, transcript: str) -> None:
        """Callback when the assistant (Pipecat bot) speaks.

        This is the transcript of what ElevenLabs heard from the assistant.

        Args:
            transcript: The text that the assistant said
        """
        self._reset_keepalive_counter()
        logger.info(f"🤖 Assistant: {transcript}")

        self.event_logger.log_event(
            "assistant_speech",
            {
                "text": transcript,
                "source": "pipecat_assistant",
            },
        )

    def _record_audio(self, source: str, audio_data: bytes) -> None:
        """Record audio for later analysis.

        Args:
            source: "user", "assistant", or "user_clean"
            audio_data: Raw audio bytes
        """
        if source == "user":
            self._user_audio_chunks.append(audio_data)
        elif source == "assistant":
            self._assistant_audio_chunks.append(audio_data)
        elif source == "user_clean":
            self._user_clean_audio_chunks.append(audio_data)

    def get_recorded_audio(self) -> tuple[bytes, bytes]:
        """Get the recorded audio.

        Returns:
            Tuple of (user_audio, assistant_audio) as raw bytes
        """
        user_audio = b"".join(self._user_audio_chunks)
        assistant_audio = b"".join(self._assistant_audio_chunks)
        return user_audio, assistant_audio
