"""Audio-LLM agentic system - extends AgenticSystem with audio input support.

Used with self-hosted model (via vLLM) where the model accepts audio input
+ text context and returns text output. All user turns include their original
audio so the model has full conversational context across turns.
"""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.agentic.system import AgenticSystem
from eva.assistant.pipeline.alm_base import BaseALMClient
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)


class AudioLLMAgenticSystem(AgenticSystem):
    """AgenticSystem variant that sends audio to the LLM for every user turn.

    Retains audio from all turns so the model sees full conversational context.

    Conversation context format:
    - [system_prompt, user_audio_1, assistant_1, ..., user_audio_N, (tool_results)]
    - All user messages contain their original audio (not text placeholders).
    - The audit log stores ``[user audio]`` placeholders for text transcripts;
      the actual audio is kept in ``_turn_audio_history``.
    """

    def __init__(
        self,
        current_date_time: str,
        agent: AgentConfig,
        tool_handler: ToolExecutor,
        audit_log: AuditLog,
        alm_client: BaseALMClient,
        output_dir: Path | None = None,
        pre_tool_speech: str = "off",
        llm_streaming: bool = False,
        full_audio_context: bool = False,
    ):
        super().__init__(
            current_date_time=current_date_time,
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            llm_client=alm_client,
            output_dir=output_dir,
            pre_tool_speech=pre_tool_speech,
            llm_streaming=llm_streaming,
        )
        self.alm_client: BaseALMClient = alm_client
        # When True, every user turn is sent as audio (full audio context). When False
        # (default), only the current turn carries audio and prior turns rely on their
        # text transcriptions, keeping context small.
        self.full_audio_context = full_audio_context

        # Override system prompt with audio-LLM specific version
        self.system_prompt = self.prompt_manager.get_prompt(
            "audio_llm_agent.system_prompt",
            agent_personality=agent.description,
            agent_instructions=agent.instructions,
            datetime=current_date_time,
        )
        # Reuse the shared pre-tool lead-in prompt appended to the system prompt.
        if self.pre_tool_speech == "auto":
            self.system_prompt += "\n\n" + self.prompt_manager.get_prompt("agent.pre_tool_speech")

        # Per-turn audio history: list of (audio_bytes, sample_rate)
        self._turn_audio_history: list[tuple[bytes, int]] = []
        # Optional text prepended to the current turn's audio message (consumed once). Used by the
        # turn-end fallback to have the model acknowledge it may not have heard the audio perfectly.
        self._turn_text_hint: str = ""

    def set_turn_audio(self, audio_bytes: bytes, sample_rate: int) -> None:
        """Record audio data for the current user turn.

        Called by the processor before process_query_with_audio().
        Audio is appended to the history and retained for all future LLM calls.
        """
        self._turn_audio_history.append((audio_bytes, sample_rate))

    def set_turn_text_hint(self, hint: str) -> None:
        """Set a text hint prepended to the next audio user message (consumed once).

        Used by the turn-end fallback so the model is told the audio may be incomplete /
        imperfectly heard and should acknowledge that before answering.
        """
        self._turn_text_hint = hint

    async def process_query_with_audio(self, user_text: str) -> AsyncGenerator[str, None]:
        """Process a user turn that has audio data.

        Args:
            user_text: Text label for this user turn in the audit log.
                      Typically a placeholder like ``[user audio]`` since the
                      audio-LLM model receives the raw audio directly and
                      no separate transcription is performed.

        Yields:
            Text responses to be sent to TTS.
        """
        logger.info(f"Processing audio query: {user_text}")

        # Note: User input is already added to audit log in AudioLLMProcessor.process_complete_user_turn()
        # before this method is called. This ensures the transcription callback can update it.

        # Execute agent with audio-aware message building
        async for response in self._execute_agent_with_audio(self.agent):
            yield response

    async def _execute_agent_with_audio(self, agent: AgentConfig) -> AsyncGenerator[str, None]:
        """Build messages with audio on user turns, then run tool loop.

        By default only the current (last) user turn is sent as audio; previous
        user turns remain as text (transcriptions updated via the parallel
        transcription pipeline), keeping context manageable. When
        ``full_audio_context`` is enabled, every user turn is sent as audio so the
        model has full conversational context without relying on transcriptions —
        at the cost of a much larger context.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Get conversation history (includes the current user input we just appended)
        conversation_history = self.audit_log.get_conversation_messages(max_messages=30)
        history_dicts = [msg.to_dict() for msg in conversation_history]

        # Text hint for the CURRENT turn's audio (e.g. a turn-end fallback acknowledgment).
        # Consumed once so it only applies to this turn.
        turn_text_hint = self._turn_text_hint
        self._turn_text_hint = ""

        if self._turn_audio_history:
            if self.full_audio_context:
                # Replace ALL user messages with their corresponding audio
                user_indices = [i for i, msg in enumerate(history_dicts) if msg.get("role") == "user"]
                for turn_idx, msg_idx in enumerate(user_indices):
                    if turn_idx < len(self._turn_audio_history):
                        audio_bytes, sample_rate = self._turn_audio_history[turn_idx]
                        # Only the current (last) turn gets the hint.
                        hint = turn_text_hint if turn_idx == len(user_indices) - 1 else ""
                        history_dicts[msg_idx] = self.alm_client.build_audio_user_message(
                            audio_bytes=audio_bytes,
                            source_sample_rate=sample_rate,
                            text_hint=hint,
                        )
            else:
                # Replace only the LAST user message with audio (current turn)
                last_user_idx = None
                for i in range(len(history_dicts) - 1, -1, -1):
                    if history_dicts[i].get("role") == "user":
                        last_user_idx = i
                        break

                if last_user_idx is not None:
                    audio_bytes, sample_rate = self._turn_audio_history[-1]
                    history_dicts[last_user_idx] = self.alm_client.build_audio_user_message(
                        audio_bytes=audio_bytes,
                        source_sample_rate=sample_rate,
                        text_hint=turn_text_hint,
                    )

        messages.extend(history_dicts)

        # Run the standard tool loop with audio-augmented messages
        async for response in self._run_tool_loop(messages, agent):
            yield response
