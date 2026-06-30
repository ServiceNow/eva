"""Provider-neutral user simulator contract and shared behavior."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import yaml
from pipecat.transcriptions.language import Language

from eva.models.config import LANGUAGE_DISPLAY_NAMES, PerturbationConfig
from eva.user_simulator.event_logger import UserSimulatorEventLogger
from eva.user_simulator.perturbation import AudioPerturbator
from eva.utils.culture import add_user_language_directive
from eva.utils.logging import current_record_id, get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

_BEHAVIORS_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "user_behaviors.yaml"

# Shared across realtime-style callers (OpenAI Realtime, Gemini Live) that expose
# an ``end_call`` function tool to let the simulated caller hang up.
END_CALL_DESCRIPTION = """Use this to end the phone call and hang up.

Call this function when it is time to end the call and one of the following is true:
1. The agent has confirmed your request is resolved, all steps are completed, and you have said goodbye.
2. The agent has initiated a transfer to a live agent.
3. The agent has been unable to make progress for at least 5 consecutive turns.
4. The agent says goodbye or indicates the conversation is over.
5. The agent indicates that the remainder of your request cannot be fulfilled.
6. The assistant reports an unrecoverable processing error.

Never call this tool in the same turn that you provide the agent with data, an identifier,
an approval to proceed, a transfer request, or any other information. Say a brief goodbye first."""


@lru_cache(maxsize=1)
def load_behavior_prompts() -> dict:
    """Load the shared user behavior prompt fragments."""
    with open(_BEHAVIORS_PATH) as f:
        return yaml.safe_load(f)


class AbstractUserSimulator(ABC):
    """Common lifecycle and artifact contract for simulated caller providers."""

    provider: str

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
        language: str = "en",
        *,
        provider: str,
    ) -> None:
        self.provider = provider
        self.persona_config = persona_config
        self.goal = goal
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.current_date_time = current_date_time
        self.agent_id = agent_id
        self._perturbation_config = perturbation_config
        self._language = language
        self._perturbator = (
            AudioPerturbator(perturbation_config)
            if perturbation_config is not None
            and (perturbation_config.background_noise is not None or perturbation_config.connection_degradation)
            else None
        )

        self._audio_interface = None
        self._end_reason = "unknown"
        self._conversation_done = asyncio.Event()

        self.event_logger = UserSimulatorEventLogger(
            self.output_dir / "user_simulator_events.jsonl",
            provider=provider,
        )

        self._user_audio_chunks: list[bytes] = []
        self._assistant_audio_chunks: list[bytes] = []
        self._user_clean_audio_chunks: list[bytes] = []
        self._record_id = current_record_id.get()

    @abstractmethod
    async def run_conversation(self) -> str:
        """Run until the simulated conversation reaches a terminal state."""

    def _build_prompt(self) -> str:
        behavior_prompts = load_behavior_prompts()
        if self._perturbation_config and self._perturbation_config.behavior:
            user_persona = behavior_prompts[self._perturbation_config.behavior.value]
        else:
            user_persona = behavior_prompts["default"]

        user_persona = add_user_language_directive(
            self._language,
            LANGUAGE_DISPLAY_NAMES.get(Language(self._language), self._language),
            user_persona,
        )

        domain = self.agent_id.removeprefix("agent_")
        return PromptManager().get_prompt(
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

    def _on_conversation_end(self, reason: str = "goodbye") -> None:
        if not self._conversation_done.is_set():
            self._end_reason = reason
            self._conversation_done.set()
            logger.info(f"Conversation end signaled: {reason}")

    def _on_user_speaks(self, response: str) -> None:
        current_record_id.set(self._record_id)
        self.event_logger.log_event(
            "user_speech",
            {"text": response, "source": "simulated_user"},
        )

    def _on_assistant_speaks(self, transcript: str) -> None:
        current_record_id.set(self._record_id)
        self.event_logger.log_event(
            "assistant_speech",
            {"text": transcript, "source": "assistant"},
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
        return b"".join(self._user_audio_chunks), b"".join(self._assistant_audio_chunks)
