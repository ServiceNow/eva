"""Provider-neutral user simulator contract and shared behavior."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import yaml
from pipecat.transcriptions.language import Language

from eva.models.config import LANGUAGE_DISPLAY_NAMES, PerturbationConfig
from eva.user_simulator.event_logger import UserSimulatorEventLogger
from eva.user_simulator.perturbation import AudioPerturbator
from eva.utils.audio_utils import save_audio_track
from eva.utils.culture import add_user_language_directive
from eva.utils.logging import current_record_id, get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

_BEHAVIORS_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "user_behaviors.yaml"


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

        # Set by the worker to the assistant server's notify_conversation_ending. Invoked as
        # soon as the call is known to be over — before the STT grace period and any post-hoc
        # provider API polling, which can keep the transport open well past that point. See
        # signal_conversation_ending.
        self.on_conversation_ending: Callable[[str | None], None] | None = None
        self._ending_signaled = False

        self.event_logger = UserSimulatorEventLogger(
            self.output_dir / "user_simulator_events.jsonl",
            provider=provider,
        )

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

    def signal_conversation_ending(self, reason: str | None = None) -> None:
        """Tell the assistant the call is over, before transport teardown begins.

        Idempotent and never raises: teardown paths call this best-effort, and the assistant
        side treats it as advisory. Call it from *every* terminal path (end_call, timeout,
        error), since the assistant's silence-triggered behavior doesn't care why the call
        ended — only that no further user speech is coming.
        """
        if self._ending_signaled:
            return
        self._ending_signaled = True
        if self.on_conversation_ending is None:
            return
        try:
            self.on_conversation_ending(reason or self._end_reason)
        except Exception as e:
            logger.warning(f"Failed to signal conversation ending to the assistant: {e}")

    def _on_conversation_end(self, reason: str = "goodbye") -> None:
        if not self._conversation_done.is_set():
            self._end_reason = reason
            self._conversation_done.set()
            logger.info(f"Conversation end signaled: {reason}")
            self.signal_conversation_ending(reason)

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

        Only the clean (unperturbed) user track is persisted — it is the one
        artifact the assistant server never sees and therefore cannot record.
        Other sources are captured by the assistant server's own recording path.

        Args:
            source: recording channel; only "user_clean" is retained
            audio_data: Raw audio bytes
        """
        if source == "user_clean":
            self._user_clean_audio_chunks.append(audio_data)

    def _save_clean_user_audio(self, sample_rate: int) -> None:
        """Persist the recorded clean user track to ``audio_user_clean.wav``.

        Shared by all providers; skips writing when no clean audio was recorded.
        """
        if save_audio_track(
            self._user_clean_audio_chunks,
            self.output_dir / "audio_user_clean.wav",
            sample_rate,
        ):
            logger.info(f"Saved clean user audio to {self.output_dir / 'audio_user_clean.wav'}")
