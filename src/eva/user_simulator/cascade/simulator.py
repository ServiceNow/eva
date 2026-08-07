"""Self-hosted STT/LLM/TTS caller driven by the tick scheduler."""

from __future__ import annotations

import re
from pathlib import Path

import websockets

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
from eva.user_simulator.cascade.constants import (
    TICK_DURATION_MS,
    TRANSCRIPT_WAIT_MS,
    CALLER_SAMPLE_RATE,
    ms_to_ticks,
)
from eva.user_simulator.cascade.scheduler import TickScheduler
from eva.user_simulator.cascade.stt_livekit import LiveKitStreamingSTT
from eva.user_simulator.cascade.tts import CartesiaTTS

# Shared with the OpenAI Realtime provider so both simulators hang up on the same rules.
from eva.user_simulator.openai_realtime import END_CALL_DESCRIPTION
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)

MISSED_UTTERANCE_DIRECTIVE = """You did not hear what the agent just said — the audio did not
come through. Do NOT repeat your previous message. Say briefly that you did not catch that and
ask them to repeat it, the way anyone would on a bad phone line."""

END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": END_CALL_DESCRIPTION,
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def parse_turn_response(raw: str) -> str:
    """Return the spoken line from the model's reply, stripping any stray code fence."""
    return _FENCE.sub("", raw).strip()


def _flip_role(role: str) -> str:
    """Swap user/assistant so the caller LLM sees its own lines tagged assistant."""
    return "assistant" if role == "user" else "user"


def extract_turn(message: object) -> tuple[str, bool]:
    """Read (utterance, end_call) from whatever LiteLLMClient returned.

    ``complete()`` returns a bare ``str`` when the model made no tool call and a
    message object when it did, so both shapes must be handled.
    """
    if isinstance(message, str):
        return parse_turn_response(message), False
    content = getattr(message, "content", None) or ""
    calls = getattr(message, "tool_calls", None) or []
    end_call = any(getattr(call.function, "name", "") == "end_call" for call in calls)
    return parse_turn_response(content), end_call


class CascadeUserSimulator(AbstractUserSimulator):
    """Simulated caller built from independently chosen STT, LLM, and TTS models."""

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
        simulator_config: CascadeSimulatorConfig,
    ) -> None:
        super().__init__(
            current_date_time=current_date_time,
            persona_config=persona_config,
            goal=goal,
            server_url=server_url,
            output_dir=output_dir,
            agent_id=agent_id,
            timeout=timeout,
            perturbation_config=perturbation_config,
            language=language,
            provider="cascade",
        )
        self._warn_unsupported_perturbation(perturbation_config)
        self._config = simulator_config
        self._stt = LiveKitStreamingSTT(simulator_config.stt, simulator_config.stt_params, language=language)
        self._tts = CartesiaTTS(simulator_config.tts_params, language=language)
        self._llm = LiteLLMClient(model=simulator_config.llm)
        self._voice_id = self._tts.voice_for_persona(persona_config)
        self._history: list[dict[str, str]] = []
        self._ticks_awaiting_transcript = 0
        self._missed_transcripts = 0

    async def run_conversation(self) -> str:
        """Run the tick loop until the call ends, and return the end reason."""
        try:
            await self._run()
        except Exception as exc:
            logger.exception(f"Cascade simulator failed: {exc}")
            self._end_reason = "error"
            self.event_logger.log_error(str(exc))
        finally:
            self._save_clean_user_audio(CALLER_SAMPLE_RATE)
            self.event_logger.save()
        return self._end_reason

    async def _run(self) -> None:
        """Drive the scheduler until end_call, timeout, or disconnect."""
        websocket = await websockets.connect(self.server_url)
        adapter = RealtimeWSAdapter(
            websocket=websocket,
            conversation_id=self._record_id or "cascade",
        )
        scheduler = TickScheduler(adapter)

        await adapter.start()
        await self._stt.start()
        self.event_logger.log_connection_state("connected", {"server_url": self.server_url})

        max_ticks = self.timeout * 1000 // TICK_DURATION_MS
        assistant_was_speaking = False
        try:
            while scheduler.tick < max_ticks and not self._conversation_done.is_set():
                result = await scheduler.run_tick()
                # Fed on every tick, speech or silence, so Scribe sees a continuous stream
                # and never idles out; committed exactly on the speech->silence transition,
                # which is what closes the utterance so take_committed() below isn't starved.
                commit = assistant_was_speaking and not result.has_assistant_speech
                if result.has_assistant_speech != assistant_was_speaking:
                    logger.debug(
                        f"tick {scheduler.tick}: assistant speech "
                        f"{'started' if result.has_assistant_speech else 'ended'} "
                        f"(raw={result.assistant_audio_raw_bytes}B)"
                    )
                await self._stt.feed(result.assistant_audio, commit=commit)
                assistant_was_speaking = result.has_assistant_speech
                if result.has_assistant_speech:
                    continue
                if scheduler.caller_is_speaking or not scheduler.may_take_turn():
                    continue
                heard, waiting = self._collect_heard_text(scheduler)
                if waiting:
                    continue
                if await self._take_turn(scheduler, heard):
                    break
            else:
                if not self._conversation_done.is_set():
                    self._on_conversation_end("timeout")
        finally:
            await self._stt.stop()
            await adapter.stop()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

    def _collect_heard_text(self, scheduler: TickScheduler) -> tuple[str, bool]:
        """Return what the assistant said and whether to keep waiting for it.

        Finalization is not instantaneous, so an empty buffer at the first turn
        opportunity usually means "not ready yet" rather than "nothing was said".
        Retrying on later ticks is the wait; the in-flight partial is the fallback
        once that budget is spent.
        """
        heard = self._stt.buffer.take_committed()
        if heard:
            self._ticks_awaiting_transcript = 0
            return heard, False

        self._ticks_awaiting_transcript += 1
        if self._ticks_awaiting_transcript <= ms_to_ticks(TRANSCRIPT_WAIT_MS):
            return "", True

        self._ticks_awaiting_transcript = 0
        partial = self._stt.buffer.in_flight
        self._stt.buffer.in_flight = ""
        if partial:
            logger.warning(
                f"tick {scheduler.tick}: no final transcript after {TRANSCRIPT_WAIT_MS}ms; "
                f"falling back to the in-flight partial: {partial[:120]!r}"
            )
            self.event_logger.log_event("transcript_partial_fallback", {"text": partial, "tick_index": scheduler.tick})
            return partial, False

        self._missed_transcripts += 1
        logger.error(
            f"tick {scheduler.tick}: heard nothing at all from the assistant this turn "
            f"(missed {self._missed_transcripts} so far); asking it to repeat"
        )
        self.event_logger.log_event("transcript_missed", {"tick_index": scheduler.tick})
        return "", False

    async def _take_turn(self, scheduler: TickScheduler, heard: str) -> bool:
        """Generate, synthesize, and queue one caller turn. Returns True to hang up."""
        if heard:
            self._history.append({"role": "assistant", "content": heard})
            self._on_assistant_speaks(heard)

        message, _stats = await self._llm.complete(
            messages=self._messages(missed_utterance=not heard), tools=[END_CALL_TOOL]
        )
        utterance, end_call = extract_turn(message)

        if utterance:
            self._history.append({"role": "user", "content": utterance})
            self._on_user_speaks(utterance)
            audio = await self._tts.synthesize(utterance, voice_id=self._voice_id)
            self._record_audio("user_clean", audio)
            scheduler.enqueue_utterance(audio)
            self.event_logger.log_event("caller_turn", {"text": utterance, "tick_index": scheduler.tick})

        if end_call:
            self._on_conversation_end("goodbye")
            return True
        return False

    @staticmethod
    def _warn_unsupported_perturbation(perturbation_config: PerturbationConfig | None) -> None:
        """Warn when outbound-audio perturbations are configured but unsupported by cascade.

        `RealtimeWSAdapter` does not yet apply perturbation to outbound audio, so
        `background_noise` and `connection_degradation` are silently dropped without this.
        `snr_db` only has an effect alongside `background_noise` and defaults to 15.0, so it
        is flagged only when `background_noise` is also set, not on its default value alone.
        """
        if perturbation_config is None:
            return
        unsupported = []
        if perturbation_config.background_noise is not None:
            unsupported.extend(["background_noise", "snr_db"])
        if perturbation_config.connection_degradation:
            unsupported.append("connection_degradation")
        if unsupported:
            logger.warning(
                f"Cascade simulator does not yet apply audio perturbation: ignoring {', '.join(unsupported)}. "
                "Behavior and accent perturbations are unaffected."
            )

    def _messages(self, *, missed_utterance: bool = False) -> list[dict[str, str]]:
        """Build the message list: the shared per-domain caller prompt plus flipped history.

        The system prompt is `_build_prompt()` unmodified — the same per-domain prompt the other
        providers use, which already carries the persona, goal, and end_call rules.

        `self._history` is kept in conversation-truth roles (assistant said by the agent, user
        said by the caller) since it also feeds logging. This LLM is itself the assistant
        in its own frame, so that history must be flipped here or a message tagged "assistant"
        reads to the model as its own prior output and it echoes it back.
        """
        messages = [{"role": "system", "content": self._build_prompt()}]
        messages += [{"role": _flip_role(turn["role"]), "content": turn["content"]} for turn in self._history]
        if missed_utterance:
            # History is unchanged since the last turn, so without this the model would
            # regenerate its previous utterance verbatim.
            messages.append({"role": "system", "content": MISSED_UTTERANCE_DIRECTIVE})
        return messages
