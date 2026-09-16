"""Self-hosted STT/LLM/TTS caller driven by the tick scheduler."""

from __future__ import annotations

import audioop
import re
from pathlib import Path

import websockets

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
from eva.user_simulator.cascade.adapter.tick_driven import MAX_INACTIVE_SECONDS, TickDrivenAdapter
from eva.user_simulator.cascade.constants import (
    CALLER_SAMPLE_RATE,
    INACTIVITY_TIMEOUT_MS,
    TICK_DURATION_MS,
    TRANSCRIPT_WAIT_MS,
    ms_to_ticks,
)
from eva.user_simulator.cascade.decision_log import DecisionLog
from eva.user_simulator.cascade.scheduler import TickScheduler
from eva.user_simulator.cascade.stt_livekit import LiveKitStreamingSTT
from eva.user_simulator.cascade.tick_result import TickResult
from eva.user_simulator.cascade.tts import CartesiaTTS

# Shared with the OpenAI Realtime provider so both simulators hang up on the same rules.
from eva.user_simulator.openai_realtime import END_CALL_DESCRIPTION
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)

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


TICK_DRIVEN_FRAMEWORKS = frozenset({"openai_realtime"})
"""Frameworks whose clock the caller can own. Others keep real-time streaming."""


def adapter_class_for_framework(framework: str) -> type[Adapter]:
    """Pick the adapter for a framework, defaulting to real-time streaming.

    Defaulting to real-time is deliberate: it works everywhere, whereas
    tick-driving requires the assistant to have no wall-clock timers of its own.
    """
    if framework in TICK_DRIVEN_FRAMEWORKS:
        return TickDrivenAdapter
    return RealtimeWSAdapter


class CascadeUserSimulator(AbstractUserSimulator):
    """Simulated caller built from independently chosen STT, LLM, and TTS models."""

    _ticks_awaiting_transcript = 0
    _ticks_assistant_silent = 0

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
        framework: str = "pipecat",
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
        self._config = simulator_config
        self._framework = framework
        self._stt = LiveKitStreamingSTT(simulator_config.stt, simulator_config.stt_params, language=language)
        self._tts = CartesiaTTS(simulator_config.tts_params, language=language)
        self._llm = LiteLLMClient(model=simulator_config.llm)
        self._voice_id = self._tts.voice_for_persona(persona_config)
        self._history: list[dict[str, str]] = []
        self._decision_log = DecisionLog(self.output_dir / "user_simulator_decisions.jsonl")

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
            self._decision_log.save()
            logger.info(f"Caller decision trace: {self._decision_log.summary()}")
        return self._end_reason

    async def _run(self) -> None:
        """Drive the scheduler until end_call, timeout, or disconnect."""
        websocket = await websockets.connect(self.server_url)
        adapter_cls = adapter_class_for_framework(self._framework)
        adapter = adapter_cls(
            websocket=websocket,
            conversation_id=self._record_id or "cascade",
            perturbator=self._perturbator,
        )
        scheduler = TickScheduler(adapter)

        await adapter.start()
        await self._stt.start()
        self.event_logger.log_connection_state("connected", {"server_url": self.server_url})

        max_ticks = self.timeout * 1000 // TICK_DURATION_MS
        assistant_was_speaking = False
        caller_was_speaking = False
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
                self._log_audio_boundaries(scheduler, result, assistant_was_speaking, caller_was_speaking)
                caller_was_speaking = scheduler.caller_spoke_this_tick
                assistant_was_speaking = result.has_assistant_speech
                if result.provider_stalled:
                    # Distinct from inactivity_timeout, which is a *legitimate* end the
                    # metrics treat as definitive when the user spoke last. A stall is a
                    # dead peer: the record is invalid and the runner should retry it,
                    # which is what the reason not being "goodbye" already means to it.
                    logger.error(
                        f"tick {scheduler.tick}: no assistant audio for "
                        f"{MAX_INACTIVE_SECONDS}s; abandoning the conversation as unusable"
                    )
                    self._on_conversation_end("provider_stalled")
                    break
                if self._assistant_is_inactive(scheduler, result):
                    logger.warning(
                        f"tick {scheduler.tick}: assistant silent for "
                        f"{INACTIVITY_TIMEOUT_MS // 1000}s; ending the conversation"
                    )
                    self._on_conversation_end("inactivity_timeout")
                    break
                self._log_tick_state(scheduler, result)
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

    def _assistant_is_inactive(self, scheduler: TickScheduler, result: TickResult) -> bool:
        """Whether the assistant has produced no audio for INACTIVITY_TIMEOUT_MS *contiguously*.

        Mirrors ElevenLabsUserSimulator's keep-alive rule so both providers record the
        same terminal state: conversation_valid_end treats inactivity_timeout with the
        user speaking last as a definitive end, not a failure.

        Must be called on every tick, speech or silence. It was previously reached only on
        silent ticks, which made the reset below dead code: the counter then measured
        *cumulative* silence over the whole call and killed healthy conversations once their
        quiet ticks happened to total two minutes.
        """
        if result.has_assistant_speech:
            self._ticks_assistant_silent = 0
            return False
        self._ticks_assistant_silent += 1
        return scheduler.assistant_has_spoken and self._ticks_assistant_silent > ms_to_ticks(INACTIVITY_TIMEOUT_MS)

    def _log_audio_boundaries(
        self,
        scheduler: TickScheduler,
        result: TickResult,
        assistant_was_speaking: bool,
        caller_was_speaking: bool,
    ) -> None:
        """Emit audio_start/audio_end for both roles, which is how metrics number turns.

        The caller's boundaries are authored rather than detected: the playout queue
        drains on a known tick, so these stamp the real edges instead of a
        silence-threshold estimate that has to be back-dated (see
        BotToBotAudioBridge, whose end detection lags by ~600ms).
        """
        seconds = result.wall_clock_ms / 1000
        caller_speaking = scheduler.caller_spoke_this_tick
        if caller_speaking and not caller_was_speaking:
            self.event_logger.log_audio_start("simulated_user", seconds)
        elif not caller_speaking and caller_was_speaking:
            self.event_logger.log_audio_end("simulated_user", seconds)
        if result.has_assistant_speech and not assistant_was_speaking:
            self.event_logger.log_audio_start("assistant", seconds)
        elif not result.has_assistant_speech and assistant_was_speaking:
            self.event_logger.log_audio_end("assistant", seconds)

    def _log_tick_state(self, scheduler: TickScheduler, result: TickResult) -> None:
        """Trace speech state and audio levels for each simulator tick."""
        raw = result.assistant_audio[: result.assistant_audio_raw_bytes]
        self._decision_log.log(
            "tick",
            tick=scheduler.tick,
            has_assistant_speech=result.has_assistant_speech,
            raw_bytes=result.assistant_audio_raw_bytes,
            rms=audioop.rms(raw, 2) if len(raw) >= 2 else 0,
            caller_is_speaking=scheduler.caller_is_speaking,
            caller_spoke_this_tick=scheduler.caller_spoke_this_tick,
            ticks_assistant_silent=self._ticks_assistant_silent,
        )

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

        partial = self._stt.buffer.in_flight
        self._stt.buffer.in_flight = ""
        if partial:
            logger.warning(
                f"tick {scheduler.tick}: no final transcript after {TRANSCRIPT_WAIT_MS}ms; "
                f"falling back to the in-flight partial: {partial[:120]!r}"
            )
            self.event_logger.log_event("transcript_partial_fallback", {"text": partial, "tick_index": scheduler.tick})
            return partial, False

        # Nothing was heard at all. Keep waiting rather than speaking into the void:
        # an assistant that never replies is an inactivity timeout, handled in _run.
        return "", True

    async def _take_turn(self, scheduler: TickScheduler, heard: str) -> bool:
        """Generate, synthesize, and queue one caller turn. Returns True to hang up."""
        if heard:
            self._history.append({"role": "assistant", "content": heard})
            self._on_assistant_speaks(heard)

        message, _stats = await self._llm.complete(messages=self._messages(), tools=[END_CALL_TOOL])
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

    def _messages(self) -> list[dict[str, str]]:
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
        return messages
