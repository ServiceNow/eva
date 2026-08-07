"""Self-hosted STT/LLM/TTS caller driven by the tick scheduler."""

from __future__ import annotations

import random
import re
from pathlib import Path

import websockets

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
from eva.user_simulator.cascade.constants import (
    BACKCHANNEL_PHRASES,
    BARGE_IN_OPENERS,
    CALLER_SAMPLE_RATE,
    INACTIVITY_TIMEOUT_MS,
    MAX_INTERRUPT_SLIP_MS,
    SELF_CORRECTION_DELAY_MS,
    SELF_CORRECTION_RATE,
    TICK_DURATION_MS,
    TRANSCRIPT_WAIT_MS,
    ms_to_ticks,
)
from eva.user_simulator.cascade.decisions import ListenerDecisions, parse_yes_no
from eva.user_simulator.cascade.phrase_cache import PhraseCache
from eva.user_simulator.cascade.scheduler import TickScheduler
from eva.user_simulator.cascade.stt_livekit import LiveKitStreamingSTT
from eva.user_simulator.cascade.tick_result import TickResult
from eva.user_simulator.cascade.tts import CartesiaTTS

# Shared with the OpenAI Realtime provider so both simulators hang up on the same rules.
from eva.user_simulator.openai_realtime import END_CALL_DESCRIPTION
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

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


def extract_correction(message: object) -> str:
    """Read the self-correction line from its own dedicated call.

    It is a bare spoken line, not a JSON field: the caller's turn call keeps the
    plain-text contract Plan 1 settled on, since demanding JSON there suppressed
    the end_call tool call entirely.
    """
    content = message if isinstance(message, str) else (getattr(message, "content", None) or "")
    line = _FENCE.sub("", content).strip()
    return "" if line.upper() == "NONE" else line


def should_fire_self_correction(*, ticks_since_assistant_started: int, assistant_speaking: bool) -> bool:
    """Whether an armed correction should play now."""
    if not assistant_speaking:
        return False
    return ticks_since_assistant_started >= ms_to_ticks(SELF_CORRECTION_DELAY_MS)


def interrupt_slip_ms(*, intended_tick: int, actual_tick: int) -> int:
    """How far past its intended tick a barge-in actually landed."""
    return max(0, actual_tick - intended_tick) * TICK_DURATION_MS


def should_drop_interrupt(*, slip_ms: int, assistant_still_speaking: bool) -> bool:
    """Whether a barge-in has gone stale and should be abandoned."""
    return slip_ms > MAX_INTERRUPT_SLIP_MS or not assistant_still_speaking


async def candidate_is_relevant(llm, *, candidate: str, heard: str) -> bool:
    """Whether a pre-generated interruption still fits what the assistant is saying.

    Fails closed: an unusable answer means we fall back to generating fresh
    content rather than firing something that may have gone stale.
    """
    prompt = PromptManager().get_prompt("user_simulator.cascade_relevance_gate", candidate=candidate, heard=heard)
    try:
        reply = await llm.decide(prompt)
    except Exception as exc:
        logger.warning(f"Relevance gate failed, discarding candidate: {exc}")
        return False
    return parse_yes_no(reply)


def play_backchannel(scheduler, cache, phrases: list[str]) -> str:
    """Queue a cached continuer and return the phrase that was chosen."""
    phrase = cache.choose(phrases)
    scheduler.enqueue_utterance(cache.get(phrase))
    return phrase


class _DecisionClient:
    """Adapts LiteLLMClient to the single-prompt interface the checks expect."""

    def __init__(self, client: LiteLLMClient) -> None:
        self._client = client

    async def decide(self, prompt: str) -> str:
        """Ask one YES/NO question and return the raw reply."""
        message, _stats = await self._client.complete(messages=[{"role": "user", "content": prompt}])
        if isinstance(message, str):
            return message
        return getattr(message, "content", None) or ""


class CascadeUserSimulator(AbstractUserSimulator):
    """Simulated caller built from independently chosen STT, LLM, and TTS models."""

    _ticks_awaiting_transcript = 0
    _ticks_assistant_silent = 0
    _ticks_since_assistant_started = 0

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
        self._config = simulator_config
        self._stt = LiveKitStreamingSTT(simulator_config.stt, simulator_config.stt_params, language=language)
        self._tts = CartesiaTTS(simulator_config.tts_params, language=language)
        self._llm = LiteLLMClient(model=simulator_config.llm)
        self._voice_id = self._tts.voice_for_persona(persona_config)
        self._history: list[dict[str, str]] = []
        # Shared by the listener checks and the relevance gate so both cost one client.
        self._decision_client = _DecisionClient(LiteLLMClient(model=simulator_config.decision_llm))
        self._phrase_cache: PhraseCache | None = None
        self._decisions: ListenerDecisions | None = None
        self._rng = random.Random(0)
        self._armed_correction: bytes = b""
        self._armed_correction_text = ""
        self._candidate_text = ""
        self._candidate_audio = b""

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
            perturbator=self._perturbator,
        )
        scheduler = TickScheduler(adapter)

        await adapter.start()
        await self._stt.start()
        await self._prepare_listener_behaviors()
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
                if result.has_assistant_speech:
                    self._ticks_since_assistant_started += 1
                    if self._armed_correction and should_fire_self_correction(
                        ticks_since_assistant_started=self._ticks_since_assistant_started,
                        assistant_speaking=True,
                    ):
                        self._fire_self_correction(scheduler)
                        continue
                    if scheduler.is_check_tick():
                        await self._run_checks(scheduler)
                    continue
                self._ticks_since_assistant_started = 0
                if self._assistant_is_inactive(scheduler, result):
                    logger.warning(
                        f"tick {scheduler.tick}: assistant silent for "
                        f"{INACTIVITY_TIMEOUT_MS // 1000}s; ending the conversation"
                    )
                    self._on_conversation_end("inactivity_timeout")
                    break
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

    async def _prepare_listener_behaviors(self) -> None:
        """Pre-render the fixed vocabularies and build the checks, when any is enabled.

        Rendering happens once at connect time rather than on demand: a cached phrase
        is the only way a 300ms reaction lands on the tick its check chose.
        """
        vocabulary: list[str] = []
        if self._config.enable_backchannel:
            vocabulary += BACKCHANNEL_PHRASES
        if self._config.enable_interruptions:
            vocabulary += BARGE_IN_OPENERS
        if not vocabulary:
            return

        self._phrase_cache = PhraseCache(self._tts, voice_id=self._voice_id)
        await self._phrase_cache.prerender(vocabulary)
        prompts = PromptManager()
        self._decisions = ListenerDecisions(
            self._decision_client,
            interrupt_prompt=prompts.get_template("user_simulator.interruption_decision"),
            backchannel_prompt=prompts.get_template("user_simulator.backchannel_decision"),
        )

    async def _run_checks(self, scheduler: TickScheduler) -> None:
        """Run the listener-reaction checks and act on the verdict."""
        if self._decisions is None or self._phrase_cache is None:
            return
        verdict = await self._decisions.evaluate(
            self._stt.buffer.current_text(),
            allow_interrupt=self._config.enable_interruptions,
            allow_backchannel=self._config.enable_backchannel,
        )
        if verdict.should_interrupt:
            await self._play_interruption(scheduler)
            return
        if verdict.should_backchannel:
            phrase = play_backchannel(scheduler, self._phrase_cache, BACKCHANNEL_PHRASES)
            # Recorded too, or the saved clean track diverges from what went on the wire.
            self._record_audio("user_clean", self._phrase_cache.get(phrase))
            self.event_logger.log_event("backchannel", {"text": phrase, "tick_index": scheduler.tick})

    async def _play_interruption(self, scheduler: TickScheduler) -> None:
        """Voice a cached opener immediately, then stream the real content behind it.

        The opener buys the lead time the content generation costs. If the content
        still arrives too late to be a barge-in, it is dropped rather than emitted
        stale — and both ticks are logged either way, because that gap is the
        empirical risk this design carries.
        """
        if self._phrase_cache is None:
            return
        intended_tick = scheduler.tick
        opener = self._phrase_cache.choose(BARGE_IN_OPENERS)
        opener_audio = self._phrase_cache.get(opener)
        scheduler.enqueue_utterance(opener_audio)
        self._record_audio("user_clean", opener_audio)

        if self._config.speculative_generation and self._candidate_audio:
            candidate, audio = self._candidate_text, self._candidate_audio
            self._candidate_text, self._candidate_audio = "", b""
            if await candidate_is_relevant(
                self._decision_client, candidate=candidate, heard=self._stt.buffer.current_text()
            ):
                scheduler.enqueue_utterance(audio)
                self._record_audio("user_clean", audio)
                self._history.append({"role": "user", "content": candidate})
                self._on_user_speaks(candidate)
                self.event_logger.log_event(
                    "interruption",
                    {
                        "text": candidate,
                        "opener": opener,
                        "intended_tick": intended_tick,
                        "actual_tick": scheduler.tick,
                        "slip_ms": interrupt_slip_ms(intended_tick=intended_tick, actual_tick=scheduler.tick),
                        "speculative": True,
                        "dropped": False,
                    },
                )
                return
            self.event_logger.log_event("interruption_candidate_rejected", {"text": candidate})

        # Consumed, not peeked: leaving it in the buffer would re-append the same
        # assistant prefix at the next ordinary turn and duplicate it in the history.
        heard = self._stt.buffer.current_text()
        self._stt.buffer.take_committed()
        self._stt.buffer.in_flight = ""
        if heard:
            self._history.append({"role": "assistant", "content": heard})
            self._on_assistant_speaks(heard)
        message, _stats = await self._llm.complete(messages=self._messages(), tools=[END_CALL_TOOL])
        utterance, _end_call = extract_turn(message)

        slip = interrupt_slip_ms(intended_tick=intended_tick, actual_tick=scheduler.tick)
        dropped = should_drop_interrupt(slip_ms=slip, assistant_still_speaking=scheduler.assistant_is_speaking)
        self.event_logger.log_event(
            "interruption",
            {
                "text": utterance,
                "opener": opener,
                "intended_tick": intended_tick,
                "actual_tick": scheduler.tick,
                "slip_ms": slip,
                "dropped": dropped,
            },
        )
        if dropped or not utterance:
            return

        self._history.append({"role": "user", "content": utterance})
        self._on_user_speaks(utterance)
        async for chunk in self._tts.stream(utterance, voice_id=self._voice_id):
            self._record_audio("user_clean", chunk)
            scheduler.enqueue_utterance(chunk)

    def _assistant_is_inactive(self, scheduler: TickScheduler, result: TickResult) -> bool:
        """Whether the assistant has produced no audio for INACTIVITY_TIMEOUT_MS.

        Mirrors ElevenLabsUserSimulator's keep-alive rule so both providers record the
        same terminal state: conversation_valid_end treats inactivity_timeout with the
        user speaking last as a definitive end, not a failure.
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

        self._drop_stale_correction()
        message, _stats = await self._llm.complete(messages=self._messages(), tools=[END_CALL_TOOL])
        utterance, end_call = extract_turn(message)

        if utterance and not end_call:
            utterance = await self._maybe_arm_self_correction(utterance)

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

        # After the audio is queued: the caller is now speaking, so this generation
        # runs behind its own outgoing audio and costs no conversational latency.
        if utterance:
            await self._prerender_candidate(utterance)
        return False

    def _fire_self_correction(self, scheduler: TickScheduler) -> None:
        """Play the armed correction over the assistant's reply and clear the arming."""
        scheduler.enqueue_utterance(self._armed_correction)
        self._record_audio("user_clean", self._armed_correction)
        self._history.append({"role": "user", "content": self._armed_correction_text})
        self._on_user_speaks(self._armed_correction_text)
        self.event_logger.log_event(
            "self_correction",
            {"text": self._armed_correction_text, "tick_index": scheduler.tick},
        )
        self._armed_correction = b""
        self._armed_correction_text = ""

    async def _prerender_candidate(self, utterance: str) -> None:
        """Pre-generate and pre-render the line the caller would barge in with.

        Done on the caller's own turn, where the latency is already hidden behind
        its outgoing audio, so a later barge-in can fire without waiting on
        generation. The relevance gate is what keeps this from degrading into a
        scripted interruption that lands as a non-sequitur.
        """
        self._candidate_text, self._candidate_audio = "", b""
        if not self._config.speculative_generation:
            return
        prompt = PromptManager().get_prompt("user_simulator.cascade_next_interruption", utterance=utterance)
        try:
            message, _stats = await self._llm.complete(
                messages=[*self._messages(), {"role": "user", "content": prompt}]
            )
        except Exception as exc:
            logger.warning(f"Speculative interruption generation failed: {exc}")
            return
        candidate = extract_correction(message)
        if not candidate:
            return
        self._candidate_text = candidate
        self._candidate_audio = await self._tts.synthesize(candidate, voice_id=self._voice_id)

    def _drop_stale_correction(self) -> None:
        """Abandon an armed correction whose assistant turn never arrived.

        Carrying it into a later turn would read as a non-sequitur, since it refers
        to an utterance that is now several exchanges back.
        """
        if not self._armed_correction:
            return
        self.event_logger.log_event("self_correction_dropped", {"text": self._armed_correction_text})
        self._armed_correction = b""
        self._armed_correction_text = ""

    async def _maybe_arm_self_correction(self, utterance: str) -> str:
        """Maybe misspeak: return a wrong variant to say now, arming `utterance` as the fix.

        The wrong-then-right ordering is the design, not a detail. The generated slip
        is spoken first and the model's own goal-consistent line lands as the
        correction, so the conversation's end state still satisfies must_have_criteria
        by construction and this behavior cannot make a record unachievable.

        Asked as its own call rather than as an extra JSON field on the turn call: a
        JSON contract there suppressed the end_call tool entirely (Plan 1), and this
        way a failure degrades to an ordinary turn instead of a broken one.
        """
        if not self._config.enable_self_correction or self._rng.random() >= SELF_CORRECTION_RATE:
            return utterance

        prompt = PromptManager().get_prompt("user_simulator.cascade_self_correction", utterance=utterance)
        try:
            message, _stats = await self._llm.complete(
                messages=[*self._messages(), {"role": "user", "content": prompt}]
            )
        except Exception as exc:
            logger.warning(f"Self-correction generation failed, speaking the turn unchanged: {exc}")
            return utterance

        slip = extract_correction(message)
        if not slip or slip == utterance:
            return utterance

        self._armed_correction = await self._tts.synthesize(utterance, voice_id=self._voice_id)
        self._armed_correction_text = utterance
        self.event_logger.log_event("self_correction_armed", {"slip": slip, "correction": utterance})
        return slip

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
