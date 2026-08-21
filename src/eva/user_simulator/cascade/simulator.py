"""Self-hosted STT/LLM/TTS caller driven by the tick scheduler."""

from __future__ import annotations

import audioop
import re
import time
from pathlib import Path

import websockets

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.adapter.base import Adapter
from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
from eva.user_simulator.cascade.adapter.tick_driven import TickDrivenAdapter
from eva.user_simulator.cascade.constants import (
    BACKCHANNEL_PHRASES,
    BARGE_IN_OPENERS,
    CALLER_SAMPLE_RATE,
    INACTIVITY_TIMEOUT_MS,
    TICK_DURATION_MS,
    TRANSCRIPT_WAIT_MS,
    WAIT_TO_RESPOND_OTHER_MS,
    ms_to_ticks,
)
from eva.user_simulator.cascade.decision_log import DecisionLog
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


def extract_optional_line(message: object) -> str:
    """Read a bare spoken line from a dedicated call, empty when the model declined.

    A bare line rather than a JSON field: demanding JSON on a caller call suppressed
    the end_call tool call entirely (Plan 1).
    """
    content = message if isinstance(message, str) else (getattr(message, "content", None) or "")
    line = _FENCE.sub("", content).strip()
    return "" if line.upper() == "NONE" else line


def interrupt_slip_ms(*, elapsed_s: float) -> int:
    """How far past its intended moment a barge-in actually landed.

    Measured on the wall clock, not the tick counter. `run_tick` is only pumped by
    the `_run` loop, so while `_play_interruption` awaits its generation no tick can
    advance — a tick-delta slip is structurally always zero and the staleness check
    built on it never engages.
    """
    return max(0, int(elapsed_s * 1000))


def summarize_goal(goal: dict) -> str:
    """State what the caller wants and what would end the call, for the listener checks.

    Field meanings live in the prompt template rather than here, so the explanations stay
    static across the many calls this decision makes rather than being rebuilt per call.
    `edge_cases` and `information_required` are deliberately omitted: they are long and
    describe how to answer questions, not whether the goal is finished.
    """
    tree = goal.get("decision_tree", {}) or {}
    sections = [
        ("GOAL", goal.get("high_level_user_goal")),
        ("MUST HAVE", tree.get("must_have_criteria")),
        ("NICE TO HAVE", tree.get("nice_to_have_criteria")),
        ("HOW THEY EVALUATE OPTIONS", tree.get("negotiation_behavior")),
        ("RESOLVED WHEN", tree.get("resolution_condition")),
        ("FAILED WHEN", tree.get("failure_condition")),
        ("ESCALATION", tree.get("escalation_behavior")),
    ]
    lines: list[str] = []
    for label, value in sections:
        if not value:
            continue
        if isinstance(value, list):
            lines.append(f"{label}:")
            lines += [f"- {item}" for item in value]
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def is_new_assistant_turn(*, ticks_silent_before: int) -> bool:
    """Whether assistant audio arriving now starts a new turn or resumes the current one.

    A pause shorter than the turn-end threshold is a gap *inside* one turn — between
    sentences, or a hole in the audio stream — not a new turn. Treating any single quiet
    tick as a boundary re-armed the interruption cap mid-utterance, so one assistant turn
    could collect several barge-ins.
    """
    return ticks_silent_before >= ms_to_ticks(WAIT_TO_RESPOND_OTHER_MS)


def should_drop_interrupt(*, assistant_still_speaking: bool, same_assistant_turn: bool) -> bool:
    """Whether a barge-in has gone stale and should be abandoned.

    Staleness is a fact about the assistant, not about how long generation took: a line is
    still a real interruption whenever the assistant is mid-utterance, however many wall-clock
    seconds elapsed. Both conditions are needed — "still speaking" alone is satisfied by a
    *later* turn, which would land the line as a non-sequitur against speech it never heard.
    """
    return not (assistant_still_speaking and same_assistant_turn)


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
    """Queue a cached continuer and return the phrase that was chosen.

    Queued as a backchannel, not an utterance: it must not consume the caller's
    turn, or the caller waits for a reply the continuer never earns.
    """
    phrase = cache.choose(phrases)
    scheduler.enqueue_backchannel(cache.get(phrase))
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
    _ticks_since_assistant_started = 0
    _may_interrupt_this_turn = False
    _assistant_turn_index = 0

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
        # Shared by the listener checks and the relevance gate so both cost one client.
        self._decision_client = _DecisionClient(LiteLLMClient(model=simulator_config.decision_llm))
        self._phrase_cache: PhraseCache | None = None
        self._decisions: ListenerDecisions | None = None
        self._candidate_text = ""
        self._candidate_audio = b""
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
                # Captured before the inactivity check, which clears it on a speech tick.
                silent_before = self._ticks_assistant_silent
                if self._assistant_is_inactive(scheduler, result):
                    logger.warning(
                        f"tick {scheduler.tick}: assistant silent for "
                        f"{INACTIVITY_TIMEOUT_MS // 1000}s; ending the conversation"
                    )
                    self._on_conversation_end("inactivity_timeout")
                    break
                if result.has_assistant_speech:
                    if is_new_assistant_turn(ticks_silent_before=silent_before):
                        self._ticks_since_assistant_started = 0
                        # One roll per assistant turn, so a turn carries at most one barge-in.
                        self._may_interrupt_this_turn = self._config.enable_interruptions
                        self._assistant_turn_index += 1
                        self._decision_log.log(
                            "assistant_turn_start",
                            tick=scheduler.tick,
                            turn_index=self._assistant_turn_index,
                            ticks_silent_before=silent_before,
                            armed_interrupt=self._may_interrupt_this_turn,
                        )
                    self._ticks_since_assistant_started += 1
                    is_check = scheduler.is_check_tick()
                    self._log_tick_state(scheduler, result, is_check_tick=is_check)
                    if is_check and await self._run_checks(scheduler):
                        break
                    continue
                self._log_tick_state(scheduler, result, is_check_tick=False)
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
            user_goal=summarize_goal(self.goal),
        )

    async def _run_checks(self, scheduler: TickScheduler) -> bool:
        """Run the listener-reaction checks and act on the verdict. True means hang up."""
        if self._decisions is None or self._phrase_cache is None:
            return False
        history = self._stt.buffer.current_text()
        verdict = await self._decisions.evaluate(
            history,
            allow_interrupt=self._may_interrupt_this_turn,
            allow_backchannel=self._config.enable_backchannel,
        )
        self._decision_log.log(
            "listener_check",
            tick=scheduler.tick,
            allow_interrupt=self._may_interrupt_this_turn,
            allow_backchannel=self._config.enable_backchannel,
            heard_chars=len(history),
            heard=history,
            interrupt_ran=verdict.interrupt_trace.ran,
            interrupt_raw=verdict.interrupt_trace.raw,
            interrupt_latency_ms=verdict.interrupt_trace.latency_ms,
            interrupt_error=verdict.interrupt_trace.error,
            backchannel_ran=verdict.backchannel_trace.ran,
            backchannel_raw=verdict.backchannel_trace.raw,
            backchannel_error=verdict.backchannel_trace.error,
            should_interrupt=verdict.should_interrupt,
            should_backchannel=verdict.should_backchannel,
        )
        if verdict.should_interrupt:
            self._may_interrupt_this_turn = False
            return await self._play_interruption(scheduler)
        if verdict.should_backchannel:
            phrase = play_backchannel(scheduler, self._phrase_cache, BACKCHANNEL_PHRASES)
            # Recorded too, or the saved clean track diverges from what went on the wire.
            self._record_audio("user_clean", self._phrase_cache.get(phrase))
            self.event_logger.log_event("backchannel", {"text": phrase, "tick_index": scheduler.tick})
        return False

    async def _play_interruption(self, scheduler: TickScheduler) -> bool:
        """Decide what to say, then voice the opener and the content together.

        Returns True when the caller decided to hang up.

        The content is generated *before* anything reaches the wire. Emitting the opener
        first hid its ~1s latency, but committed the caller to barging in before knowing
        whether it had anything to say: a hang-up or a dropped-as-stale line then left an
        orphaned "Actually—" hanging with nothing behind it, which is worse than either a
        late line or silence. Generating first means "say nothing" is actually available.
        """
        if self._phrase_cache is None:
            return False
        intended_tick = scheduler.tick
        intended_turn = self._assistant_turn_index
        started_at = time.monotonic()
        opener = self._phrase_cache.choose(BARGE_IN_OPENERS)
        opener_audio = self._phrase_cache.get(opener)

        if self._config.speculative_generation and self._candidate_audio:
            candidate, audio = self._candidate_text, self._candidate_audio
            self._candidate_text, self._candidate_audio = "", b""
            relevant = await candidate_is_relevant(
                self._decision_client, candidate=candidate, heard=self._stt.buffer.current_text()
            )
            self._decision_log.log("relevance_gate", tick=intended_tick, candidate=candidate, relevant=relevant)
            if relevant:
                # Tell the adapter the next tick that reaches the wire cuts the
                # assistant off, so a tick-driven transport can truncate the audio
                # the caller never heard. Ignored on the real-time path.
                scheduler.arm_barge_in()
                scheduler.enqueue_utterance(opener_audio)
                self._record_audio("user_clean", opener_audio)
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
                        "slip_ms": interrupt_slip_ms(elapsed_s=time.monotonic() - started_at),
                        "speculative": True,
                        "dropped": False,
                    },
                )
                self._decision_log.log(
                    "interruption", tick=intended_tick, outcome="spoken", speculative=True, text=candidate
                )
                return False
            self.event_logger.log_event("interruption_candidate_rejected", {"text": candidate})
            self._decision_log.log("interruption", tick=intended_tick, outcome="candidate_rejected", text=candidate)

        # Consumed, not peeked: leaving it in the buffer would re-append the same
        # assistant prefix at the next ordinary turn and duplicate it in the history.
        heard = self._stt.buffer.heard_text()
        self._stt.buffer.take_committed()
        self._stt.buffer.in_flight = ""
        if heard:
            self._history.append({"role": "assistant", "content": heard})
            self._on_assistant_speaks(heard)
        message, _stats = await self._llm.complete(messages=self._messages(), tools=[END_CALL_TOOL])
        utterance, end_call = extract_turn(message)

        slip = interrupt_slip_ms(elapsed_s=time.monotonic() - started_at)
        # A hang-up is never stale: the caller has decided the call is over, and
        # dropping it here is what left conversations looping until the timeout.
        dropped = not end_call and should_drop_interrupt(
            assistant_still_speaking=scheduler.assistant_is_speaking,
            same_assistant_turn=self._assistant_turn_index == intended_turn,
        )
        self.event_logger.log_event(
            "interruption",
            {
                "text": utterance,
                "opener": opener,
                "intended_tick": intended_tick,
                "actual_tick": scheduler.tick,
                "slip_ms": slip,
                "dropped": dropped,
                "end_call": end_call,
            },
        )
        self._decision_log.log(
            "interruption",
            tick=intended_tick,
            outcome="dropped" if dropped else ("end_call" if end_call else "spoken"),
            text=utterance,
            slip_ms=slip,
            assistant_still_speaking=scheduler.assistant_is_speaking,
            intended_turn=intended_turn,
            actual_turn=self._assistant_turn_index,
        )
        # Nothing has reached the wire yet, so a stale line or a hang-up costs no audio.
        if dropped:
            return False

        if utterance:
            scheduler.arm_barge_in()
            scheduler.enqueue_utterance(opener_audio)
            self._record_audio("user_clean", opener_audio)
            self._history.append({"role": "user", "content": utterance})
            self._on_user_speaks(utterance)
            async for chunk in self._tts.stream(utterance, voice_id=self._voice_id):
                self._record_audio("user_clean", chunk)
                scheduler.enqueue_utterance(chunk)

        if end_call:
            self._on_conversation_end("goodbye")
        return end_call

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

    def _log_tick_state(self, scheduler: TickScheduler, result: TickResult, *, is_check_tick: bool) -> None:
        """Trace one tick's speech state, so a check that never ran can be traced to its gate.

        `rms` is why this exists: `has_assistant_speech` is true for any non-zero bytes,
        digital silence included, so a transport that pads with silence reads as continuous
        speech. Recording both lets that be measured rather than inferred.
        """
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
            ticks_since_assistant_started=self._ticks_since_assistant_started,
            may_interrupt_this_turn=self._may_interrupt_this_turn,
            is_check_tick=is_check_tick,
            has_candidate=bool(self._candidate_audio),
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

        # After the audio is queued: the caller is now speaking, so this generation
        # runs behind its own outgoing audio and costs no conversational latency.
        if utterance:
            await self._prerender_candidate(utterance)
        return False

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
        started = time.monotonic()
        prompt = PromptManager().get_prompt("user_simulator.cascade_next_interruption", utterance=utterance)
        try:
            message, _stats = await self._llm.complete(
                messages=[*self._messages(), {"role": "user", "content": prompt}]
            )
        except Exception as exc:
            logger.warning(f"Speculative interruption generation failed: {exc}")
            self._decision_log.log("candidate_generation", ok=False, error=str(exc))
            return
        candidate = extract_optional_line(message)
        if not candidate:
            raw = message if isinstance(message, str) else (getattr(message, "content", None) or "")
            self._decision_log.log("candidate_generation", ok=False, declined=True, after=utterance, raw=raw[:400])
            return
        self._candidate_text = candidate
        self._candidate_audio = await self._tts.synthesize(candidate, voice_id=self._voice_id)
        self._decision_log.log(
            "candidate_generation",
            ok=True,
            after=utterance,
            candidate=candidate,
            audio_bytes=len(self._candidate_audio),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

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
