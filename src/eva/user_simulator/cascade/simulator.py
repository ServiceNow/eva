"""Cascade (streaming STT -> LLM -> TTS) implementation of the EVA simulated caller."""

from __future__ import annotations

import asyncio
import json
import random
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.audio_bridge import BotToBotAudioBridge
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.barge_in import BargeIn, trim_silence
from eva.user_simulator.cascade.interruptor import DecisionsInterruptor
from eva.user_simulator.cascade.stt import ScribeStreamingSTT
from eva.user_simulator.cascade.tts import TTS_SAMPLE_RATE, CartesiaStreamingTTS
from eva.utils import router
from eva.utils.logging import get_logger

logger = get_logger(__name__)

TURN_SETTLE_SECONDS = 1.0
"""Assistant silence required before the caller replies."""
TRANSCRIPT_WAIT_SECONDS = 2.5
"""How long after a commit to wait for the final transcript before failing the call.

The commit is sent at the start of the pause, so this leaves about 1.5s past ``TURN_SETTLE_SECONDS``
for a slow final. Scribe usually answers within a few hundred ms; only a stalled final costs this wait."""
INACTIVITY_TIMEOUT_SECONDS = 120.0
"""Assistant silence that ends the call, matching ElevenLabsUserSimulator's 12 keep-alives."""
PLAYBACK_DRAIN_SECONDS = 15.0
FINAL_FLUSH_SECONDS = 4.0
POLL_SECONDS = 0.05

MAX_UNTRANSCRIBED_IN_A_ROW = 2
"""Agent lines in a row the caller could not transcribe before the call fails: one is a bad line,
repeated ones mean the caller cannot hear the agent, and the conversation would not be a fair test."""

INAUDIBLE = "[inaudible]"
"""Stands in for an agent line the caller heard but could not transcribe."""
INAUDIBLE_NOTE = f"""The agent just said something you could not make out ({INAUDIBLE} above). \
Ask them briefly to repeat it, the way a caller on a bad line would."""

ASSISTANT_AUDIO_TICK_S = 0.02
"""Bridge input tick: the agent's audio reaches the STT this often, as it arrives, not in 250ms batches."""

TTS_WARM_CONNECTIONS = 2
"""Pooled TTS connections opened before the first turn: one per TTS request that can render at once.
A barge-in renders its first clause and the next sentence concurrently (see ``next_segment``)."""

DECISION_WARM_CONNECTIONS = 2
"""Pooled decision connections opened before the first turn: consecutive transcript updates'
decisions can overlap."""

OPENERS = ["Sorry, wait\u2014"]
"""Pre-rendered at startup and played the moment a decision says interrupt, masking the reply's latency.
One phrase for now, so barge-in timings are measured against a fixed opener duration."""

BARGE_IN_NOTE = """The agent is still speaking and their last line above is cut off where you stopped \
listening. You are cutting them off now, and you have ALREADY said "{opener}" out loud. Write only \
the words that come right after it, picking up mid-sentence. Do not repeat "{opener}" or say anything \
like it: no apology, no "wait", and no other interjection or filler to get their attention, in any \
language. Start directly with what you need to say."""

CLAUSE_MIN_WORDS = 3
"""The first clause is spoken early only once it has this many words, so TTS has enough to phrase."""
CLAUSE_BREAKS = ",;:\u2014\u2013"
SENTENCE_BREAKS = ".?!"

_PERSONA_GENDER = {1: "F", 2: "M"}
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)

END_CALL_DESCRIPTION = """Use this to end the phone call and hang up.

Call this function when it is time to end the call and one of the following is true:
1. The agent has confirmed your request is resolved and all steps are completed.
2. The agent has initiated a transfer to a live agent.
3. The agent has been unable to make progress for at least 5 consecutive turns.
4. The agent says goodbye or indicates the conversation is over.
5. The agent indicates that the remainder of your request cannot be fulfilled.
6. The assistant reports an unrecoverable processing error.

Only once one of these is true, say a brief goodbye and call this tool in that same response.
Your goodbye is spoken in full before the call hangs up, so do not wait for another turn to call it.

That same-response rule is only for goodbyes. Never call this tool in a response that gives the
agent data, an identifier, an approval to proceed, a transfer request, or any other information,
or that asks the agent anything: that response needs the agent's reply first. A response that
calls this tool contains only your goodbye."""
"""Written for the cascade, not shared with OpenAI Realtime: the caller hangs up in the goodbye turn
itself, since the cascade speaks a response's text before acting on its end_call, and waiting for a
later turn stalls when the agent doesn't answer the goodbye."""

END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": END_CALL_DESCRIPTION,
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


KEEP_LISTENING_DESCRIPTION = """Use this instead of replying when the agent has not finished its turn.

Agents pause mid-turn while they look something up, so what you heard may be only a first part: a
filler like "Let me check that", "One moment", or "I'll look that up", or an announcement of what
they are about to do. A caller waits through those in silence. Call this tool, with no text, to keep
listening; the rest of the agent's turn will follow.

Never use it when the agent asked you a question, gave you a result, or said goodbye: those need
your reply."""

KEEP_LISTENING_TOOL = {
    "type": "function",
    "function": {
        "name": "keep_listening",
        "description": KEEP_LISTENING_DESCRIPTION,
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

MAX_KEEP_LISTENING_IN_A_ROW = 2
"""After this many silent waits the tool is withheld, so a caller that keeps waiting still answers
rather than stalling the call until the inactivity timeout."""


def next_segment(buffer: str, *, first: bool, final: bool) -> tuple[str, str]:
    """Split the next speakable segment off streamed text; returns (segment, rest).

    Not used on regular turns, which speak the whole line in one request (see ``_speak``).
    Kept for barge-ins, where speaking the first clause early beats waiting for the full line.

    The first segment may end at a clause break, so audio starts as soon as a few words exist;
    later ones end at sentences, so TTS gets whole sentences to phrase. A break only counts once
    the following character is whitespace, so "4.5" or "MRC-07" never split. ``final`` flushes.
    """
    breaks = SENTENCE_BREAKS + (CLAUSE_BREAKS if first else "")
    for index, char in enumerate(buffer[:-1]):
        if char not in breaks or not buffer[index + 1].isspace():
            continue
        words = buffer[:index].split()
        if char in CLAUSE_BREAKS and len(words) < CLAUSE_MIN_WORDS:
            continue
        # A spelled-out letter or digit ("F F W,") said last in its own TTS request sounds clipped.
        if char in CLAUSE_BREAKS and len(words[-1]) == 1:
            continue
        return buffer[: index + 1].strip(), buffer[index + 1 :]
    if final:
        return buffer.strip(), ""
    return "", buffer


def extract_turn(message: Any) -> tuple[str, bool]:
    """Read (utterance, end_call) from LiteLLMClient output: a bare str, or a message with tool calls."""
    if isinstance(message, str):
        return _FENCE.sub("", message).strip(), False
    content = getattr(message, "content", None) or ""
    calls = getattr(message, "tool_calls", None) or []
    end_call = any(getattr(call.function, "name", "") == "end_call" for call in calls)
    return _FENCE.sub("", content).strip(), end_call


def calls_tool(message: Any, name: str) -> bool:
    """Whether a LiteLLMClient message called the tool ``name``."""
    return any(getattr(call.function, "name", "") == name for call in getattr(message, "tool_calls", None) or [])


class CascadeUserSimulator(AbstractUserSimulator):
    """Simulated caller built from independently chosen streaming STT, LLM, and TTS models."""

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
        if perturbation_config and perturbation_config.accent is not None:
            raise ValueError("Cascade caller does not support ElevenLabs-specific accent variants")
        self.simulator_config = simulator_config
        self._stt = ScribeStreamingSTT(simulator_config.stt_model, language=language)
        self._tts = CartesiaStreamingTTS(simulator_config.tts_model, language=language)
        self._llm = LiteLLMClient(model=simulator_config.model)
        self._barge_in_llm = LiteLLMClient(model=simulator_config.barge_in_model or simulator_config.model)
        # None is a commit marker: commit the STT segment once the audio queued ahead of it is sent.
        self._assistant_audio_queue: asyncio.Queue[tuple[bytes, float] | None] = asyncio.Queue()
        self._commit_after_next_chunk = False
        self._history: list[dict[str, str]] = []
        self._system_prompt: str | None = None
        self._kept_listening = 0
        self._interruptor = DecisionsInterruptor(simulator_config.interruption_model)
        self._openers: list[tuple[str, bytes]] = []
        self._hung_up = False
        """The caller called end_call: it never speaks, decides or barges in again."""
        self._barge_in_warm_up: asyncio.Task | None = None
        self._turn_active = False
        """A caller turn (regular or barge-in) is being generated or spoken; no new barge-in starts."""
        self._barge_ins = 0
        """Barge-ins so far; a decision requested before the latest one is stale and never fires."""
        self._barge_in_warm_up_turn: int | None = None
        """The agent turn the barge-in LLM was last warmed for (see ``_warm_up_barge_in_llm``)."""
        self._tails_recorded = 0
        """Talked-over tails of interrupted agent lines recorded so far (see ``_record_talked_over_tail``)."""
        self._caller_turns = 0
        """Caller replies so far (regular or barge-in): each one ends an agent turn."""

    @property
    def caller_voice(self) -> str:
        gender = _PERSONA_GENDER.get(self.persona_config.get("user_persona_id"))
        if gender == "M":
            return self.simulator_config.male_voice
        return self.simulator_config.female_voice

    async def run_conversation(self) -> str:
        try:
            await self._run_cascade_conversation()
        except Exception as exc:
            logger.error(f"Cascade caller simulation error: {exc}", exc_info=True)
            self._end_reason = "error"
            self.event_logger.log_error(str(exc))
            self.signal_conversation_ending("error")
        finally:
            # Logged last, after any error, so the final event carries the true end reason.
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})
            self.event_logger.save()
        return self._end_reason

    async def _run_cascade_conversation(self) -> None:
        self._audio_interface = BotToBotAudioBridge(
            websocket_uri=self.server_url,
            conversation_id=self.output_dir.name,
            record_callback=self._record_audio,
            event_logger=self.event_logger,
            conversation_done_callback=self._on_conversation_end,
            perturbator=self._perturbator,
            disconnect_reason="assistant_disconnect",
            input_chunk_duration=ASSISTANT_AUDIO_TICK_S,
        )
        tasks: list[asyncio.Task] = []
        try:
            await self._stt.start()
            await self._tts.warm_up(self.caller_voice, connections=TTS_WARM_CONNECTIONS)
            await self._interruptor.warm_up(self._decision_goal(), connections=DECISION_WARM_CONNECTIONS)
            self._openers = await self._render_openers()
            await self._audio_interface.start_async()
            self._audio_interface.start(self._on_assistant_audio)
            self.event_logger.log_connection_state(
                "connected",
                {
                    "server_url": self.server_url,
                    "caller_provider": self.provider,
                    "caller_stt": f"{self.simulator_config.stt}/{self.simulator_config.stt_model}",
                    "caller_llm": self.simulator_config.model,
                    "caller_barge_in_llm": self.simulator_config.barge_in_model or self.simulator_config.model,
                    "caller_interruption_model": self.simulator_config.interruption_model,
                    "caller_tts": f"{self.simulator_config.tts}/{self.simulator_config.tts_model}",
                    "caller_voice": self.caller_voice,
                },
            )
            completion_task = asyncio.create_task(self._wait_for_conversation_end())
            tasks = [completion_task, asyncio.create_task(self._forward_assistant_audio())]
            tasks.append(asyncio.create_task(self._run_turns()))
            tasks.append(asyncio.create_task(self._watch_for_interruptions()))

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if completion_task not in done and not self._conversation_done.is_set():
                finished = next(iter(done))
                # exception() raises CancelledError on a cancelled task, which would skip the error path below.
                error = None if finished.cancelled() else finished.exception()
                raise error or RuntimeError("Cascade caller task stopped unexpectedly")

            # Mark the bridge stopped, as ElevenLabs' end_session() does: an assistant that hangs up on
            # signal_conversation_ending (e.g. Cartesia) drops the socket during the flush below, and a
            # still-running bridge would report that as a disconnect mid-conversation.
            self._audio_interface.stop()
            # Allow final goodbye audio and transcripts to flush before closing.
            await asyncio.sleep(FINAL_FLUSH_SECONDS)
        except Exception:
            # Tell the assistant before teardown, as the base contract asks of every terminal path.
            self.signal_conversation_ending("error")
            raise
        finally:
            if self._barge_in_warm_up is not None:
                tasks.append(self._barge_in_warm_up)
            for task in tasks:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            await self._stt.stop()
            await self._tts.close()
            await self._interruptor.close()
            await self._audio_interface.stop_async()
            self._save_clean_user_audio(TTS_SAMPLE_RATE)

    async def _wait_for_conversation_end(self) -> None:
        try:
            await asyncio.wait_for(self._conversation_done.wait(), timeout=self.timeout)
        except TimeoutError:
            self.event_logger.log_event("timeout", {"duration": self.timeout})
            self._on_conversation_end("timeout")

    def _on_assistant_audio(self, mulaw_audio: bytes) -> None:
        # Forwarded unconditionally, silence included, so the caller keeps hearing while it speaks.
        # Stamped on arrival, when the caller has heard it, so STT word times are not skewed by the queue.
        if mulaw_audio:
            self._assistant_audio_queue.put_nowait((mulaw_audio, asyncio.get_running_loop().time()))
            if self._commit_after_next_chunk:
                self._commit_after_next_chunk = False
                self._assistant_audio_queue.put_nowait(None)

    def _request_commit(self) -> None:
        """Commit once the next bridge chunk is forwarded.

        The bridge flags the assistant's audio as ended before it hands over the tick that holds
        the tail, so committing right away would cut the last word into the next segment.
        A request already waiting for that chunk covers this one too.
        """
        if self._commit_after_next_chunk:
            return
        self._commit_after_next_chunk = True
        self._stt.expect_commit()

    async def _forward_assistant_audio(self) -> None:
        while True:
            item = await self._assistant_audio_queue.get()
            if item is None:
                await self._stt.commit()
            else:
                await self._stt.send(item[0], heard_at=item[1])

    async def _run_turns(self) -> None:
        """Reply once the assistant's turn has settled; end the call if it goes quiet for too long."""
        loop = asyncio.get_running_loop()
        bridge = self._audio_interface
        assistant_spoke = False
        committed_at: float | None = None
        last_activity = loop.time()
        untranscribed = 0
        tails_recorded = self._tails_recorded

        while not self._conversation_done.is_set():
            await asyncio.sleep(POLL_SECONDS)
            now = loop.time()
            if self._tails_recorded != tails_recorded:
                # A barge-in answered what the assistant had said; wait for it to speak again.
                tails_recorded, assistant_spoke, committed_at, last_activity = self._tails_recorded, False, None, now
            if bridge.is_assistant_playing():
                assistant_spoke, committed_at, last_activity = True, None, now
                continue
            if bridge.is_caller_playing():
                last_activity = now
                continue
            if not assistant_spoke:
                if now - last_activity > INACTIVITY_TIMEOUT_SECONDS:
                    self._on_conversation_end("inactivity_timeout")
                    return
                continue

            # Commit each pause, so a mid-turn pause closes a segment and the rest accumulates.
            if committed_at is None:
                self._request_commit()
                committed_at = now
            # The caller starting to speak clears the bridge's end time, so a line the assistant finished
            # during the caller's turn settles from when this pause was first seen instead.
            ended_at = bridge.assistant_audio_ended_at or committed_at
            if now - ended_at < TURN_SETTLE_SECONDS:
                continue

            if self._turn_active:
                continue  # a barge-in is replying; this pause is part of it
            heard = await self._collect_heard_text(committed_at)
            assistant_spoke = False
            if heard:
                untranscribed = 0
            else:
                # Staying silent would stall the call until the inactivity timeout; a caller asks again.
                self.event_logger.log_event("assistant_audio_untranscribed", {})
                untranscribed += 1
                if untranscribed >= MAX_UNTRANSCRIBED_IN_A_ROW:
                    raise RuntimeError(f"Caller could not transcribe {untranscribed} agent lines in a row")
            starts = bridge.assistant_audio_starts
            if await self._take_turn(heard or None):
                return
            last_activity = loop.time()
            if bridge.assistant_audio_starts != starts:
                # The assistant spoke while the caller was generating (e.g. "One moment."). The polls above
                # never saw it play, so without this it would go uncommitted, merge into the next turn, or
                # leave the caller silent until the inactivity timeout if the assistant waits for an answer.
                self.event_logger.log_event("assistant_spoke_during_caller_turn", {})
                assistant_spoke, committed_at = True, None

    async def _collect_heard_text(self, committed_at: float) -> str:
        """Return what the assistant said; a final that never arrives fails the call."""
        remaining = TRANSCRIPT_WAIT_SECONDS - (asyncio.get_running_loop().time() - committed_at)
        try:
            await asyncio.wait_for(self._stt.committed_event.wait(), timeout=max(remaining, 0))
        except TimeoutError:
            # Acting on the partial would answer a cut-off line; failing lets the call be rerun.
            raise RuntimeError(f"No final transcript within {TRANSCRIPT_WAIT_SECONDS}s of the commit") from None
        return self._stt.take()

    async def _take_turn(self, heard: str | None, *, barge_in: BargeIn | None = None) -> bool:
        """Generate and speak one caller turn. Returns True when the caller hangs up.

        ``heard`` is None when the agent spoke but nothing was transcribed: the caller asks them to repeat.
        ``barge_in`` marks a barge-in: its opener is already playing, and the reply follows it.
        """
        self._turn_active = True
        try:
            return await self._generate_turn(heard, barge_in)
        finally:
            self._turn_active = False

    async def _generate_turn(self, heard: str | None, barge_in: BargeIn | None) -> bool:
        self._history.append({"role": "assistant", "content": heard or INAUDIBLE})
        if heard:
            self._on_assistant_speaks(heard)

        messages = self._messages()
        if not heard:
            messages.append({"role": "system", "content": INAUDIBLE_NOTE})
        if barge_in is None:
            message, stats, spoke = await self._speak(messages)
        else:
            messages.append({"role": "system", "content": BARGE_IN_NOTE.format(opener=barge_in.opener)})
            message, stats, spoke = await self._speak_barge_in(messages, barge_in)
        utterance, end_call = extract_turn(message)
        self._log_llm_response(messages, message, stats)
        if barge_in is not None:
            # The opener is part of what the caller said, so history keeps it too.
            utterance, spoke = f"{barge_in.opener} {barge_in.content}".strip(), True

        if calls_tool(message, "keep_listening") and not end_call:
            self._kept_listening += 1
            return False
        self._kept_listening = 0
        self._caller_turns += 1
        if utterance:
            self._history.append({"role": "user", "content": utterance})
            self._on_user_speaks(utterance)
        if spoke:
            # A trailing partial chunk is what lets the bridge detect the end of the utterance.
            self._audio_interface.output(b"\x00\x00")
            self._audio_interface.notify_user_utterance_complete()

        if end_call:
            self._hung_up = True
            self.event_logger.log_event("tool_call", {"name": "end_call", "arguments": "{}"})
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wait_for_caller_playback(), timeout=PLAYBACK_DRAIN_SECONDS)
            self._on_conversation_end("goodbye")
            return True
        return False

    async def _speak(self, messages: list[dict[str, Any]]) -> tuple[Any, dict[str, Any], bool]:
        """Generate the whole caller line, then speak it in one TTS request with the audio streamed back.

        One request per line keeps the audio gap-free (Cartesia renders faster than real time) and
        the prosody continuous, the way an ElevenLabs caller sounds. Splitting the text at
        ``next_segment`` breaks would start audio before the line is written, but a stall between
        segments can let the bridge end the utterance mid-sentence. Splitting is reserved for
        barge-ins (``_speak_barge_in``), where the first clause must start before the rest of the line exists.

        Returns the assembled final message, its stats, and whether any audio was sent.
        """
        message: Any = ""
        stats: dict[str, Any] = {}
        tools = [END_CALL_TOOL]
        if self._kept_listening < MAX_KEEP_LISTENING_IN_A_ROW:
            tools.append(KEEP_LISTENING_TOOL)
        async for kind, payload in self._llm.complete_stream(messages=messages, tools=tools):
            if kind == "final":
                message, stats = payload
        utterance, _ = extract_turn(message)
        spoke = False
        if utterance and not calls_tool(message, "keep_listening"):
            async for chunk in self._tts.stream(utterance, voice_id=self.caller_voice):
                self._audio_interface.output(chunk)
                spoke = True
        return message, stats, spoke

    async def _speak_barge_in(
        self, messages: list[dict[str, Any]], barge_in: BargeIn
    ) -> tuple[Any, dict[str, Any], bool]:
        """Speak a barge-in reply as it streams: each ``next_segment`` goes to TTS the moment it is written.

        The first clause starts rendering while the rest is still being generated. Segments render
        concurrently but play in order; one that is late leaves a gap rather than reordering speech.
        The reply's leading silence is dropped as it arrives. Keep-listening is not offered: the
        opener is already out.

        Returns the assembled final message, its stats, and whether any audio was sent.
        """
        loop = asyncio.get_running_loop()
        segments: asyncio.Queue[asyncio.Queue[bytes | None] | None] = asyncio.Queue()
        renders: list[asyncio.Task] = []

        async def render(text: str, out: asyncio.Queue[bytes | None]) -> None:
            try:
                async for chunk in self._tts.stream(text, voice_id=self.caller_voice):
                    out.put_nowait(chunk)
            finally:
                out.put_nowait(None)

        def submit(text: str) -> None:
            out: asyncio.Queue[bytes | None] = asyncio.Queue()
            barge_in.segment_sent(loop.time())
            renders.append(asyncio.create_task(render(text, out)))
            segments.put_nowait(out)

        async def play() -> bool:
            spoke = False
            while (out := await segments.get()) is not None:
                while (chunk := await out.get()) is not None:
                    if chunk := barge_in.audible(chunk, loop.time()):
                        self._audio_interface.output(chunk)
                        spoke = True
            return spoke

        player = asyncio.create_task(play())
        message: Any = ""
        stats: dict[str, Any] = {}
        buffer, first = "", True
        try:
            async for kind, payload in self._barge_in_llm.complete_stream(messages=messages, tools=[END_CALL_TOOL]):
                if kind == "final":
                    message, stats = payload
                    continue
                barge_in.first_token(loop.time())
                buffer += payload
                while True:
                    segment, buffer = next_segment(buffer, first=first, final=False)
                    if not segment:
                        break
                    submit(segment)
                    first = False
            segment, _ = next_segment(buffer, first=first, final=True)
            if segment:
                submit(segment)
            segments.put_nowait(None)
            spoke = await player
            await asyncio.gather(*renders)  # surface a failed render
        finally:
            for task in [player, *renders]:
                task.cancel()
        barge_in.content, _ = extract_turn(message)
        return message, stats, spoke

    def _can_interrupt(self) -> bool:
        """Barge in only on the agent's speech: not over the caller's own, a turn in progress, or a hang-up."""
        bridge = self._audio_interface
        return (
            bridge.is_assistant_playing()
            and not bridge.is_caller_playing()
            and not self._turn_active
            and not self._hung_up
        )

    async def _watch_for_interruptions(self) -> None:
        """Ask for a decision on every transcript update while the agent speaks.

        No decision is requested where no barge-in may fire: the first ``interruption_min_updates``
        updates of each agent turn, agent turns before ``interruption_first_turn``, the
        ``interruption_cooldown_updates`` updates after a barge-in, and once
        ``interruption_max_per_call`` barge-ins have fired.
        """
        config = self.simulator_config
        goal = self._decision_goal()
        decisions: set[asyncio.Task] = set()
        last_text = ""
        turn, updates = self._caller_turns, 0
        fired, since_barge_in = 0, 0
        try:
            while not self._conversation_done.is_set():
                await self._stt.updated.wait()
                self._stt.updated.clear()
                text = self._stt.heard_text()
                if not text or text == last_text or not self._can_interrupt():
                    continue
                last_text = text
                if self._caller_turns != turn:
                    turn, updates = self._caller_turns, 0
                updates += 1
                if self._barge_ins != fired:
                    fired, since_barge_in = self._barge_ins, 0
                since_barge_in += 1
                if (
                    updates <= config.interruption_min_updates
                    or turn < config.interruption_first_turn
                    or (fired and since_barge_in <= config.interruption_cooldown_updates)
                    or (config.interruption_max_per_call is not None and fired >= config.interruption_max_per_call)
                ):
                    continue
                if self._barge_in_warm_up_turn != turn:  # once per turn, as its first decision is requested
                    self._barge_in_warm_up_turn = turn
                    self._warm_up_barge_in_llm()
                # Where the update's last word sits, so its end can be read off the STT word timestamps.
                barge_in = BargeIn(
                    heard_at=asyncio.get_running_loop().time(),
                    decision_ms=0.0,
                    p_interrupt=0.0,
                    segment_start=self._stt.segment_start,
                    trigger_words=len(self._stt.partial.split()),
                    forced=self._forces_interruption(turn, updates),
                )
                # Not awaited: the next update's decision must not queue behind this one.
                task = asyncio.create_task(self._decide(goal, text, barge_in))
                decisions.add(task)
                task.add_done_callback(decisions.discard)
        finally:
            for task in decisions:
                task.cancel()

    def _warm_up_barge_in_llm(self) -> None:
        """Once per agent turn, as barge-ins become possible: prime the barge-in model with one token.

        Sent with the conversation so far, so it warms both the connection (a cold first request
        measured ~2.4s to first text) and the provider's prompt cache for the reply's long prefix.
        """
        if self._barge_in_warm_up is not None and not self._barge_in_warm_up.done():
            return

        async def warm_up() -> None:
            try:
                request: dict[str, Any] = {"messages": self._messages(), "tools": [END_CALL_TOOL], "max_tokens": 1}
                await router.get().acompletion(model=self._barge_in_llm.model, **request)
            except Exception as exc:
                logger.debug(f"Barge-in LLM warm-up failed: {exc}")

        self._barge_in_warm_up = asyncio.create_task(warm_up())

    def _forces_interruption(self, turn: int, updates: int) -> bool:
        """Whether forced_interruption_every_turns overrides this decision.

        ``turn`` is the 0-based agent turn, ``updates`` the transcript updates heard in it so far.
        """
        every = self.simulator_config.forced_interruption_every_turns
        return (
            every is not None
            and (turn + 1) % every == 0
            and updates >= self.simulator_config.forced_interruption_after_updates
        )

    async def _decide(self, goal: str, heard: str, barge_in: BargeIn) -> None:
        """One decision; an ``interrupt`` fires if barging in is still allowed and nothing fired since.

        ``barge_in`` holds when and where the transcript update ``heard`` came from, and times the barge-in if it fires.
        """
        loop = asyncio.get_running_loop()
        requested_at = self._barge_ins
        asked_at = loop.time()
        try:
            choice, p_interrupt = await self._interruptor.decide(goal, list(self._history), heard)
        except Exception as exc:
            logger.warning(f"Interruption decision failed: {exc}")
            return
        decision_ms = (loop.time() - asked_at) * 1000.0
        fire = (choice == "interrupt" or barge_in.forced) and self._can_interrupt() and self._barge_ins == requested_at
        self.event_logger.log_event(
            "interruption_decision",
            {
                "heard": heard,
                "choice": choice,
                "p_interrupt": p_interrupt,
                "decision_ms": decision_ms,
                "forced": barge_in.forced,
                "fired": fire,
            },
        )
        if fire:
            barge_in.decision_ms, barge_in.p_interrupt = decision_ms, p_interrupt
            await self._interrupt(barge_in)

    async def _interrupt(self, barge_in: BargeIn) -> None:
        """Sequential barge-in: play an opener now, then generate and speak the reply behind it.

        The agent may keep talking. A failure ends the call as an error, as a failed regular turn does.
        """
        self._turn_active = True  # before any await, so a concurrent decision cannot fire too
        self._barge_ins += 1
        bridge = self._audio_interface
        starts, caller_starts = bridge.assistant_audio_starts, bridge.user_audio_starts
        phrase, pcm = random.choice(self._openers)
        bridge.output(pcm)
        barge_in.play_opener(phrase, pcm, asyncio.get_running_loop().time())
        # Close the segment being heard, so its final does not repeat into the next turn.
        self._request_commit()
        barge_in.heard = self._stt.take(include_partial=True)
        try:
            await self._take_turn(barge_in.heard, barge_in=barge_in)
        except Exception as exc:
            logger.error(f"Cascade caller barge-in failed: {exc}", exc_info=True)
            self.event_logger.log_error(f"Barge-in failed: {exc}")
            self._on_conversation_end("error")
            return
        if bridge.user_audio_starts == caller_starts + 1 and bridge.user_audio_started_at is not None:
            barge_in.opener_on_wire(bridge.user_audio_started_at)  # the utterance the opener started
        self._record_talked_over_tail(starts)
        # The cut-off segment's timestamped final has arrived by now: it was committed as the opener started.
        barge_in.word_end_at = self._stt.word_end(barge_in.segment_start, barge_in.trigger_words - 1)
        latency = barge_in.latency()
        self.barge_ins.append(latency)
        self.event_logger.log_event("barge_in", latency.model_dump())
        logger.info(f"Barge-in timing: {latency.model_dump(exclude={'heard', 'content'})}")

    def _record_talked_over_tail(self, starts: int) -> None:
        """Record what the agent said over the barge-in, if that was only the rest of the interrupted line.

        It goes into history after the caller's barge-in, since the caller heard it, but gets no reply
        of its own: the caller already answered that line, and a tail with no words (the agent stopping
        a moment after the opener) would read as an inaudible line. Left for a regular turn when the
        agent is still talking or has started a new line (``starts`` is the bridge's count at the
        barge-in), since the caller owes that a reply.
        """
        bridge = self._audio_interface
        if bridge.assistant_audio_starts != starts or bridge.is_assistant_playing():
            return
        self._request_commit()
        tail = self._stt.take(include_partial=True)
        if tail:
            self._history.append({"role": "assistant", "content": tail})
        self._tails_recorded += 1
        self.event_logger.log_event("barge_in_tail", {"text": tail})

    async def _render_openers(self) -> list[tuple[str, bytes]]:
        async def render(phrase: str) -> tuple[str, bytes]:
            # Trimmed, so every millisecond of playout covers the reply and the gap is not padded.
            pcm = b"".join([chunk async for chunk in self._tts.stream(phrase, voice_id=self.caller_voice)])
            return phrase, trim_silence(pcm)

        return list(await asyncio.gather(*(render(phrase) for phrase in OPENERS)))

    def _decision_goal(self) -> str:
        """The caller's goal as the decision model reads it: what they want and what they hold."""
        tree = self.goal.get("decision_tree", {})
        return "\n".join(
            [
                f"GOAL: {self.goal.get('high_level_user_goal', '')}",
                f"MUST HAVE: {tree.get('must_have_criteria', '')}",
                f"INFORMATION YOU HAVE READY: {self.goal.get('information_required', '')}",
                f"RESOLVED WHEN: {tree.get('resolution_condition', '')}",
            ]
        )

    def _log_llm_response(self, messages: list[dict[str, str]], message: Any, stats: dict[str, Any]) -> None:
        """Record exactly what the caller LLM was sent and returned, for diagnosing turn decisions."""
        calls = getattr(message, "tool_calls", None) or []
        self.event_logger.log_event(
            "caller_llm_response",
            {
                "response_type": type(message).__name__,
                "content": message if isinstance(message, str) else getattr(message, "content", None),
                "tool_calls": [
                    {
                        "name": getattr(call.function, "name", None),
                        "arguments": getattr(call.function, "arguments", None),
                    }
                    for call in calls
                ],
                "finish_reason": stats.get("finish_reason"),
                "model": stats.get("model"),
                "parameters": json.loads(json.dumps(stats.get("parameters"), default=str)),
                "message_roles": [m["role"] for m in messages],
                "last_message": messages[-1]["content"] if len(messages) > 1 else None,
                "latency": stats.get("latency"),
                "prompt_tokens": stats.get("prompt_tokens"),
            },
        )

    async def _wait_for_caller_playback(self) -> None:
        while self._audio_interface.is_caller_playing():
            await asyncio.sleep(POLL_SECONDS)

    def _messages(self) -> list[dict[str, str]]:
        """The shared caller prompt plus history, with roles flipped into the caller LLM's own frame.

        ``_history`` keeps conversation-truth roles; the caller LLM is the "assistant" in its own
        frame, so without the flip it reads the agent's lines as its own prior output.
        """
        flipped = [
            {"role": "assistant" if turn["role"] == "user" else "user", "content": turn["content"]}
            for turn in self._history
        ]
        if self._system_prompt is None:
            # Built once: PromptManager re-reads its YAML (~35ms, blocking the loop), which on a barge-in
            # landed between queuing the opener and the bridge sending it.
            self._system_prompt = self._build_prompt()
        return [{"role": "system", "content": self._system_prompt}, *flipped]
