"""Cascade (streaming STT -> LLM -> TTS) implementation of the EVA simulated caller."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.audio_bridge import BotToBotAudioBridge
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.stt import ScribeStreamingSTT
from eva.user_simulator.cascade.tts import TTS_SAMPLE_RATE, CartesiaStreamingTTS
from eva.utils.logging import get_logger

logger = get_logger(__name__)

TURN_SETTLE_SECONDS = 1.0
"""Assistant silence required before the caller replies."""
TRANSCRIPT_WAIT_SECONDS = 1.5
"""How long after a commit to wait for the final transcript before falling back to the partial."""
INACTIVITY_TIMEOUT_SECONDS = 120.0
"""Assistant silence that ends the call, matching ElevenLabsUserSimulator's 12 keep-alives."""
PLAYBACK_DRAIN_SECONDS = 15.0
FINAL_FLUSH_SECONDS = 4.0
POLL_SECONDS = 0.05

TTS_WARM_CONNECTIONS = 1
"""Pooled TTS connections opened before the first turn: one per TTS request that can render at once.
Barge-ins that split the first clause off (see ``next_segment``) will want 2."""

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
        if char in CLAUSE_BREAKS and len(buffer[:index].split()) < CLAUSE_MIN_WORDS:
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
        self._llm = LiteLLMClient(model=simulator_config.llm)
        # None is a commit marker: commit the STT segment once the audio queued ahead of it is sent.
        self._assistant_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._commit_after_next_chunk = False
        self._history: list[dict[str, str]] = []

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
        )
        tasks: list[asyncio.Task] = []
        try:
            await self._stt.start()
            await self._tts.warm_up(self.caller_voice, connections=TTS_WARM_CONNECTIONS)
            await self._audio_interface.start_async()
            self._audio_interface.start(self._on_assistant_audio)
            self.event_logger.log_connection_state(
                "connected",
                {
                    "server_url": self.server_url,
                    "caller_provider": self.provider,
                    "caller_stt": f"{self.simulator_config.stt}/{self.simulator_config.stt_model}",
                    "caller_llm": self.simulator_config.llm,
                    "caller_tts": f"{self.simulator_config.tts}/{self.simulator_config.tts_model}",
                    "caller_voice": self.caller_voice,
                },
            )
            completion_task = asyncio.create_task(self._wait_for_conversation_end())
            tasks = [completion_task, asyncio.create_task(self._forward_assistant_audio())]
            tasks.append(asyncio.create_task(self._run_turns()))

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if completion_task not in done and not self._conversation_done.is_set():
                finished = next(iter(done))
                raise finished.exception() or RuntimeError("Cascade caller task stopped unexpectedly")

            # Allow final goodbye audio and transcripts to flush before closing.
            await asyncio.sleep(FINAL_FLUSH_SECONDS)
        except Exception:
            # Tell the assistant before teardown, as the base contract asks of every terminal path.
            self.signal_conversation_ending("error")
            raise
        finally:
            for task in tasks:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            await self._stt.stop()
            await self._tts.close()
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
        if mulaw_audio:
            self._assistant_audio_queue.put_nowait(mulaw_audio)
            if self._commit_after_next_chunk:
                self._commit_after_next_chunk = False
                self._assistant_audio_queue.put_nowait(None)

    def _request_commit(self) -> None:
        """Commit once the next bridge chunk is forwarded.

        The bridge flags the assistant's audio as ended before it hands over the tick that holds
        the tail, so committing right away would cut the last word into the next segment.
        """
        self._commit_after_next_chunk = True
        # Until the commit is sent, an earlier turn's set event must not count as this segment's final.
        self._stt.committed_event.clear()

    async def _forward_assistant_audio(self) -> None:
        while True:
            item = await self._assistant_audio_queue.get()
            if item is None:
                await self._stt.commit()
            else:
                await self._stt.send(item)

    async def _run_turns(self) -> None:
        """Reply once the assistant's turn has settled; end the call if it goes quiet for too long."""
        loop = asyncio.get_running_loop()
        bridge = self._audio_interface
        assistant_spoke = False
        committed_at: float | None = None
        last_activity = loop.time()

        while not self._conversation_done.is_set():
            await asyncio.sleep(POLL_SECONDS)
            now = loop.time()
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
            ended_at = bridge.assistant_audio_ended_at
            if ended_at is None or now - ended_at < TURN_SETTLE_SECONDS:
                continue

            heard = await self._collect_heard_text(committed_at)
            assistant_spoke = False
            if not heard:
                self.event_logger.log_event("assistant_audio_untranscribed", {})
                continue
            if await self._take_turn(heard):
                return
            last_activity = loop.time()

    async def _collect_heard_text(self, committed_at: float) -> str:
        """Return what the assistant said, falling back to the partial if the final is late."""
        remaining = TRANSCRIPT_WAIT_SECONDS - (asyncio.get_running_loop().time() - committed_at)
        try:
            await asyncio.wait_for(self._stt.committed_event.wait(), timeout=max(remaining, 0))
        except TimeoutError:
            text = self._stt.take(include_partial=True)
            logger.warning(f"No final transcript after {TRANSCRIPT_WAIT_SECONDS}s; using partial: {text[:120]!r}")
            self.event_logger.log_event("transcript_partial_fallback", {"text": text})
            return text
        return self._stt.take()

    async def _take_turn(self, heard: str) -> bool:
        """Generate and speak one caller turn. Returns True when the caller hangs up."""
        self._history.append({"role": "assistant", "content": heard})
        self._on_assistant_speaks(heard)

        messages = self._messages()
        message, stats, spoke = await self._speak(messages)
        utterance, end_call = extract_turn(message)
        self._log_llm_response(messages, message, stats)

        if utterance:
            self._history.append({"role": "user", "content": utterance})
            self._on_user_speaks(utterance)
        if spoke:
            # A trailing partial chunk is what lets the bridge detect the end of the utterance.
            self._audio_interface.output(b"\x00\x00")
            self._audio_interface.notify_user_utterance_complete()

        if end_call:
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
        barge-ins, where the first clause must start before the rest of the line exists.

        Returns the assembled final message, its stats, and whether any audio was sent.
        """
        message: Any = ""
        stats: dict[str, Any] = {}
        # Streamed even though only the final is used: barge-ins will consume the deltas.
        async for kind, payload in self._llm.complete_stream(messages=messages, tools=[END_CALL_TOOL]):
            if kind == "final":
                message, stats = payload
        utterance, _ = extract_turn(message)
        spoke = False
        if utterance:
            async for chunk in self._tts.stream(utterance, voice_id=self.caller_voice):
                self._audio_interface.output(chunk)
                spoke = True
        return message, stats, spoke

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
        return [{"role": "system", "content": self._build_prompt()}, *flipped]
