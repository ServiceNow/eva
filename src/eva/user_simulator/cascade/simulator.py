"""Self-hosted STT/LLM/TTS caller simulator driven by the tick scheduler."""

from __future__ import annotations

import json
import re
from pathlib import Path

import websockets

from eva.assistant.services.llm import LiteLLMClient
from eva.models.config import CascadeSimulatorConfig, PerturbationConfig
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.cascade.adapter.realtime_ws import RealtimeWSAdapter
from eva.user_simulator.cascade.constants import CALLER_SAMPLE_RATE, TICK_DURATION_MS
from eva.user_simulator.cascade.scheduler import TickScheduler
from eva.user_simulator.cascade.stt import ScribeStreamingSTT
from eva.user_simulator.cascade.tts import CartesiaTTS
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

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

END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": END_CALL_DESCRIPTION,
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def parse_turn_response(raw: str) -> str:
    """Extract the utterance from the caller LLM's JSON reply.

    Falls back to treating the whole response as the utterance when it is not
    JSON, so a malformed reply degrades into a plain turn rather than silence.
    """
    stripped = _FENCE.sub("", raw).strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(payload, dict):
        return stripped
    utterance = payload.get("utterance", "")
    if not isinstance(utterance, str):
        raise ValueError(f"Caller LLM returned a non-string utterance: {utterance!r}")
    return utterance


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
        self._config = simulator_config
        self._stt = ScribeStreamingSTT(simulator_config.stt_params, language=language)
        self._tts = CartesiaTTS(simulator_config.tts_params, language=language)
        self._llm = LiteLLMClient(model=simulator_config.llm)
        self._voice_id = self._tts.voice_for_persona(persona_config)
        self._history: list[dict[str, str]] = []

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
                if result.has_assistant_speech:
                    await self._stt.feed(result.assistant_audio)
                    assistant_was_speaking = True
                    continue
                if assistant_was_speaking:
                    # Assistant just stopped: close the utterance so Scribe emits a
                    # committed_transcript. With commit_strategy=manual nothing is
                    # ever finalized unless we say so, and take_committed() below
                    # would return empty forever.
                    await self._stt.feed(result.assistant_audio, commit=True)
                    assistant_was_speaking = False
                    continue
                if scheduler.caller_is_speaking or not scheduler.may_take_turn():
                    continue
                if await self._take_turn(scheduler):
                    break
            else:
                if not self._conversation_done.is_set():
                    self._on_conversation_end("timeout")
        finally:
            await self._stt.stop()
            await adapter.stop()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

    async def _take_turn(self, scheduler: TickScheduler) -> bool:
        """Generate, synthesize, and queue one caller turn. Returns True to hang up."""
        heard = self._stt.buffer.take_committed()
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
        """Build the caller LLM message list: persona/goal prompt, JSON contract, history."""
        system = self._build_prompt() + "\n\n" + PromptManager().get_prompt("user_simulator.cascade_turn_contract")
        return [{"role": "system", "content": system}, *self._history]
