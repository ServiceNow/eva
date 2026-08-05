"""``UserRole``: the simulated-caller role (one generic class).

A single concrete, provider-agnostic role that holds a ``Backend`` and works
with any backend. Provider specifics (session, audio format, event parsing)
live in the backend; this class owns only the role-common concerns shared by
every user simulator regardless of provider:

- the user-simulator system prompt (persona + goal / decision tree);
- the single caller-side ``end_call`` tool;
- the event logger (``user_simulator_events.jsonl``) and clean-user-audio WAV;
- the counterparty transport: the audio-bridge **client** that dials the
  assistant, plus a generic "respond after the assistant's turn settles"
  sequencing policy that asks the backend to speak via ``trigger_response()``
  (a no-op for self-driving backends).

It consumes only the normalized ``BackendEvent`` stream, never a raw provider
event. Interpreting the *inbound* transcript as the assistant and the *output*
transcript as the caller is the one role-specific reading of the role-agnostic
events.

Plug-in point: ``run()`` mirrors ``AbstractUserSimulator.run_conversation()``
(drives to completion, returns the end-reason) and exposes the same
``on_conversation_ending`` hook, so the worker swap is 1:1. Wired behind the
``USE_ROLE_BACKEND_OPENAI_REALTIME`` gate in the worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from pipecat.transcriptions.language import Language
from websockets.exceptions import ConnectionClosedOK

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[import-not-found,no-redef]

from eva.backend.base import Backend, BackendEvent, BackendEventType, ToolCallRequest, ToolCallResult
from eva.models.config import LANGUAGE_DISPLAY_NAMES, PerturbationConfig
from eva.role.base import Role
from eva.user_simulator.audio_bridge import BotToBotAudioBridge
from eva.user_simulator.base import load_behavior_prompts
from eva.user_simulator.event_logger import UserSimulatorEventLogger
from eva.user_simulator.perturbation import AudioPerturbator
from eva.utils.audio_utils import save_audio_track
from eva.utils.culture import add_user_language_directive
from eva.utils.logging import current_record_id, get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

BRIDGE_SAMPLE_RATE = 16000
CALLER_RESPONSE_SETTLE_SECONDS = 2.0
CALLER_RESPONSE_POLL_SECONDS = 0.05
CALLER_PLAYBACK_DRAIN_SECONDS = 15.0
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


class UserRole(Role):
    """Generic simulated-caller role. Drives any ``Backend``; owns the bridge-client transport."""

    # Generic phone-caller session conventions (provider-agnostic): telephony
    # input, manual turn-taking (respond after the assistant settles), caller VAD
    # tuning, and no parallel tool calls. Unpacked by the worker into the caller
    # backend's args, so the caller session shape lives with the caller role.
    CALLER_BACKEND_DEFAULTS: dict[str, Any] = {
        "input_format": "pcmu",
        "manual_turn_taking": True,
        "vad_settings": {"threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500},
        "parallel_tool_calls": False,
    }

    def __init__(
        self,
        *,
        backend: Backend,
        current_date_time: str,
        persona_config: dict[str, Any],
        goal: dict[str, Any],
        server_url: str,
        output_dir: Path,
        agent_id: str,
        provider: str = "unknown",
        timeout: int = 600,
        perturbation_config: PerturbationConfig | None = None,
        language: str = "en",
    ) -> None:
        super().__init__(backend=backend)
        self.persona_config = persona_config
        self.goal = goal
        self.current_date_time = current_date_time
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.agent_id = agent_id
        self.provider = provider
        self.timeout = timeout
        self._language = language
        self._perturbation_config = perturbation_config
        self._perturbator = (
            AudioPerturbator(perturbation_config)
            if perturbation_config is not None
            and (perturbation_config.background_noise is not None or perturbation_config.connection_degradation)
            else None
        )

        self._audio_interface: BotToBotAudioBridge | None = None
        self._end_reason = "unknown"
        self._conversation_done = asyncio.Event()
        self._ending_signaled = False
        # Set by the worker to the assistant role's notify_conversation_ending.
        self.on_conversation_ending: Callable[[str | None], None] | None = None
        self.event_logger = UserSimulatorEventLogger(self.output_dir / "user_simulator_events.jsonl", provider=provider)
        self._user_clean_audio = bytearray()
        self._record_id = current_record_id.get()

        # Caller response-sequencing state.
        self._assistant_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._caller_audio_seen = False
        self._caller_playback_pending = False
        self._caller_response_active = False
        self._caller_response_pending = False
        self._caller_response_task: asyncio.Task[None] | None = None
        self._assistant_transcript_ready = False
        self._end_call_pending = False
        self._resampler_state: Any = None

    # ── Role seams (prompt / tools / recording) ───────────────────────

    def build_prompt(self) -> str:
        """Build the user-simulator system prompt from persona + goal."""
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

    def _end_call_tool_spec(self) -> dict[str, Any]:
        """Provider-agnostic ``end_call`` tool spec (backend formats to its schema)."""
        return {
            "name": "end_call",
            "description": END_CALL_DESCRIPTION,
            "parameters": {"type": "object", "properties": {}},
        }

    async def handle_tool_call_request(self, request: ToolCallRequest) -> ToolCallResult:
        """Handle a caller-side tool call. Only ``end_call`` exists; it arms hang-up.

        The caller never returns a function_call_output to the provider --
        ``end_call`` is terminal intent, consumed locally (the ``run()`` loop
        does not relay this result back to the backend).
        """
        if request.name == "end_call":
            self.event_logger.log_event("tool_call", {"name": "end_call", "arguments": request.arguments})
            self._end_call_pending = True
        return ToolCallResult(call_id=request.call_id, result={})

    def record_audio(self, source: str, audio_data: bytes) -> None:
        """Retain only the clean (unperturbed) user track; other sources are the assistant's."""
        if source == "user_clean":
            self._user_clean_audio.extend(audio_data)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def run(self) -> str:
        try:
            await self._run_conversation()
        except Exception as exc:
            logger.error(f"User caller simulation error: {exc}", exc_info=True)
            self._end_reason = "error"
            self.event_logger.log_error(str(exc))
            if self._audio_interface is not None:
                with suppress(Exception):
                    await self._audio_interface.stop_async()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})
        finally:
            self.event_logger.save()
        return self._end_reason

    def get_end_reason(self) -> str:
        return self._end_reason

    def _connection_info(self) -> dict[str, Any]:
        """Connection metadata for the ``connected`` event, built from what the role knows.

        Generic caller session facts (transport, sequencing, sample rates) --
        the OpenAI-specific labels the old simulator logged are intentionally
        not reproduced here (a sub-macro logging diff).
        """
        d = self.CALLER_BACKEND_DEFAULTS
        return {
            "server_url": self.server_url,
            "caller_provider": self.provider,
            "caller_input_format": d["input_format"],
            "caller_turn_detection": {**d["vad_settings"], "manual_turn_taking": d["manual_turn_taking"]},
            "caller_input_sample_rate": self.backend.input_sample_rate,
            "caller_output_sample_rate": self.backend.output_sample_rate,
        }

    async def _run_conversation(self) -> None:
        self._audio_interface = BotToBotAudioBridge(
            websocket_uri=self.server_url,
            conversation_id=self.output_dir.name,
            record_callback=self.record_audio,
            event_logger=self.event_logger,
            conversation_done_callback=self._on_conversation_end,
            perturbator=self._perturbator,
            disconnect_reason="assistant_disconnect",
        )
        await self._audio_interface.start_async()
        self._audio_interface.start(self._on_assistant_audio)
        self.event_logger.log_connection_state("connected", self._connection_info())

        forward_task: asyncio.Task[Any] | None = None
        listener_task: asyncio.Task[Any] | None = None
        completion_task: asyncio.Task[Any] | None = None
        session = await self.backend.open(system_prompt=self.build_prompt(), tools=[self._end_call_tool_spec()])
        self.event_logger.log_connection_state("session_started")
        try:
            forward_task = asyncio.create_task(self._forward_assistant_audio(session))
            listener_task = asyncio.create_task(self._listen_for_caller_events(session))
            completion_task = asyncio.create_task(self._wait_for_conversation_end())

            await self._wait_for_session_completion(completion_task, forward_task, listener_task)
            # Allow final goodbye audio + transcripts to flush before closing.
            await asyncio.sleep(4.0)
        finally:
            if self._caller_response_task is not None:
                await self._cancel_background_task(self._caller_response_task)
            for task in (completion_task, forward_task, listener_task):
                if task is not None:
                    await self._cancel_background_task(task)
            await self.backend.close(session)
            await self._audio_interface.stop_async()
            self._save_clean_user_audio(BRIDGE_SAMPLE_RATE)
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

    @staticmethod
    async def _cancel_background_task(task: asyncio.Task[Any]) -> None:
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    async def _wait_for_conversation_end(self) -> None:
        try:
            await asyncio.wait_for(self._conversation_done.wait(), timeout=self.timeout)
        except TimeoutError:
            self.event_logger.log_event("timeout", {"duration": self.timeout})
            self._on_conversation_end("timeout")

    async def _wait_for_session_completion(
        self,
        completion_task: asyncio.Task[Any],
        forward_task: asyncio.Task[Any],
        listener_task: asyncio.Task[Any],
    ) -> None:
        done, _ = await asyncio.wait(
            {completion_task, forward_task, listener_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if completion_task in done:
            return
        finished_task = next(iter(done))
        if self._conversation_done.is_set():
            await completion_task
            return
        exception = finished_task.exception()
        if exception is not None:
            raise exception
        task_name = "listener" if finished_task is listener_task else "audio forwarder"
        raise RuntimeError(f"Caller {task_name} stopped unexpectedly")

    # ── Messaging hooks ────────────────────────────────────────────────

    def signal_conversation_ending(self, reason: str | None = None) -> None:
        """Advise the assistant the call is over, before transport teardown. Idempotent, never raises."""
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
        self.event_logger.log_event("user_speech", {"text": response, "source": "simulated_user"})

    def _on_assistant_speaks(self, transcript: str) -> None:
        current_record_id.set(self._record_id)
        self.event_logger.log_event("assistant_speech", {"text": transcript, "source": "assistant"})

    def _save_clean_user_audio(self, sample_rate: int) -> None:
        if save_audio_track(bytes(self._user_clean_audio), self.output_dir / "audio_user_clean.wav", sample_rate):
            logger.info(f"Saved clean user audio to {self.output_dir / 'audio_user_clean.wav'}")

    # ── Assistant audio -> caller backend ─────────────────────────────

    def _on_assistant_audio(self, mulaw_audio: bytes) -> None:
        if mulaw_audio and not self._caller_response_active and not self._caller_audio_is_playing():
            self._assistant_audio_queue.put_nowait(mulaw_audio)

    def _caller_audio_is_playing(self) -> bool:
        return self._audio_interface is not None and self._audio_interface.is_caller_playing()

    async def _forward_assistant_audio(self, session: Any) -> None:
        while True:
            mulaw_audio = await self._assistant_audio_queue.get()
            if not mulaw_audio:
                continue
            try:
                await self.backend.send(session, audio=mulaw_audio)
            except ConnectionClosedOK:
                return

    # ── Caller event processing (normalized events only) ──────────────

    async def _listen_for_caller_events(self, session: Any) -> None:
        try:
            async for event in self.backend.receive(session):
                await self._handle_caller_event(event, session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Caller event loop error: {exc}", exc_info=True)
            self.event_logger.log_error(str(exc))
            self._on_conversation_end("error")

    async def _handle_caller_event(self, event: BackendEvent, session: Any) -> None:
        meta = event.metadata
        match event.event_type:
            case BackendEventType.INPUT_SPEECH_STOPPED:
                self._schedule_caller_response(session, trigger="vad_speech_stopped")

            case BackendEventType.TRANSCRIPT:
                if meta.get("stream") == "input":
                    # From the caller's POV the inbound party is the assistant.
                    if event.transcript:
                        self._assistant_transcript_ready = True
                        self._on_assistant_speaks(event.transcript)
                elif meta.get("stream") == "output" and event.transcript:
                    self._on_user_speaks(event.transcript)

            case BackendEventType.AUDIO_OUTPUT:
                if event.audio and self._audio_interface is not None:
                    pcm16_16k, self._resampler_state = audioop.ratecv(
                        event.audio, 2, 1, self.backend.output_sample_rate, BRIDGE_SAMPLE_RATE, self._resampler_state
                    )
                    self._audio_interface.output(pcm16_16k)
                    self._caller_audio_seen = True
                    self._caller_playback_pending = True

            case BackendEventType.OUTPUT_TURN_STARTED:
                self._caller_response_active = True

            case BackendEventType.OUTPUT_AUDIO_DONE:
                self._flush_caller_output()
                self._resampler_state = None

            case BackendEventType.TOOL_CALL_REQUEST:
                if event.tool_call_request is not None:
                    await self.handle_tool_call_request(event.tool_call_request)

            case BackendEventType.TURN_END:
                self._flush_caller_output()
                await self._finish_caller_response(session)

            case BackendEventType.ERROR:
                if meta.get("code") == "conversation_already_has_active_response":
                    self._caller_response_active = True
                    self._caller_response_pending = True
                    self.event_logger.log_event(
                        "caller_response_coalesced", {"trigger": "active_response_error", "error": event.error}
                    )
                    return
                self.event_logger.log_error(str(event.error))
                self._on_conversation_end("error")

    def _schedule_caller_response(self, session: Any, *, trigger: str, require_settled_turn: bool = True) -> None:
        if self._conversation_done.is_set():
            return
        if self._caller_response_task is not None and not self._caller_response_task.done():
            self.event_logger.log_event("caller_response_coalesced", {"trigger": trigger})
            return
        self._caller_response_task = asyncio.create_task(
            self._create_caller_response_when_ready(session, trigger, require_settled_turn=require_settled_turn)
        )

    async def _create_caller_response_when_ready(
        self, session: Any, trigger: str, *, require_settled_turn: bool = True
    ) -> None:
        try:
            while not self._conversation_done.is_set():
                turn_ready = not require_settled_turn or self._assistant_turn_is_settled()
                if not self._caller_response_active and turn_ready:
                    self._caller_response_active = True
                    self._assistant_transcript_ready = False
                    self.event_logger.log_event("caller_response_created", {"trigger": trigger})
                    try:
                        await self.backend.trigger_response(session)
                    except Exception as exc:
                        self._caller_response_active = False
                        self.event_logger.log_error(
                            "Failed to request caller response", {"trigger": trigger, "error": str(exc)}
                        )
                        self._on_conversation_end("error")
                    return
                await asyncio.sleep(CALLER_RESPONSE_POLL_SECONDS)
        finally:
            self._caller_response_task = None

    def _assistant_turn_is_settled(self) -> bool:
        if not self._assistant_transcript_ready:
            return False
        if self._audio_interface is None or self._audio_interface.is_assistant_playing():
            return False
        ended_time = self._audio_interface.assistant_audio_ended_at
        if ended_time is None:
            return False
        return asyncio.get_running_loop().time() - ended_time >= CALLER_RESPONSE_SETTLE_SECONDS

    async def _wait_for_caller_playback_complete(self) -> None:
        if self._audio_interface is None or not self._caller_playback_pending:
            return
        while True:
            if not self._audio_interface.is_caller_playing():
                await asyncio.sleep(0.7)
                if not self._audio_interface.is_caller_playing():
                    self._caller_playback_pending = False
                    return
            await asyncio.sleep(0.05)

    async def _finish_caller_response(self, session: Any) -> None:
        self._caller_response_active = False
        with suppress(TimeoutError):
            await asyncio.wait_for(self._wait_for_caller_playback_complete(), timeout=CALLER_PLAYBACK_DRAIN_SECONDS)
        if self._end_call_pending:
            self._end_call_pending = False
            self._on_conversation_end("goodbye")
        elif self._caller_response_pending:
            self._caller_response_pending = False
            self._schedule_caller_response(session, trigger="pending_after_response_done", require_settled_turn=False)

    def _flush_caller_output(self) -> None:
        if self._caller_audio_seen and self._audio_interface is not None:
            self._audio_interface.output(b"\x00\x00")
            self._caller_audio_seen = False
