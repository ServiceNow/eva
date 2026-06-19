"""OpenAI Realtime implementation of the EVA simulated caller."""

from __future__ import annotations

import asyncio
import base64
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosedOK

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.models.config import OpenAIRealtimeSimulatorConfig, PerturbationConfig
from eva.user_simulator.audio_bridge import BotToBotAudioBridge
from eva.user_simulator.base import AbstractUserSimulator
from eva.utils.audio_utils import save_pcm_as_wav
from eva.utils.logging import get_logger

logger = get_logger(__name__)

OPENAI_SAMPLE_RATE = 24000
BRIDGE_SAMPLE_RATE = 16000
_PERSONA_GENDER = {1: "F", 2: "M"}
CALLER_INPUT_FORMAT = {"type": "audio/pcmu"}
CALLER_OUTPUT_FORMAT = {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE}
CALLER_TURN_DETECTION = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "create_response": False,
    "interrupt_response": False,
    "idle_timeout_ms": 15_000,
}
CALLER_TRANSCRIPTION_MODEL = "whisper-1"
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


class OpenAIRealtimeUserSimulator(AbstractUserSimulator):
    """Use a second OpenAI Realtime session as EVA's simulated caller."""

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
        simulator_config: OpenAIRealtimeSimulatorConfig,
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
            provider="openai_realtime",
        )
        if perturbation_config and perturbation_config.accent is not None:
            raise ValueError("OpenAI Realtime caller does not support ElevenLabs-specific accent variants")
        self.simulator_config = simulator_config
        self._assistant_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._caller_transcript_parts: list[str] = []
        self._caller_audio_seen = False
        self._caller_playback_pending = False
        self._caller_response_active = False
        self._caller_response_pending = False
        self._caller_response_task: asyncio.Task | None = None
        self._assistant_transcript_ready = False
        self._end_call_pending = False
        self._resampler_state = None

    @property
    def caller_model(self) -> str:
        return self.simulator_config.model

    @property
    def caller_voice(self) -> str:
        gender = _PERSONA_GENDER.get(self.persona_config.get("user_persona_id"))
        if gender == "M":
            return self.simulator_config.male_voice
        return self.simulator_config.female_voice

    def _build_session_config(self) -> dict[str, Any]:
        return {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self._build_prompt(),
            "audio": {
                "output": {
                    "voice": self.caller_voice,
                    "format": CALLER_OUTPUT_FORMAT.copy(),
                },
                "input": {
                    "format": CALLER_INPUT_FORMAT.copy(),
                    "turn_detection": CALLER_TURN_DETECTION.copy(),
                    "transcription": {"model": CALLER_TRANSCRIPTION_MODEL, "language": self._language},
                },
            },
            "parallel_tool_calls": False,
            "tools": [
                {
                    "type": "function",
                    "name": "end_call",
                    "description": END_CALL_DESCRIPTION,
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

    def _create_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key)

    async def run_conversation(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            await self._run_openai_conversation(api_key)
        except Exception as exc:
            logger.error(f"OpenAI caller simulation error: {exc}", exc_info=True)
            self._end_reason = "error"
            self.event_logger.log_error(str(exc))
            if self._audio_interface is not None:
                with suppress(Exception):
                    await self._audio_interface.stop_async()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})
        finally:
            self.event_logger.save()
        return self._end_reason

    async def _run_openai_conversation(self, api_key: str) -> None:
        conversation_id = self.output_dir.name
        self._audio_interface = BotToBotAudioBridge(
            websocket_uri=self.server_url,
            conversation_id=conversation_id,
            record_callback=self._record_audio,
            event_logger=self.event_logger,
            conversation_done_callback=self._on_conversation_end,
            perturbator=self._perturbator,
            disconnect_reason="assistant_disconnect",
        )
        await self._audio_interface.start_async()
        self._audio_interface.start(self._on_assistant_audio)
        self.event_logger.log_connection_state(
            "connected",
            {
                "server_url": self.server_url,
                "caller_provider": self.provider,
                "caller_model": self.caller_model,
                "caller_voice": self.caller_voice,
                "caller_input_format": CALLER_INPUT_FORMAT,
                "caller_output_format": CALLER_OUTPUT_FORMAT,
                "assistant_input_transport": "audio/pcmu_8000hz",
                "caller_turn_detection": CALLER_TURN_DETECTION,
                "caller_response_sequencing": "manual_transcript_gated_noninterruptible",
                "caller_transcription_model": CALLER_TRANSCRIPTION_MODEL,
                "caller_end_call_profile": "paper_v2",
            },
        )

        client = self._create_client(api_key)
        forward_task: asyncio.Task | None = None
        listener_task: asyncio.Task | None = None
        completion_task: asyncio.Task | None = None
        try:
            async with client.realtime.connect(model=self.caller_model) as conn:
                await conn.session.update(session=self._build_session_config())
                self.event_logger.log_connection_state("session_started")
                forward_task = asyncio.create_task(self._forward_assistant_audio(conn))
                listener_task = asyncio.create_task(self._listen_for_caller_events(conn))
                completion_task = asyncio.create_task(self._wait_for_conversation_end())

                await self._wait_for_session_completion(completion_task, forward_task, listener_task)

                # Allow final goodbye audio and transcripts to flush before closing.
                await asyncio.sleep(4.0)
        finally:
            if self._caller_response_task is not None:
                await self._cancel_background_task(self._caller_response_task)
            for task in (completion_task, forward_task, listener_task):
                if task is not None:
                    await self._cancel_background_task(task)
            await client.close()
            await self._audio_interface.stop_async()
            self._save_user_audio()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

    @staticmethod
    async def _cancel_background_task(task: asyncio.Task) -> None:
        """Cancel and consume a background task without interrupting cleanup."""
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
        completion_task: asyncio.Task,
        forward_task: asyncio.Task,
        listener_task: asyncio.Task,
    ) -> None:
        done, _ = await asyncio.wait(
            {completion_task, forward_task, listener_task},
            return_when=asyncio.FIRST_COMPLETED,
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
        raise RuntimeError(f"OpenAI Realtime {task_name} stopped unexpectedly")

    def _on_assistant_audio(self, mulaw_audio: bytes) -> None:
        if mulaw_audio and not self._caller_response_active and not self._caller_audio_is_playing():
            self._assistant_audio_queue.put_nowait(mulaw_audio)

    def _caller_audio_is_playing(self) -> bool:
        if self._audio_interface is None:
            return False
        return self._audio_interface.is_caller_playing()

    async def _forward_assistant_audio(self, conn: Any) -> None:
        while True:
            mulaw_audio = await self._assistant_audio_queue.get()
            if not mulaw_audio:
                continue
            try:
                await conn.input_audio_buffer.append(audio=base64.b64encode(mulaw_audio).decode("ascii"))
            except ConnectionClosedOK:
                return

    async def _listen_for_caller_events(self, conn: Any) -> None:
        try:
            async for event in conn:
                event_type = getattr(event, "type", "")
                if event_type == "input_audio_buffer.speech_stopped":
                    self._schedule_caller_response(conn, trigger="vad_speech_stopped")
                elif event_type == "conversation.item.input_audio_transcription.completed" and getattr(
                    event, "transcript", ""
                ):
                    self._assistant_transcript_ready = True
                elif event_type == "response.created":
                    self._caller_response_active = True
                await self._handle_caller_event(event)
                if event_type == "response.done":
                    await self._finish_caller_response(conn)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"OpenAI caller event loop error: {exc}", exc_info=True)
            self.event_logger.log_error(str(exc))
            self._on_conversation_end("error")

    def _schedule_caller_response(self, conn: Any, *, trigger: str, require_settled_turn: bool = True) -> None:
        if self._conversation_done.is_set():
            return
        if self._caller_response_task is not None and not self._caller_response_task.done():
            self.event_logger.log_event("caller_response_coalesced", {"trigger": trigger})
            return
        self._caller_response_task = asyncio.create_task(
            self._create_caller_response_when_ready(conn, trigger, require_settled_turn=require_settled_turn)
        )

    async def _create_caller_response_when_ready(
        self,
        conn: Any,
        trigger: str,
        *,
        require_settled_turn: bool = True,
    ) -> None:
        try:
            while not self._conversation_done.is_set():
                turn_ready = not require_settled_turn or self._assistant_turn_is_settled()
                if not self._caller_response_active and turn_ready:
                    self._caller_response_active = True
                    self._assistant_transcript_ready = False
                    self.event_logger.log_event("caller_response_created", {"trigger": trigger})
                    try:
                        await conn.response.create()
                    except Exception as exc:
                        self._caller_response_active = False
                        self.event_logger.log_error(
                            "Failed to request OpenAI caller response",
                            {"trigger": trigger, "error": str(exc)},
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

    async def _finish_caller_response(self, conn: Any) -> None:
        self._caller_response_active = False
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self._wait_for_caller_playback_complete(),
                timeout=CALLER_PLAYBACK_DRAIN_SECONDS,
            )
        if self._end_call_pending:
            self._end_call_pending = False
            self._on_conversation_end("goodbye")
        elif self._caller_response_pending:
            self._caller_response_pending = False
            self._schedule_caller_response(
                conn,
                trigger="pending_after_response_done",
                require_settled_turn=False,
            )

    async def _handle_caller_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = getattr(event, "transcript", "")
            if transcript:
                self._on_assistant_speaks(transcript)
        elif event_type == "response.output_audio.delta":
            delta = getattr(event, "delta", "")
            if delta and self._audio_interface is not None:
                pcm16_24k = base64.b64decode(delta)
                pcm16_16k, self._resampler_state = audioop.ratecv(
                    pcm16_24k,
                    2,
                    1,
                    OPENAI_SAMPLE_RATE,
                    BRIDGE_SAMPLE_RATE,
                    self._resampler_state,
                )
                self._audio_interface.output(pcm16_16k)
                self._caller_audio_seen = True
                self._caller_playback_pending = True
        elif event_type == "response.output_audio_transcript.delta":
            self._caller_transcript_parts.append(getattr(event, "delta", ""))
        elif event_type == "response.output_audio_transcript.done":
            transcript = getattr(event, "transcript", "") or "".join(self._caller_transcript_parts)
            self._caller_transcript_parts.clear()
            if transcript:
                self._on_user_speaks(transcript)
        elif event_type == "response.function_call_arguments.done" and getattr(event, "name", "") == "end_call":
            self.event_logger.log_event("tool_call", {"name": "end_call", "arguments": getattr(event, "arguments", "")})
            self._end_call_pending = True
        elif event_type in {"response.output_audio.done", "response.done"}:
            self._flush_caller_output()
            if event_type == "response.output_audio.done":
                self._resampler_state = None
        elif event_type == "error":
            error = getattr(event, "error", "")
            if getattr(error, "code", None) == "conversation_already_has_active_response":
                self._caller_response_active = True
                self._caller_response_pending = True
                self.event_logger.log_event(
                    "caller_response_coalesced",
                    {"trigger": "active_response_error", "error": str(error)},
                )
                return
            self.event_logger.log_error(str(error))
            self._on_conversation_end("error")

    def _flush_caller_output(self) -> None:
        if self._caller_audio_seen and self._audio_interface is not None:
            self._audio_interface.output(b"\x00\x00")
            self._caller_audio_seen = False

    def _save_user_audio(self) -> None:
        if not self._user_clean_audio_chunks:
            return
        save_pcm_as_wav(
            b"".join(self._user_clean_audio_chunks),
            self.output_dir / "audio_user_clean.wav",
            sample_rate=BRIDGE_SAMPLE_RATE,
            num_channels=1,
        )
