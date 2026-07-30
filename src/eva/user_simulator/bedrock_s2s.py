"""Amazon Bedrock (Nova Sonic) speech-to-speech implementation of the EVA caller.

This is an audio-native S2S caller, the AWS analog of ``OpenAIRealtimeUserSimulator``:
EVA streams the agent's audio straight to **Amazon Nova Sonic** over Bedrock's
bidirectional streaming API and streams the caller's spoken reply back — there is no
separate STT/LLM/TTS cascade. Nova Sonic also emits both transcripts itself
(agent ASR as ``role: USER``; the caller's own speech as ``role: ASSISTANT``).

Audio path (the bridge speaks μ-law 8 kHz; Nova speaks LPCM):
- **agent → Nova:** μ-law 8 kHz → ``audioop.ulaw2lin`` → PCM16 8 kHz → resample to
  16 kHz → base64 ``audioInput``. Fed only while the caller isn't speaking (half-duplex).
- **Nova → bridge:** ``audioOutput`` is LPCM at ``output_sample_rate`` (pinned to
  16 kHz = the bridge's ``output()`` rate, so no resample) → played over the bridge.

Turn-taking differs from the OpenAI caller: Nova Sonic has no "don't auto-respond"
switch, so we rely on its native endpointing (``endpointingSensitivity``) plus
half-duplex suppression (agent audio is dropped while the caller is speaking) rather
than manual transcript-gated response creation.

The experimental ``aws-sdk-bedrock-runtime`` client (Python >=3.12) is imported lazily
so this module and its unit tests load without the optional dependency installed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import audioop
except ImportError:  # Python 3.13 removed stdlib audioop
    import audioop_lts as audioop  # type: ignore[import-not-found, no-redef]

from eva.user_simulator.audio_bridge import BotToBotAudioBridge
from eva.user_simulator.base import AbstractUserSimulator
from eva.utils.audio_utils import save_pcm_as_wav
from eva.utils.logging import get_logger

if TYPE_CHECKING:
    from eva.models.config import BedrockS2SSimulatorConfig, PerturbationConfig

logger = get_logger(__name__)

MULAW_SAMPLE_RATE = 8000  # agent audio arrives from the bridge as μ-law 8 kHz
NOVA_INPUT_RATE = 16000  # we send LPCM 16 kHz to Nova Sonic
BRIDGE_OUTPUT_RATE = 16000  # BotToBotAudioBridge.output() expects PCM16 16 kHz
_PERSONA_GENDER = {1: "F", 2: "M"}
_FINAL_DRAIN_SECONDS = 4.0
# Nova Sonic drops the session if it receives no audio/interactive content for ~55s. Send a
# small silence frame if the input stream has been idle this long (e.g. agent gone quiet
# mid-call or at end-of-call) so a live-but-quiet conversation isn't killed.
_KEEPALIVE_CHECK_SECONDS = 5.0
_KEEPALIVE_IDLE_SECONDS = 15.0
_KEEPALIVE_SILENCE = base64.b64encode(b"\x00" * (NOVA_INPUT_RATE // 50 * 2)).decode("ascii")  # 20 ms PCM16 16 kHz

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


class BedrockS2SUserSimulator(AbstractUserSimulator):
    """Play EVA's simulated caller with Amazon Nova Sonic (Bedrock bidirectional streaming)."""

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
        simulator_config: BedrockS2SSimulatorConfig,
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
            provider="bedrock_s2s",
        )
        if perturbation_config and perturbation_config.accent is not None:
            raise ValueError("Bedrock S2S caller does not support ElevenLabs-specific accent variants")
        self.simulator_config = simulator_config

        # Nova Sonic event identifiers (a promptName ties the session together; each
        # content block gets its own contentName).
        self._prompt_name = uuid.uuid4().hex
        self._audio_content_name = uuid.uuid4().hex

        self._stream: Any = None
        self._is_active = False
        self._agent_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._input_resampler_state: Any = None
        # Loop time of the last audioInput frame sent to Nova (drives the keepalive).
        self._last_input_sent = 0.0

        # Caller-turn state: True from the moment Nova starts emitting the caller's audio
        # until that audio has drained, so agent audio is suppressed while the caller speaks.
        self._caller_response_active = False
        self._end_call_pending = False

        # Text-output tracking: role + generation stage come on the TEXT contentStart and
        # apply to the following textOutput (Nova emits agent ASR as USER, caller text as
        # ASSISTANT with generationStage SPECULATIVE preview then FINAL). The caller's text is
        # buffered and flushed once per turn.
        self._current_text_role: str | None = None
        self._current_text_stage: str | None = None
        self._pending_caller_text: str | None = None

    # ── config-derived helpers ───────────────────────────────────────────────
    @property
    def caller_model(self) -> str:
        return self.simulator_config.model_id

    @property
    def caller_voice(self) -> str:
        gender = _PERSONA_GENDER.get(self.persona_config.get("user_persona_id"))
        if gender == "M":
            return self.simulator_config.male_voice
        return self.simulator_config.female_voice

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def run_conversation(self) -> str:
        try:
            await self._run_nova_conversation()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Bedrock S2S caller simulation error: {exc}", exc_info=True)
            self._end_reason = "error"
            self.event_logger.log_error(str(exc))
            if self._audio_interface is not None:
                with suppress(Exception):
                    await self._audio_interface.stop_async()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})
        finally:
            self.event_logger.save()
        return self._end_reason

    async def _run_nova_conversation(self) -> None:
        self._audio_interface = BotToBotAudioBridge(
            websocket_uri=self.server_url,
            conversation_id=self.output_dir.name,
            record_callback=self._record_audio,
            event_logger=self.event_logger,
            conversation_done_callback=self._on_conversation_end,
            perturbator=self._perturbator,
            disconnect_reason="assistant_disconnect",
        )
        await self._audio_interface.start_async()
        self._audio_interface.start(self._on_agent_audio)
        self.event_logger.log_connection_state(
            "connected",
            {
                "server_url": self.server_url,
                "caller_provider": self.provider,
                "caller_model": self.caller_model,
                "caller_voice": self.caller_voice,
                "caller_input_transport": "audio/lpcm_16000hz",
                "caller_output_transport": f"audio/lpcm_{self.simulator_config.output_sample_rate}hz",
                "assistant_input_transport": "audio/pcmu_8000hz",
                "caller_response_sequencing": "native_endpointing_halfduplex",
                "caller_endpointing_sensitivity": self.simulator_config.endpointing_sensitivity,
            },
        )

        client = self._create_client()
        forward_task: asyncio.Task | None = None
        listener_task: asyncio.Task | None = None
        completion_task: asyncio.Task | None = None
        keepalive_task: asyncio.Task | None = None
        try:
            self._stream = await client.invoke_model_with_bidirectional_stream(
                self._stream_input(client, self.caller_model)
            )
            self._is_active = True
            self._last_input_sent = asyncio.get_running_loop().time()
            await self._start_session()
            self.event_logger.log_connection_state("session_started")

            forward_task = asyncio.create_task(self._forward_agent_audio())
            listener_task = asyncio.create_task(self._process_responses())
            completion_task = asyncio.create_task(self._wait_for_conversation_end())
            keepalive_task = asyncio.create_task(self._keepalive_audio())

            await self._wait_for_session_completion(completion_task, forward_task, listener_task)

            # Allow the final goodbye audio and transcript to flush before closing.
            await asyncio.sleep(_FINAL_DRAIN_SECONDS)
        finally:
            self._is_active = False
            with suppress(Exception):
                await self._end_session()
            for task in (completion_task, forward_task, listener_task, keepalive_task):
                if task is not None:
                    await self._cancel_background_task(task)
            await self._audio_interface.stop_async()
            self._save_user_audio()
            self.event_logger.log_connection_state("session_ended", {"reason": self._end_reason})

    @staticmethod
    async def _cancel_background_task(task: asyncio.Task) -> None:
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
        raise RuntimeError(f"Nova Sonic {task_name} stopped unexpectedly")

    # ── Bedrock client + event encoding (experimental SDK, imported lazily) ───
    def _create_client(self) -> Any:
        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
        from smithy_aws_core.identity import EnvironmentCredentialsResolver

        region = self.simulator_config.region
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
        )
        return BedrockRuntimeClient(config=config)

    @staticmethod
    def _stream_input(client: Any, model_id: str) -> Any:  # noqa: ARG004
        from aws_sdk_bedrock_runtime.client import InvokeModelWithBidirectionalStreamOperationInput

        return InvokeModelWithBidirectionalStreamOperationInput(model_id=model_id)

    def _encode_chunk(self, payload: bytes) -> Any:
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        return InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart(bytes_=payload))

    async def _send_event(self, event: dict[str, Any]) -> None:
        if self._stream is None:
            return
        payload = json.dumps(event).encode("utf-8")
        await self._stream.input_stream.send(self._encode_chunk(payload))

    # ── session setup / teardown (Nova Sonic event sequence) ──────────────────
    async def _start_session(self) -> None:
        cfg = self.simulator_config
        await self._send_event(
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": cfg.max_tokens,
                            "topP": cfg.top_p,
                            "temperature": cfg.temperature,
                        },
                        "turnDetectionConfiguration": {"endpointingSensitivity": cfg.endpointing_sensitivity},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "promptStart": {
                        "promptName": self._prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": cfg.output_sample_rate,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": self.caller_voice,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                        "toolUseOutputConfiguration": {"mediaType": "application/json"},
                        "toolConfiguration": {
                            "tools": [
                                {
                                    "toolSpec": {
                                        "name": "end_call",
                                        "description": END_CALL_DESCRIPTION,
                                        "inputSchema": {"json": json.dumps({"type": "object", "properties": {}})},
                                    }
                                }
                            ]
                        },
                    }
                }
            }
        )
        # System prompt (persona + goal template) as a one-shot SYSTEM text block.
        system_content_name = uuid.uuid4().hex
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": system_content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": system_content_name,
                        "content": self._build_prompt(),
                    }
                }
            }
        )
        await self._send_event(
            {"event": {"contentEnd": {"promptName": self._prompt_name, "contentName": system_content_name}}}
        )
        # Open the single, long-lived audio input block; all agent audio streams into it.
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_INPUT_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )

    async def _end_session(self) -> None:
        await self._send_event(
            {"event": {"contentEnd": {"promptName": self._prompt_name, "contentName": self._audio_content_name}}}
        )
        await self._send_event({"event": {"promptEnd": {"promptName": self._prompt_name}}})
        await self._send_event({"event": {"sessionEnd": {}}})
        if self._stream is not None:
            with suppress(Exception):
                await self._stream.input_stream.close()

    # ── agent audio in (bridge → Nova) ───────────────────────────────────────
    def _on_agent_audio(self, mulaw_audio: bytes) -> None:
        """Bridge callback: queue agent audio (μ-law 8 kHz) for Nova as LPCM 16 kHz.

        Half-duplex: drop agent audio while the caller's own audio is still playing, so the
        caller's line isn't fed back to Nova as the agent's turn. Gated solely on the bridge's
        ``is_caller_playing()`` (not the ``_caller_response_active`` flag) so a missed Nova
        turn-end event can't wedge suppression on and starve the input stream.
        """
        if not mulaw_audio or self._caller_audio_is_playing():
            return
        try:
            pcm8k = audioop.ulaw2lin(mulaw_audio, 2)
            pcm16k, self._input_resampler_state = audioop.ratecv(
                pcm8k, 2, 1, MULAW_SAMPLE_RATE, NOVA_INPUT_RATE, self._input_resampler_state
            )
        except Exception:  # noqa: BLE001
            return
        self._agent_audio_queue.put_nowait(pcm16k)

    def _caller_audio_is_playing(self) -> bool:
        if self._audio_interface is None:
            return False
        return self._audio_interface.is_caller_playing()

    async def _forward_agent_audio(self) -> None:
        while True:
            pcm16k = await self._agent_audio_queue.get()
            if not pcm16k:
                continue
            await self._send_audio_input(base64.b64encode(pcm16k).decode("ascii"))

    async def _send_audio_input(self, content_b64: str) -> None:
        await self._send_event(
            {
                "event": {
                    "audioInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "content": content_b64,
                    }
                }
            }
        )
        self._last_input_sent = asyncio.get_running_loop().time()

    async def _keepalive_audio(self) -> None:
        """Send a silence frame if the input stream has been idle, so Nova doesn't drop us.

        Nova Sonic ends the session on a >~55s gap with no audio/interactive content. When the
        agent is quiet (thinking, or the call has wound down) neither real agent audio nor the
        bridge's silence may be flowing, so top up the stream with a brief silence frame.
        """
        while self._is_active:
            await asyncio.sleep(_KEEPALIVE_CHECK_SECONDS)
            if self._conversation_done.is_set() or self._stream is None:
                continue
            idle = asyncio.get_running_loop().time() - self._last_input_sent
            if idle >= _KEEPALIVE_IDLE_SECONDS and not self._caller_audio_is_playing():
                with suppress(Exception):
                    await self._send_audio_input(_KEEPALIVE_SILENCE)

    # ── Nova responses out (Nova → bridge) ───────────────────────────────────
    async def _process_responses(self) -> None:
        try:
            while self._is_active:
                output = await self._stream.await_output()
                result = await output[1].receive()
                value = getattr(result, "value", None)
                raw = getattr(value, "bytes_", None) if value is not None else None
                if not raw:
                    continue
                await self._handle_event(json.loads(raw.decode("utf-8")))
        except asyncio.CancelledError:
            raise
        except StopAsyncIteration:
            return
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Nova Sonic event loop error: {exc}", exc_info=True)
            self.event_logger.log_error(str(exc))
            self._on_conversation_end("error")

    async def _handle_event(self, data: dict[str, Any]) -> None:
        event = data.get("event")
        if not event:
            return
        if "contentStart" in event:
            self._on_content_start(event["contentStart"])
        elif "textOutput" in event:
            self._on_text_output(event["textOutput"])
        elif "audioOutput" in event:
            self._on_audio_output(event["audioOutput"])
        elif "toolUse" in event:
            self._on_tool_use(event["toolUse"])
            # Nova may invoke end_call as a pure tool with no trailing audio END_TURN, so
            # hang up here (after any in-flight caller audio drains) rather than waiting for
            # a turn-end event that never comes.
            if self._end_call_pending:
                await self._finish_caller_response()
        elif "contentEnd" in event:
            await self._on_content_end(event["contentEnd"])
        elif "completionEnd" in event:
            await self._finish_caller_response()

    def _on_content_start(self, content_start: dict[str, Any]) -> None:
        content_type = content_start.get("type")
        role = content_start.get("role")
        if content_type == "TEXT":
            self._current_text_role = role
            stage = None
            fields = content_start.get("additionalModelFields")
            if fields:
                with suppress(Exception):
                    stage = json.loads(fields).get("generationStage")
            self._current_text_stage = stage
        elif content_type == "AUDIO" and role == "ASSISTANT":
            # The caller has started speaking — suppress agent audio until it drains.
            self._caller_response_active = True

    def _on_text_output(self, text_output: dict[str, Any]) -> None:
        text = (text_output.get("content") or "").strip()
        if not text:
            return
        role = text_output.get("role") or self._current_text_role
        if role == "USER":
            # Nova's ASR of the agent-under-test's speech.
            self._on_assistant_speaks(text)
        elif role == "ASSISTANT":
            # The caller's own words. Buffer and emit once at turn end, preferring the FINAL
            # post-audio transcript but falling back to the SPECULATIVE preview if that's all
            # Nova sends for a short turn (a FINAL always overwrites a buffered SPECULATIVE).
            if self._current_text_stage == "FINAL" or self._pending_caller_text is None:
                self._pending_caller_text = text

    def _on_audio_output(self, audio_output: dict[str, Any]) -> None:
        content = audio_output.get("content")
        if not content or self._audio_interface is None:
            return
        pcm = base64.b64decode(content)
        if pcm:
            self._audio_interface.output(pcm)
            self._caller_response_active = True

    def _on_tool_use(self, tool_use: dict[str, Any]) -> None:
        if tool_use.get("toolName") == "end_call":
            self.event_logger.log_event("tool_call", {"name": "end_call", "arguments": tool_use.get("content", "")})
            self._end_call_pending = True

    async def _on_content_end(self, content_end: dict[str, Any]) -> None:
        stop_reason = content_end.get("stopReason")
        if content_end.get("type") == "AUDIO" and stop_reason in {"END_TURN", "INTERRUPTED"}:
            await self._finish_caller_response()

    async def _finish_caller_response(self) -> None:
        """The caller's turn ended: emit its transcript, drain its audio, release / hang up."""
        if self._conversation_done.is_set():
            return  # idempotent: a trailing turn-end event after end_call must not re-run this
        if self._pending_caller_text:
            self._on_user_speaks(self._pending_caller_text)
            self._pending_caller_text = None
        # Tell the bridge no more caller audio is coming so it finalizes the turn even when the
        # audio ends on an exact chunk boundary; otherwise is_caller_playing() stays True, which
        # would wedge half-duplex suppression on and starve Nova's input stream (a hard freeze).
        if self._audio_interface is not None:
            self._audio_interface.notify_user_utterance_complete()
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self._wait_for_caller_playback_complete(),
                timeout=self.simulator_config.playback_drain_seconds,
            )
        self._caller_response_active = False
        if self._end_call_pending:
            self._end_call_pending = False
            self._on_conversation_end("goodbye")

    async def _wait_for_caller_playback_complete(self) -> None:
        if self._audio_interface is None:
            return
        while True:
            if not self._audio_interface.is_caller_playing():
                await asyncio.sleep(0.7)
                if not self._audio_interface.is_caller_playing():
                    return
            await asyncio.sleep(0.05)

    def _save_user_audio(self) -> None:
        if not self._user_clean_audio_chunks:
            return
        save_pcm_as_wav(
            b"".join(self._user_clean_audio_chunks),
            self.output_dir / "audio_user_clean.wav",
            sample_rate=BRIDGE_OUTPUT_RATE,
            num_channels=1,
        )
