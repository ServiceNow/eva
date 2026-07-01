"""Smallest Hydra speech-to-speech AssistantServer for EVA-Bench.

Bridges between the Twilio-framed WebSocket (user simulator) and Smallest's
Hydra S2S model over a raw WebSocket. Audio flows:

    User simulator (8 kHz mulaw)
        -> 16 kHz PCM16 -> Hydra input (input_audio_buffer.append)
    Hydra output (48 kHz PCM16, response.output_audio.delta)
        -> 8 kHz mulaw -> User simulator

Hydra's wire protocol is OpenAI-Realtime-shaped: a ``session.configure``
handshake, ``input_audio_buffer.append`` for audio in, ``response.output_audio
.delta`` for audio out, and ``response.function_call_arguments.done`` /
``conversation.item.create`` (function_call_output) for client-side tools.

Transcripts: Hydra streams a native **assistant** transcript
(``response.output_audio_transcript.delta``), which we use directly — it is more
accurate than re-transcribing the audio. It emits no **user** transcript, so each
completed user utterance is batch-transcribed with a configurable STT provider
(see ``s2s_transcription.py``). The audit log is timestamp-sorted, so the
asynchronous user transcriptions land in the right order regardless of timing.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_16k,
    mulaw_8k_to_pcm16_48k,
    parse_twilio_media_message,
    pcm16_48k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.s2s_transcription import BatchTranscriber, create_transcriber
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# Hydra streams 48 kHz PCM16 output; record at that native rate for best
# fidelity and upsample the 8 kHz user track to match.
_RECORDING_SAMPLE_RATE = 48000
# Rate Hydra expects for input audio.
_HYDRA_INPUT_RATE = 16000
# Skip batch-transcribing user utterances shorter than this — they are usually
# near-silence that STT models hallucinate full sentences on (16-bit mono => 2
# bytes/sample, so 300 ms @ 16 kHz = 9600 bytes).
_MIN_USER_UTTERANCE_BYTES = int(_HYDRA_INPUT_RATE * 0.3) * 2

_HYDRA_WS_URL = "wss://api.smallest.ai/waves/v1/s2s"
_VALID_VOICES = {"wren", "sloane", "marlowe", "reed", "knox", "tate"}

# Audio output pacing: 160-byte mulaw chunks (20 ms at 8 kHz) at real-time rate
# so the user simulator's silence detection works correctly.
MULAW_CHUNK_SIZE = 160
MULAW_CHUNK_DURATION_S = 0.02


# ---------------------------------------------------------------------------
# Tool schema helper
# ---------------------------------------------------------------------------


def _agent_tools_to_hydra(agent: AgentConfig) -> list[dict[str, Any]] | None:
    """Convert EVA AgentConfig tools to Hydra ``session.tools`` (client-side).

    Hydra uses OpenAI-style function declarations. Executing them client-side
    (no endpoint) makes Hydra emit ``response.function_call_arguments.done``.
    """
    if not agent.tools:
        return None

    functions: list[dict[str, Any]] = []
    for tool in agent.tools:
        functions.append(
            {
                "type": "function",
                "name": tool.function_name,
                "description": f"{tool.name}: {tool.description}",
                "parameters": {
                    "type": "object",
                    "properties": tool.get_parameter_properties(),
                    "required": tool.get_required_param_names(),
                },
            }
        )
    return functions or None


# ---------------------------------------------------------------------------
# Smallest Hydra AssistantServer
# ---------------------------------------------------------------------------


class SmallestHydraAssistantServer(AbstractAssistantServer):
    """Bridges the Twilio WebSocket <-> Smallest Hydra S2S API."""

    def __init__(
        self,
        current_date_time: str,
        pipeline_config: ModelConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
        language: str = "en",
    ):
        super().__init__(
            current_date_time=current_date_time,
            pipeline_config=pipeline_config,
            agent=agent,
            agent_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            output_dir=output_dir,
            port=port,
            conversation_id=conversation_id,
            language=language,
        )

        self._audio_sample_rate = _RECORDING_SAMPLE_RATE

        s2s_params = self.pipeline_config.s2s_params or {}
        self._model = s2s_params.get("model", "hydra")
        self._api_key = s2s_params.get("api_key", "")
        if not self._api_key:
            raise ValueError("Missing Smallest API key in s2s_params['api_key']")

        self._voice = s2s_params.get("voice", "wren")
        if self._voice not in _VALID_VOICES:
            logger.warning(f"Unknown Hydra voice {self._voice!r}; valid voices: {sorted(_VALID_VOICES)}")
        self._generate_initial_response = bool(s2s_params.get("generate_initial_response", True))

        # Build system prompt (same pattern as the other realtime/S2S servers).
        prompt_manager = PromptManager()
        self._system_prompt = prompt_manager.get_prompt(
            "realtime_agent.system_prompt",
            agent_personality=agent.description,
            agent_instructions=agent.instructions,
            datetime=self.current_date_time,
        )
        self._hydra_tools = _agent_tools_to_hydra(agent)

        # Hydra generates its own opening line; steer it to EVA's canned greeting
        # so the conversation opens consistently with the other frameworks.
        greeting_steer = (
            f"\n\nBegin the conversation by greeting the user with: {self.initial_message}"
            if self._generate_initial_response
            else ""
        )

        # Hydra silently drops audio output when the session.configure payload
        # exceeds ~18 KB, and also when total context (instructions + tools +
        # conversation history) grows too large on subsequent turns.  Cap the
        # instructions so that instructions + tools JSON fits comfortably in a
        # single session.configure message and leaves headroom for conversation.
        _MAX_PAYLOAD_BYTES = 18_000
        _tools_bytes = len(json.dumps(self._hydra_tools)) if self._hydra_tools else 0
        # Reserve room for the greeting steer so it is never truncated away.
        max_prompt_chars = max(1000, _MAX_PAYLOAD_BYTES - _tools_bytes - len(greeting_steer) - 500)
        if len(self._system_prompt) > max_prompt_chars:
            logger.warning(
                f"Hydra: truncating system prompt from {len(self._system_prompt)} to "
                f"{max_prompt_chars} chars ({len(self._hydra_tools or [])} tools take "
                f"{_tools_bytes} bytes) to stay within Hydra's context budget"
            )
            self._system_prompt = self._system_prompt[:max_prompt_chars]
        # Append the greeting AFTER truncation so it always survives.
        self._system_prompt += greeting_steer

        # Per-turn batch transcription (Hydra emits no transcript on the wire).
        self._transcriber: BatchTranscriber = create_transcriber(s2s_params, self.language)
        self._pending_transcriptions: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the FastAPI WebSocket server (non-blocking)."""
        if self._running:
            logger.warning("Server already running")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Smallest Hydra server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the Hydra server."""
        if not self._running:
            return
        self._running = False

        await self._transcriber.aclose()

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._server_task
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        logger.info(f"Smallest Hydra server stopped on port {self.port}")

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:  # noqa: C901
        """Bridge a single Twilio WebSocket session with Hydra."""
        logger.info("Client connected to Smallest Hydra server")

        stream_sid: str = self.conversation_id
        twilio_connected = True

        # Per-turn state (shared across tasks via nonlocal).
        _in_model_turn = False
        _user_speaking = False
        _user_speech_start_ts: str | None = None  # simulator VAD: user speech start (wall ms)
        _user_speech_stop_ts: str | None = None  # simulator VAD: user speech stop (wall ms)
        # Hydra's own end-of-user-speech (wall ms when we receive speech_stopped).
        # Preferred latency reference: Hydra responds within ~10 ms of this, before
        # the simulator's local user_speech_stop fires, so relying on the simulator
        # event alone drops nearly every turn's latency (same race as ElevenLabs).
        _hydra_speech_stop_wall_ms: str | None = None
        _assistant_turn_start_ts: str | None = None  # wall ms of first audio chunk this turn

        # User utterances have no native transcript, so we batch-STT the audio we
        # sent to Hydra. The assistant transcript comes natively from Hydra's
        # response.output_audio_transcript.delta stream (more accurate than STT).
        _user_utt_pcm = bytearray()  # 16 kHz PCM sent to Hydra for the current user utterance
        _asst_text = ""  # accumulated native assistant transcript for the current response

        # Outbound mulaw chunks; the pacer drains at real-time rate so the event
        # processor never blocks on sleep.
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        url = f"{_HYDRA_WS_URL}?model={self._model}&api_key={self._api_key}"

        def _transcribe_user_async(pcm: bytes, ts: str | None) -> None:
            """Spawn a background STT of a user utterance that appends to the audit log."""
            if len(pcm) < _MIN_USER_UTTERANCE_BYTES:
                # Too short to be real speech; skip to avoid STT hallucinations.
                return
            stamp = ts or str(int(round(time.time() * 1000)))

            async def _run() -> None:
                text = await self._transcriber.transcribe(pcm, _HYDRA_INPUT_RATE)
                if not text:
                    return
                logger.info(f"User transcript: {text}")
                self.audit_log.append_user_input(text, timestamp_ms=stamp)

            self._pending_transcriptions.append(asyncio.create_task(_run()))

        try:
            async with websockets.connect(url, max_size=None) as hydra_ws:
                logger.info(f"Hydra session connected (model={self._model}, voice={self._voice})")

                # Handshake: server sends session.created, then we configure.
                with contextlib.suppress(TimeoutError):
                    created = await asyncio.wait_for(hydra_ws.recv(), timeout=10.0)
                    logger.info(f"Hydra handshake: {created!r}")

                session: dict[str, Any] = {
                    "instructions": self._system_prompt,
                    "voice": self._voice,
                    "generate_initial_response": self._generate_initial_response,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                }
                if self._hydra_tools:
                    session["tools"] = self._hydra_tools
                configure_msg = json.dumps({"type": "session.configure", "session": session})
                logger.info(
                    f"Sending session.configure ({len(configure_msg)} bytes, {len(session.get('tools', []))} tools)"
                )
                await hydra_ws.send(configure_msg)

                # Wait for session.configured before starting concurrent tasks
                with contextlib.suppress(TimeoutError):
                    configured_raw = await asyncio.wait_for(hydra_ws.recv(), timeout=10.0)
                    configured_msg = json.loads(configured_raw)
                    sess = configured_msg.get("session", {})
                    key_fields = {k: v for k, v in sess.items() if k not in ("instructions", "tools")}
                    logger.info(f"Hydra session.configured: {key_fields}, tools_count={len(sess.get('tools', []))}")

                if self._generate_initial_response:
                    self._fw_log.turn_start()

                # ----- Concurrent tasks -----

                async def _forward_user_audio() -> None:
                    """Read Twilio WS messages, convert audio, send to Hydra."""
                    nonlocal stream_sid, twilio_connected
                    nonlocal _user_speech_start_ts, _user_speech_stop_ts, _user_speaking, _in_model_turn
                    nonlocal _last_audio_sent

                    def _flush_user_utterance() -> None:
                        nonlocal _user_utt_pcm
                        if _user_utt_pcm:
                            _transcribe_user_async(bytes(_user_utt_pcm), _user_speech_start_ts)
                            _user_utt_pcm = bytearray()

                    try:
                        while twilio_connected and self._running:
                            try:
                                raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                            except TimeoutError:
                                continue

                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            event = msg.get("event")
                            if event == "start":
                                stream_sid = msg.get("start", {}).get("streamSid", stream_sid)
                                logger.info(f"Twilio stream started: {stream_sid}")
                            elif event == "stop":
                                logger.info("Twilio stream stopped")
                                twilio_connected = False
                                break
                            elif event == "user_speech_start":
                                # New utterance: flush any unfinished prior one first
                                # (the simulator's stop event occasionally races).
                                _flush_user_utterance()
                                _user_speech_start_ts = msg.get("timestamp_ms")
                                _user_speaking = True
                                _in_model_turn = False
                                logger.info(f"User speech start: {_user_speech_start_ts}")
                            elif event == "user_speech_stop":
                                _user_speech_stop_ts = msg.get("timestamp_ms")
                                _user_speaking = False
                                logger.info(f"User speech stop: {_user_speech_stop_ts}")
                                _flush_user_utterance()
                            elif event == "media":
                                mulaw_bytes = parse_twilio_media_message(raw)
                                if mulaw_bytes is None:
                                    continue

                                # Record user audio at 48 kHz (recording rate).
                                pcm_48k = mulaw_8k_to_pcm16_48k(mulaw_bytes)
                                if not _in_model_turn:
                                    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                                self.user_audio_buffer.extend(pcm_48k)

                                # Accumulate the 16 kHz copy for utterance transcription.
                                pcm_16k = mulaw_8k_to_pcm16_16k(mulaw_bytes)
                                _user_utt_pcm.extend(pcm_16k)

                                # Send 16 kHz PCM16 to Hydra.
                                await hydra_ws.send(
                                    json.dumps(
                                        {
                                            "type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(pcm_16k).decode("ascii"),
                                        }
                                    )
                                )
                                _last_audio_sent = time.monotonic()
                    except WebSocketDisconnect:
                        logger.info("Twilio WebSocket disconnected")
                        twilio_connected = False
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in user audio forwarder: {e}", exc_info=True)
                    finally:
                        _flush_user_utterance()
                        twilio_connected = False

                async def _pace_audio_output() -> None:
                    """Drain audio_output_queue and forward chunks at real-time rate."""
                    nonlocal twilio_connected
                    next_send_time = time.monotonic()
                    try:
                        while self._running:
                            try:
                                chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                            except TimeoutError:
                                continue

                            try:
                                await websocket.send_text(create_twilio_media_message(stream_sid, chunk))
                            except Exception:
                                twilio_connected = False
                                return

                            now = time.monotonic()
                            if next_send_time <= now:
                                next_send_time = now
                            next_send_time += MULAW_CHUNK_DURATION_S
                            sleep_duration = next_send_time - time.monotonic()
                            if sleep_duration > 0:
                                await asyncio.sleep(sleep_duration)
                    except asyncio.CancelledError:
                        pass

                async def _process_hydra_events() -> None:  # noqa: C901
                    """Consume events from the Hydra WebSocket."""
                    nonlocal _in_model_turn, _user_speaking
                    nonlocal _user_speech_stop_ts, _hydra_speech_stop_wall_ms
                    nonlocal _assistant_turn_start_ts, _asst_text
                    nonlocal twilio_connected

                    def _flush_assistant(interrupted: bool) -> None:
                        nonlocal _asst_text, _in_model_turn, _assistant_turn_start_ts
                        # Collapse the phrase-level deltas Hydra streams into one line.
                        text = " ".join(_asst_text.split())
                        if text:
                            stamp = _assistant_turn_start_ts or str(int(round(time.time() * 1000)))
                            logger.info(f"Assistant transcript: {text}")
                            label = f"{text} [interrupted]" if interrupted else text
                            self.audit_log.append_assistant_output(label, timestamp_ms=stamp)
                            if interrupted:
                                self._fw_log.s2s_transcript(text)
                            else:
                                self._fw_log.llm_response(text)
                        self._fw_log.turn_end(was_interrupted=interrupted)
                        _asst_text = ""
                        _in_model_turn = False
                        _assistant_turn_start_ts = None

                    try:
                        while self._running:
                            try:
                                raw = await asyncio.wait_for(hydra_ws.recv(), timeout=2.0)
                            except TimeoutError:
                                continue
                            except websockets.exceptions.ConnectionClosed:
                                logger.info("Hydra WebSocket closed")
                                break

                            try:
                                msg = json.loads(raw)
                            except (json.JSONDecodeError, TypeError):
                                if isinstance(raw, bytes):
                                    logger.debug(f"Hydra binary frame: {len(raw)} bytes")
                                else:
                                    logger.debug(f"Hydra non-JSON text: {raw[:200]!r}")
                                continue

                            etype = msg.get("type")
                            if etype in ("response.output_audio.delta", "response.output_audio_transcript.delta"):
                                logger.debug(f"Hydra {etype} ({len(msg.get('delta', ''))} chars)")
                            elif etype == "session.configured":
                                sess = msg.get("session", {})
                                key_fields = {k: v for k, v in sess.items() if k not in ("instructions", "tools")}
                                logger.info(
                                    f"Hydra session.configured: {key_fields}, tools={len(sess.get('tools', []))}"
                                )
                            else:
                                logger.info(f"Hydra event: {etype} — {json.dumps(msg, ensure_ascii=False)[:500]}")

                            if etype == "response.output_audio.delta":
                                pcm_48k = base64.b64decode(msg.get("delta", ""))
                                if len(pcm_48k) < 6:
                                    continue

                                if not _in_model_turn:
                                    _in_model_turn = True
                                    _assistant_turn_start_ts = str(int(round(time.time() * 1000)))
                                    self._fw_log.turn_start()
                                    # Model response latency: user speech end -> first audio.
                                    # Prefer Hydra's speech_stopped (fires just before the
                                    # response); fall back to the simulator's user_speech_stop.
                                    # Absent on the initial greeting turn (model-initiated).
                                    _user_end_ref = _hydra_speech_stop_wall_ms or _user_speech_stop_ts
                                    if _user_end_ref and self._metrics_log:
                                        latency_ms = int(_assistant_turn_start_ts) - int(_user_end_ref)
                                        # Floor at 50 ms: Hydra occasionally emits speech_stopped
                                        # only after it has begun generating, collapsing the gap to
                                        # ~0 — sub-RTT values aren't real response latency.
                                        if 50 <= latency_ms < 30_000:
                                            self._metrics_log.write_latency(
                                                "model_response", latency_ms / 1000, self._model
                                            )
                                    _user_speech_stop_ts = None
                                    _hydra_speech_stop_wall_ms = None

                                # Record assistant track (48 kHz native).
                                if not _user_speaking:
                                    sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                                self.assistant_audio_buffer.extend(pcm_48k)

                                # Convert to 8 kHz mulaw and enqueue in 20 ms chunks.
                                if twilio_connected:
                                    try:
                                        mulaw = pcm16_48k_to_mulaw_8k(pcm_48k)
                                    except Exception as conv_err:
                                        logger.warning(f"Audio conversion error ({len(pcm_48k)} bytes): {conv_err}")
                                        continue
                                    offset = 0
                                    while offset < len(mulaw):
                                        await audio_output_queue.put(mulaw[offset : offset + MULAW_CHUNK_SIZE])
                                        offset += MULAW_CHUNK_SIZE

                            elif etype == "response.output_audio_transcript.delta":
                                # Native assistant transcript (more accurate than STT).
                                delta = msg.get("delta", "")
                                if delta:
                                    _asst_text = f"{_asst_text} {delta}" if _asst_text else delta

                            elif etype == "input_audio_buffer.speech_started":
                                # User barged in; Hydra cancels its in-flight response.
                                _user_speaking = True

                            elif etype == "input_audio_buffer.speech_stopped":
                                _user_speaking = False
                                # Anchor for model-response latency (see note at declaration).
                                _hydra_speech_stop_wall_ms = str(int(round(time.time() * 1000)))

                            elif etype == "response.function_call_arguments.done":
                                name = msg.get("name", "")
                                call_id = msg.get("call_id", "")
                                try:
                                    arguments = json.loads(msg.get("arguments") or "{}")
                                except json.JSONDecodeError:
                                    arguments = {}
                                logger.info(f"Tool call: {name}({json.dumps(arguments, ensure_ascii=False)})")
                                result = await self.execute_tool(name, arguments)
                                await hydra_ws.send(
                                    json.dumps(
                                        {
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "function_call_output",
                                                "role": "user",
                                                "call_id": call_id,
                                                "output": json.dumps(result, ensure_ascii=False),
                                            },
                                        }
                                    )
                                )
                                # Ask Hydra to narrate the tool result.
                                await hydra_ws.send(json.dumps({"type": "response.create", "response": {}}))

                            elif etype == "response.done":
                                response = msg.get("response", {}) or {}
                                status = response.get("status", "completed")
                                reason = (response.get("status_details") or {}).get("reason", "")
                                interrupted = status in ("cancelled", "incomplete") or reason in (
                                    "interrupted",
                                    "client_cancelled",
                                )
                                if _in_model_turn or _asst_text:
                                    _flush_assistant(interrupted)

                                usage = response.get("usage") or {}
                                prompt_tokens = usage.get("input_tokens", 0) or 0
                                completion_tokens = usage.get("output_tokens", 0) or 0
                                if (prompt_tokens or completion_tokens) and self._metrics_log:
                                    self._metrics_log.write_token_usage(
                                        processor="smallest_hydra",
                                        model=self._model,
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens,
                                    )

                            elif etype == "error":
                                err = msg.get("error", {})
                                logger.error(f"Hydra error: {err.get('code')} - {err.get('message')}")

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in Hydra event processor: {e}", exc_info=True)

                # Hydra closes the socket after ~30 s of silence.
                # Send silence frames every few seconds to keep it alive.
                _KEEPALIVE_INTERVAL = 5.0  # seconds
                _SILENCE_16K = base64.b64encode(b"\x00" * 3200).decode("ascii")  # 100 ms
                _last_audio_sent = time.monotonic()

                async def _hydra_keepalive() -> None:
                    """Send silence to Hydra when no user audio is flowing."""
                    nonlocal _last_audio_sent
                    try:
                        while self._running:
                            await asyncio.sleep(_KEEPALIVE_INTERVAL)
                            elapsed = time.monotonic() - _last_audio_sent
                            if elapsed >= _KEEPALIVE_INTERVAL:
                                await hydra_ws.send(
                                    json.dumps({"type": "input_audio_buffer.append", "audio": _SILENCE_16K})
                                )
                                _last_audio_sent = time.monotonic()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass  # non-critical; let other tasks drive shutdown

                user_task = asyncio.create_task(_forward_user_audio())
                hydra_task = asyncio.create_task(_process_hydra_events())
                pacer_task = asyncio.create_task(_pace_audio_output())
                keepalive_task = asyncio.create_task(_hydra_keepalive())

                done, pending = await asyncio.wait(
                    [user_task, hydra_task, pacer_task, keepalive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                def _task_name(t: asyncio.Task) -> str:
                    if t is user_task:
                        return "user_audio"
                    if t is hydra_task:
                        return "hydra_events"
                    if t is keepalive_task:
                        return "keepalive"
                    return "audio_pacer"

                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        logger.error(f"Task '{_task_name(task)}' failed: {exc}", exc_info=exc)
                    else:
                        logger.info(f"Task '{_task_name(task)}' completed normally")

                for task in pending:
                    logger.info(f"Cancelling pending task '{_task_name(task)}'")
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        except Exception as e:
            logger.error(f"Smallest Hydra session error: {e}", exc_info=True)
        finally:
            # Let outstanding transcriptions finish so the audit log is complete
            # before stop()/save_outputs() runs.
            if self._pending_transcriptions:
                logger.info(f"Awaiting {len(self._pending_transcriptions)} pending transcription(s)")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(*self._pending_transcriptions, return_exceptions=True),
                        timeout=60.0,
                    )
            logger.info("Client disconnected from Smallest Hydra server")
