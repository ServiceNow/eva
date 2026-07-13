"""ElevenLabs AssistantServer for EVA-Bench.

Bridges between Twilio-framed WebSocket (user simulator) and ElevenLabs
Conversational AI via the elevenlabs Python SDK.  Audio flows:

    User simulator (8 kHz mulaw) -> ElevenLabs input
    User simulator (8 kHz mulaw) -> 16 kHz PCM16 (local WAV recording only)
    ElevenLabs output (16 kHz PCM16) -> 8 kHz mulaw -> User simulator

All tool calls are executed locally via ToolExecutor (through ClientTools);
transcription events from ElevenLabs populate the audit log.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import httpx
import uvicorn
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    AsyncConversation,
    ClientTools,
    ConversationInitiationData,
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_16k,
    parse_twilio_media_message,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.elevenlabs_audio_interface import TwilioAudioBridge
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_RECORDING_SAMPLE_RATE = 16000

# Audio output pacing: send 160-byte mulaw chunks (20ms at 8kHz) at real-time
# rate so the user simulator's silence detection works correctly.
MULAW_CHUNK_SIZE = 160  # bytes per chunk (20ms at 8kHz, 1 byte per sample)
MULAW_CHUNK_DURATION_S = 0.02  # 20ms per chunk

# 20ms of recording-rate (16kHz) 16-bit mono PCM that corresponds to one
# MULAW_CHUNK_SIZE mulaw chunk: 160 mulaw samples @8kHz -> 320 PCM @16kHz -> 640 bytes.
PCM_CHUNK_SIZE = MULAW_CHUNK_SIZE * 4  # 640 bytes


# ---------------------------------------------------------------------------
# Audio conversion helper
# ---------------------------------------------------------------------------


def _pcm16_16k_to_mulaw_8k(pcm_16k: bytes) -> bytes:
    """Convert 16 kHz 16-bit PCM mono to 8 kHz mulaw."""
    pcm_8k, _ = audioop.ratecv(pcm_16k, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(pcm_8k, 2)


def _pad_buffer_to_walltime(buffer: bytearray, elapsed_s: float, sample_rate: int) -> None:
    """Pad *buffer* with silence so its length reflects *elapsed_s* of real time.

    Both recording channels are anchored to a single session start, so the
    mixed/stereo output preserves real inter-turn timing instead of collapsing
    silence gaps (which previously made the recording shorter than reality and
    overlapped turns). 16-bit mono PCM => 2 bytes per sample.
    """
    target_bytes = int(elapsed_s * sample_rate) * 2
    if len(buffer) < target_bytes:
        buffer.extend(b"\x00" * (target_bytes - len(buffer)))


# ---------------------------------------------------------------------------
# Tool conversion helper
# ---------------------------------------------------------------------------


def _agent_tools_to_client_tools(
    agent: AgentConfig,
    execute_tool_fn,
) -> ClientTools | None:
    """Convert EVA AgentConfig tools to ElevenLabs ClientTools.

    Each AgentTool is registered as a client tool and sets the handler
    as self.execute_tool
    """
    if not agent.tools:
        return None

    client_tools = ClientTools()

    for tool in agent.tools:
        func_name = tool.function_name

        async def _handle(parameters: dict, _name: str = func_name) -> str:
            # tool_call_id is injected by ClientTools.execute_tool; strip it
            # before forwarding to the domain tool handler.
            args = {k: v for k, v in parameters.items() if k != "tool_call_id"}
            result = await execute_tool_fn(_name, args)
            return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)

        client_tools.register(func_name, _handle, is_async=True)

    return client_tools


# ---------------------------------------------------------------------------
# ElevenLabs AssistantServer
# ---------------------------------------------------------------------------


class ElevenLabsAssistantServer(AbstractAssistantServer):
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

        # Recording sample rate (ElevenLabs operates at 16 kHz)
        self._audio_sample_rate = _RECORDING_SAMPLE_RATE

        s2s_params = self.pipeline_config.s2s_params or {}
        self.s2s_params = s2s_params
        self._model = s2s_params.get("model", "elevenlabs")

        self._system_prompt = self._build_system_prompt()

        # Build ElevenLabs client tools from agent config
        self._client_tools = _agent_tools_to_client_tools(agent, self.execute_tool)

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

        logger.info(f"Elevenlabs server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the server (framework-specific teardown)."""
        if not self._running:
            return
        self._running = False

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        logger.info(f"ElevenLabs server stopped on port {self.port}")

    async def _handle_session(self, websocket: WebSocket) -> None:  # noqa: C901
        """Bridge a single Twilio WebSocket session with ElevenLabs."""
        logger.info("Client connected to ElevenLabs server")

        stream_sid: str = self.conversation_id
        twilio_connected = True

        # Per-turn state
        _in_model_turn = False
        _is_first_turn = True  # prevents timestamp issues on first turn
        _user_speaking = False
        _user_speech_start_ts: str | None = None
        _user_speech_stop_ts: str | None = None
        # ElevenLabs' own end-of-user-turn signal (wall-clock ms), stamped when the
        # user transcript arrives. ElevenLabs runs its own server-side VAD and
        # responds off that, often before the user sim's local end-of-speech
        # detection (and its user_speech_stop event) fires. Using this as the
        # model-response-latency reference avoids the race that drops most turns.
        _user_turn_end_ts: str | None = None
        _assistant_turn_start_ts: str | None = None
        # Shared recording anchor: set on the first recorded audio of either
        # channel. Both tracks pad silence relative to this so the mixed output
        # keeps real wall-clock timing (set lazily; see _pad_buffer_to_walltime).
        _record_t0: float | None = None
        # Pad the USER track to wall-clock only at the start of a user turn, then
        # append contiguously within the turn (mirrors the assistant pacer's
        # turn_start handling). Padding on EVERY frame — the previous behavior —
        # injected silence whenever frames were processed late under concurrency,
        # fabricating mid-utterance "chops" in audio_user.wav even though the
        # audio and the conversation were fine. True inter-turn gaps (e.g. the
        # assistant's speaking period, when the user sim sends no frames) are
        # still filled by the one pad at the next user-turn start.
        _pad_user_on_turn_start: bool = True

        # Queue of (mulaw_chunk, pcm16k_chunk, turn_start) items; the pacer drains
        # at real-time rate, sends the mulaw and records the PCM at playback time.
        audio_output_queue: asyncio.Queue[tuple[bytes, bytes, bool]] = asyncio.Queue()

        # Signalled when ElevenLabs ends the session
        session_ended = asyncio.Event()

        # -- Audio bridge --------------------------------------------------

        audio_bridge = TwilioAudioBridge()

        # -- ElevenLabs callbacks ------------------------------------------

        async def _on_agent_response(text: str) -> None:
            nonlocal _assistant_turn_start_ts, _is_first_turn
            logger.info(f"Agent response: {text}")
            self.audit_log.append_assistant_output(text, timestamp_ms=_assistant_turn_start_ts)
            self._fw_log.llm_response(text)
            self._fw_log.turn_end(was_interrupted=False)
            if _is_first_turn:
                _is_first_turn = False
            # NOTE: do not reset _in_model_turn here. The agent_response event
            # carries the full text but audio is still streaming; resetting the
            # flag mid-stream would make the recorder treat the rest of the same
            # utterance as a new turn (re-anchoring to wall-clock and injecting
            # silence). _in_model_turn resets at the next user_speech_start.
            _assistant_turn_start_ts = None

        async def _on_agent_response_correction(original: str, corrected: str) -> None:
            nonlocal _assistant_turn_start_ts
            logger.info(f"Agent response corrected: {original!r} -> {corrected!r}")
            if corrected:
                self.audit_log.append_assistant_output(
                    corrected + " [interrupted]",
                    timestamp_ms=_assistant_turn_start_ts,
                )
                self._fw_log.s2s_transcript(corrected)
            self._fw_log.turn_end(was_interrupted=True)
            # Same as _on_agent_response: do not reset _in_model_turn here.
            _assistant_turn_start_ts = None

        async def _on_user_transcript(text: str) -> None:
            nonlocal _user_speech_start_ts, _user_speaking, _user_turn_end_ts
            logger.info(f"User transcript: {text}")
            _user_speaking = False
            # ElevenLabs has finished hearing the user — this is its end-of-turn,
            # the reference for model-response latency to the first agent audio.
            _user_turn_end_ts = str(int(round(time.time() * 1000)))
            self.audit_log.append_user_input(text, timestamp_ms=_user_speech_start_ts)
            _user_speech_start_ts = None

        async def _on_end_session() -> None:
            logger.info("ElevenLabs session ended")
            session_ended.set()

        # -- ElevenLabs client setup ---------------------------------------

        http_client = httpx.Client(verify=False, timeout=30.0)
        client = ElevenLabs(
            api_key=self.s2s_params.get("api_key"),
            timeout=30.0,
            httpx_client=http_client,
        )

        conv_config = ConversationInitiationData(
            dynamic_variables={
                "system_prompt": self._system_prompt,
                "initial_message": self.initial_message,
            },
        )

        agent_id = self.s2s_params.get("assistant_agent_id")
        if not agent_id:
            raise ValueError("Missing ElevenLabs assistant agent ID in s2s_params")
        logger.info(f"Using assistant agent ID: {agent_id}")

        conversation = AsyncConversation(
            client,
            agent_id,
            requires_auth=True,
            audio_interface=audio_bridge,
            config=conv_config,
            client_tools=self._client_tools,
            callback_agent_response=_on_agent_response,
            callback_agent_response_correction=_on_agent_response_correction,
            callback_user_transcript=_on_user_transcript,
            callback_end_session=_on_end_session,
        )

        try:
            await conversation.start_session()
            logger.info(f"ElevenLabs conversation session started: {conversation._conversation_id}")

            # Wait for the ElevenLabs WS to connect before sending the prompt
            for _ in range(50):  # up to 5s
                if conversation._ws:
                    break
                await asyncio.sleep(0.1)
            logger.info("ElevenLabs WebSocket connected, sending initial message")

            self._fw_log.turn_start()

            # ----- Concurrent tasks -----

            async def _forward_user_audio() -> None:
                """Read Twilio WS messages, convert audio, send to ElevenLabs."""
                nonlocal stream_sid, twilio_connected
                nonlocal _user_speech_start_ts, _user_speech_stop_ts
                nonlocal _user_speaking, _in_model_turn, _record_t0, _pad_user_on_turn_start
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
                            _user_speech_start_ts = msg.get("timestamp_ms")
                            _user_speaking = True
                            _in_model_turn = False
                            # New user turn: allow one wall-clock pad to place this
                            # utterance on the real timeline (fills the assistant's
                            # speaking gap). Cleared after the first frame so the
                            # rest of the turn records contiguously.
                            _pad_user_on_turn_start = True
                            logger.info(f"User speech start: {_user_speech_start_ts}")
                        elif event == "user_speech_stop":
                            _user_speech_stop_ts = msg.get("timestamp_ms")
                            logger.info(f"User speech stop: {_user_speech_stop_ts}")
                        elif event == "media":
                            mulaw_bytes = parse_twilio_media_message(raw)
                            if mulaw_bytes is None:
                                continue

                            # Record user audio as 16 kHz PCM for the WAV file,
                            # anchored to real time so gaps are preserved.
                            pcm_16k = mulaw_8k_to_pcm16_16k(mulaw_bytes)
                            now = time.monotonic()
                            if _record_t0 is None:
                                _record_t0 = now
                            # Pad to wall-clock only at the start of a user turn;
                            # within the turn append contiguously so processing-time
                            # jitter under concurrency cannot inject mid-utterance
                            # silence (the choppy-audio bug). See _pad_user_on_turn_start.
                            if _pad_user_on_turn_start:
                                _pad_buffer_to_walltime(
                                    self.user_audio_buffer,
                                    now - _record_t0,
                                    self._audio_sample_rate,
                                )
                                _pad_user_on_turn_start = False
                            self.user_audio_buffer.extend(pcm_16k)

                            # Feed raw 8 kHz mulaw to ElevenLabs — the agent
                            # is configured to accept mulaw input directly. The
                            # bridge keeps ElevenLabs' VAD fed with silence during
                            # gaps, so no separate silence task is needed here.
                            await audio_bridge.feed_user_audio(mulaw_bytes)

                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket disconnected")
                    twilio_connected = False
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in user audio forwarder: {e}", exc_info=True)
                finally:
                    twilio_connected = False

            async def _forward_assistant_audio() -> None:
                """Pull audio from bridge, convert, and enqueue for the pacer."""
                nonlocal _in_model_turn, _assistant_turn_start_ts
                nonlocal _user_speech_stop_ts, _user_turn_end_ts
                try:
                    while self._running:
                        pcm_16k = await audio_bridge.get_output_audio(timeout=1.0)
                        if pcm_16k is None:
                            continue
                        if len(pcm_16k) < 4:
                            continue

                        # First audio chunk of a new model turn
                        new_turn = False
                        if not _in_model_turn:
                            _in_model_turn = True
                            new_turn = True
                            _assistant_turn_start_ts = str(int(round(time.time() * 1000)))
                            self._fw_log.turn_start()

                            # Model response latency: user turn end -> first audio.
                            # Prefer ElevenLabs' own end-of-turn (user transcript),
                            # which is what triggered the response and is available
                            # before the agent audio; fall back to the user sim's
                            # user_speech_stop event. Absent on the greeting turn.
                            _user_end_ref = _user_turn_end_ts or _user_speech_stop_ts
                            if _user_end_ref and self._metrics_log:
                                latency_ms = int(_assistant_turn_start_ts) - int(_user_end_ref)
                                if 0 < latency_ms < 30_000:
                                    self._metrics_log.write_latency(
                                        "model_response",
                                        latency_ms / 1000,
                                        self._model,
                                    )
                            _user_speech_stop_ts = None
                            _user_turn_end_ts = None

                        # Convert to mulaw and enqueue for pacer. Recording is
                        # deferred to the pacer so the assistant track is placed
                        # at playback (real) time rather than this bursty receive
                        # time — each mulaw chunk is paired with its 16kHz PCM so
                        # the pacer can record without a lossy re-conversion. Only
                        # the first chunk of a turn carries turn_start=True; the
                        # pacer pads to wall-clock just at the turn boundary and
                        # then records contiguously, so ElevenLabs' bursty intra-turn
                        # delivery does not inject silence mid-utterance.
                        if twilio_connected:
                            try:
                                mulaw = _pcm16_16k_to_mulaw_8k(pcm_16k)
                            except Exception as conv_err:
                                logger.warning(f"Audio conversion error ({len(pcm_16k)} bytes): {conv_err}")
                                continue

                            m_off = 0
                            p_off = 0
                            while m_off < len(mulaw):
                                m_chunk = mulaw[m_off : m_off + MULAW_CHUNK_SIZE]
                                p_chunk = pcm_16k[p_off : p_off + PCM_CHUNK_SIZE]
                                m_off += MULAW_CHUNK_SIZE
                                p_off += PCM_CHUNK_SIZE
                                turn_start = new_turn and m_off == MULAW_CHUNK_SIZE
                                await audio_output_queue.put((m_chunk, p_chunk, turn_start))

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in assistant audio forwarder: {e}", exc_info=True)

            async def _pace_audio_output() -> None:
                """Drain audio_output_queue and send to Twilio at real-time rate."""
                nonlocal twilio_connected, _record_t0
                next_send_time = time.monotonic()
                try:
                    while self._running:
                        try:
                            mulaw_chunk, pcm_chunk, turn_start = await asyncio.wait_for(
                                audio_output_queue.get(), timeout=1.0
                            )
                        except TimeoutError:
                            continue

                        # Stamp before send so WAV position matches when the chunk
                        # starts transmission (when the user sim can first receive it).
                        now = time.monotonic()

                        # Pad assistant track to wall-clock only at turn boundaries;
                        # append contiguously within a turn to avoid mid-utterance gaps.
                        if _record_t0 is None:
                            _record_t0 = now
                        if turn_start:
                            _pad_buffer_to_walltime(
                                self.assistant_audio_buffer,
                                now - _record_t0,
                                self._audio_sample_rate,
                            )
                        self.assistant_audio_buffer.extend(pcm_chunk)

                        twilio_msg = create_twilio_media_message(stream_sid, mulaw_chunk)
                        try:
                            await websocket.send_text(twilio_msg)
                        except Exception:
                            twilio_connected = False
                            return

                        if next_send_time <= now:
                            next_send_time = now
                        next_send_time += MULAW_CHUNK_DURATION_S
                        sleep_duration = next_send_time - time.monotonic()
                        if sleep_duration > 0:
                            await asyncio.sleep(sleep_duration)
                except asyncio.CancelledError:
                    pass

            async def _watch_session_end() -> None:
                """Wait for ElevenLabs to signal end-of-session."""
                await session_ended.wait()

            # Run all tasks; when any exits, cancel the others
            user_task = asyncio.create_task(_forward_user_audio())
            assistant_task = asyncio.create_task(_forward_assistant_audio())
            pacer_task = asyncio.create_task(_pace_audio_output())
            end_task = asyncio.create_task(_watch_session_end())

            all_tasks = [user_task, assistant_task, pacer_task, end_task]

            done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

            def _task_name(t: asyncio.Task) -> str:
                if t is user_task:
                    return "user_audio"
                if t is assistant_task:
                    return "assistant_audio"
                if t is pacer_task:
                    return "audio_pacer"
                return "session_end"

            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    logger.error(f"Task '{_task_name(task)}' failed: {exc}", exc_info=exc)
                else:
                    logger.info(f"Task '{_task_name(task)}' completed normally")

            for task in pending:
                logger.info(f"Cancelling pending task '{_task_name(task)}'")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"ElevenLabs session error: {e}", exc_info=True)
        finally:
            try:
                await conversation.end_session()
                await conversation.wait_for_session_end()
            except Exception as e:
                logger.warning(f"Error ending ElevenLabs session: {e}")
            logger.info("Client disconnected from ElevenLabs server")
