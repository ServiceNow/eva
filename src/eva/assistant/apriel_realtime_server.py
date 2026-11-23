"""Apriel Realtime assistant bridge, isolated from the upstream OpenAI adapter.

Uses the OpenAI Python SDK's Realtime API (client.realtime.connect())
to bridge audio between a Twilio-framed WebSocket (user simulator) and the
Apriel Realtime endpoint. Handles tool calls via the local ToolExecutor and
records all conversation events in the audit log.
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.utils.audio_utils import (
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# OpenAI Realtime operates at 24 kHz 16-bit mono PCM
OPENAI_SAMPLE_RATE = 24000

# Audio output pacing: send 160-byte mulaw chunks (20ms at 8kHz) at real-time rate
# so the user simulator's silence detection works correctly.
MULAW_CHUNK_SIZE = 160  # bytes per chunk (20ms at 8kHz, 1 byte per sample)
# Don't pad the user track to align with the assistant when real user audio
# arrived within this window. The speaking-state flag can go stale under
# event-loop jitter, and padding then injects silence into an active user
# utterance (the choppy-audio bug). This guard only ever *skips* a pad, so it
# can never add silence or worsen alignment — during genuine user silence the
# timestamp is old and padding proceeds normally.
USER_ACTIVE_GUARD_S = 0.3
# TTS audio starts with near-silence. When estimating how much text the caller
# heard, this much audio at the start of a response is treated as containing no
# speech (checked against recordings: a 160 ms cut had no audible speech).
LEADING_SILENCE_PCM_BYTES = int(0.2 * OPENAI_SAMPLE_RATE * 2)  # 200 ms of PCM16


def _wall_ms() -> str:
    """Return current wall-clock time as epoch-milliseconds string."""
    return str(int(round(time.time() * 1000)))


@dataclass
class _UserTurnRecord:
    """Tracks state for a single user speech turn."""

    item_id: str = ""
    speech_started_wall_ms: str = ""
    speech_stopped_wall_ms: str = ""
    transcript: str = ""
    flushed: bool = False
    # Set once the assistant has replied to this turn. A turn that was answered
    # but still has no transcript is not going to get one.
    answered: bool = False
    # Audit-log turn id, used to replace a placeholder with the real transcript.
    turn_id: int | None = None
    placeholder_written: bool = False


@dataclass
class _AssistantResponseState:
    """Accumulates state for the current assistant response."""

    transcript_parts: list[str] = field(default_factory=list)
    transcript_done_text: str = ""  # Final text from response.audio_transcript.done
    first_audio_wall_ms: str | None = None
    responding: bool = False
    has_function_calls: bool = False


@dataclass
class _DeferredAssistantOutput:
    """Assistant text held back until its paced playback finishes or is cut off."""

    content: str
    timestamp_ms: str


@dataclass
class _AudioPlayback:
    """Playback can outlive response.done and must remain interruptible."""

    interrupted: bool = False
    pending_frames: int = 0
    # First assistant audio item queued for this response; the truncate target
    # when an interruption happens before any of its frames are sent.
    item_id: str = ""
    content_index: int = 0
    # Assistant audio item currently being sent and how much of it was sent.
    played_item_id: str = ""
    played_content_index: int = 0
    played_pcm_bytes: int = 0
    # Whole-response audio totals (PCM16 bytes), used to estimate how much of
    # the text the caller heard when playback is cut off.
    queued_pcm_bytes: int = 0
    sent_pcm_bytes: int = 0
    deferred_output: _DeferredAssistantOutput | None = None
    output_logged: bool = False


@dataclass
class _OutputAudioChunk:
    playback: _AudioPlayback
    mulaw: bytes
    pcm16: bytes
    item_id: str = ""
    content_index: int = 0


@dataclass
class _PendingToolCall:
    """A function call streamed by one Realtime response, run at response.done."""

    call_id: str
    response_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


class AprielRealtimeAssistantServer(AbstractAssistantServer):
    """Assistant server for Apriel through the OpenAI-compatible Realtime API.

    Exposes a local WebSocket at ``ws://localhost:{port}/ws`` using the Twilio
    frame format so the user simulator can connect as if talking to Twilio.
    Internally bridges audio between Twilio (8 kHz mulaw) and OpenAI Realtime
    (24 kHz PCM16 base64).
    """

    _service_name: str = "Apriel Realtime"
    _metrics_processor_name: str = "openai_realtime"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._audio_sample_rate = OPENAI_SAMPLE_RATE
        # Monotonic time of the last real user-audio frame appended; used to
        # guard against padding the user track mid-utterance (see USER_ACTIVE_GUARD_S).
        self._last_user_audio_mono: float = 0.0

        self._system_prompt: str = self._build_system_prompt()

        self._realtime_tools: list[dict] = self._build_realtime_tools()

        self._user_turn: _UserTurnRecord | None = None
        self._user_turns_by_item_id: dict[str, _UserTurnRecord] = {}
        self._assistant_state = _AssistantResponseState()
        self._stream_sid: str = ""
        self._reset_audio_playback()

        self._user_speaking: bool = False
        self._bot_speaking: bool = False
        self._user_frame_count: int = 0

        # Tool calls are streamed before response.done. Record them as they
        # arrive and run them only once the originating response completes, so
        # a cancelled response never changes scenario state.
        self._active_response_id: str | None = None
        self._pending_tool_calls: dict[str, _PendingToolCall] = {}

        # User speech start timestamp from audio_interface (source of truth)
        self._audio_interface_speech_start_ts: str | None = None

        s2s_params = self.pipeline_config.s2s_params or {}
        self._model: str = s2s_params["model"]

    async def start(self) -> None:
        """Start the FastAPI WebSocket server."""
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
            if self._server_task.done():
                # serve() returned without starting, e.g. the port is already in use.
                self._running = False
                self._server = None
                self._server_task = None
                raise RuntimeError(f"{self._service_name} server failed to start on port {self.port}")
            await asyncio.sleep(0.01)

        logger.info(f"{self._service_name} server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the OpenAI Realtime server."""
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

        logger.info(f"{self._service_name} server stopped on port {self.port}")

    def _default_voice(self) -> str | None:
        """Default voice ID when `s2s_params.voice` is not set.

        Apriel's voice catalogue is endpoint-specific, so no voice is sent
        unless `s2s_params.voice` is configured.
        """
        return None

    def _build_session_config(self) -> dict[str, Any]:
        """Construct the `session.update` payload for the realtime connection.

        Subclasses override to adjust service-specific fields (e.g. drop the
        `transcription.model` selector for xAI, which doesn't expose one).
        """
        s2s = self.pipeline_config.s2s_params or {}
        vad = s2s.get("vad_settings", {}) or {}
        transcription_model = s2s.get("transcription_model", "whisper-1")

        session_config: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self._system_prompt,
            "audio": {
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                },
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": vad.get("type", "server_vad"),
                        "threshold": vad.get("threshold", 0.5),
                        "prefix_padding_ms": vad.get("prefix_padding_ms", 300),
                        "silence_duration_ms": vad.get("silence_duration_ms", 200),
                        "create_response": True,
                    },
                    "transcription": {"model": transcription_model} if transcription_model else None,
                },
            },
            "tools": self._realtime_tools,
        }

        voice = s2s.get("voice", self._default_voice())
        if voice:
            session_config["audio"]["output"]["voice"] = voice

        reasoning_effort = s2s.get("reasoning_effort")
        if reasoning_effort:
            session_config["reasoning"] = {"effort": reasoning_effort}

        if self.pipeline_config.parallel_tool_calls is not None:
            session_config["parallel_tool_calls"] = self.pipeline_config.parallel_tool_calls

        # OpenAI-compatible providers can expose additional session controls.
        # Only forward explicitly configured fields so the default OpenAI
        # payload remains unchanged.
        for field_name in ("enable_thinking", "pre_tool_speech"):
            if field_name in s2s:
                value = s2s[field_name]
                if not isinstance(value, bool):
                    # bool("false") is True, which would silently invert the setting.
                    raise ValueError(f"s2s_params.{field_name} must be a JSON boolean, got {value!r}")
                session_config[field_name] = value
        if "apriel" in s2s:
            session_config["apriel"] = s2s["apriel"]

        return session_config

    def _create_client(self) -> AsyncOpenAI:
        """Construct the SDK client for the configured Apriel endpoint."""
        s2s = self.pipeline_config.s2s_params or {}
        api_key = s2s.get("api_key")
        if not api_key:
            raise ValueError(f"API key required for {self._service_name}")

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        for field_name in ("base_url", "websocket_base_url"):
            if value := s2s.get(field_name):
                client_kwargs[field_name] = value
        return AsyncOpenAI(**client_kwargs)

    def _websocket_options(self) -> dict[str, Any]:
        """Keep proxy credentials in fields redacted by RunConfig serialization."""
        params = self.pipeline_config.s2s_params or {}
        options = dict(params.get("websocket_connection_options") or {})
        if key := params.get("apriel_api_key"):
            auth_subprotocols = ["realtime", "openai-insecure-api-key." + key]
            user_subprotocols = [p for p in (options.get("subprotocols") or []) if p not in auth_subprotocols]
            options["subprotocols"] = auth_subprotocols + user_subprotocols
        # The SDK derives the websocket URL from base_url when websocket_base_url is unset.
        host = urlsplit(params.get("websocket_base_url") or params.get("base_url") or "").hostname or ""
        if "_" in host:
            options["server_hostname"] = host.replace("_", "-")
        return options

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Handle a single WebSocket session.

        1. Accept Twilio WS connection
        2. Connect to OpenAI Realtime API
        3. Configure session (instructions, tools, voice, VAD)
        4. Run two concurrent tasks:
           a. Forward user audio: Twilio WS -> decode mulaw -> PCM16 24kHz base64 -> OpenAI
           b. Process OpenAI events: async for event in conn -> handle each type
        5. On tool call: execute via self.tool_handler, send result back
        6. On audio: decode base64 PCM16 -> record -> encode mulaw -> send to Twilio WS
        """
        logger.info(f"Client connected to {self._service_name} server")

        # Reset per-session state
        self._user_turn = None
        self._user_turns_by_item_id = {}
        self._next_turn_id = 0
        self._assistant_state = _AssistantResponseState()
        self._stream_sid = self.conversation_id
        self._reset_audio_playback()
        self._user_speaking = False
        self._bot_speaking = False
        self._reset_tool_tracking()

        client = self._create_client()

        try:
            logger.info(f"Starting {self._service_name} session (model={self._model})")
            websocket_options = self._websocket_options()

            async with client.realtime.connect(
                model=self._model, websocket_connection_options=websocket_options
            ) as conn:
                # Configure the session
                session_config = self._build_session_config()
                await conn.session.update(session=session_config)

                # Trigger the initial greeting
                await conn.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Say: '{self.initial_message}'",
                            }
                        ],
                    }
                )
                await conn.response.create()

                # Run all three tasks; when any exits, cancel the others
                audio_output_queue: asyncio.Queue[_OutputAudioChunk] = asyncio.Queue()
                forward_task = asyncio.create_task(self._forward_user_audio(websocket, conn))
                receive_task = asyncio.create_task(self._process_openai_events(conn, websocket, audio_output_queue))
                pacer_task = asyncio.create_task(self._pace_audio_output(websocket, audio_output_queue))

                done, pending = await asyncio.wait(
                    [forward_task, receive_task, pacer_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                task_names = {
                    forward_task: "forward_user_audio",
                    receive_task: "process_openai_events",
                    pacer_task: "pace_audio_output",
                }
                for task in done:
                    outcome = (
                        "cancelled"
                        if task.cancelled()
                        else (repr(task.exception()) if task.exception() else "returned")
                    )
                    logger.info(f"{self._service_name} session task finished first: {task_names[task]} ({outcome})")

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check for exceptions in completed tasks
                for task in done:
                    if not task.cancelled() and task.exception():
                        logger.error(f"Session task failed: {task.exception()}")

        except Exception as e:
            logger.error(f"{self._service_name} session error: {e}", exc_info=True)
        finally:
            self._flush_user_turn_without_transcript()
            self._flush_deferred_outputs_on_disconnect()
            self._reset_tool_tracking()
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Failed to close {self._service_name} client: {e}")
            logger.info(f"Client disconnected from {self._service_name} server")

    # ── Audio output pacer (OpenAI -> Twilio WS at real-time rate) ───

    async def _pace_audio_output(
        self, websocket: WebSocket, audio_output_queue: asyncio.Queue[_OutputAudioChunk]
    ) -> None:
        """Drain audio_output_queue and forward chunks at real-time rate.

        Runs as its own task so _process_openai_events never blocks on sleep
        and can read the next OpenAI event immediately.
        """
        next_send_time = time.monotonic()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                try:
                    # Serialize with interruption: a dequeued frame may have
                    # been invalidated before the socket becomes writable.
                    async with self._playback_lock:
                        if chunk.playback.interrupted:
                            continue
                        await websocket.send_text(create_twilio_media_message(self._stream_sid, chunk.mulaw))
                        playback = chunk.playback
                        playback.pending_frames -= 1
                        if chunk.item_id and chunk.item_id != playback.played_item_id:
                            playback.played_item_id = chunk.item_id
                            playback.played_content_index = chunk.content_index
                            playback.played_pcm_bytes = 0
                        playback.played_pcm_bytes += len(chunk.pcm16)
                        playback.sent_pcm_bytes += len(chunk.pcm16)
                        self._record_played_audio(chunk.pcm16)
                        if playback.pending_frames <= 0 and playback.deferred_output:
                            deferred = playback.deferred_output
                            self._log_assistant_output(playback, deferred.content, deferred.timestamp_ms)
                except Exception as e:
                    logger.error(f"Error sending audio to Twilio WS: {e}")
                    return
                finally:
                    audio_output_queue.task_done()

                now = time.monotonic()
                if next_send_time <= now:
                    next_send_time = now
                next_send_time += len(chunk.mulaw) / 8000
                sleep_duration = next_send_time - time.monotonic()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            pass

    def _reset_audio_playback(self) -> None:
        self._audio_playbacks: dict[str, _AudioPlayback] = {}
        self._playback_response_id = ""
        self._playback_lock = asyncio.Lock()

    def _audio_playback(self, event: Any) -> _AudioPlayback:
        response_id = getattr(event, "response_id", "") or self._playback_response_id
        return self._audio_playbacks.setdefault(response_id, _AudioPlayback())

    async def _clear_audio_playback(
        self,
        queue: asyncio.Queue[_OutputAudioChunk],
        response_id: str | None = None,
    ) -> list[_AudioPlayback]:
        """Stop queued playback and reject its late deltas.

        A confirmed speech start clears all old playback. Provider clear/cancel
        events with a response ID only clear that response, never a newer one.
        Tool execution/continuation is managed independently.

        Returns the playbacks that still had unsent audio when they were cut off.
        Text held back for those playbacks is logged as interrupted.
        """
        cut: list[_AudioPlayback] = []
        async with self._playback_lock:
            self._audio_playbacks.setdefault(response_id or self._playback_response_id, _AudioPlayback())
            for key, playback in self._audio_playbacks.items():
                if (response_id is not None and key != response_id) or playback.interrupted:
                    continue
                if playback.pending_frames > 0:
                    cut.append(playback)
                playback.interrupted = True
                playback.pending_frames = 0
                if playback.deferred_output:
                    deferred = playback.deferred_output
                    self._log_assistant_output(playback, deferred.content, deferred.timestamp_ms, interrupted=True)
            remaining = []
            while not queue.empty():
                chunk = queue.get_nowait()
                queue.task_done()
                if not chunk.playback.interrupted:
                    remaining.append(chunk)
            for chunk in remaining:
                queue.put_nowait(chunk)
            self._bot_speaking = bool(remaining)
        return cut

    async def _truncate_interrupted_items(self, conn: Any, playbacks: list[_AudioPlayback]) -> None:
        """Tell the provider how much of each cut-off assistant item was actually sent.

        Without this the provider's conversation keeps the full assistant message,
        so the model assumes the caller heard all of it.
        """
        for playback in playbacks:
            item_id = playback.played_item_id or playback.item_id
            if not item_id:
                continue
            if playback.played_item_id:
                content_index = playback.played_content_index
                audio_end_ms = int(playback.played_pcm_bytes * 1000 / (OPENAI_SAMPLE_RATE * 2))
            else:
                content_index = playback.content_index
                audio_end_ms = 0
            try:
                await conn.conversation.item.truncate(
                    item_id=item_id, content_index=content_index, audio_end_ms=audio_end_ms
                )
                logger.debug(f"Truncated assistant item {item_id} at {audio_end_ms}ms")
            except Exception as e:
                logger.warning(f"Failed to truncate assistant item {item_id}: {e}")

    def _log_assistant_output(
        self, playback: _AudioPlayback | None, content: str, timestamp_ms: str, *, interrupted: bool = False
    ) -> None:
        """Write one assistant turn to the audit and framework logs."""
        if playback is not None:
            playback.deferred_output = None
            playback.output_logged = True
        if interrupted:
            heard = self._estimate_heard_text(playback, content)
            text = f"{heard} [interrupted]" if heard else "[interrupted]"
            self.audit_log.append_assistant_output(text, timestamp_ms=timestamp_ms)
            if self._fw_log:
                self._fw_log.s2s_transcript(text)
                self._fw_log.turn_end(was_interrupted=True)
            logger.debug(f"Logged interrupted assistant response: {text[:60]}...")
            return
        self.audit_log.append_assistant_output(content, timestamp_ms=timestamp_ms)
        if self._fw_log:
            self._fw_log.llm_response(content)
            self._fw_log.turn_end(was_interrupted=False)
        logger.debug(f"Logged assistant response: '{content[:60]}...'")

    @staticmethod
    def _estimate_heard_text(playback: _AudioPlayback | None, content: str) -> str:
        """Approximate the part of `content` the caller heard before playback was cut off.

        The provider gives no alignment between text and audio, so this assumes
        speech is spread evenly over the audio after the leading silence: the
        heard fraction of the text is the fraction of speech bytes sent. The cut
        is moved back to a word boundary and then one more word is dropped, so
        the estimate errs towards logging less than was heard rather than words
        the caller never heard. Without audio information the full text is kept.
        """
        if playback is None or playback.queued_pcm_bytes <= 0:
            return content
        if playback.sent_pcm_bytes >= playback.queued_pcm_bytes:
            return content
        speech_bytes = playback.queued_pcm_bytes - LEADING_SILENCE_PCM_BYTES
        sent_speech_bytes = playback.sent_pcm_bytes - LEADING_SILENCE_PCM_BYTES
        if speech_bytes <= 0 or sent_speech_bytes <= 0:
            return ""
        cut = len(content) * sent_speech_bytes // speech_bytes
        if cut <= 0:
            return ""
        head = content[:cut]
        if not content[cut].isspace():
            # Drop the partial word at the cut.
            head = head.rpartition(" ")[0]
        # Drop one more word to stay on the side of what was actually heard.
        return head.strip().rpartition(" ")[0].strip()

    def _flush_deferred_outputs_on_disconnect(self) -> None:
        """Log text whose playback never finished because the session ended.

        The caller hanging up is not a barge-in, so only the heard part is
        logged, as an ordinary uninterrupted turn. Nothing is logged if no
        speech was heard.
        """
        for playback in self._audio_playbacks.values():
            if not playback.deferred_output:
                continue
            deferred = playback.deferred_output
            heard = self._estimate_heard_text(playback, deferred.content)
            if heard:
                self._log_assistant_output(playback, heard, deferred.timestamp_ms)
            else:
                playback.deferred_output = None
                playback.output_logged = True

    def _record_played_audio(self, pcm16: bytes) -> None:
        """Record only audio sent to EVA, so discarded speech is not graded."""
        user_recently_active = (time.monotonic() - self._last_user_audio_mono) <= USER_ACTIVE_GUARD_S
        if not self._user_speaking and not user_recently_active:
            sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
        self.assistant_audio_buffer.extend(pcm16)
        self._bot_speaking = any(
            not playback.interrupted and playback.pending_frames > 0 for playback in self._audio_playbacks.values()
        )

    # ── User audio forwarding (Twilio WS -> OpenAI) ──────────────────

    async def _forward_user_audio(self, websocket: WebSocket, conn: Any) -> None:
        """Read Twilio media frames and forward audio to OpenAI Realtime."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                event_type = data.get("event")

                if event_type == "start":
                    # Twilio stream start - extract streamSid
                    self._stream_sid = data.get("start", {}).get("streamSid", self.conversation_id)
                    logger.debug(f"Twilio stream started: streamSid={self._stream_sid}")
                    continue

                if event_type == "stop":
                    logger.debug("Twilio stream stopped")
                    break

                if event_type == "user_speech_start":
                    # Timestamp from audio_interface when user audio actually started
                    self._audio_interface_speech_start_ts = data.get("timestamp_ms")
                    logger.debug(f"User speech start timestamp received: {self._audio_interface_speech_start_ts}")
                    continue

                if event_type != "media":
                    continue

                # Extract raw mulaw audio bytes
                mulaw_bytes = parse_twilio_media_message(raw)
                if mulaw_bytes is None:
                    continue

                # Convert 8kHz mulaw -> 24kHz PCM16
                pcm16_24k = mulaw_8k_to_pcm16_24k(mulaw_bytes)

                asst_before = len(self.assistant_audio_buffer)
                synced = 0
                if not self._bot_speaking:
                    sync_target = len(self.user_audio_buffer)
                    sync_buffer_to_position(self.assistant_audio_buffer, sync_target)
                    synced = len(self.assistant_audio_buffer) - asst_before
                self.user_audio_buffer.extend(pcm16_24k)
                self._last_user_audio_mono = time.monotonic()
                self._user_frame_count += 1
                if self._user_frame_count % 50 == 0:
                    diff = len(self.user_audio_buffer) - len(self.assistant_audio_buffer)
                    diff_ms = diff / (OPENAI_SAMPLE_RATE * 2) * 1000
                    logger.debug(
                        f"[ALIGN DEBUG] user_frame #{self._user_frame_count}: "
                        f"user={len(self.user_audio_buffer)} asst={len(self.assistant_audio_buffer)} "
                        f"diff={diff}({diff_ms:.0f}ms) bot_spk={self._bot_speaking} "
                        f"usr_spk={self._user_speaking} added={len(pcm16_24k)} synced={synced}"
                    )

                # Encode as base64 and send to OpenAI
                audio_b64 = base64.b64encode(pcm16_24k).decode("ascii")
                await conn.input_audio_buffer.append(audio=audio_b64)

        except WebSocketDisconnect:
            logger.debug("Twilio WebSocket disconnected")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error forwarding user audio: {e}", exc_info=True)

    # ── OpenAI event processing ───────────────────────────────────────

    async def _process_openai_events(
        self, conn: Any, websocket: WebSocket, audio_output_queue: asyncio.Queue[_OutputAudioChunk]
    ) -> None:
        """Process events from the OpenAI Realtime connection."""
        try:
            async for event in conn:
                try:
                    await self._handle_openai_event(event, conn, websocket, audio_output_queue)
                except Exception as e:
                    logger.error(f"Error handling event {getattr(event, 'type', '?')}: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in OpenAI event loop: {e}", exc_info=True)

    async def _handle_openai_event(
        self, event: Any, conn: Any, websocket: WebSocket, audio_output_queue: asyncio.Queue[_OutputAudioChunk]
    ) -> None:
        """Dispatch a single OpenAI Realtime event."""
        event_type = getattr(event, "type", "")

        match event_type:
            case "session.created":
                logger.info(f"{self._service_name} session created")

            case "session.updated":
                logger.debug(f"{self._service_name} session updated")

            case "response.created":
                response = getattr(event, "response", None)
                self._active_response_id = getattr(response, "id", None)
                self._playback_response_id = self._active_response_id or ""
                if self._playback_response_id:
                    self._audio_playbacks.setdefault(self._playback_response_id, _AudioPlayback())
                else:
                    # Some compatible providers omit response IDs. Only a new
                    # response.created event may reopen their interrupted output.
                    self._audio_playbacks[""] = _AudioPlayback()

            case "input_audio_buffer.speech_started":
                # Clear first so a cut-off assistant turn is logged before the
                # new user turn starts.
                cut_playbacks = await self._clear_audio_playback(audio_output_queue)
                await self._on_speech_started(event)
                await self._truncate_interrupted_items(conn, cut_playbacks)

            case "output_audio_buffer.cleared":
                await self._clear_audio_playback(audio_output_queue, getattr(event, "response_id", None))

            case "input_audio_buffer.speech_stopped":
                await self._on_speech_stopped(event)

            case "conversation.item.input_audio_transcription.completed":
                await self._on_transcription_completed(event)

            case "conversation.item.input_audio_transcription.delta":
                logger.debug(f"Transcription delta: {getattr(event, 'delta', '')}")

            case "conversation.item.input_audio_transcription.failed":
                error_info = getattr(event, "error", "")
                logger.warning(f"Transcription failed: {error_info}")
                # Gracefully handle transcription failure (e.g. API key lacks
                # whisper-1 access).  If a user turn was active but has no
                # transcript yet, record a placeholder so the turn is not lost.
                user_turn = self._user_turn_for_event(event)
                if user_turn and not user_turn.flushed:
                    self._write_missing_transcript_placeholder(user_turn)

            case "response.output_audio.delta":
                await self._on_audio_delta(event, audio_output_queue)

            case "response.output_audio_transcript.delta":
                if not self._audio_playback(event).interrupted:
                    self._on_transcript_delta(event)

            case "response.output_audio_transcript.done":
                if not self._audio_playback(event).interrupted:
                    self._on_transcript_done(event)

            case "response.output_item.added" | "response.output_item.done":
                self._on_response_output_item(event)

            case "response.function_call_arguments.done":
                self._on_function_call_done(event)

            case "response.done":
                response = getattr(event, "response", None)
                if getattr(response, "status", None) in {"cancelled", "failed"}:
                    await self._clear_audio_playback(audio_output_queue, getattr(response, "id", None))
                await self._on_response_done(event, conn)

            case "error":
                error_data = getattr(event, "error", None)
                logger.error(f"{self._service_name} error: {error_data}")

            case _:
                logger.debug(f"Unhandled {self._service_name} event: {event_type}")

    # ── Event handlers ────────────────────────────────────────────────

    @staticmethod
    def _event_item_id(event: Any) -> str:
        """Return the conversation item ID carried by a speech/ASR event."""
        return getattr(event, "item_id", "") or ""

    def _new_user_turn(self, **fields: Any) -> _UserTurnRecord:
        """Create a user turn with a fresh audit-log turn id."""
        self._next_turn_id += 1
        return _UserTurnRecord(turn_id=self._next_turn_id, **fields)

    def _write_missing_transcript_placeholder(self, turn: _UserTurnRecord) -> None:
        """Record that a user turn happened but its transcript is unavailable.

        The entry carries the turn id so a late transcription replaces it instead of
        adding a second entry for the same spoken turn.
        """
        timestamp_ms = turn.speech_started_wall_ms or None
        self.audit_log.append_user_input(
            "[user speech - transcription unavailable]", timestamp_ms=timestamp_ms, turn_id=turn.turn_id
        )
        turn.flushed = True
        turn.placeholder_written = True

    def _flush_user_turn_without_transcript(self) -> None:
        """Write a placeholder for a spoken user turn that never received a transcript.

        Apriel may emit no input transcription at all, in which case nothing else
        writes the turn. Skipped when a transcript is buffered or the turn never
        recorded a speech start.
        """
        turn = self._user_turn
        if turn and not turn.flushed and not turn.transcript and turn.speech_started_wall_ms:
            logger.warning("User turn ended without a transcription; recording a placeholder")
            self._write_missing_transcript_placeholder(turn)

    def _user_turn_for_event(self, event: Any, *, create: bool = False) -> _UserTurnRecord | None:
        """Resolve a speech/ASR event to its user turn without relying on ASR timing.

        Realtime input transcription can finish after a later speech event, and
        compatible providers such as Apriel may emit no input transcription at
        all. Conversation item IDs provide the provider-neutral association.
        """
        item_id = self._event_item_id(event)
        if not item_id:
            return self._user_turn

        if turn := self._user_turns_by_item_id.get(item_id):
            return turn

        if self._user_turn and not self._user_turn.item_id:
            self._user_turn.item_id = item_id
            self._user_turns_by_item_id[item_id] = self._user_turn
            return self._user_turn

        if not create:
            return None

        turn = self._new_user_turn(item_id=item_id)
        self._user_turns_by_item_id[item_id] = turn
        if self._user_turn is None:
            self._user_turn = turn
        return turn

    async def _on_speech_started(self, event: Any) -> None:
        """Handle input_audio_buffer.speech_started."""
        self._user_speaking = True
        diff = len(self.user_audio_buffer) - len(self.assistant_audio_buffer)
        diff_ms = diff / (OPENAI_SAMPLE_RATE * 2) * 1000
        logger.debug(
            f"[ALIGN DEBUG] speech_started: user={len(self.user_audio_buffer)} "
            f"asst={len(self.assistant_audio_buffer)} diff={diff}({diff_ms:.0f}ms) "
            f"bot_spk={self._bot_speaking}"
        )
        wall = _wall_ms()

        # If the assistant was still generating, flush the interrupted response.
        # Mark its playback as logged so a later response.done does not log it again.
        if self._assistant_state.responding and self._assistant_state.transcript_parts:
            self._log_assistant_output(
                self._audio_playbacks.get(self._playback_response_id),
                "".join(self._assistant_state.transcript_parts),
                self._assistant_state.first_audio_wall_ms or wall,
                interrupted=True,
            )
            self._assistant_state = _AssistantResponseState()

        # The assistant already replied to the previous turn and no transcript has
        # arrived, so none is coming: record a placeholder rather than losing the turn.
        if self._user_turn and self._user_turn.answered:
            self._flush_user_turn_without_transcript()

        item_id = self._event_item_id(event)
        known_turn = self._user_turns_by_item_id.get(item_id) if item_id else None
        starts_new_turn = known_turn is None and (
            not self._user_turn
            or self._user_turn.flushed
            or (bool(item_id) and item_id != self._user_turn.item_id)
            # Fallback for compatible providers that omit item_id: a VAD stop
            # closes the prior speech item even when no ASR event follows.
            or (not item_id and bool(self._user_turn.speech_stopped_wall_ms))
        )

        if starts_new_turn:
            # Use timestamp from audio_interface if available (source of truth)
            start_ts = self._audio_interface_speech_start_ts or wall
            self._user_turn = self._new_user_turn(item_id=item_id, speech_started_wall_ms=start_ts)
            if item_id:
                self._user_turns_by_item_id[item_id] = self._user_turn
            if self._fw_log:
                self._fw_log.turn_start(timestamp_ms=int(start_ts))
            logger.debug(
                f"Speech started at {start_ts} (new turn, from_audio_interface={self._audio_interface_speech_start_ts is not None})"
            )
        elif known_turn is not None:
            self._user_turn = known_turn
            logger.debug(f"Speech started at {wall} (continuing item {item_id})")
        else:
            logger.debug(f"Speech started at {wall} (continuing existing turn)")
        self._audio_interface_speech_start_ts = None  # Reset for next turn

    async def _on_speech_stopped(self, event: Any) -> None:
        """Handle input_audio_buffer.speech_stopped."""
        self._user_speaking = False
        diff = len(self.user_audio_buffer) - len(self.assistant_audio_buffer)
        diff_ms = diff / (OPENAI_SAMPLE_RATE * 2) * 1000
        logger.debug(
            f"[ALIGN DEBUG] speech_stopped: user={len(self.user_audio_buffer)} "
            f"asst={len(self.assistant_audio_buffer)} diff={diff}({diff_ms:.0f}ms) "
            f"bot_spk={self._bot_speaking}"
        )
        wall = _wall_ms()
        user_turn = self._user_turn_for_event(event, create=True)
        if user_turn:
            self._user_turn = user_turn
            user_turn.speech_stopped_wall_ms = wall

        logger.debug(f"Speech stopped at {wall}")

    async def _on_transcription_completed(self, event: Any) -> None:
        """Handle conversation.item.input_audio_transcription.completed."""
        transcript = getattr(event, "transcript", "") or ""
        transcript = transcript.strip()

        if not transcript:
            logger.debug("Empty transcription, skipping")
            return

        timestamp_ms = None
        user_turn = self._user_turn_for_event(event, create=True)
        if user_turn:
            timestamp_ms = user_turn.speech_started_wall_ms or None
            user_turn.transcript = transcript
            user_turn.flushed = True

        if user_turn and user_turn.placeholder_written and user_turn.turn_id is not None:
            # A placeholder was already written for this turn; replace it rather than
            # recording the same spoken turn twice.
            user_turn.placeholder_written = False
            if self.audit_log.update_user_input_by_turn_id(user_turn.turn_id, transcript):
                logger.debug(f"Replaced placeholder with late transcription: {transcript[:60]}...")
                return

        self.audit_log.append_user_input(
            transcript, timestamp_ms=timestamp_ms, turn_id=user_turn.turn_id if user_turn else None
        )
        logger.debug(f"User transcription: {transcript}...")

    async def _on_audio_delta(self, event: Any, audio_output_queue: asyncio.Queue[_OutputAudioChunk]) -> None:
        """Handle response.audio.delta - assistant audio chunk."""
        playback = self._audio_playback(event)
        if playback.interrupted:
            return
        delta_b64 = getattr(event, "delta", "") or ""
        if not delta_b64:
            return

        pcm16_bytes = base64.b64decode(delta_b64)
        item_id = getattr(event, "item_id", "") or ""
        content_index = getattr(event, "content_index", 0) or 0
        if item_id and not playback.item_id:
            playback.item_id = item_id
            playback.content_index = content_index

        if self._assistant_state.first_audio_wall_ms is None:
            self._assistant_state.first_audio_wall_ms = _wall_ms()
            self._assistant_state.responding = True
            self._bot_speaking = True

            # Record model response latency: user speech end → first audio chunk.
            # speech_stopped_wall_ms may be absent on the initial greeting turn.
            if self._user_turn and self._user_turn.speech_stopped_wall_ms and self._metrics_log:
                latency_ms = int(self._assistant_state.first_audio_wall_ms) - int(
                    self._user_turn.speech_stopped_wall_ms
                )
                if 0 < latency_ms < 30_000:
                    self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)

        # Convert 24kHz PCM16 -> 8kHz mulaw and enqueue for real-time pacing.
        # _pace_audio_output owns the timing loop so this method returns immediately,
        # allowing the OpenAI event loop to process the next event without delay.
        try:
            mulaw_bytes = pcm16_24k_to_mulaw_8k(pcm16_bytes)
            for offset in range(0, len(mulaw_bytes), MULAW_CHUNK_SIZE):
                end = min(offset + MULAW_CHUNK_SIZE, len(mulaw_bytes))
                # One 8 kHz mu-law sample represents three 24 kHz PCM16
                # samples (six bytes). Preserve the final resampling remainder.
                pcm_end = end * 6 if end < len(mulaw_bytes) else len(pcm16_bytes)
                pcm16_chunk = pcm16_bytes[offset * 6 : pcm_end]
                audio_output_queue.put_nowait(
                    _OutputAudioChunk(playback, mulaw_bytes[offset:end], pcm16_chunk, item_id, content_index)
                )
                playback.pending_frames += 1
                playback.queued_pcm_bytes += len(pcm16_chunk)
        except Exception as e:
            logger.error(f"Error converting audio for output queue: {e}")

    def _on_transcript_delta(self, event: Any) -> None:
        """Handle response.audio_transcript.delta - incremental assistant text."""
        delta = getattr(event, "delta", "") or ""
        if delta:
            self._assistant_state.transcript_parts.append(delta)

    def _on_transcript_done(self, event: Any) -> None:
        """Handle response.audio_transcript.done - full assistant transcript.

        This is the most reliable source of what the model actually said.
        Store it so _on_response_done can use it if delta accumulation failed.
        """
        transcript = getattr(event, "transcript", "") or ""
        if transcript:
            self._assistant_state.transcript_done_text = transcript.strip()
            logger.debug(f"Assistant transcript done: {transcript}...")

    def _on_response_output_item(self, event: Any) -> None:
        """Record function call metadata from response.output_item events.

        Compatible providers may omit the name from
        response.function_call_arguments.done, so the output item is also used
        as a source. Registering is idempotent.
        """
        item = getattr(event, "item", None)
        if not item or getattr(item, "type", "") != "function_call":
            return

        call = self._register_tool_call(
            call_id=getattr(item, "call_id", ""),
            response_id=getattr(event, "response_id", "") or self._active_response_id or "",
            name=getattr(item, "name", ""),
            arguments_raw=getattr(item, "arguments", None),
        )
        if call:
            self._assistant_state.has_function_calls = True

    def _on_function_call_done(self, event: Any) -> None:
        """Record a function call's complete arguments; it runs at response.done."""
        call = self._register_tool_call(
            call_id=getattr(event, "call_id", ""),
            response_id=getattr(event, "response_id", "") or self._active_response_id or "",
            name=getattr(event, "name", ""),
            arguments_raw=getattr(event, "arguments", "{}") or "{}",
        )
        if call:
            self._assistant_state.has_function_calls = True

    async def _on_response_done(self, event: Any, conn: Any | None = None) -> None:
        """Record a finished response and, if it completed, run its tools and continue."""
        response_id, tool_calls = self._take_response_tool_calls(event)
        response = getattr(event, "response", None)
        status = getattr(response, "status", None)

        await self._record_response_done(event)

        if self._active_response_id == response_id:
            self._active_response_id = None

        if not tool_calls:
            return

        if status != "completed":
            # The tools have not run, so the scenario state stays consistent
            # with what the model knows.
            logger.warning(
                f"Not running {len(tool_calls)} tool call(s) from {self._service_name} "
                f"response {response_id or '<unknown>'} with status={status!r}"
            )
            return

        if conn is None:
            logger.error("Cannot return Realtime tool results without an active connection")
            return

        await self._run_tool_batch(response_id, tool_calls, conn, create_response=not self._user_speaking)

    async def _record_response_done(self, event: Any) -> None:
        """Handle response.done - assistant response complete.

        Following the pipecat InstrumentedRealtimeLLMService pattern:
        - Only call append_assistant_output() (no append_llm_call)
        - Token usage goes to pipecat_metrics.jsonl only
        """
        # Extract usage metrics
        response = getattr(event, "response", None)
        if response:
            usage = getattr(response, "usage", None)
            if usage and self._metrics_log:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                self._metrics_log.write_token_usage(
                    processor=self._metrics_processor_name,
                    model=self._model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )

        # Skip cancelled responses - these were interrupted and not fully spoken
        if response and getattr(response, "status", None) == "cancelled":
            logger.debug("response_done: cancelled response, skipping transcript entry")
            self._reset_assistant_state()
            return

        has_function_calls = self._response_has_function_calls(event) or self._assistant_state.has_function_calls

        # Build transcript text from best available source:
        # 1. response.audio_transcript.done text (most reliable)
        # 2. Accumulated response.audio_transcript.delta parts
        # 3. Text extracted from response.done output items
        content = self._assistant_state.transcript_done_text
        if not content:
            content = "".join(self._assistant_state.transcript_parts).strip()
        if not content:
            content = self._extract_response_text(event)

        audio_was_streamed = self._assistant_state.first_audio_wall_ms is not None

        if self._user_turn and (content or audio_was_streamed) and not has_function_calls:
            self._user_turn.answered = True

        # Skip tool-call-only responses (nothing spoken)
        if not content and has_function_calls:
            logger.debug("response_done: tool-call-only response, skipping assistant entry")
            self._reset_assistant_state()
            return

        # Skip mixed responses where audio was not streamed
        if content and not audio_was_streamed and has_function_calls:
            logger.debug(f"response_done: mixed response with no audio, skipping: '{content[:60]}...'")
            self._reset_assistant_state()
            return

        # If audio was streamed but we have no transcript at all, skip rather
        # than pollute the audit log with a placeholder.  The audio recording
        # still captures what was said.
        if not content and audio_was_streamed:
            logger.debug("response_done: audio streamed but no transcript available, skipping text entry")
            self._reset_assistant_state()
            return

        if not content:
            # No audio, no text, no function calls — nothing to log
            self._reset_assistant_state()
            return

        # Log assistant output (single entry — no append_llm_call). Generation
        # usually finishes long before paced playback, so while frames are still
        # queued the entry is held until playback drains (logged as spoken) or
        # is cut off (logged as interrupted).
        timestamp = self._assistant_state.first_audio_wall_ms or _wall_ms()
        response_id = getattr(response, "id", "") or self._playback_response_id
        playback = self._audio_playbacks.get(response_id)
        if playback is None:
            self._log_assistant_output(None, content, timestamp)
        elif playback.output_logged:
            logger.debug("response_done: interrupted response already logged, skipping")
        elif playback.interrupted:
            self._log_assistant_output(playback, content, timestamp, interrupted=True)
        elif playback.pending_frames > 0:
            playback.deferred_output = _DeferredAssistantOutput(content, timestamp)
            logger.debug(f"response_done: holding '{content[:60]}...' until playback drains")
        else:
            self._log_assistant_output(playback, content, timestamp)
        self._reset_assistant_state()

    def _register_tool_call(
        self,
        *,
        call_id: str,
        response_id: str,
        name: str,
        arguments_raw: Any = None,
    ) -> _PendingToolCall | None:
        """Create or enrich a pending tool call from a streamed event."""
        if not call_id:
            logger.warning(f"Ignoring {self._service_name} function call without call_id")
            return None

        call = self._pending_tool_calls.get(call_id)
        if call is None:
            call = _PendingToolCall(call_id=call_id)
            self._pending_tool_calls[call_id] = call

        if response_id:
            call.response_id = response_id
        if name:
            call.name = name
        if arguments_raw is not None:
            call.arguments = self._parse_tool_arguments(arguments_raw)
        return call

    @staticmethod
    def _parse_tool_arguments(arguments_raw: Any) -> dict[str, Any]:
        """Parse function arguments while preserving the previous empty-object fallback."""
        if isinstance(arguments_raw, dict):
            return arguments_raw
        if not isinstance(arguments_raw, str):
            return {}
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            return {}
        return arguments if isinstance(arguments, dict) else {}

    async def _execute_pending_tool(self, call: _PendingToolCall) -> Any:
        """Execute and audit one pending tool, converting unexpected failures to output."""
        logger.info(f"Tool call: {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        try:
            result = await self.execute_tool(call.name, call.arguments)
        except asyncio.CancelledError:
            # The session is shutting down mid-tool. execute_tool already
            # recorded the call, so complete the audit pair before re-raising.
            self.audit_log.append_tool_response(
                call.name,
                {"status": "error", "error_type": "cancelled", "message": "Tool execution was cancelled"},
            )
            raise
        except Exception as exc:
            logger.error(f"Tool execution failed for {call.name}: {exc}", exc_info=True)
            result = {
                "status": "error",
                "error_type": "execution_error",
                "message": f"Tool execution failed: {exc}",
            }
            # execute_tool records the call before invoking the handler. Complete
            # the audit pair when the handler itself raises unexpectedly.
            self.audit_log.append_tool_response(call.name, result)

        if self._fw_log:
            self._fw_log.write(
                "tool_call",
                {
                    "frame": "tool_call",
                    "tool_name": call.name,
                    "arguments": call.arguments,
                    "result": result,
                },
            )
        return result

    def _take_response_tool_calls(self, event: Any) -> tuple[str, list[_PendingToolCall]]:
        """Return pending calls in response output order and remove them from tracking."""
        response = getattr(event, "response", None)
        response_id = getattr(response, "id", "") or self._active_response_id or ""
        ordered: list[_PendingToolCall] = []
        seen_call_ids: set[str] = set()

        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", "") != "function_call":
                continue
            call = self._register_tool_call(
                call_id=getattr(item, "call_id", ""),
                response_id=response_id,
                name=getattr(item, "name", ""),
                arguments_raw=getattr(item, "arguments", None),
            )
            if not call:
                continue
            ordered.append(call)
            seen_call_ids.add(call.call_id)

        # Compatible providers may omit output items from response.done. Include
        # calls already associated with this response in their arrival order.
        for call in self._pending_tool_calls.values():
            if call.call_id in seen_call_ids:
                continue
            if call.response_id == response_id or (not call.response_id and not response_id):
                ordered.append(call)
                seen_call_ids.add(call.call_id)

        for call in ordered:
            self._pending_tool_calls.pop(call.call_id, None)
        return response_id, ordered

    async def _run_tool_batch(
        self,
        response_id: str,
        tool_calls: list[_PendingToolCall],
        conn: Any,
        *,
        create_response: bool = True,
    ) -> None:
        """Run a completed response's tools in order, return all results, then request one continuation.

        When create_response is False (the user is speaking), the outputs are still sent
        and server VAD creates the reply once the user's turn ends.
        """
        results: list[Any] = []
        for call in tool_calls:
            if not call.name:
                logger.error(f"Cannot execute Realtime tool call {call.call_id}: function name was unavailable")
                results.append(
                    {
                        "status": "error",
                        "error_type": "function_not_found",
                        "message": "Function name was unavailable in the Realtime response",
                    }
                )
                continue
            results.append(await self._execute_pending_tool(call))

        for call, result in zip(tool_calls, results, strict=True):
            await conn.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

        if not create_response:
            logger.debug(
                f"Returned {len(tool_calls)} tool result(s) for {self._service_name} "
                f"response {response_id or '<unknown>'}; user is speaking, leaving the reply to server VAD"
            )
            return

        await conn.response.create()
        logger.debug(
            f"Returned {len(tool_calls)} tool result(s) for {self._service_name} "
            f"response {response_id or '<unknown>'} and requested one continuation"
        )

    def _reset_tool_tracking(self) -> None:
        """Reset per-session tool bookkeeping."""
        self._active_response_id = None
        self._pending_tool_calls = {}

    # ── Helpers ───────────────────────────────────────────────────────

    def _reset_assistant_state(self) -> None:
        """Clear accumulated assistant response state."""
        audio_was_streamed = self._assistant_state.first_audio_wall_ms is not None
        diff = len(self.user_audio_buffer) - len(self.assistant_audio_buffer)
        diff_ms = diff / (OPENAI_SAMPLE_RATE * 2) * 1000
        logger.debug(
            f"[ALIGN DEBUG] reset_state: user={len(self.user_audio_buffer)} "
            f"asst={len(self.assistant_audio_buffer)} diff={diff}({diff_ms:.0f}ms) "
            f"audio_streamed={audio_was_streamed} bot_spk={self._bot_speaking}"
        )
        # Generation completion is independent of the paced playback tail.
        self._assistant_state = _AssistantResponseState()

    def _build_realtime_tools(self) -> list[dict]:
        """Convert agent tools to OpenAI Realtime session tool format.

        The Realtime API session.tools expects a flat structure:
        {type, name, description, parameters: {type, properties, required}}
        """
        tools: list[dict] = []
        if not self.agent.tools:
            return tools

        for tool in self.agent.tools:
            tools.append(
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
        return tools

    @staticmethod
    def _response_has_function_calls(event: Any) -> bool:
        """Return True if the response.done event contains function_call outputs."""
        response = getattr(event, "response", None)
        if not response:
            return False
        output_items = getattr(response, "output", None) or []
        return any(getattr(item, "type", "") == "function_call" for item in output_items)

    @staticmethod
    def _extract_response_text(event: Any) -> str:
        """Extract text content from response.done output items."""
        response = getattr(event, "response", None)
        if not response:
            return ""

        output_items = getattr(response, "output", None) or []
        text_parts: list[str] = []

        for item in output_items:
            content_list = getattr(item, "content", None) or []
            for part in content_list:
                part_type = getattr(part, "type", "")
                if part_type in ("audio", "text"):
                    transcript = getattr(part, "transcript", None) or getattr(part, "text", None) or ""
                    if transcript:
                        text_parts.append(transcript)

        return "".join(text_parts).strip()
