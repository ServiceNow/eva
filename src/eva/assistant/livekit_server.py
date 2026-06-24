"""LiveKitAssistantServer — an EVA assistant server (``framework=livekit``).

EVA's orchestrator constructs one of these per scenario record, calls
``start()`` to bring up a Twilio-framed WebSocket on a port from the pool,
EVA's user-simulator dials it, and we proxy audio + lifecycle to a real
LiveKit voice agent (the chariot intake agent, running in docker).

Unlike the in-tree servers (OpenAI Realtime, Gemini, …) the "model" under
test is not an API we call directly — it's an agent that joins a LiveKit
room. So per session we:

  1. Accept EVA's Twilio WS, read the ``start`` event for ``streamSid``.
  2. Create a unique LiveKit room, dispatch the agent into it, and join as a
     participant carrying the SIP attributes the agent reads to take its
     inbound-call code path.
  3. Publish a 24 kHz mono track sourced from EVA's mulaw stream, subscribe to
     the agent's track (resampled to 24 kHz), and re-encode it back to Twilio
     mulaw frames for EVA.
  4. Capture the agent's transcriptions (LiveKit ``lk.transcription`` text
     streams) into EVA's AuditLog so the metrics pipeline has a transcript.
  5. On EVA ``stop`` / disconnect, tear the room down.

Audio runs at 24 kHz throughout (EVA's ``SAMPLE_RATE``) so we reuse
``audio_bridge``'s mulaw<->pcm helpers and the inherited audio recording /
mixing in ``AbstractAssistantServer``.

Configuration: LiveKit connection comes from ``LIVEKIT_URL`` /
``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET`` (env, standard for any LiveKit
deployment). Agent-specific values (agent name, SIP attributes, caller/trunk
numbers) are read from ``pipeline_config.s2s_params`` when present, else from
env vars, else chariot defaults — see ``_resolve_config``.

Dev / staging only — never run with prod LiveKit credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import uvicorn
from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.utils.logging import get_logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from livekit import api, rtc

logger = get_logger(__name__)

# EVA runs PCM at 24 kHz (base_server.SAMPLE_RATE); we match it so the inherited
# audio mixing/recording and audio_bridge helpers work without resampling.
LIVEKIT_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1
# 20 ms @ 24 kHz mono int16 = 480 samples = 960 bytes. Matches Twilio's 20 ms cadence.
_FRAME_DURATION_MS = 20
_FRAME_SAMPLES = LIVEKIT_SAMPLE_RATE * _FRAME_DURATION_MS // 1000  # 480
_FRAME_BYTES = _FRAME_SAMPLES * 2  # 960
# Twilio media frames are 160 bytes (20 ms @ 8 kHz mulaw, 1 byte/sample).
_MULAW_CHUNK_SIZE = 160
_MULAW_CHUNK_DURATION_S = 0.02

# Chariot defaults — overridable via s2s_params or env (see _resolve_config).
_DEFAULT_AGENT_NAME = "chariot-agent"
_DEFAULT_FROM_NUMBER = "+15555550100"
_DEFAULT_TO_NUMBER = "+15555550101"

# LiveKit Agents forwards transcriptions over a text stream on this topic, one
# stream per speech segment (see livekit.agents.voice.room_io._output). Mirrors
# livekit.agents.types.{TOPIC_TRANSCRIPTION, ATTRIBUTE_*}.
_TRANSCRIPTION_TOPIC = "lk.transcription"
_ATTR_SEGMENT_ID = "lk.segment_id"
# The track that was transcribed — i.e. the speaker's own audio track. This is
# how we tell the agent's speech from the caller's: the stream's publisher is
# always the agent, but this attribute points at whoever actually spoke.
_ATTR_TRANSCRIBED_TRACK_ID = "lk.transcribed_track_id"


def _wall_ms() -> int:
    return int(round(time.time() * 1000))


class LiveKitAssistantServer(AbstractAssistantServer):
    """Bridge between EVA's Twilio-WS user simulator and a real LiveKit agent."""

    _service_name: str = "LiveKit"
    _metrics_processor_name: str = "livekit"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audio_sample_rate = LIVEKIT_SAMPLE_RATE

        # Per-session LiveKit state — recreated each connect.
        self._room: rtc.Room | None = None
        self._lk_api: api.LiveKitAPI | None = None
        self._room_name: str | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._agent_subscriber_task: asyncio.Task | None = None
        self._stream_sid: str | None = None
        self._ws: WebSocket | None = None
        # Outbound (agent→EVA) mulaw chunks, drained by a real-time-paced sender.
        self._outbound_mulaw: asyncio.Queue[bytes] | None = None
        self._outbound_sender_task: asyncio.Task | None = None
        # Inbound (EVA→agent) PCM frames, drained by a separate task. Kept off
        # the WebSocket receive loop because AudioSource.capture_frame paces at
        # real time and blocks when full — blocking it inline stalls socket
        # reads, starves the keepalive PONG, and drops EVA's connection (1011).
        self._inbound_frames: asyncio.Queue[bytes] | None = None
        self._inbound_capture_task: asyncio.Task | None = None

        # Track alignment: keep user/assistant PCM buffers positionally synced so
        # the mixed recording isn't temporally skewed (mirrors the OpenAI server).
        self._user_speaking = False
        self._bot_speaking = False
        # Latency: user speech-stop → first agent audio chunk.
        self._user_speech_stopped_wall_ms: int | None = None

        # Transcript capture. The sid of the audio track we publish (the EVA
        # caller) labels caller turns vs the agent. Latest text per segment id
        # wins, so interim updates collapse to the final utterance; flushed to
        # the audit log at session teardown.
        self._user_track_sid: str | None = None
        self._transcript_segments: dict[str, dict[str, str]] = {}
        self._transcription_tasks: set[asyncio.Task] = set()
        self._fwlog_turn_counter = 0

    # ---- Config resolution ----------------------------------------------

    def _cfg(self, key: str, env: str, default: Any = None) -> Any:
        """Read a setting from s2s_params, falling back to env, then default."""
        params = self.pipeline_config.s2s_params or {}
        val = params.get(key)
        if val not in (None, ""):
            return val
        val = os.environ.get(env)
        return val if val not in (None, "") else default

    # ---- Lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            logger.warning(f"{self._service_name} server already running")
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
        logger.info(
            f"{self._service_name} server started on ws://localhost:{self.port}"
        )

    async def _shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._cleanup_livekit()
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5)
            except asyncio.TimeoutError:
                self._server_task.cancel()
        logger.info(f"{self._service_name} server stopped on port {self.port}")

    # ---- Per-session glue -----------------------------------------------

    async def _handle_session(self, ws: WebSocket) -> None:
        logger.info(f"[{self.conversation_id}] EVA WebSocket connected")
        self._ws = ws
        self._stream_sid = None
        self._user_speaking = False
        self._bot_speaking = False
        self._user_speech_stopped_wall_ms = None
        self._fwlog_turn_counter = 0
        self._outbound_mulaw = asyncio.Queue(maxsize=200)
        self._outbound_sender_task = asyncio.create_task(self._send_outbound_to_eva())
        self._inbound_frames = asyncio.Queue(maxsize=200)
        self._inbound_capture_task = asyncio.create_task(self._capture_inbound_to_room())
        try:
            await self._spin_up_livekit_room()
            audio_in_buf = bytearray()

            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")

                if event == "start":
                    self._stream_sid = msg.get("streamSid") or msg.get(
                        "start", {}
                    ).get("streamSid", self.conversation_id)
                    logger.info(
                        f"[{self.conversation_id}] stream started: {self._stream_sid}"
                    )
                    # The agent greets via its own on_enter / TTS, which appears
                    # on the LiveKit track — nothing to inject from our side.

                elif event == "media":
                    mulaw = parse_twilio_media_message(raw)
                    if not mulaw:
                        continue
                    pcm = mulaw_8k_to_pcm16_24k(mulaw)
                    # Keep tracks aligned: pad the assistant track to the user
                    # position before extending the user track (unless the bot
                    # is mid-utterance, in which case its own frames pad it).
                    if not self._bot_speaking:
                        sync_buffer_to_position(
                            self.assistant_audio_buffer, len(self.user_audio_buffer)
                        )
                    self.user_audio_buffer.extend(pcm)
                    # Chunk into 20 ms frames and hand off to the capture task.
                    # Enqueue is non-blocking so the socket keeps draining.
                    audio_in_buf.extend(pcm)
                    while len(audio_in_buf) >= _FRAME_BYTES:
                        frame_bytes = bytes(audio_in_buf[:_FRAME_BYTES])
                        del audio_in_buf[:_FRAME_BYTES]
                        if self._inbound_frames is None:
                            continue
                        try:
                            self._inbound_frames.put_nowait(frame_bytes)
                        except asyncio.QueueFull:
                            try:
                                self._inbound_frames.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self._inbound_frames.put_nowait(frame_bytes)

                elif event == "user_speech_start":
                    self._user_speaking = True
                    self._bot_speaking = False

                elif event == "user_speech_stop":
                    self._user_speaking = False
                    self._user_speech_stopped_wall_ms = _wall_ms()

                elif event == "stop":
                    logger.info(f"[{self.conversation_id}] EVA sent stop")
                    break

                else:
                    logger.debug(f"[{self.conversation_id}] ignored event: {event}")

        except WebSocketDisconnect:
            logger.info(f"[{self.conversation_id}] EVA WebSocket disconnected")
        except Exception:
            logger.exception(f"[{self.conversation_id}] bridge session error")
        finally:
            await self._cleanup_livekit()

    # ---- LiveKit room mgmt ----------------------------------------------

    async def _spin_up_livekit_room(self) -> None:
        lk_url = self._cfg("url", "LIVEKIT_URL")
        lk_key = self._cfg("api_key", "LIVEKIT_API_KEY")
        lk_secret = self._cfg("api_secret", "LIVEKIT_API_SECRET")
        if not (lk_url and lk_key and lk_secret):
            raise RuntimeError(
                "LiveKit connection not configured: set LIVEKIT_URL / "
                "LIVEKIT_API_KEY / LIVEKIT_API_SECRET (or s2s_params url/"
                "api_key/api_secret)."
            )
        agent_name = self._cfg("agent_name", "LIVEKIT_AGENT_NAME", _DEFAULT_AGENT_NAME)
        from_num = self._cfg(
            "from_number", "CHARIOT_TEST_FROM_NUMBER", _DEFAULT_FROM_NUMBER
        )
        to_num = self._cfg("to_number", "CHARIOT_TEST_TRUNK_NUMBER", _DEFAULT_TO_NUMBER)
        params = self.pipeline_config.s2s_params or {}
        attributes = params.get("participant_attributes") or {
            "sip.phoneNumber": from_num,
            "sip.trunkPhoneNumber": to_num,
        }
        should_dispatch = params.get("dispatch", True)

        # Unique room per scenario run.
        self._room_name = f"eva-{self.conversation_id}-{secrets.token_hex(4)}"
        self._lk_api = api.LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_secret)

        # Pre-create the room — explicit dispatch requires the room to exist.
        await self._lk_api.room.create_room(api.CreateRoomRequest(name=self._room_name))

        # Dispatch the agent into this room. This is what SIP ingress normally
        # does for us; we do it manually because the bridge is not a real SIP
        # trunk. Skippable via s2s_params.dispatch=false for agents that join
        # rooms by their own rule.
        if should_dispatch:
            await self._lk_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=self._room_name,
                    metadata=json.dumps(
                        {"source": "eva", "conversation_id": self.conversation_id}
                    ),
                )
            )

        # Mint a participant token with the SIP attributes the agent reads. We
        # deliberately do NOT set kind=SIP via with_kind() — the Python SDK
        # serialises the enum as an int and LiveKit's server rejects it. The
        # agent's only hard dependency on SIP kind is its hangup-status handler,
        # which we don't need: the bridge controls hangup via _cleanup_livekit().
        token = (
            api.AccessToken(lk_key, lk_secret)
            .with_identity(f"sip-{self._room_name}")
            .with_name("EVA caller")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_attributes(attributes)
            .to_jwt()
        )

        self._room = rtc.Room()
        self._room.on("track_subscribed", self._on_track_subscribed)
        # Capture transcriptions the agent forwards into the room. Registered
        # before connect() so we don't miss the agent's on_enter greeting.
        self._room.register_text_stream_handler(
            _TRANSCRIPTION_TOPIC, self._on_transcription_stream
        )

        await self._room.connect(
            lk_url, token, options=rtc.RoomOptions(auto_subscribe=True)
        )
        logger.info(
            f"[{self.conversation_id}] connected to LiveKit room {self._room_name}"
        )

        # Publish a 24 kHz mono track sourced from EVA's (upsampled) mulaw stream.
        self._audio_source = rtc.AudioSource(LIVEKIT_SAMPLE_RATE, _NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track(
            "caller-audio", self._audio_source
        )
        publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        # The agent tags caller transcriptions with this track's sid.
        self._user_track_sid = publication.sid

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        # Only consume audio from the agent — skip our own self-subscription.
        if participant.identity.startswith("sip-"):
            return
        logger.info(
            f"[{self.conversation_id}] subscribed to agent audio "
            f"from {participant.identity}"
        )
        # AudioStream resamples the agent's track to our 24 kHz mono buffer rate.
        self._agent_subscriber_task = asyncio.create_task(
            self._pump_agent_audio_to_eva(
                rtc.AudioStream(
                    track,
                    sample_rate=LIVEKIT_SAMPLE_RATE,
                    num_channels=_NUM_CHANNELS,
                )
            )
        )

    # ---- Transcript capture ---------------------------------------------

    def _on_transcription_stream(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ) -> None:
        """Text-stream handler — reading is async, so hand off to a task."""
        task = asyncio.create_task(
            self._consume_transcription(reader, participant_identity)
        )
        self._transcription_tasks.add(task)
        task.add_done_callback(self._transcription_tasks.discard)

    async def _consume_transcription(
        self, reader: rtc.TextStreamReader, publisher_identity: str
    ) -> None:
        """Read one transcription segment; keep the latest full text per segment.

        The stream's publisher is always the agent, so we identify the speaker
        from the transcribed track id: it equals the caller-audio track we
        publish for the EVA caller's turns, and the agent's track otherwise.
        """
        try:
            start_ts_ms = _wall_ms()
            info = reader.info
            attrs = dict(getattr(info, "attributes", None) or {})
            seg_id = attrs.get(_ATTR_SEGMENT_ID) or getattr(info, "id", None)
            track_id = attrs.get(_ATTR_TRANSCRIBED_TRACK_ID)
            text = (await reader.read_all() or "").strip()
            end_ts_ms = _wall_ms()
            if not text or seg_id is None:
                return
            is_user = track_id is not None and track_id == self._user_track_sid
            self._transcript_segments[seg_id] = {
                "role": "user" if is_user else "assistant",
                "text": text,
            }
            if not is_user and self._fw_log:
                # Per-assistant-turn events for the metrics processor. User turns
                # are recovered from the user-simulator events on the other side.
                self._fwlog_turn_counter += 1
                self._fw_log.turn_start(timestamp_ms=start_ts_ms)
                self._fw_log.s2s_transcript(text, timestamp_ms=end_ts_ms)
                self._fw_log.turn_end(was_interrupted=False, timestamp_ms=end_ts_ms)
        except Exception:
            logger.exception(
                f"[{self.conversation_id}] error reading transcription stream"
            )

    async def _drain_and_flush_transcripts(self) -> None:
        """Let in-flight reads finish, then append captured turns to the audit log.

        Called during teardown while the room is still connected. Segments are
        emitted in first-seen order, which tracks the conversation order.
        """
        if self._transcription_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._transcription_tasks, return_exceptions=True),
                    timeout=3,
                )
            except asyncio.TimeoutError:
                for t in list(self._transcription_tasks):
                    t.cancel()
        for seg in self._transcript_segments.values():
            if seg["role"] == "user":
                self.audit_log.append_user_input(seg["text"])
            else:
                self.audit_log.append_assistant_output(seg["text"])
        if self._transcript_segments:
            logger.info(
                f"[{self.conversation_id}] flushed "
                f"{len(self._transcript_segments)} transcript segments to audit log"
            )
        self._transcript_segments = {}

    # ---- Audio pumps -----------------------------------------------------

    async def _pump_agent_audio_to_eva(self, audio_stream: rtc.AudioStream) -> None:
        """Read agent audio frames (24 kHz) → record + mulaw 8 kHz → enqueue.

        A separate sender task drains the queue at 20 ms cadence so we don't
        blast EVA's WebSocket faster than real time (its user simulator relies
        on real-time pacing for turn detection).
        """
        carry = bytearray()
        async for ev in audio_stream:
            pcm_bytes = bytes(ev.frame.data)
            # First agent audio after a user turn → record model-response latency.
            if not self._bot_speaking:
                self._bot_speaking = True
                if (
                    self._user_speech_stopped_wall_ms is not None
                    and self._metrics_log
                ):
                    latency_ms = _wall_ms() - self._user_speech_stopped_wall_ms
                    if 0 < latency_ms < 30_000:
                        self._metrics_log.write_latency(
                            "model_response", latency_ms / 1000, self._service_name
                        )
                    self._user_speech_stopped_wall_ms = None
            # Keep tracks aligned: pad the user track to the assistant position
            # before extending the assistant track (unless the user is speaking).
            if not self._user_speaking:
                sync_buffer_to_position(
                    self.user_audio_buffer, len(self.assistant_audio_buffer)
                )
            self.assistant_audio_buffer.extend(pcm_bytes)
            mulaw = pcm16_24k_to_mulaw_8k(pcm_bytes)
            carry.extend(mulaw)
            while len(carry) >= _MULAW_CHUNK_SIZE and self._outbound_mulaw is not None:
                chunk = bytes(carry[:_MULAW_CHUNK_SIZE])
                del carry[:_MULAW_CHUNK_SIZE]
                try:
                    self._outbound_mulaw.put_nowait(chunk)
                except asyncio.QueueFull:
                    try:
                        self._outbound_mulaw.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                self._outbound_mulaw.put_nowait(chunk)

    async def _capture_inbound_to_room(self) -> None:
        """Drain decoded caller frames into the LiveKit audio source.

        capture_frame paces at real time and blocks when its buffer is full, so
        it runs here rather than inline in the receive loop — otherwise it would
        stall socket reads and trip EVA's keepalive timeout (1011).
        """
        try:
            while True:
                frame_bytes = await self._inbound_frames.get()  # type: ignore[union-attr]
                if self._audio_source is None:
                    continue
                frame = rtc.AudioFrame(
                    data=frame_bytes,
                    sample_rate=LIVEKIT_SAMPLE_RATE,
                    num_channels=_NUM_CHANNELS,
                    samples_per_channel=_FRAME_SAMPLES,
                )
                await self._audio_source.capture_frame(frame)
        except asyncio.CancelledError:
            return

    async def _send_outbound_to_eva(self) -> None:
        """Drain the outbound queue at 20 ms cadence, sending Twilio `media` frames."""
        try:
            while True:
                chunk = await self._outbound_mulaw.get()  # type: ignore[union-attr]
                if self._ws is None or self._stream_sid is None:
                    continue
                try:
                    await self._ws.send_text(
                        create_twilio_media_message(self._stream_sid, chunk)
                    )
                except Exception:
                    return
                await asyncio.sleep(_MULAW_CHUNK_DURATION_S)
        except asyncio.CancelledError:
            return

    # ---- Teardown --------------------------------------------------------

    async def _cleanup_livekit(self) -> None:
        # Flush transcripts first — needs the room still connected so pending
        # read_all() calls can complete before we disconnect.
        await self._drain_and_flush_transcripts()
        if self._outbound_sender_task:
            self._outbound_sender_task.cancel()
            self._outbound_sender_task = None
        if self._inbound_capture_task:
            self._inbound_capture_task.cancel()
            self._inbound_capture_task = None
        if self._agent_subscriber_task:
            self._agent_subscriber_task.cancel()
            self._agent_subscriber_task = None
        if self._room:
            try:
                await self._room.disconnect()
            except Exception:
                logger.exception("error disconnecting room")
            self._room = None
        if self._lk_api and self._room_name:
            try:
                await self._lk_api.room.delete_room(
                    api.DeleteRoomRequest(room=self._room_name)
                )
            except Exception:
                logger.exception("error deleting room")
        if self._lk_api:
            await self._lk_api.aclose()
            self._lk_api = None
        self._room_name = None
        self._audio_source = None
