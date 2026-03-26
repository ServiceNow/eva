"""Telephony bridge server for benchmarking external voice assistants."""

import asyncio
import base64
import io
import json
import struct
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.models.config import TelephonyBridgeConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2
# Maximum silence gap before starting a new speech segment. Controls turn
# boundary detection for latency measurement: too short splits natural pauses
# into separate turns (inflating latency count), too long merges distinct
# turns (missing measurements). 750ms matches typical conversational cadence.
SEGMENT_GAP_SECONDS = 0.75

# RMS energy threshold for speech detection on 16-bit PCM audio at 24kHz
# (after upsampling). Chunks below this are treated as silence for segment
# boundary purposes. Tuned for telephony audio; increase if background
# noise causes false speech detection.
SPEECH_ENERGY_THRESHOLD = 300


def _pcm16k_to_pcm24k(pcm_16khz: bytes) -> bytes:
    """Upsample 16kHz 16-bit PCM audio to 24kHz 16-bit PCM."""
    if not pcm_16khz:
        return b""

    pcm_24khz, _ = audioop.ratecv(
        pcm_16khz,
        PCM_SAMPLE_WIDTH,
        1,
        INPUT_SAMPLE_RATE,
        OUTPUT_SAMPLE_RATE,
        None,
    )
    return pcm_24khz


def _pcm_to_wav_bytes(audio_data: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """Encode PCM audio as a mono WAV file."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        return buffer.getvalue()


def _append_timed_audio(buffer: bytearray, audio_data: bytes, offset_seconds: float) -> None:
    """Mix audio into a PCM buffer using a monotonic time offset."""
    if not audio_data:
        return

    start_frame = max(0, round(offset_seconds * OUTPUT_SAMPLE_RATE))
    start_byte = start_frame * PCM_SAMPLE_WIDTH
    end_byte = start_byte + len(audio_data)

    if len(buffer) < end_byte:
        buffer.extend(b"\x00" * (end_byte - len(buffer)))

    existing = bytes(buffer[start_byte:end_byte])
    if existing and any(existing):
        buffer[start_byte:end_byte] = audioop.add(existing, audio_data, PCM_SAMPLE_WIDTH)
    else:
        buffer[start_byte:end_byte] = audio_data


@dataclass(slots=True)
class AudioSegment:
    """Buffered audio for a single role turn."""

    role: str
    started_at: float
    ended_at: float
    audio: bytearray = field(default_factory=bytearray)

    def append(self, audio_data: bytes, timestamp: float) -> None:
        self.audio.extend(audio_data)
        duration_seconds = len(audio_data) / (OUTPUT_SAMPLE_RATE * PCM_SAMPLE_WIDTH)
        self.ended_at = max(self.ended_at, timestamp + duration_seconds)


class BaseSegmentTranscriber(ABC):
    """Transcribe PCM audio segments into text."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        """Return text for the provided audio segment."""


class NoopSegmentTranscriber(BaseSegmentTranscriber):
    """Fallback transcriber when no STT service is configured."""

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        return ""


class DeepgramSegmentTranscriber(BaseSegmentTranscriber):
    """Deepgram prerecorded transcription client."""

    def __init__(self, params: dict[str, Any]):
        self._params = dict(params)
        self._api_key = self._params["api_key"]
        self._endpoint = self._params.get("url", "https://api.deepgram.com/v1/listen")

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        if not audio_data:
            return ""

        request_params = {
            "model": self._params.get("model", "nova-2"),
            "smart_format": self._params.get("smart_format", "true"),
            "punctuate": self._params.get("punctuate", "true"),
        }
        if language := self._params.get("language"):
            request_params["language"] = language

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._endpoint,
                params=request_params,
                headers=headers,
                content=_pcm_to_wav_bytes(audio_data, sample_rate=sample_rate),
            )
            response.raise_for_status()

        payload = response.json()
        return (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
            .strip()
        )


def create_segment_transcriber(config: TelephonyBridgeConfig) -> BaseSegmentTranscriber:
    """Create a segment transcriber for the bridge transcript.jsonl output."""
    if not config.stt:
        return NoopSegmentTranscriber()
    if config.stt.lower().startswith("deepgram"):
        return DeepgramSegmentTranscriber(config.stt_params)

    logger.warning(f"Unsupported telephony bridge STT provider '{config.stt}', transcript.jsonl will be best-effort")
    return NoopSegmentTranscriber()


class BaseTelephonyTransport(ABC):
    """Abstract audio transport for external assistant connectivity."""

    def __init__(self, sip_uri: str, conversation_id: str, webhook_base_url: str):
        self.sip_uri = sip_uri
        self.conversation_id = conversation_id
        self.webhook_base_url = webhook_base_url
        self._audio_handler: Callable[[bytes], Awaitable[None]] | None = None

    @property
    def external_call_id(self) -> str | None:
        """Return the platform-specific call identifier, if available.

        For Call Control transports this is the call_control_id assigned by
        Telnyx after the call is placed. Used for conversation API lookups.
        """
        return None

    @property
    def call_leg_id(self) -> str | None:
        """Return the A-leg call_leg_id, if available.

        Used for conversation lookup — the A-leg and B-leg call_leg_ids are
        sequential UUIDs generated within milliseconds of each other.
        """
        return None

    @property
    def eva_call_id(self) -> str | None:
        """Return the EVA-generated call ID for tool webhook routing.

        Generated before dialing and passed as a custom SIP header
        (``X-Eva-Call-Id``). The Telnyx AI assistant resolves it as the
        dynamic variable ``{{eva_call_id}}`` in webhook URLs, providing a
        deterministic routing key that EVA controls.
        """
        return None

    def set_audio_handler(self, handler: Callable[[bytes], Awaitable[None]]) -> None:
        self._audio_handler = handler

    async def emit_audio(self, audio_data: bytes) -> None:
        if self._audio_handler is not None:
            await self._audio_handler(audio_data)

    @abstractmethod
    async def start(self) -> None:
        """Open the transport session to the external assistant."""

    @abstractmethod
    async def stop(self) -> None:
        """Close the transport session."""

    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Send 16kHz 16-bit PCM (L16) audio to the external assistant."""


@dataclass(slots=True)
class _SessionState:
    """Accumulated audio and timing state for a bridge session."""

    started_at: datetime
    user_audio: bytearray = field(default_factory=bytearray)
    assistant_audio: bytearray = field(default_factory=bytearray)
    mixed_audio: bytearray = field(default_factory=bytearray)
    user_segments: list[AudioSegment] = field(default_factory=list)
    assistant_segments: list[AudioSegment] = field(default_factory=list)

    def add_chunk(self, role: str, pcm_data: bytes, offset_seconds: float) -> None:
        pcm_24khz = _pcm16k_to_pcm24k(pcm_data)
        if not pcm_24khz:
            return

        is_speech = self._is_speech(pcm_24khz)

        if role == "user":
            _append_timed_audio(self.user_audio, pcm_24khz, offset_seconds)
            if is_speech:
                self._append_segment(self.user_segments, role, pcm_24khz, offset_seconds)
        else:
            _append_timed_audio(self.assistant_audio, pcm_24khz, offset_seconds)
            if is_speech:
                self._append_segment(self.assistant_segments, role, pcm_24khz, offset_seconds)

        _append_timed_audio(self.mixed_audio, pcm_24khz, offset_seconds)

    @staticmethod
    def _is_speech(pcm_data: bytes) -> bool:
        """Simple energy-based speech detection on 16-bit PCM audio.

        Returns True if the RMS energy of the audio chunk exceeds
        SPEECH_ENERGY_THRESHOLD. This filters out silence and low-level
        noise from continuous media streaming, enabling proper turn
        boundary detection for latency measurement.
        """
        if len(pcm_data) < 2:
            return False
        n_samples = len(pcm_data) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2])
        rms = (sum(s * s for s in samples) / n_samples) ** 0.5
        return rms > SPEECH_ENERGY_THRESHOLD

    def _append_segment(
        self,
        segments: list[AudioSegment],
        role: str,
        audio_data: bytes,
        offset_seconds: float,
    ) -> None:
        if not segments or offset_seconds - segments[-1].ended_at > SEGMENT_GAP_SECONDS:
            segment = AudioSegment(role=role, started_at=offset_seconds, ended_at=offset_seconds)
            segment.append(audio_data, offset_seconds)
            segments.append(segment)
            return

        segments[-1].append(audio_data, offset_seconds)


class TelephonyBridgeServer:
    """WebSocket server that bridges the user simulator to an external assistant transport."""

    def __init__(
        self,
        current_date_time: str,
        bridge_config: TelephonyBridgeConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
        transport_factory: Callable[[TelephonyBridgeConfig, str], BaseTelephonyTransport] | None = None,
        segment_transcriber: BaseSegmentTranscriber | None = None,
    ):
        self.current_date_time = current_date_time
        self.bridge_config = bridge_config
        self.agent = agent
        self.agent_config_path = agent_config_path
        self.scenario_db_path = scenario_db_path
        self.output_dir = Path(output_dir)
        self.port = port
        self.conversation_id = conversation_id

        self.audit_log = AuditLog()
        self.tool_handler = ToolExecutor(
            tool_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            tool_module_path=self.agent.tool_module_path,
            current_date_time=self.current_date_time,
        )

        self._transport_factory = transport_factory or self._default_transport_factory
        self._segment_transcriber = segment_transcriber or create_segment_transcriber(bridge_config)
        self._session_state: _SessionState | None = None
        # Saved before transport cleanup for post-call enrichment
        self._enrichment_call_control_id: str | None = None
        self._enrichment_call_leg_id: str | None = None
        self._enrichment_eva_call_id: str | None = None
        self._session_started_monotonic: float | None = None
        self._transport: BaseTelephonyTransport | None = None
        self._tool_webhook_register_callback: Callable[[str], Any] | None = None
        self._conversation_id_resolver: Callable[[str], str | None] | None = None

        self._app = FastAPI()
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._running = False

        self._register_routes()

    @property
    def app(self) -> FastAPI:
        """Return the FastAPI app for tests."""
        return self._app

    async def start(self) -> None:
        """Start the bridge server."""
        if self._running:
            logger.warning("Telephony bridge server already running")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        self._running = True

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Telephony bridge started on ws://localhost:{self.port}")

    async def stop(self) -> None:
        """Stop the bridge server and save outputs."""
        if not self._running:
            await self._save_outputs()
            return

        self._running = False

        if self._transport is not None:
            await self._transport.stop()
            self._transport = None

        if self._server:
            self._server.should_exit = True

        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass

        self._server = None
        self._server_task = None

        await self._save_outputs()
        logger.info(f"Telephony bridge stopped on port {self.port}")

    def _register_routes(self) -> None:
        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

    def _default_transport_factory(self, config: TelephonyBridgeConfig, conversation_id: str) -> BaseTelephonyTransport:
        from eva.assistant.transports import CallControlTransport

        return CallControlTransport(
            api_key=config.telnyx_api_key,
            to=config.sip_uri,
            app_id=config.call_control_app_id,
            from_number=config.call_control_from,
            conversation_id=conversation_id,
            webhook_base_url=config.webhook_base_url,
        )

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Bridge a user simulator websocket to the configured transport."""
        logger.info(f"Telephony bridge client connected for {self.conversation_id}")

        self._session_state = _SessionState(started_at=datetime.now(UTC))
        self._session_started_monotonic = asyncio.get_event_loop().time()
        self._transport = self._transport_factory(self.bridge_config, self.conversation_id)

        assistant_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def on_transport_audio(audio_data: bytes) -> None:
            """Audio from Telnyx (L16 16kHz PCM) → forward to user simulator."""
            assert self._session_state is not None
            offset_seconds = self._elapsed_seconds()
            self._session_state.add_chunk("assistant", audio_data, offset_seconds)
            await assistant_audio_queue.put(audio_data)

        self._transport.set_audio_handler(on_transport_audio)

        async def send_assistant_audio() -> None:
            while True:
                audio_data = await assistant_audio_queue.get()
                if audio_data is None:
                    break
                await websocket.send_json(
                    {
                        "event": "media",
                        "conversation_id": self.conversation_id,
                        "media": {"payload": base64.b64encode(audio_data).decode("utf-8")},
                    }
                )

        sender_task = asyncio.create_task(send_assistant_audio())

        try:
            await self._transport.start()

            # After transport starts, register the eva_call_id with the webhook
            # so tool calls routed by {{eva_call_id}} find the right executor.
            # eva_call_id is generated by EVA before dialing and passed as a
            # custom SIP header, so it's deterministic and known upfront.
            eva_id = self._transport.eva_call_id
            if eva_id:
                if self._tool_webhook_register_callback:
                    await self._tool_webhook_register_callback(eva_id)
                    logger.info("Registered eva_call_id %s for tool webhooks", eva_id)

            # Monitor both the user sim WebSocket and the transport disconnect event.
            # When the Telnyx assistant hangs up, the media stream closes and
            # _disconnected_event fires — we should end the session cleanly
            # instead of waiting for the user sim to time out.
            transport_done = asyncio.create_task(self._transport._disconnected_event.wait())
            try:
                while True:
                    ws_receive = asyncio.create_task(websocket.receive_text())
                    done, _ = await asyncio.wait(
                        [ws_receive, transport_done],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if transport_done in done:
                        ws_receive.cancel()
                        logger.info(
                            "Telnyx call ended (assistant hung up) for %s — ending session",
                            self.conversation_id,
                        )
                        # Send a clean stop event so the user sim knows the call
                        # ended normally (assistant hangup), not as an error.
                        try:
                            await websocket.send_json({
                                "event": "stop",
                                "conversation_id": self.conversation_id,
                                "reason": "assistant_hangup",
                            })
                        except Exception:
                            pass  # Best-effort — WS may already be closing
                        break

                    message = ws_receive.result()
                    payload = json.loads(message)
                    event = payload.get("event")

                    if event == "media":
                        audio_base64 = payload.get("media", {}).get("payload", "")
                        pcm_audio = base64.b64decode(audio_base64) if audio_base64 else b""
                        if pcm_audio:
                            assert self._session_state is not None
                            offset_seconds = self._elapsed_seconds()
                            self._session_state.add_chunk("user", pcm_audio, offset_seconds)
                            await self._transport.send_audio(pcm_audio)
                    elif event == "stop":
                        logger.info(f"Received stop event for {self.conversation_id}")
                        break
                    else:
                        logger.debug(f"Ignoring telephony bridge event '{event}'")
            finally:
                transport_done.cancel()

        except WebSocketDisconnect:
            logger.info(f"Telephony bridge websocket disconnected for {self.conversation_id}")
        except Exception as exc:
            logger.error(f"Telephony bridge session error: {exc}", exc_info=True)
        finally:
            # Log audio capture stats
            if self._session_state is not None:
                user_bytes = len(self._session_state.user_audio)
                asst_bytes = len(self._session_state.assistant_audio)
                logger.info(
                    "Session audio captured: user=%d bytes, assistant=%d bytes",
                    user_bytes, asst_bytes,
                )

            await assistant_audio_queue.put(None)
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

            # Save call identifiers before cleaning up the transport — we need
            # them later for conversation API enrichment in _save_outputs().
            if self._transport is not None:
                self._enrichment_call_control_id = self._transport.external_call_id
                self._enrichment_call_leg_id = getattr(self._transport, "call_leg_id", None)
                self._enrichment_eva_call_id = self._transport.eva_call_id
                await self._transport.stop()
                self._transport = None

            logger.info(f"Telephony bridge client finished for {self.conversation_id}")

    def _elapsed_seconds(self) -> float:
        if self._session_started_monotonic is None:
            return 0.0
        return max(0.0, asyncio.get_event_loop().time() - self._session_started_monotonic)

    async def _generate_transcript(self) -> list[dict[str, str]]:
        """Generate transcript.jsonl entries from recorded user and assistant segments."""
        if self._session_state is None:
            return []

        transcript_entries: list[dict[str, str]] = []
        all_segments = sorted(
            [*self._session_state.user_segments, *self._session_state.assistant_segments],
            key=lambda segment: segment.started_at,
        )

        session_started_at = self._session_state.started_at

        for segment in all_segments:
            text = await self._segment_transcriber.transcribe(bytes(segment.audio), OUTPUT_SAMPLE_RATE)
            if not text:
                continue

            timestamp_dt = session_started_at + timedelta(seconds=segment.started_at)
            timestamp_ms = str(int(timestamp_dt.timestamp() * 1000))

            if segment.role == "user":
                self.audit_log.append_user_input(text, timestamp_ms=timestamp_ms)
            else:
                self.audit_log.append_assistant_output(text, timestamp_ms=timestamp_ms)

            transcript_entries.append(
                {
                    "timestamp": timestamp_dt.isoformat(),
                    "role": segment.role,
                    "content": text,
                }
            )

        return transcript_entries

    async def _save_outputs(self) -> None:
        """Persist bridge outputs in the same shape as AssistantServer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving outputs to %s (session_state=%s, user_audio=%d, assistant_audio=%d)",
            self.output_dir,
            self._session_state is not None,
            len(self._session_state.user_audio) if self._session_state else 0,
            len(self._session_state.assistant_audio) if self._session_state else 0,
        )

        transcript_entries = await self._generate_transcript()
        transcript_path = self.output_dir / "transcript.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as transcript_file:
            for entry in transcript_entries:
                transcript_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Enrich audit log with conversation messages from Telnyx API
        # before saving, so the audit log has user/assistant speech events
        # that the metrics processor needs for turn extraction.
        await self._enrich_audit_log_from_api()

        audit_path = self.output_dir / "audit_log.json"
        self.audit_log.save(audit_path)

        self._save_audio()

        initial_db_path = self.output_dir / "initial_scenario_db.json"
        with open(initial_db_path, "w", encoding="utf-8") as initial_db_file:
            json.dump(self.get_initial_scenario_db(), initial_db_file, indent=2, sort_keys=True, default=str)

        final_db_path = self.output_dir / "final_scenario_db.json"
        with open(final_db_path, "w", encoding="utf-8") as final_db_file:
            json.dump(self.get_final_scenario_db(), final_db_file, indent=2, sort_keys=True, default=str)

        # Write an empty pipecat_logs.jsonl — no local TTS pipeline exists for telephony
        # bridge calls.  The metrics processor will fall back to audit_log assistant entries
        # (Conversations API text) as the authoritative source for intended_assistant_turns.
        # Previously this file was populated from ElevenLabs assistant_speech transcriptions,
        # but those are STT output (what ElevenLabs *heard*) and can garble alphanumeric
        # strings, numbers, and formatting — producing incorrect "intended" text.
        pipecat_logs_path = self.output_dir / "pipecat_logs.jsonl"
        pipecat_logs_path.write_text("")

        # Generate response_latencies.json from client-side audio segment timing.
        # Mirrors Pipecat's UserStoppedSpeaking → BotStartedSpeaking measurement.
        self._generate_response_latencies()

        logger.info(f"Telephony bridge outputs saved to {self.output_dir}")

    async def _enrich_audit_log_from_api(self) -> None:
        """Fetch conversation messages from Telnyx Conversations API and rebuild audit log.

        The Telnyx AI assistant records full user/assistant message history. We fetch it
        and rebuild the audit log transcript so that EVA's metrics processor can extract
        proper conversation turns. The local audit log only has tool call events, which
        isn't enough for LLM judge metrics.

        Lookup strategy (in priority order):
        1. Dynamic variables webhook mapping: eva_call_id → conversation_id (fastest, no API call)
        2. Call Events API fallback: query conversation_created event on the A-leg
        """
        logger.info("Starting post-call enrichment from Conversations API...")

        api_key = self.bridge_config.telnyx_api_key
        if not api_key:
            logger.warning("No telnyx_api_key configured — skipping enrichment")
            return

        base_url = "https://api.telnyx.com/v2/ai/conversations"
        headers = {"Authorization": f"Bearer {api_key}"}

        # Wait for the Conversations API to fully index the call's messages.
        # Voice calls need more time than chat — transcription + message storage.
        await asyncio.sleep(5)

        # Try to resolve conversation_id from the DV webhook mapping first
        # (instant, no API call needed — the DV webhook stored it during the call).
        conv_id: str | None = None
        eva_call_id = getattr(self, "_enrichment_eva_call_id", None)
        if eva_call_id and self._conversation_id_resolver:
            conv_id = self._conversation_id_resolver(eva_call_id)
            if conv_id:
                logger.info(
                    "Resolved conversation_id=%s from DV webhook mapping (eva_call_id=%s)",
                    conv_id, eva_call_id,
                )

        # Fallback: query Call Events API
        if not conv_id:
            call_control_id = getattr(self, "_enrichment_call_control_id", None)
            if not call_control_id:
                logger.warning("No call_control_id or eva_call_id available — cannot enrich")
                return

            logger.info("DV mapping miss — falling back to Call Events API lookup")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if not conv_id:
                    conv_id = await self._find_conversation_id_from_events(
                        client, call_control_id, headers
                    )

                if not conv_id:
                    logger.warning(
                        "Could not find conversation_id via DV mapping or Call Events API "
                        "(eva_call_id=%s, call_control_id=%s)",
                        eva_call_id, call_control_id,
                    )
                    return

                # Fetch messages
                msg_resp = await client.get(
                    f"{base_url}/{conv_id}/messages",
                    params={"page[size]": 100},
                    headers=headers,
                )
                msg_resp.raise_for_status()
                messages = [m for m in msg_resp.json().get("data", []) if m is not None]
                messages.reverse()  # API returns reverse chronological

                if not messages:
                    logger.warning("No messages found for conversation %s", conv_id)
                    return

                # Rebuild the audit log transcript from API messages
                transcript = self._messages_to_audit_transcript(messages)
                self.audit_log.replace_transcript(transcript)
                logger.info(
                    "Enriched audit log from conversations API: %d messages → %d transcript events",
                    len(messages),
                    len(transcript),
                )

        except Exception as e:
            logger.warning("Failed to fetch conversation messages: %s", e)

    @staticmethod
    async def _find_conversation_id_from_events(
        client: httpx.AsyncClient,
        call_control_id: str,
        headers: dict[str, str],
    ) -> str | None:
        """Find the conversation_id by querying the Call Events API.

        The ``conversation_created`` event on the A-leg call contains the
        ``conversation_id`` that maps to the Conversations messages API.
        """
        from urllib.parse import quote

        try:
            resp = await client.get(
                "https://api.telnyx.com/v2/call_events",
                params={
                    "filter[call_control_id]": call_control_id,
                    "page[size]": 50,
                },
                headers=headers,
            )
            resp.raise_for_status()
            events = resp.json().get("data", [])

            for event in events:
                if event.get("name") != "conversation_created":
                    continue
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    inner = payload.get("payload", payload)
                    if isinstance(inner, dict):
                        conv_id = inner.get("conversation_id")
                        if conv_id:
                            logger.info(
                                "Found conversation_id=%s from call events for cc_id=%s",
                                conv_id,
                                call_control_id[:40],
                            )
                            return conv_id

            logger.warning(
                "No conversation_created event found for call_control_id=%s (checked %d events)",
                call_control_id[:40],
                len(events),
            )
        except Exception as e:
            logger.warning("Failed to query call events: %s", e)

        return None

    @staticmethod
    def _uuid_v1_time_diff_seconds(uuid_a: str, uuid_b: str) -> float:
        """Return the absolute time difference in seconds between two UUID v1 values.

        UUID v1 encodes a 60-bit timestamp (100ns intervals since 1582-10-15)
        split across time_low, time_mid, and time_hi_and_version fields.
        """
        def _ts(u: str) -> int:
            parts = u.split("-")
            time_low = int(parts[0], 16)
            time_mid = int(parts[1], 16)
            time_hi = int(parts[2], 16) & 0x0FFF  # mask version nibble
            return (time_hi << 48) | (time_mid << 32) | time_low

        return abs(_ts(uuid_a) - _ts(uuid_b)) / 10_000_000

    @staticmethod
    def _messages_to_audit_transcript(messages: list[dict]) -> list[dict]:
        """Convert Telnyx Conversations API messages to audit log transcript format."""
        transcript = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            text = msg.get("text", "")
            tool_calls = msg.get("tool_calls") or []
            tool_call_id = msg.get("tool_call_id")
            sent_at = msg.get("sent_at", msg.get("created_at", ""))

            ts = 0
            if sent_at:
                try:
                    dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
                    ts = int(dt.timestamp() * 1000)
                except (ValueError, TypeError):
                    ts = i

            if role == "user" and text:
                transcript.append(
                    {
                        "timestamp": ts,
                        "message_type": "user",
                        "type": "user",
                        "displayName": "User",
                        "content": text,
                        "value": text,
                    }
                )
            elif role == "assistant":
                if text:
                    transcript.append(
                        {
                            "timestamp": ts,
                            "message_type": "assistant",
                            "type": "assistant",
                            "displayName": "Assistant",
                            "content": text,
                            "value": text,
                        }
                    )
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    transcript.append(
                        {
                            "timestamp": ts,
                            "message_type": "tool_call",
                            "type": "tool_call",
                            "displayName": "Tool",
                            "value": {
                                "tool": fn.get("name", ""),
                                "parameters": args,
                                "tool_call_id": tc.get("id", ""),
                            },
                        }
                    )
            elif role == "tool":
                tool_name = ""
                if tool_call_id:
                    for prev in reversed(transcript):
                        if (
                            prev.get("message_type") == "tool_call"
                            and prev.get("value", {}).get("tool_call_id") == tool_call_id
                        ):
                            tool_name = prev["value"]["tool"]
                            break
                transcript.append(
                    {
                        "timestamp": ts,
                        "message_type": "tool_response",
                        "type": "tool_response",
                        "displayName": "Tool Response",
                        "value": {
                            "tool": tool_name,
                            "response": text,
                            "tool_call_id": tool_call_id or "",
                        },
                    }
                )

        return transcript

    def _generate_response_latencies(self) -> None:
        """Generate response_latencies.json from client-side audio segment timing.

        Measures the gap between the end of each user speech segment and the start
        of the next assistant speech segment. This mirrors how Pipecat's
        ``UserBotLatencyObserver`` measures ``UserStoppedSpeaking → BotStartedSpeaking``
        and gives an apples-to-apples comparison with Pipecat-based benchmarks.

        Unlike server-side ``end_user_perceived_latency_ms`` from the Conversations API,
        this includes the full round-trip: network latency between the bridge and
        Telnyx, plus any media streaming overhead.
        """
        latencies_path = self.output_dir / "response_latencies.json"
        latencies: list[float] = []

        if self._session_state is not None:
            user_segs = self._session_state.user_segments
            asst_segs = self._session_state.assistant_segments

            # For each user segment, find the next assistant segment that starts after it ends.
            # This pairs conversational turns the same way Pipecat's VAD-based observer does.
            asst_idx = 0
            for user_seg in user_segs:
                # Advance past assistant segments that started before this user segment ended
                while asst_idx < len(asst_segs) and asst_segs[asst_idx].started_at <= user_seg.ended_at:
                    asst_idx += 1

                if asst_idx < len(asst_segs):
                    gap = asst_segs[asst_idx].started_at - user_seg.ended_at
                    if gap > 0:
                        latencies.append(round(gap, 4))

        latencies_data = {
            "latencies": latencies,
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else 0,
            "max": round(max(latencies), 4) if latencies else 0,
            "min": round(min(latencies), 4) if latencies else 0,
            "count": len(latencies),
            "source": "client_audio_segments",
        }
        with open(latencies_path, "w") as f:
            json.dump(latencies_data, f, indent=2)

        logger.info(
            "Generated client-side response latencies: %d measurements, mean=%.3fs",
            len(latencies),
            latencies_data["mean"],
        )

    def _save_audio(self) -> None:
        if self._session_state is None:
            return

        self._save_wav_file(bytes(self._session_state.mixed_audio), self.output_dir / "audio_mixed.wav")
        self._save_wav_file(bytes(self._session_state.user_audio), self.output_dir / "audio_user.wav")
        self._save_wav_file(bytes(self._session_state.assistant_audio), self.output_dir / "audio_assistant.wav")

    @staticmethod
    def _save_wav_file(audio_data: bytes, file_path: Path) -> None:
        try:
            with wave.open(str(file_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
                wav_file.setframerate(OUTPUT_SAMPLE_RATE)
                wav_file.writeframes(audio_data)
        except Exception as exc:
            logger.error(f"Error saving audio to {file_path}: {exc}")

    def get_conversation_stats(self) -> dict[str, Any]:
        """Get pre-metric conversation statistics."""
        return self.audit_log.get_stats()

    def get_initial_scenario_db(self) -> dict[str, Any]:
        """Get the scenario DB state before tool execution."""
        return self.tool_handler.original_db

    def get_final_scenario_db(self) -> dict[str, Any]:
        """Get the scenario DB state after tool execution."""
        return self.tool_handler.db
