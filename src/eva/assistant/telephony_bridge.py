"""Telephony bridge server for benchmarking external voice assistants."""

import asyncio
import base64
import io
import json
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
SEGMENT_GAP_SECONDS = 0.75


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

        if role == "user":
            _append_timed_audio(self.user_audio, pcm_24khz, offset_seconds)
            self._append_segment(self.user_segments, role, pcm_24khz, offset_seconds)
        else:
            _append_timed_audio(self.assistant_audio, pcm_24khz, offset_seconds)
            self._append_segment(self.assistant_segments, role, pcm_24khz, offset_seconds)

        _append_timed_audio(self.mixed_audio, pcm_24khz, offset_seconds)

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
        self._session_started_monotonic: float | None = None
        self._transport: BaseTelephonyTransport | None = None
        self._tool_webhook_register_callback: Callable[[str], Any] | None = None

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

            # After transport starts, register the call_control_id with the webhook
            # so tool calls routed by {{call_control_id}} find the right executor
            if hasattr(self._transport, '_call_control_id') and self._transport._call_control_id:
                cc_id = self._transport._call_control_id
                if self._tool_webhook_register_callback:
                    await self._tool_webhook_register_callback(cc_id)
                    logger.info("Registered call_control_id %s for tool webhooks", cc_id)

            while True:
                message = await websocket.receive_text()
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

            if self._transport is not None:
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

        audit_path = self.output_dir / "audit_log.json"
        self.audit_log.save(audit_path)

        self._save_audio()

        initial_db_path = self.output_dir / "initial_scenario_db.json"
        with open(initial_db_path, "w", encoding="utf-8") as initial_db_file:
            json.dump(self.get_initial_scenario_db(), initial_db_file, indent=2, sort_keys=True, default=str)

        final_db_path = self.output_dir / "final_scenario_db.json"
        with open(final_db_path, "w", encoding="utf-8") as final_db_file:
            json.dump(self.get_final_scenario_db(), final_db_file, indent=2, sort_keys=True, default=str)

        logger.info(f"Telephony bridge outputs saved to {self.output_dir}")

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
