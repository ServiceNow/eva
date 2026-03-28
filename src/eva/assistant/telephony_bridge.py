"""Telephony bridge server for benchmarking external voice assistants."""

import asyncio
import base64
import io
import json
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.bridge_vad_observer import BridgeVADObserver
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.models.config import TelephonyBridgeConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2
_TELNYX_CONVERSATIONS_API_BASE_URL = "https://api.telnyx.com/v2"


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

    def add_chunk(self, role: str, pcm_data: bytes, offset_seconds: float) -> None:
        pcm_24khz = _pcm16k_to_pcm24k(pcm_data)
        if not pcm_24khz:
            return

        if role == "user":
            _append_timed_audio(self.user_audio, pcm_24khz, offset_seconds)
        else:
            _append_timed_audio(self.assistant_audio, pcm_24khz, offset_seconds)

        _append_timed_audio(self.mixed_audio, pcm_24khz, offset_seconds)


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
        observer_factory: Callable[[Path, BaseSegmentTranscriber, AuditLog], BridgeVADObserver] | None = None,
        telnyx_conversation_lookup: Callable[[str], str | None] | None = None,
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
        self._observer_factory = observer_factory or (
            lambda output_dir, transcriber, audit_log: BridgeVADObserver(output_dir, transcriber, audit_log)
        )
        self._session_state: _SessionState | None = None
        self._vad_observer: BridgeVADObserver | None = None
        self._session_started_monotonic: float | None = None
        self._session_end_reason: str | None = None
        self._transport: BaseTelephonyTransport | None = None
        self._tool_webhook_register_callback: Callable[[str], Any] | None = None
        self._telnyx_conversation_lookup = telnyx_conversation_lookup
        self._eva_call_id: str | None = None

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
        self._session_end_reason = None
        self._transport = self._transport_factory(self.bridge_config, self.conversation_id)
        self._vad_observer = self._observer_factory(self.output_dir, self._segment_transcriber, self.audit_log)

        assistant_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def on_transport_audio(audio_data: bytes) -> None:
            """Audio from Telnyx (L16 16kHz PCM) → forward to user simulator."""
            assert self._session_state is not None
            offset_seconds = self._elapsed_seconds()
            self._session_state.add_chunk("assistant", audio_data, offset_seconds)
            if self._vad_observer is not None:
                await self._vad_observer.feed_assistant_audio(audio_data, self._current_epoch_seconds())
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
                self._eva_call_id = eva_id
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
                        self._session_end_reason = "assistant_hangup"
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
                            if self._vad_observer is not None:
                                await self._vad_observer.feed_user_audio(pcm_audio, self._current_epoch_seconds())
                            await self._transport.send_audio(pcm_audio)
                    elif event == "stop":
                        self._session_end_reason = payload.get("reason") or self._session_end_reason
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

            if self._transport is not None:
                await self._transport.stop()
                self._transport = None

            if self._vad_observer is not None:
                await self._vad_observer.finalize(reason=self._infer_session_end_reason())

            logger.info(f"Telephony bridge client finished for {self.conversation_id}")

    def _elapsed_seconds(self) -> float:
        if self._session_started_monotonic is None:
            return 0.0
        return max(0.0, asyncio.get_event_loop().time() - self._session_started_monotonic)

    @staticmethod
    def _current_epoch_seconds() -> float:
        return time.time()

    def _infer_session_end_reason(self) -> str | None:
        # Check for end_call tool invocation first — if the assistant
        # called end_call, the session ended with a proper goodbye
        # regardless of whether the transport reported "assistant_hangup".
        for entry in reversed(self.audit_log.transcript):
            if entry.get("message_type") not in {"tool_call", "tool_response"}:
                continue
            value = entry.get("value", {})
            if isinstance(value, dict) and value.get("tool") == "end_call":
                return "goodbye"

        if self._session_end_reason:
            return self._session_end_reason

        return None

    async def _fetch_intended_assistant_speech(self, conversation_id: str) -> list[dict[str, Any]]:
        """Fetch assistant messages from Telnyx Conversations API in chronological order."""
        api_key = self.bridge_config.telnyx_api_key
        if not api_key:
            logger.warning("Skipping Telnyx conversation message fetch: missing API key")
            return []

        url = f"{_TELNYX_CONVERSATIONS_API_BASE_URL}/ai/conversations/{conversation_id}/messages?page[size]=100"
        timeout = aiohttp.ClientTimeout(total=30.0)

        try:
            async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            ) as session:
                async with session.get(url) as response:
                    try:
                        payload = await response.json()
                    except aiohttp.ContentTypeError:
                        payload = {"message": await response.text()}

                    if response.status >= 400:
                        logger.warning(
                            "Failed to fetch Telnyx conversation messages for %s: %s %s",
                            conversation_id,
                            response.status,
                            payload,
                        )
                        return []
        except Exception as exc:
            logger.warning("Failed to fetch Telnyx conversation messages for %s: %s", conversation_id, exc)
            return []

        data = payload.get("data", [])
        if not isinstance(data, list):
            logger.warning(
                "Unexpected Telnyx conversation messages payload for %s: data was %s",
                conversation_id,
                type(data).__name__,
            )
            return []

        intended_speech: list[dict[str, Any]] = []
        for message in reversed(data):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue

            text = str(message.get("text", "")).strip()
            if not text:
                continue

            sent_at = message.get("sent_at")
            if not isinstance(sent_at, str) or not sent_at.strip():
                logger.warning("Skipping assistant message without sent_at for conversation %s", conversation_id)
                continue

            try:
                timestamp_ms = self._iso8601_to_epoch_ms(sent_at)
            except ValueError:
                logger.warning(
                    "Skipping assistant message with invalid sent_at '%s' for conversation %s",
                    sent_at,
                    conversation_id,
                )
                continue

            intended_speech.append({"text": text, "timestamp_ms": timestamp_ms})

        return intended_speech

    @staticmethod
    def _iso8601_to_epoch_ms(value: str) -> int:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)

    def _resolve_telnyx_conversation_id(self) -> str | None:
        if self._eva_call_id and self._telnyx_conversation_lookup is not None:
            conversation_id = self._telnyx_conversation_lookup(self._eva_call_id)
            if conversation_id:
                return conversation_id
        return None

    def _read_assistant_speech_timestamps(self) -> list[int]:
        """Read assistant_speech timestamps from the bridge VAD observer's event log.

        These timestamps reflect when the assistant's speech was actually heard
        (via Deepgram transcription), as opposed to Conversations API sent_at
        timestamps which reflect when the LLM generated the text.
        """
        events_path = self.output_dir / "elevenlabs_events.jsonl"
        if not events_path.exists():
            return []
        timestamps: list[int] = []
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "assistant_speech":
                    ts = event.get("timestamp", 0)
                    if ts:
                        timestamps.append(ts)
        return timestamps

    async def _write_pipecat_tts_events(self, pipecat_logs_path: Path) -> None:
        intended_speech: list[dict[str, Any]] = []
        telnyx_conversation_id = self._resolve_telnyx_conversation_id()
        if telnyx_conversation_id:
            intended_speech = await self._fetch_intended_assistant_speech(telnyx_conversation_id)
        elif self._eva_call_id:
            logger.warning("No Telnyx conversation_id found for eva_call_id %s", self._eva_call_id)

        # Use bridge VAD observer timestamps (when speech was actually heard)
        # instead of Conversations API sent_at (when LLM generated the text).
        # This ensures pipecat events interleave correctly with user speech
        # events for proper turn boundary detection.
        spoken_timestamps = self._read_assistant_speech_timestamps()
        for i, item in enumerate(intended_speech):
            if i < len(spoken_timestamps):
                item["timestamp_ms"] = spoken_timestamps[i]

        with open(pipecat_logs_path, "w", encoding="utf-8") as file_obj:
            for i, item in enumerate(intended_speech):
                # Insert turn_end between consecutive tts_text events so the
                # pipecat log aggregator doesn't merge separate assistant
                # messages into one giant chunk.
                if i > 0:
                    file_obj.write(
                        json.dumps(
                            {
                                "type": "turn_end",
                                "timestamp": item["timestamp_ms"] - 1,
                                "start_timestamp": item["timestamp_ms"] - 1,
                                "data": {"frame": ""},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                file_obj.write(
                    json.dumps(
                        {
                            "type": "tts_text",
                            "timestamp": item["timestamp_ms"],
                            "start_timestamp": item["timestamp_ms"],
                            "data": {"frame": item["text"]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

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

        transcript_entries = self._vad_observer.get_transcript_entries() if self._vad_observer is not None else []
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

        pipecat_logs_path = self.output_dir / "pipecat_logs.jsonl"
        await self._write_pipecat_tts_events(pipecat_logs_path)

        response_latencies_path = self.output_dir / "response_latencies.json"
        if not response_latencies_path.exists():
            with open(response_latencies_path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "latencies": [],
                        "mean": 0,
                        "max": 0,
                        "min": 0,
                        "count": 0,
                        "source": "vad_observer",
                    },
                    file_obj,
                    indent=2,
                )

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
