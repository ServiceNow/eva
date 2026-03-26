"""Telnyx Call Control transport backed by media streaming websockets.

We REQUEST L16 16kHz for the bidirectional stream, but Telnyx may send audio
back in a different codec (e.g. PCMU 8kHz). The ``media_format`` in the
``start`` event tells us the actual inbound codec; we convert to 16kHz PCM
before emitting so the rest of the pipeline always sees uniform L16.
"""

import asyncio
try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # Python 3.13+ removed audioop
import base64
import json
from typing import Any

import aiohttp


from eva.assistant.telephony_bridge import BaseTelephonyTransport
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_TELNYX_API_BASE_URL = "https://api.telnyx.com/v2"
_CALL_CONNECT_TIMEOUT_SECONDS = 60.0
_REQUEST_TIMEOUT_SECONDS = 30.0

# Global registry: conversation_id → CallControlTransport
# The ToolWebhookService uses this to route incoming media stream connections.
# Protected by _registry_lock for safe concurrent access from multiple async tasks.
_active_transports: dict[str, "CallControlTransport"] = {}
_registry_lock = asyncio.Lock()


async def get_active_transport(conversation_id: str) -> "CallControlTransport | None":
    """Look up the transport for a conversation (used by media stream WS handler)."""
    async with _registry_lock:
        return _active_transports.get(conversation_id)


class CallControlTransport(BaseTelephonyTransport):
    """Telnyx Call Control transport using bidirectional media streaming.

    Places an outbound call via Call Control API, then receives/sends audio
    through a WebSocket media stream that Telnyx opens to our webhook server.
    """

    def __init__(
        self,
        api_key: str,
        to: str,
        app_id: str,
        from_number: str,
        conversation_id: str,
        webhook_base_url: str,
        *,
        connect_timeout_seconds: float = _CALL_CONNECT_TIMEOUT_SECONDS,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        api_base_url: str = _TELNYX_API_BASE_URL,
    ):
        super().__init__(sip_uri=to, conversation_id=conversation_id, webhook_base_url=webhook_base_url)
        self.api_key = api_key
        self.to = to
        self.app_id = app_id
        self.from_number = from_number
        self.connect_timeout_seconds = connect_timeout_seconds
        self.api_base_url = api_base_url.rstrip("/")
        self._request_timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)

        self._session: aiohttp.ClientSession | None = None
        self._stream_ws: Any | None = None  # WebSocket connection from Telnyx
        self._call_control_id: str | None = None
        self._stream_id: str | None = None
        self._connected_event = asyncio.Event()
        self._disconnected_event = asyncio.Event()
        self._send_lock = asyncio.Lock()

        # Inbound codec — updated from the "start" event's media_format.
        # Defaults assume L16 16kHz; overridden once the stream reports actual format.
        self._inbound_encoding: str = "L16"
        self._inbound_sample_rate: int = 16000

    @property
    def external_call_id(self) -> str | None:
        """Return the Telnyx call_control_id once the call is placed."""
        return self._call_control_id

    async def start(self) -> None:
        if self._session is not None:
            logger.warning("Call Control transport already started for %s", self.to)
            return

        logger.info("Starting Telnyx Call Control transport: dialing %s", self.to)
        self._connected_event.clear()
        self._disconnected_event.clear()

        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._request_timeout,
        )

        # Register ourselves so the webhook WS handler can route the stream to us
        async with _registry_lock:
            _active_transports[self.conversation_id] = self

        try:
            # Build WSS URL for the media stream on the shared webhook server
            stream_wss_url = self.webhook_base_url.replace("https://", "wss://").replace("http://", "ws://")
            stream_wss_url = f"{stream_wss_url}/media-stream/{self.conversation_id}"

            logger.info("Call Control: placing call to %s, media stream URL: %s", self.to, stream_wss_url)

            response = await self._post(
                "/calls",
                {
                    "connection_id": self.app_id,
                    "to": self.to,
                    "from": self.from_number,
                    "stream_url": stream_wss_url,
                    "stream_track": "both_tracks",
                    "stream_bidirectional_mode": "rtp",
                    "stream_bidirectional_codec": "L16",
                    "stream_bidirectional_sampling_rate": 16000,
                },
            )
            data = response.get("data", {})
            self._call_control_id = data.get("call_control_id")
            if not self._call_control_id:
                raise RuntimeError("Telnyx call creation response did not include data.call_control_id")

            logger.info("Call placed: call_control_id=%s, waiting for media stream...", self._call_control_id)
            await asyncio.wait_for(self._connected_event.wait(), timeout=self.connect_timeout_seconds)
            logger.info("Telnyx Call Control media stream connected for %s", self.to)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        # Unregister from global registry
        async with _registry_lock:
            _active_transports.pop(self.conversation_id, None)

        if self._call_control_id and not self._disconnected_event.is_set():
            try:
                await self._post(f"/calls/{self._call_control_id}/actions/hangup", {})
            except Exception as exc:
                logger.warning("Failed to hang up Telnyx call %s: %s", self._call_control_id, exc)

        if self._stream_ws is not None:
            try:
                await self._stream_ws.close()
            except Exception as exc:
                logger.debug("Ignoring stream websocket close error: %s", exc)
            self._stream_ws = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        self._stream_id = None
        self._call_control_id = None
        self._connected_event.clear()

    def _convert_outbound_audio(self, pcm_16khz: bytes) -> bytes:
        """Downsample 16kHz PCM to match the stream's actual sample rate if needed."""
        if self._inbound_sample_rate >= 16000:
            return pcm_16khz
        # Downsample 16kHz → stream rate (typically 8kHz)
        converted, _ = audioop.ratecv(
            pcm_16khz, 2, 1, 16000, self._inbound_sample_rate, None
        )
        return converted

    async def send_audio(self, audio_data: bytes) -> None:
        if not audio_data:
            return

        async with self._send_lock:
            if self._stream_ws is None:
                logger.debug("Dropping outbound audio because Telnyx media stream is not connected")
                return

            try:
                outbound = self._convert_outbound_audio(audio_data)
                await self._stream_ws.send_text(
                    json.dumps({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(outbound).decode("ascii"),
                        },
                    })
                )
            except Exception:
                logger.info("Telnyx media stream disconnected while sending audio for %s", self.to)

    # ------------------------------------------------------------------
    # Called by the webhook server when Telnyx connects the media stream
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inbound audio conversion helpers
    # ------------------------------------------------------------------

    def _convert_inbound_audio(self, raw_audio: bytes) -> bytes:
        """Convert inbound audio to 16kHz 16-bit PCM based on detected codec.

        If Telnyx sends PCMU (μ-law 8kHz), decode and upsample.
        If L16 16kHz, pass through. Otherwise best-effort passthrough.
        """
        enc = self._inbound_encoding
        rate = self._inbound_sample_rate

        if enc == "PCMU" or enc == "audio/x-mulaw":
            # μ-law → 16-bit linear PCM
            pcm_audio = audioop.ulaw2lin(raw_audio, 2)
            if rate and rate != 16000:
                # Upsample (typically 8000 → 16000)
                pcm_audio, _ = audioop.ratecv(pcm_audio, 2, 1, rate, 16000, None)
            return pcm_audio
        elif enc == "L16" or enc == "audio/L16":
            if rate and rate != 16000:
                pcm_audio, _ = audioop.ratecv(raw_audio, 2, 1, rate, 16000, None)
                return pcm_audio
            return raw_audio
        else:
            # Unknown codec — pass through and hope for the best
            return raw_audio

    async def handle_media_stream(self, websocket: Any) -> None:
        """Handle the incoming media stream WebSocket from Telnyx.

        Called by the ToolWebhookService when a WS connects to
        /media-stream/{conversation_id}. Uses FastAPI WebSocket interface.
        """
        self._stream_ws = websocket
        self._connected_event.set()
        self._disconnected_event.clear()
        logger.info("Telnyx media stream connected for conversation %s", self.conversation_id)

        try:
            while True:
                raw_message = await websocket.receive_text()
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message on media stream")
                    continue

                event_type = message.get("event")
                if event_type == "media":
                    media_obj = message.get("media", {})
                    payload_b64 = media_obj.get("payload")
                    track = media_obj.get("track", "unknown")
                    logger.debug(
                        "Media: track=%s, %d bytes, codec=%s@%dHz",
                        track,
                        len(payload_b64) if payload_b64 else 0,
                        self._inbound_encoding,
                        self._inbound_sample_rate,
                    )
                    if payload_b64:
                        raw_audio = base64.b64decode(payload_b64)
                        pcm_16khz = self._convert_inbound_audio(raw_audio)
                        await self.emit_audio(pcm_16khz)
                elif event_type == "start":
                    self._stream_id = message.get("stream_id")
                    start_info = message.get("start", {})
                    media_format = start_info.get("media_format", {})
                    # Detect actual inbound codec from Telnyx
                    if media_format:
                        self._inbound_encoding = media_format.get("encoding", "L16")
                        self._inbound_sample_rate = int(media_format.get("sample_rate", 16000))
                    logger.info(
                        "Media stream started: stream_id=%s, inbound_codec=%s@%dHz (raw media_format=%s)",
                        self._stream_id,
                        self._inbound_encoding,
                        self._inbound_sample_rate,
                        media_format,
                    )
                    if self._inbound_encoding != "L16":
                        logger.warning(
                            "Inbound codec is %s, not L16 — will convert to 16kHz PCM on receive path",
                            self._inbound_encoding,
                        )
                elif event_type == "stop":
                    logger.info("Media stream stopped for conversation %s", self.conversation_id)
                    break
                elif event_type == "connected":
                    logger.info("Media stream signaled 'connected' for %s", self.conversation_id)
                else:
                    logger.debug("Unknown media stream event: %s", event_type)
        except Exception as exc:
            logger.info("Media stream WebSocket closed for %s: %s", self.conversation_id, exc)
        finally:
            self._stream_ws = None
            self._disconnected_event.set()
            logger.info("Media stream handler exiting for conversation %s", self.conversation_id)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("HTTP session not initialized")

        url = f"{self.api_base_url}{path}"
        async with self._session.post(url, json=payload) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"Telnyx API {resp.status} at {url}: {json.dumps(body)}")
            return body
