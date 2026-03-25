"""Telnyx Call Control transport backed by media streaming websockets."""

import asyncio
import base64
import json
import socket
from typing import Any

import aiohttp
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from eva.assistant.telephony_bridge import BaseTelephonyTransport
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_TELNYX_API_BASE_URL = "https://api.telnyx.com/v2"
_CALL_CONNECT_TIMEOUT_SECONDS = 60.0
_REQUEST_TIMEOUT_SECONDS = 30.0


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(probe.getsockname()[1])


class CallControlTransport(BaseTelephonyTransport):
    """Telnyx Call Control transport using bidirectional media streaming."""

    def __init__(
        self,
        api_key: str,
        to: str,
        stream_url: str,
        connection_id: str,
        from_number: str,
        conversation_id: str,
        webhook_base_url: str,
        *,
        local_ws_host: str = "0.0.0.0",
        local_ws_port: int | None = None,
        connect_timeout_seconds: float = _CALL_CONNECT_TIMEOUT_SECONDS,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        api_base_url: str = _TELNYX_API_BASE_URL,
    ):
        super().__init__(sip_uri=to, conversation_id=conversation_id, webhook_base_url=webhook_base_url)
        self.api_key = api_key
        self.to = to
        self.stream_url = stream_url
        self.connection_id = connection_id
        self.from_number = from_number
        self.local_ws_host = local_ws_host
        self.local_ws_port = local_ws_port or _find_available_port()
        self.connect_timeout_seconds = connect_timeout_seconds
        self.api_base_url = api_base_url.rstrip("/")
        self._request_timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)

        self._session: aiohttp.ClientSession | None = None
        self._server = None
        self._stream_connection: ServerConnection | None = None
        self._call_control_id: str | None = None
        self._stream_id: str | None = None
        self._connected_event = asyncio.Event()
        self._disconnected_event = asyncio.Event()
        self._send_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._server is not None:
            logger.warning("Call Control transport already started for %s", self.to)
            return

        logger.info("Starting Telnyx Call Control transport for %s", self.to)
        self._connected_event.clear()
        self._disconnected_event.clear()
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._request_timeout,
        )
        self._server = await serve(
            self._handle_stream_connection,
            self.local_ws_host,
            self.local_ws_port,
            compression=None,
        )

        try:
            response = await self._post(
                "/calls",
                {
                    "connection_id": self.connection_id,
                    "to": self.to,
                    "from": self.from_number,
                    "stream_url": self._resolved_stream_url(),
                    "stream_track": "both_tracks",
                    "stream_bidirectional_mode": "rtp",
                    "stream_bidirectional_codec": "PCMU",
                },
            )
            data = response.get("data", {})
            self._call_control_id = data.get("call_control_id")
            if not self._call_control_id:
                raise RuntimeError("Telnyx call creation response did not include data.call_control_id")

            await asyncio.wait_for(self._connected_event.wait(), timeout=self.connect_timeout_seconds)
            logger.info("Telnyx Call Control media stream connected for %s", self.to)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._call_control_id and not self._disconnected_event.is_set():
            try:
                await self._post(f"/calls/{self._call_control_id}/actions/hangup", {})
            except Exception as exc:
                logger.warning("Failed to hang up Telnyx call %s: %s", self._call_control_id, exc)

        if self._stream_connection is not None:
            try:
                await self._stream_connection.close()
            except Exception as exc:
                logger.debug("Ignoring stream websocket close error: %s", exc)
            self._stream_connection = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        self._stream_id = None
        self._call_control_id = None
        self._connected_event.clear()

    async def send_audio(self, audio_data: bytes) -> None:
        if not audio_data:
            return

        async with self._send_lock:
            if self._stream_connection is None:
                logger.debug("Dropping outbound audio because Telnyx media stream is not connected")
                return

            try:
                await self._stream_connection.send(
                    json.dumps(
                        {
                            "event": "media",
                            "media": {
                                "payload": base64.b64encode(audio_data).decode("ascii"),
                            },
                        }
                    )
                )
            except ConnectionClosed:
                logger.info("Telnyx media stream disconnected while sending audio for %s", self.to)

    def _resolved_stream_url(self) -> str:
        if "{port}" in self.stream_url:
            return self.stream_url.replace("{port}", str(self.local_ws_port))
        return self.stream_url

    async def _handle_stream_connection(self, websocket: ServerConnection) -> None:
        if self._stream_connection is not None:
            logger.warning("Rejecting duplicate Telnyx media stream connection for %s", self.to)
            await websocket.close(code=1013, reason="stream already connected")
            return

        self._stream_connection = websocket
        self._connected_event.set()
        self._disconnected_event.clear()
        logger.info("Telnyx media stream connected for conversation %s", self.conversation_id)

        try:
            async for message in websocket:
                if not isinstance(message, str):
                    logger.debug("Ignoring non-text Telnyx media stream frame")
                    continue
                await self._handle_stream_message(message)
        except ConnectionClosed:
            logger.info("Telnyx media stream disconnected for conversation %s", self.conversation_id)
        finally:
            if self._stream_connection is websocket:
                self._stream_connection = None
            self._disconnected_event.set()

    async def _handle_stream_message(self, payload: str) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid JSON media stream frame from Telnyx")
            return

        event = str(message.get("event", "")).lower()
        if event == "start":
            start_payload = message.get("start", {})
            self._call_control_id = start_payload.get("call_control_id", self._call_control_id)
            self._stream_id = message.get("stream_id", self._stream_id)
            media_format = start_payload.get("media_format", {})
            encoding = media_format.get("encoding")
            sample_rate = media_format.get("sample_rate")
            if encoding and (encoding != "PCMU" or sample_rate not in (8000, "8000")):
                logger.warning("Unexpected Telnyx media format: %s", media_format)
            return

        if event == "media":
            media_payload = message.get("media", {})
            track = str(media_payload.get("track", "")).lower()
            if track and track not in {"inbound", "inbound_track"}:
                return

            encoded_audio = media_payload.get("payload")
            if not encoded_audio:
                return

            try:
                audio_data = base64.b64decode(encoded_audio)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid base64 audio payload from Telnyx")
                return

            await self.emit_audio(audio_data)
            return

        if event == "stop":
            stop_payload = message.get("stop", {})
            self._call_control_id = stop_payload.get("call_control_id", self._call_control_id)
            self._stream_id = None
            logger.info("Received Telnyx stream stop event for conversation %s", self.conversation_id)
            return

        if event == "error":
            logger.error("Received Telnyx media stream error: %s", message)
            return

        logger.debug("Ignoring Telnyx media stream event: %s", message)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Call Control HTTP session is not initialized")

        url = f"{self.api_base_url}/{path.lstrip('/')}"
        async with self._session.post(url, json=payload) as response:
            response_text = await response.text()
            if response.status >= 400:
                raise RuntimeError(
                    f"Telnyx API request failed with status {response.status}: {response_text or response.reason}"
                )
            if not response_text:
                return {}
            return json.loads(response_text)
