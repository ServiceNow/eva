"""Streaming speech synthesis for the cascade caller."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress

import httpx
import websockets
from websockets.protocol import State

TTS_SAMPLE_RATE = 16000
"""Matches what BotToBotAudioBridge.output() expects."""

CARTESIA_SSE_URL = "https://api.cartesia.ai/tts/sse"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2025-11-04"

STREAM_BUFFER_DELAY_MS = 500
"""Cartesia's ``max_buffer_delay_ms`` for streamed text: the longest it holds text back waiting for a
natural break before rendering. The 3000ms default would stall a barge-in's first words; lower risks
choppier phrasing."""

CALLER_SPEED = 1.2
"""Cartesia generation_config.speed for barge-ins (openers and replies); regular turns use the voice's
default. People speed up when they cut in, and it shortens pre-rendered openers. Fixed for now; promote
to config if it needs tuning."""

KEEPALIVE = httpx.Limits(max_keepalive_connections=8, keepalive_expiry=300.0)
"""httpx's 5s default would drop the warm connection during a long assistant turn, putting a
TLS handshake (~100ms) in front of the caller's next line."""


class CartesiaStreamingTTS:
    """Renders caller text to 16kHz PCM16, yielding chunks as they are generated.

    Calls Cartesia's SSE endpoint over a long-lived httpx client rather than the Cartesia SDK:
    the SDK's ``generate_sse`` (3.0.2) buffers the whole response before yielding its first
    event, so first audio waited for the entire line to render (~0.8-1.2s against ~250ms).
    """

    def __init__(self, model: str, *, language: str = "en", api_key: str | None = None) -> None:
        self.model = model
        self.language = language.split("-")[0]
        self._headers = {
            "X-API-Key": api_key or os.environ.get("CARTESIA_API_KEY", ""),
            "Cartesia-Version": CARTESIA_VERSION,
        }
        self._client = httpx.AsyncClient(headers=self._headers, timeout=30.0, limits=KEEPALIVE)
        self._ws: websockets.ClientConnection | None = None
        self._ws_lock = asyncio.Lock()
        """One streamed context at a time on the socket; barge-ins never overlap anyway."""

    def _body(self, text: str, voice_id: str, speed: float | None = None) -> dict:
        body = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "language": self.language,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": TTS_SAMPLE_RATE},
        }
        if speed is not None:
            body["generation_config"] = {"speed": speed}
        return body

    async def stream(self, text: str, *, voice_id: str, speed: float | None = None) -> AsyncIterator[bytes]:
        """Yield sample-aligned PCM16 chunks so playout can start before the whole line is rendered."""
        carry = b""
        body = self._body(text, voice_id, speed)
        async with self._client.stream("POST", CARTESIA_SSE_URL, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                if event.get("type") == "error":
                    raise RuntimeError(f"Caller TTS failed: {event.get('error') or event}")
                if event.get("type") != "chunk" or not event.get("data"):
                    continue
                pcm, carry = _align(carry + base64.b64decode(event["data"]))
                if pcm:
                    yield pcm

    async def stream_text(
        self, text: AsyncIterator[str], *, voice_id: str, speed: float | None = None
    ) -> AsyncIterator[bytes]:
        """Render text while it is still being written, yielding PCM16 chunks as they are generated.

        Each piece is sent the moment it arrives as a continuation of one Cartesia WebSocket context, so
        Cartesia decides where to break and keeps prosody across pieces in any language; no client-side
        splitting. Pieces are joined verbatim, so they must carry their own spacing (LLM deltas do).
        Yields nothing when ``text`` is empty.
        """
        async with self._ws_lock:
            ws = await self._socket()
            context_id = uuid.uuid4().hex
            out: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
            done = False

            def request(transcript: str, more: bool) -> str:
                body = self._body(transcript, voice_id, speed)
                body.update(context_id=context_id, max_buffer_delay_ms=STREAM_BUFFER_DELAY_MS)
                body["continue"] = more
                return json.dumps(body)

            async def send() -> None:
                sent = False
                async for piece in text:
                    if piece:
                        await ws.send(request(piece, True))
                        sent = True
                if not sent:
                    out.put_nowait(None)
                    return
                await ws.send(request("", False))  # no more input: render what is buffered and finish

            async def receive() -> None:
                async for raw in ws:
                    event = json.loads(raw)
                    if event.get("context_id") != context_id:
                        continue  # left over from a cancelled context
                    if event.get("type") == "error":
                        raise RuntimeError(f"Caller TTS failed: {event.get('message') or event}")
                    if event.get("type") == "chunk" and event.get("data"):
                        out.put_nowait(base64.b64decode(event["data"]))
                    if event.get("done"):
                        out.put_nowait(None)
                        return
                # Closed before ``done``: the reply was cut off, which must not pass for a finished line.
                raise RuntimeError("Caller TTS socket closed before the line finished")

            def relay_error(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is not None:
                    out.put_nowait(task.exception())

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            for task in tasks:
                task.add_done_callback(relay_error)
            carry = b""
            try:
                while (item := await out.get()) is not None:
                    if isinstance(item, BaseException):
                        raise item
                    pcm, carry = _align(carry + item)
                    if pcm:
                        yield pcm
                done = True
            finally:
                for task in tasks:
                    task.cancel()
                if not done:
                    # Stop rendering audio nobody will play; a broken socket is reopened on next use.
                    with suppress(Exception):
                        await ws.send(json.dumps({"context_id": context_id, "cancel": True}))

    async def _socket(self) -> websockets.ClientConnection:
        if self._ws is None or self._ws.state is not State.OPEN:
            self._ws = await websockets.connect(CARTESIA_WS_URL, additional_headers=self._headers)
        return self._ws

    async def warm_up(self, voice_id: str, *, connections: int = 1) -> None:
        """Open ``connections`` pooled SSE connections and the streaming socket now.

        Done before the first turn, so no handshake lands on a caller turn.
        """

        async def render() -> None:
            async for _ in self.stream("Okay.", voice_id=voice_id):
                pass

        await asyncio.gather(*(render() for _ in range(connections)), self._socket())

    async def close(self) -> None:
        await self._client.aclose()
        if self._ws is not None:
            await self._ws.close()


def _align(data: bytes) -> tuple[bytes, bytes]:
    """Split ``data`` into whole PCM16 samples and the odd trailing byte, if any.

    Chunk boundaries need not fall on samples; an odd split would shear PCM16.
    """
    cut = len(data) // 2 * 2
    return data[:cut], data[cut:]
