"""Streaming speech synthesis for the cascade caller."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator

import httpx

TTS_SAMPLE_RATE = 16000
"""Matches what BotToBotAudioBridge.output() expects."""

CARTESIA_SSE_URL = "https://api.cartesia.ai/tts/sse"
CARTESIA_VERSION = "2025-11-04"

CALLER_SPEED = 1.2  # TODO Will use this for the generated barge ins
"""Cartesia generation_config.speed for the caller. People speed up when they cut in, and it
shortens pre-rendered openers. Fixed for now; promote to config if it needs tuning."""

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
        self._client = httpx.AsyncClient(
            headers={
                "X-API-Key": api_key or os.environ.get("CARTESIA_API_KEY", ""),
                "Cartesia-Version": CARTESIA_VERSION,
            },
            timeout=30.0,
            limits=KEEPALIVE,
        )

    def _body(self, text: str, voice_id: str) -> dict:
        body = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "language": self.language,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": TTS_SAMPLE_RATE},
            # "generation_config": {"speed": CALLER_SPEED},
        }
        return body

    async def stream(self, text: str, *, voice_id: str) -> AsyncIterator[bytes]:
        """Yield sample-aligned PCM16 chunks so playout can start before the whole line is rendered."""
        carry = b""
        async with self._client.stream("POST", CARTESIA_SSE_URL, json=self._body(text, voice_id)) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                if event.get("type") == "error":
                    raise RuntimeError(f"Caller TTS failed: {event.get('error') or event}")
                if event.get("type") != "chunk" or not event.get("data"):
                    continue
                # Chunk boundaries need not fall on samples; an odd split would shear PCM16.
                data = carry + base64.b64decode(event["data"])
                cut = len(data) - len(data) % 2
                carry = data[cut:]
                if cut:
                    yield data[:cut]

    async def warm_up(self, voice_id: str, *, connections: int = 1) -> None:
        """Open ``connections`` pooled connections now, so no handshake lands on a caller turn."""

        async def render() -> None:
            async for _ in self.stream("Okay.", voice_id=voice_id):
                pass

        await asyncio.gather(*(render() for _ in range(connections)))

    async def close(self) -> None:
        await self._client.aclose()
