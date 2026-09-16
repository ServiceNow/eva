"""Caller speech synthesis via Cartesia Sonic."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from eva.user_simulator.cascade.constants import CALLER_SAMPLE_RATE
from eva.utils.logging import get_logger

logger = get_logger(__name__)

CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-06-10"
DEFAULT_FEMALE_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
DEFAULT_MALE_VOICE = "a0e99841-438c-4a64-b679-ae501e7d6091"
_FEMALE_PERSONA_ID = 1


class CartesiaTTS:
    """Renders caller text to PCM16 at the simulator's sample rate."""

    def __init__(self, params: dict[str, Any], *, language: str = "en") -> None:
        self._model = params.get("model", "sonic-3.5")
        self._api_key = params.get("api_key") or os.environ.get("CARTESIA_API_KEY", "")
        self._female_voice = params.get("female_voice", DEFAULT_FEMALE_VOICE)
        self._male_voice = params.get("male_voice", DEFAULT_MALE_VOICE)
        self._language = language

    def voice_for_persona(self, persona_config: dict[str, Any]) -> str:
        """Pick a stable voice for this persona, mirroring the existing gender scheme."""
        if persona_config.get("user_persona_id") == _FEMALE_PERSONA_ID:
            return self._female_voice
        if persona_config.get("user_persona_id") is None:
            return self._female_voice
        return self._male_voice

    async def stream(self, text: str, *, voice_id: str) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks as they render, so playout can start before the tail exists."""
        if not text:
            return
        if not self._api_key:
            raise ValueError("Cartesia API key missing: set tts_params.api_key or CARTESIA_API_KEY")

        body = {
            "model_id": self._model,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "language": self._language,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": CALLER_SAMPLE_RATE,
            },
        }
        headers = {"X-API-Key": self._api_key, "Cartesia-Version": CARTESIA_VERSION}

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", CARTESIA_URL, json=body, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    async def synthesize(self, text: str, *, voice_id: str) -> bytes:
        """Render text to raw PCM16 mono at CALLER_SAMPLE_RATE."""
        return b"".join([chunk async for chunk in self.stream(text, voice_id=voice_id)])
