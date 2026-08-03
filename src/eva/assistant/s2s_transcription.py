"""Pluggable batch transcription for S2S frameworks that emit no transcript.

Some speech-to-speech models (notably Smallest Hydra) stream audio in both
directions but never put text on the wire. EVA's audit log — which nearly all
accuracy/experience metrics read — still needs a per-turn transcript, so the
server transcribes each completed utterance (the PCM it sent or received) with a
separate batch STT call.

The provider is configurable via the ``transcription`` block inside
``s2s_params``::

    "transcription": {
        "provider": "smallest" | "openai" | "deepgram",
        "model": "...",          # provider-specific; sensible default per provider
        "api_key": "...",        # falls back to the S2S api_key (smallest) / env
        "language": "en",        # falls back to the run language
        "base_url": "..."        # optional override
    }

All providers are called over a single shared async ``httpx`` client so the
module has no heavy SDK import surface and is straightforward to mock in tests.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from eva.assistant.audio_bridge import pcm16_to_wav_bytes
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_SMALLEST_STT_URL = "https://api.smallest.ai/waves/v1/stt/"
_OPENAI_STT_URL = "https://api.openai.com/v1/audio/transcriptions"
_DEEPGRAM_STT_URL = "https://api.deepgram.com/v1/listen"

# Default model per provider. Smallest Pulse-pro is English-only; for any other
# language we fall back to the multilingual Pulse model.
_DEFAULT_MODELS = {
    "smallest": "pulse-pro",
    "openai": "whisper-1",
    "deepgram": "nova-3",
}


class BatchTranscriber:
    """Transcribe completed utterances via a configurable batch STT provider."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        language: str,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.language = language
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe raw 16-bit mono PCM. Returns "" on any failure (fail-soft).

        Transcription errors must never abort a conversation — a dropped
        transcript degrades that turn's text metrics but the audio recording and
        tool calls are preserved.
        """
        if not pcm:
            return ""
        wav = pcm16_to_wav_bytes(pcm, sample_rate)
        try:
            if self.provider == "smallest":
                return await self._transcribe_smallest(wav)
            if self.provider == "openai":
                return await self._transcribe_openai(wav)
            if self.provider == "deepgram":
                return await self._transcribe_deepgram(wav)
            logger.error(f"Unknown transcription provider: {self.provider!r}")
            return ""
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.error(f"{self.provider} transcription failed: {e}", exc_info=True)
            return ""

    async def _transcribe_smallest(self, wav: bytes) -> str:
        url = self.base_url or _SMALLEST_STT_URL
        resp = await self._client.post(
            url,
            params={"model": self.model, "language": self.language},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/octet-stream",
            },
            content=wav,
        )
        resp.raise_for_status()
        return (resp.json().get("transcription") or "").strip()

    async def _transcribe_openai(self, wav: bytes) -> str:
        url = self.base_url or _OPENAI_STT_URL
        resp = await self._client.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": self.model, "language": self.language},
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()

    async def _transcribe_deepgram(self, wav: bytes) -> str:
        url = self.base_url or _DEEPGRAM_STT_URL
        resp = await self._client.post(
            url,
            params={"model": self.model, "language": self.language, "smart_format": "true"},
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
            content=wav,
        )
        resp.raise_for_status()
        alternatives = resp.json()["results"]["channels"][0]["alternatives"]
        return (alternatives[0]["transcript"] if alternatives else "").strip()

    async def aclose(self) -> None:
        await self._client.aclose()


def create_transcriber(s2s_params: dict[str, Any], language: str) -> BatchTranscriber:
    """Build a :class:`BatchTranscriber` from ``s2s_params['transcription']``.

    Defaults to the Smallest Pulse provider keyed on the S2S ``api_key`` so an
    in-stack transcript works with no extra configuration. The Pulse-pro model is
    English-only; for any non-English run the default model falls back to the
    multilingual ``pulse`` model.
    """
    cfg = dict(s2s_params.get("transcription") or {})
    provider = cfg.get("provider", "smallest")
    lang = cfg.get("language") or language or "en"

    model = cfg.get("model")
    if not model:
        model = _DEFAULT_MODELS.get(provider, "")
        # Smallest Pulse-pro is English-only; use multilingual Pulse otherwise.
        if provider == "smallest" and not lang.startswith("en"):
            model = "pulse"

    # API key resolution: explicit transcription key, else a sensible fallback —
    # the shared S2S key for Smallest, or the provider's standard env var.
    api_key = cfg.get("api_key")
    if not api_key:
        if provider == "smallest":
            api_key = s2s_params.get("api_key", "")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "deepgram":
            api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key:
        raise ValueError(
            f"No API key for transcription provider {provider!r}. "
            f"Set s2s_params.transcription.api_key (or s2s_params.api_key for 'smallest', "
            f"OPENAI_API_KEY for 'openai', DEEPGRAM_API_KEY for 'deepgram')."
        )

    logger.info(f"S2S transcription: provider={provider} model={model} language={lang}")
    return BatchTranscriber(
        provider=provider,
        model=model,
        api_key=api_key,
        language=lang,
        base_url=cfg.get("base_url"),
    )
