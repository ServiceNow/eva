"""Magpie TTS service using the Riva HTTP API format.

Magpie TTS exposes two endpoints:
  - /v1/audio/synthesize         (batch, returns complete audio)
  - /v1/audio/synthesize_online  (streaming, returns chunked audio)

Both accept multipart/form-data with fields:
  text, voice, language, sample_rate_hz, encoding
"""

from typing import AsyncGenerator, Optional

import aiohttp
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class MagpieTTSService(TTSService):
    """TTS service for Magpie NIM servers (Riva HTTP API)."""

    def __init__(
        self,
        *,
        base_url: str,
        voice: str = "Magpie-Multilingual.EN-US.Aria",
        language: str = "en-US",
        sample_rate: int = 24000,
        streaming: bool = True,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(
                model="magpie",
                voice=voice,
                language=language,
            ),
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._language = language
        self._streaming = streaming
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _form_data(self, text: str) -> aiohttp.FormData:
        data = aiohttp.FormData()
        data.add_field("text", text)
        data.add_field("voice", self._voice)
        data.add_field("language", self._language)
        data.add_field("sample_rate_hz", str(self.sample_rate))
        data.add_field("encoding", "LINEAR_PCM")
        return data

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()
            session = await self._ensure_session()

            if self._streaming:
                url = f"{self._base_url}/v1/audio/synthesize_online"
                async with session.post(url, data=self._form_data(text)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"{self} error (status {resp.status}): {body}")
                        yield ErrorFrame(error=f"Magpie TTS error (status {resp.status}): {body}")
                        return

                    await self.start_tts_usage_metrics(text)
                    yield TTSStartedFrame(context_id=context_id)

                    async for chunk in resp.content.iter_any():
                        if chunk:
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(
                                audio=chunk,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )

                    yield TTSStoppedFrame(context_id=context_id)
            else:
                url = f"{self._base_url}/v1/audio/synthesize"
                async with session.post(url, data=self._form_data(text)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"{self} error (status {resp.status}): {body}")
                        yield ErrorFrame(error=f"Magpie TTS error (status {resp.status}): {body}")
                        return

                    await self.start_tts_usage_metrics(text)
                    yield TTSStartedFrame(context_id=context_id)

                    audio_data = await resp.read()
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

                    yield TTSStoppedFrame(context_id=context_id)

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Magpie TTS error: {e}")

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._close_session()

    async def stop(self, frame):
        await super().stop(frame)
        await self._close_session()

    async def _close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
