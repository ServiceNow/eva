"""Audio language model client for Gemini via OpenAI-compatible API.

Two auth modes are supported:

1. API key (Gemini Developer API) — uses
   https://generativelanguage.googleapis.com/v1beta/openai/

2. Vertex AI (service-account / ADC) — uses
   https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/
   locations/{LOCATION}/endpoints/openapi
   Auth is via an OAuth access token minted from Application Default
   Credentials (e.g. GOOGLE_APPLICATION_CREDENTIALS=service-account.json).

Both surfaces accept OpenAI-style chat.completions with `input_audio` content,
so the on-the-wire shape is identical and the client only differs in URL/auth.
"""

import asyncio
import datetime
import time
from typing import Any

import google.auth
import google.auth.transport.requests
from openai import AsyncOpenAI

from eva.assistant.pipeline.alm_base import (
    DEFAULT_NUM_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_TRANSCRIPTION_PROMPT,
    BaseALMClient,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
# Refresh ADC token when fewer than this many seconds remain on it.
TOKEN_REFRESH_MARGIN_SEC = 300


def _vertex_base_url(project: str, location: str) -> str:
    # The "global" location uses the un-prefixed aiplatform.googleapis.com host;
    # all regional locations use {location}-aiplatform.googleapis.com.
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/"


class ALMGeminiClient(BaseALMClient):
    """Audio-LLM client for Gemini's OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        project: str | None = None,
        location: str | None = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            initial_delay=initial_delay,
            sample_rate=sample_rate,
            num_channels=num_channels,
            sample_width=sample_width,
        )

        self._adc_credentials = None  # Set in vertex mode for token refresh
        self._is_vertex = bool(project and location)

        if self._is_vertex:
            # Vertex OpenAI-compat requires the "google/" or "publishers/google/models/"
            # prefix on the model id; auto-add the simple form if the caller passed a bare name.
            if "/" not in self.model:
                self.model = f"google/{self.model}"
            self.base_url = (base_url or _vertex_base_url(project, location)).rstrip("/") + "/"
            self._adc_credentials, detected_project = google.auth.default(scopes=VERTEX_SCOPES)
            logger.info(
                f"ALMGeminiClient using Vertex AI: project={project} (ADC project={detected_project}), "
                f"location={location}"
            )
            token = self._refresh_adc_token()
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=token,
                timeout=120.0,
            )
        else:
            if not api_key:
                raise ValueError(
                    "ALMGeminiClient requires either api_key (Gemini Developer API) "
                    "or project+location (Vertex AI via ADC)."
                )
            self.base_url = (base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/") + "/"
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=api_key,
                timeout=120.0,
            )

        logger.info(
            f"Initialized ALMGeminiClient: base_url={self.base_url}, model={self.model}, "
            f"sample_rate={self.sample_rate}, vertex={self._is_vertex}"
        )

    def _audio_content_part(self, audio_b64: str) -> dict[str, Any]:
        return {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"},
        }

    def _refresh_adc_token(self) -> str:
        """Mint a fresh OAuth access token from the cached ADC credentials."""
        if self._adc_credentials is None:
            raise RuntimeError("_refresh_adc_token called without ADC credentials")
        request = google.auth.transport.requests.Request()
        self._adc_credentials.refresh(request)
        return self._adc_credentials.token

    def _maybe_refresh_token(self) -> None:
        """Refresh the OpenAI client's bearer token if the ADC token is stale."""
        if not self._is_vertex or self._adc_credentials is None:
            return
        creds = self._adc_credentials
        needs_refresh = not creds.valid
        if not needs_refresh and creds.expiry is not None:
            now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            remaining = (creds.expiry - now).total_seconds()
            needs_refresh = remaining < TOKEN_REFRESH_MARGIN_SEC
        if needs_refresh:
            token = self._refresh_adc_token()
            self._client.api_key = token

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Chat completion against Gemini's OpenAI-compatible endpoint."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._maybe_refresh_token()
                start_time = time.time()
                response = await self._client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                message = response.choices[0].message
                usage = response.usage

                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

                stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": response.choices[0].finish_reason or "unknown",
                    "model": response.model or self.model,
                    "cost": 0.0,
                    "cost_source": "gemini_openai_compat",
                    "latency": round(elapsed, 3),
                    "reasoning": None,
                    "reasoning_content": None,
                }

                if hasattr(message, "tool_calls") and message.tool_calls:
                    return message, stats
                return message.content or "", stats

            except Exception as e:
                last_exception = e
                if self._is_retryable(e) and attempt < self.max_retries:
                    delay = self.initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"ALMGeminiClient completion failed: {e}")
                raise

        raise last_exception  # type: ignore[misc]

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_sample_rate: int,
        system_prompt: str | None = None,
    ) -> str | None:
        """Transcribe a chunk of PCM16 audio via Gemini chat completions."""
        if not audio_bytes:
            return None

        prompt = system_prompt or DEFAULT_TRANSCRIPTION_PROMPT
        user_msg = self.build_audio_user_message(audio_bytes, source_sample_rate)
        messages = [{"role": "system", "content": prompt}, user_msg]

        try:
            self._maybe_refresh_token()
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content if response.choices else None
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"ALMGeminiClient transcription failed: {e}")
            return None
