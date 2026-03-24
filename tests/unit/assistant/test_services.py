"""Tests for assistant/pipeline/services.py — service factory functions."""

from unittest.mock import MagicMock

import pytest

from eva.assistant.pipeline.services import (
    ASCIITextFilter,
    _resolve_url,
    create_stt_service,
    create_tts_service,
)


class TestResolveUrl:
    def test_single_url_returns_it_without_incrementing(self):
        url, counter = _resolve_url({"url": "https://example.com"}, 0)
        assert url == "https://example.com"
        assert counter == 0

    def test_round_robin_cycles_through_urls(self):
        params = {"urls": ["https://a.com", "https://b.com", "https://c.com"]}

        url0, c0 = _resolve_url(params, 0)
        url1, c1 = _resolve_url(params, c0)
        url2, c2 = _resolve_url(params, c1)
        url3, c3 = _resolve_url(params, c2)  # wraps

        assert [url0, url1, url2, url3] == ["https://a.com", "https://b.com", "https://c.com", "https://a.com"]
        assert c3 == 4

    def test_empty_urls_list_falls_back_to_url(self):
        url, counter = _resolve_url({"urls": [], "url": "https://fallback.com"}, 0)
        assert url == "https://fallback.com"
        assert counter == 0

    def test_no_url_or_urls_returns_none(self):
        url, counter = _resolve_url({}, 5)
        assert url is None
        assert counter == 5

    def test_non_list_urls_falls_back(self):
        """Urls must be a list to trigger round-robin."""
        url, _ = _resolve_url({"urls": "not-a-list", "url": "https://fb.com"}, 0)
        assert url == "https://fb.com"


class TestCreateSttService:
    def test_none_disables_stt(self):
        assert create_stt_service(None) is None

    def test_unknown_model_raises_with_available_list(self):
        with pytest.raises(ValueError, match="Available:.*deepgram"):
            create_stt_service("nonexistent_provider", params={"api_key": "k"})

    def test_nvidia_requires_url(self):
        with pytest.raises(ValueError, match="url required"):
            create_stt_service("nvidia", params={"api_key": "k"})

    def test_nvidia_baseten_requires_url(self):
        with pytest.raises(ValueError, match="url required"):
            create_stt_service("nvidia-baseten", params={"api_key": "k"})

    def test_deepgram_returns_deepgram_service(self):
        svc = create_stt_service("deepgram", params={"api_key": "k", "model": "nova-2"})
        assert "Deepgram" in type(svc).__name__
        assert "Flux" not in type(svc).__name__

    def test_deepgram_flux_returns_flux_variant(self):
        svc = create_stt_service("deepgram-flux", params={"api_key": "k", "model": "nova-3"})
        assert "Flux" in type(svc).__name__

    def test_openai_service_respects_custom_model(self):
        svc = create_stt_service("openai", params={"api_key": "k", "model": "whisper-2"})
        assert svc._settings.model == "whisper-2"

    def test_cartesia_service_created(self):
        svc = create_stt_service("cartesia", params={"api_key": "k", "model": "ink"})
        assert "Cartesia" in type(svc).__name__


class TestCreateTtsService:
    def test_none_disables_tts(self):
        assert create_tts_service(None) is None

    def test_unknown_model_raises_with_available_list(self):
        with pytest.raises(ValueError, match="Available:.*cartesia"):
            create_tts_service("nonexistent_provider", params={"api_key": "k"})

    def test_nvidia_baseten_requires_api_key_and_url(self):
        with pytest.raises(ValueError, match="url required"):
            create_tts_service("nvidia-baseten", params={"api_key": "k"})

    def test_gemini_unavailable_gives_clear_error(self):
        import eva.assistant.pipeline.services as svc_mod

        orig = svc_mod.GEMINI_AVAILABLE
        try:
            svc_mod.GEMINI_AVAILABLE = False
            with pytest.raises(ValueError, match="Gemini services are unavailable"):
                create_tts_service("gemini", params={"api_key": "k"})
        finally:
            svc_mod.GEMINI_AVAILABLE = orig

    def test_cartesia_returns_cartesia_service(self):
        svc = create_tts_service("cartesia", params={"api_key": "k", "model": "sonic"})
        assert "Cartesia" in type(svc).__name__

    def test_elevenlabs_returns_elevenlabs_service(self):
        svc = create_tts_service("elevenlabs", params={"api_key": "k", "model": "eleven_turbo_v2"})
        assert "ElevenLabs" in type(svc).__name__

    def test_openai_respects_voice_param(self):
        svc = create_tts_service("openai", params={"api_key": "k", "model": "tts-1", "voice": "nova"})
        assert svc._settings.voice == "nova"

    def test_openai_azure_url_uses_azure_client(self):
        svc = create_tts_service(
            "openai",
            params={"api_key": "k", "model": "tts-1", "url": "https://my-azure.openai.com/tts"},
        )
        # Azure URL should trigger AsyncAzureOpenAI client
        assert "Azure" in type(svc._client).__name__


class TestCreateRealtimeLlmService:
    def test_unknown_model_raises(self):
        from eva.assistant.pipeline.services import create_realtime_llm_service

        agent = MagicMock()
        agent.description = "Test"
        agent.instructions = "Test"
        agent.build_tools_for_realtime.return_value = None

        with pytest.raises(ValueError, match="Unknown realtime model.*Available:"):
            create_realtime_llm_service("unknown-model", agent=agent)


class TestASCIITextFilter:
    @pytest.mark.asyncio
    async def test_smart_quotes_and_dashes_normalized(self):
        filt = ASCIITextFilter()
        text = "\u201cHello\u201d \u2014 world\u2026"
        result = await filt.filter(text)
        assert result == '"Hello" - world...'

    @pytest.mark.asyncio
    async def test_unmapped_non_ascii_removed(self):
        filt = ASCIITextFilter()
        # é (U+00E9) is not in the char map → stripped
        assert await filt.filter("café") == "caf"

    @pytest.mark.asyncio
    async def test_pure_ascii_unchanged(self):
        filt = ASCIITextFilter()
        text = "Hello, world! 123 #$%"
        assert await filt.filter(text) == text

    @pytest.mark.asyncio
    async def test_non_breaking_spaces_become_regular_spaces(self):
        filt = ASCIITextFilter()
        # Both U+00A0 and U+202F should become regular space
        assert await filt.filter("A\u00a0B\u202fC") == "A B C"

    @pytest.mark.asyncio
    async def test_all_dash_variants_become_hyphen(self):
        filt = ASCIITextFilter()
        dashes = "\u2010\u2011\u2012\u2013\u2014\u2015"
        assert await filt.filter(dashes) == "------"
