"""Service factories for STT, TTS, and LLM services.

Creates Pipecat services with proper configuration.
"""

import dataclasses
import datetime
import os
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncAzureOpenAI, BadRequestError
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService, DeepgramFluxSTTSettings
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.smallest.stt import SmallestSTTService
from pipecat.services.smallest.tts import SmallestTTSService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.xai.stt import XAISTTService
from pipecat.services.xai.tts import XAITTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_filter import BaseTextFilter
from websockets.asyncio.client import connect as websocket_connect

from eva.assistant.pipeline.alm_base import BaseALMClient
from eva.assistant.pipeline.alm_gemini import ALMGeminiClient
from eva.assistant.pipeline.alm_vllm import ALMvLLMClient
from eva.assistant.pipeline.nvidia_baseten import BasetenSTTService, BasetenTTSService
from eva.assistant.pipeline.realtime_llm import InstrumentedRealtimeLLMService
from eva.models.agents import AgentConfig

# Conditional Gemini imports - may fail if google-genai package version is incompatible
try:
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
    from pipecat.services.google.tts import GeminiTTSService

    GEMINI_AVAILABLE = True
except ImportError:
    # Gemini services unavailable - will fail at runtime if requested
    GeminiLiveLLMService = None
    GeminiVADParams = None
    GeminiTTSService = None
    GEMINI_AVAILABLE = False
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.ultravox.llm import OneShotInputParams, UltravoxRealtimeLLMService

# NOTE: Speechmatics support temporarily disabled due to API incompatibility with current pipecat version
# from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.pipeline.nvidia_stt import NVidiaWebSocketSTTService
from eva.utils.llm_utils import _resolve_url
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# Default sample rate for audio (TTS output rate).
SAMPLE_RATE = 24000


def _base_language(tag: str) -> str:
    """Return the ISO 639-1 base code from a BCP 47 tag (e.g. 'es-MX' → 'es').

    Used for providers that only accept the two-letter base code (e.g. Whisper/OpenAI STT).
    """
    return tag.split("-")[0].split("_")[0]


def _to_language_enum(tag: str) -> Language:
    """Convert a BCP 47 tag to a pipecat Language enum.

    Tries the full tag first, falls back to the base code (e.g. 'fr-CA' → 'fr').
    Used for pipecat-native services whose Settings accept Language enum values.
    """
    try:
        return Language(tag)
    except ValueError:
        return Language(_base_language(tag))


# Round-robin counters for load-balanced URLs (one per service type)
_tts_url_counter: int = 0
_stt_url_counter: int = 0
_audio_llm_url_counter: int = 0


def _aws_credentials(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve AWS creds/region for the pipecat AWS services from params, then the env chain.

    pipecat's AWS services take the secret access key as ``api_key``. Any value left as None
    lets the underlying SDK fall back to its own default resolution.
    """
    return {
        "api_key": params.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "aws_access_key_id": params.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_session_token": params.get("aws_session_token") or os.environ.get("AWS_SESSION_TOKEN"),
        "region": params.get("region") or os.environ.get("AWS_REGION", "us-east-1"),
    }


def create_stt_service(
    model: str | None,
    params: dict[str, Any] | None = None,
    language_code: str = "en",
) -> STTService | None:
    """Create speech-to-text service.

    Based on create_stt_service() from chatbot.py.

    Args:
        model: STT model identifier (deepgram, deepgram-flux, openai, assemblyai, cartesia, cartesia-multilingual, nvidia)
        params: Model-specific parameters (may include 'alias' key which is ignored here)
        language_code: Language code for transcription

    Returns:
        Configured STT service or None if model is None
    """
    if model is None:
        logger.info("STT disabled")
        return None

    params = dict(params or {})
    params.pop("alias", None)  # alias is a label only; strip before passing to service constructors
    model_lower = model.lower()

    api_key = params.get("api_key")

    # Resolve URL once (supports round-robin via "urls" list)
    global _stt_url_counter
    url, _stt_url_counter = _resolve_url(params, _stt_url_counter)

    if model_lower == "assemblyai":
        logger.info(f"Using AssemblyAI STT: {params['model']}")
        assemblyai_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(AssemblyAISTTService.Settings) if (k := f.name) in params
        }
        # vad_force_turn_endpoint is a constructor arg, not a Settings field, so the dataclass
        # forwarding above won't carry it — thread it explicitly. Default True = Pipecat-mode
        # (force the endpoint on Silero VAD stop); set False to let AssemblyAI's own server-side
        # turn detection (min_turn_silence/max_turn_silence) decide turn ends.
        return AssemblyAISTTService(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            vad_force_turn_endpoint=params.get("vad_force_turn_endpoint", True),
            settings=AssemblyAISTTService.Settings(
                language=_to_language_enum(language_code),
                **assemblyai_settings_kwargs,
            ),
        )

    elif model_lower == "cartesia":
        # ink-2 provides its own turn boundaries; ModelConfig selects external endpointing.
        model_name = params["model"]
        logger.info(f"Using Cartesia STT: {model_name}")
        return CartesiaTurnsSTTService(
            api_key=api_key,
            sample_rate=params.get("sample_rate", 16000),
            should_interrupt=params.get("should_interrupt", True),
            settings=CartesiaTurnsSTTService.Settings(model=model_name),
        )

    elif model_lower == "cartesia-multilingual":
        logger.info(f"Using Cartesia multilingual STT: {params['model']}")
        return CartesiaSTTService(
            api_key=api_key,
            sample_rate=16000,
            settings=CartesiaSTTService.Settings(
                model=params["model"],
                language=_to_language_enum(language_code),
            ),
        )

    elif model_lower == "cohere":
        logger.info(f"Using Cohere STT: {params['model']}")
        return OpenAISTTService(
            api_key=api_key,
            base_url=url,
            model=params["model"],
            language=_base_language(language_code),
            sample_rate=SAMPLE_RATE,
        )

    elif model_lower.startswith("deepgram"):
        # Check if using Flux model
        if "flux" in model_lower:
            logger.info(f"Using Deepgram Flux STT: {params['model']}")
            flux_settings_kwargs: dict[str, Any] = {"model": params["model"]}
            # Flux ignores `language`; only `flux-general-multi` honors `language_hints`.
            if params["model"] == "flux-general-multi":
                if params.get("language_hints"):
                    flux_settings_kwargs["language_hints"] = params["language_hints"]
                else:
                    logger.warning("No Language hint provided. Auto detecting language for Deepgram Flux")
            return DeepgramFluxSTTService(
                api_key=api_key,
                sample_rate=SAMPLE_RATE,
                settings=DeepgramFluxSTTSettings(**flux_settings_kwargs),
            )
        logger.info(f"Using Deepgram STT: {params['model']}")
        deepgram_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(DeepgramSTTService.Settings) if (k := f.name) in params
        }
        deepgram_settings_kwargs.setdefault("interim_results", True)
        return DeepgramSTTService(
            api_key=api_key,
            settings=DeepgramSTTService.Settings(
                language=_to_language_enum(language_code),
                **deepgram_settings_kwargs,
            ),
            sample_rate=SAMPLE_RATE,
        )

    elif model_lower == "elevenlabs":
        logger.info(f"Using ElevenLabs STT {params['model']}")
        elevenlabs_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(ElevenLabsRealtimeSTTService.Settings) if (k := f.name) in params
        }
        return ElevenLabsRealtimeSTTService(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            commit_strategy=CommitStrategy.VAD,
            settings=ElevenLabsRealtimeSTTService.Settings(
                language=_base_language(language_code),
                **elevenlabs_settings_kwargs,
            ),
        )

    elif model_lower == "nvidia":
        if not url:
            raise ValueError("url required in STT_PARAMS for NVIDIA STT (WebSocket endpoint)")

        logger.info("Using NVIDIA STT via WebSocket")
        return NVidiaWebSocketSTTService(
            url=url,
            api_key=api_key,
            sample_rate=params.get("sample_rate", SAMPLE_RATE),
            verify=False,
            model=params.get("model"),
            language=None,
        )

    elif model_lower == "nvidia-baseten":
        if not url:
            raise ValueError("url required in STT_PARAMS for NVIDIA Baseten STT")

        logger.info("Using NVIDIA Baseten STT")
        return BasetenSTTService(
            api_key=api_key,
            base_url=url,
        )

    elif model_lower == "openai":
        logger.info(f"Using OpenAI STT: {params['model']}")
        # Whisper only accepts ISO 639-1 base codes (e.g. "es", not "es-MX")
        # params["language"] takes precedence if explicitly set by the user
        whisper_lang = params.get("language") or _base_language(language_code)
        stt_service = OpenAISTTService(
            api_key=api_key,
            base_url=url,
            model=params["model"],
            language=whisper_lang,
            sample_rate=SAMPLE_RATE,
        )
        if url and "azure" in url:
            stt_service._client = AsyncAzureOpenAI(
                azure_endpoint=url,
                api_key=api_key,
                api_version=params.get("api_version", "2025-03-01-preview"),
            )
        return stt_service

    elif model_lower == "smallest":
        logger.info(f"Using Smallest STT: {params['model']}")
        smallest_stt_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(SmallestSTTService.Settings) if (k := f.name) in params
        }
        return SmallestSTTService(
            api_key=api_key,
            base_url=url or "wss://api.smallest.ai",
            sample_rate=params.get("sample_rate", SAMPLE_RATE),
            settings=SmallestSTTService.Settings(
                language=_to_language_enum(language_code),
                **smallest_stt_settings_kwargs,
            ),
        )

    elif model_lower == "xai":
        logger.info("Using xAI STT")
        xai_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(XAISTTService.Settings) if (k := f.name) in params
        }
        xai_settings_kwargs.setdefault("interim_results", True)
        xai_settings_kwargs.setdefault("endpointing", 200)
        return XAISTTService(
            api_key=api_key,
            sample_rate=params.get("sample_rate", 16000),
            encoding=params.get("encoding", "pcm"),
            settings=XAISTTService.Settings(
                language=_to_language_enum(language_code),
                **xai_settings_kwargs,
            ),
        )

    elif model_lower in {"aws", "aws_transcribe", "amazon_transcribe", "transcribe"}:
        from pipecat.services.aws.stt import AWSTranscribeSTTService, AWSTranscribeSTTSettings

        logger.info("Using AWS Transcribe STT")
        return AWSTranscribeSTTService(
            sample_rate=params.get("sample_rate", SAMPLE_RATE),
            settings=AWSTranscribeSTTSettings(language=_to_language_enum(language_code)),
            **_aws_credentials(params),
        )

    else:
        raise ValueError(
            f"Unknown STT model: {model}. Available: assemblyai, aws_transcribe, cartesia, cartesia-multilingual, cohere, deepgram, deepgram-flux, elevenlabs, nvidia, nvidia-baseten, openai, smallest, xai"
        )


async def update_stt_agent_context(stt: STTService | None, text: str) -> None:
    """Feed the agent's latest spoken reply back to the STT service as conversation context.

    AssemblyAI's Universal-3 Pro streaming models (e.g. ``u3-rt-pro``, ``universal-3-5-pro``)
    support *context carryover*: seeding the agent's most recent reply improves transcription
    of the user's next turn (short answers, spelled-out entities, disambiguation). Pipecat
    exposes this via ``AssemblyAISTTService.update_agent_context()``.

    In a standard pipecat bot this fires automatically (the assistant context aggregator emits
    ``LLMContextAssistantTurnFrame`` and the upstream STT picks it up). EVA's cascade pipeline
    drives the agent turn through a custom processor that pushes ``TTSSpeakFrame`` directly and
    does not emit the standard LLM response frames, so we trigger the update explicitly from the
    assistant-response hook instead.

    No-op for STT services without the capability (Deepgram, Cartesia, …) and for non-U3-Pro
    AssemblyAI models, where pipecat treats the call as a safe no-op.
    """
    update = getattr(stt, "update_agent_context", None)
    if update is not None and text:
        await update(text)


def create_tts_service(
    model: str | None,
    params: dict[str, Any] | None = None,
    language_code: str = "en",
) -> TTSService | None:
    """Create text-to-speech service.

    Based on create_tts_service() from chatbot.py.

    Args:
        model: TTS model identifier (cartesia, elevenlabs, openai, gemini)
        params: Model-specific parameters (may include 'alias' key which is ignored here)
        language_code: Language code for speech synthesis

    Returns:
        Configured TTS service or None if model is None
    """
    if model is None:
        logger.info("TTS disabled")
        return None

    params = dict(params or {})
    params.pop("alias", None)  # alias is a label only; strip before passing to service constructors
    model_lower = model.lower()

    api_key = params.get("api_key")

    # Resolve URL once (supports round-robin via "urls" list)
    global _tts_url_counter
    url, _tts_url_counter = _resolve_url(params, _tts_url_counter)

    if model_lower == "cartesia":
        logger.info(f"Using Cartesia TTS: {params['model']}")
        cartesia_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(CartesiaTTSService.Settings) if (k := f.name) in params
        }
        return CartesiaTTSService(
            url=url or "wss://api.cartesia.ai/tts/websocket",
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            settings=CartesiaTTSService.Settings(
                voice=params.get("voice_id", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
                language=_to_language_enum(language_code),
                **cartesia_settings_kwargs,
            ),
        )

    elif model_lower == "chatterbox":
        logger.info(f"Using Chatterbox TTS: {params['model']}")
        chatterbox_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "alloy"),
            base_url=url,
        )
        chatterbox_tts._eva_extra_body = {
            "streaming_quality": "fast",
            "streaming_strategy": "word",
            "streaming_chunk_size": 80,
            "streaming_buffer_size": 1,
        }
        OpenAITTSService.run_tts = override_run_tts
        chatterbox_tts._settings.language = language_code
        return chatterbox_tts

    elif model_lower == "deepgram":
        logger.info(f"Using Deepgram TTS: {params['model']}")
        return DeepgramTTSService(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            settings=DeepgramTTSService.Settings(
                model=params["model"],
                voice=params.get("voice", "aura-2-helena-en"),
                language=_to_language_enum(language_code),
            ),
        )

    elif model_lower == "elevenlabs":
        logger.info(f"Using ElevenLabs TTS: {params['model']}")
        if (
            params["model"] not in ("eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5")
            and language_code != "en"
        ):
            raise ValueError(f"ElevenLabs model {params['model']} only supports English language")
        elevenlabs_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(ElevenLabsTTSService.Settings) if (k := f.name) in params
        }
        return ElevenLabsTTSService(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            settings=ElevenLabsTTSService.Settings(
                voice=params.get("voice_id", "hpp4J3VqNfWAUOO0d1Us"),
                language=_base_language(language_code),
                **elevenlabs_settings_kwargs,
            ),
        )

    elif model_lower == "gemini":
        if not GEMINI_AVAILABLE:
            raise ValueError(
                "Gemini TTS requested but Gemini services are unavailable. "
                "Check google-genai package installation and version compatibility."
            )

        logger.info(f"Using Gemini TTS: {params['model']}")
        # Supports gemini-2.5-flash-tts, gemini-3.1-flash-tts-preview, etc.
        return GeminiTTSService(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            settings=GeminiTTSService.Settings(
                model=params["model"],
                voice=params.get("voice_id", params.get("voice_name", "Kore")),
                language=_to_language_enum(language_code),
            ),
        )

    elif model_lower == "kokoro":
        logger.info(f"Using Kokoro TTS: {params['model']}")
        kokoro_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "alloy"),
            base_url=url,
        )
        # Kokoro sometimes accepts the 2 char codes, and sometimes doesn't
        # reference codes: https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py
        supported = ["en-us", "en-gb", "es", "fr-fr", "hi", "it", "pt-br", "ja", "zh"]
        if language_code not in supported:
            logger.warning(f"Language code {language_code} not supported by Kokoro, trying to convert to 4 char code")
            two_to_four = {"en": "en-us", "fr": "fr-fr", "fr-CA": "fr-fr", "pt": "pt-br"}
            language_code = two_to_four.get(language_code, language_code)
            if language_code not in supported:
                raise ValueError(f"Language code {language_code} not supported by Kokoro")
        kokoro_tts._eva_extra_body = {
            "stream": True,
            "streaming_quality": "fast",
            "streaming_strategy": "word",
            "streaming_chunk_size": 80,
            "streaming_buffer_size": 1,
            "lang_code": language_code,
        }
        OpenAITTSService.run_tts = override_run_tts
        return kokoro_tts

    elif model_lower == "nvidia-baseten":
        if not url:
            raise ValueError("url required in TTS_PARAMS for NVIDIA Baseten TTS")

        logger.info("Using NVIDIA Baseten TTS")
        return BasetenTTSService(
            api_key=api_key,
            base_url=url,
            voice_id=params.get("voice"),
            text_filters=[ASCIITextFilter()],
        )

    elif model_lower == "openai":
        logger.info(f"Using OpenAI TTS: {params['model']}")

        voice = params.get("voice", "alloy")
        openai_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=voice,
        )
        openai_tts._settings.language = language_code
        if url and "azure" in url:
            openai_tts._client = AsyncAzureOpenAI(
                azure_endpoint=url,
                api_key=api_key,
                api_version=params.get("api_version", "2025-03-01-preview"),
            )
            return openai_tts

        return openai_tts

    elif model_lower == "smallest":
        logger.info(f"Using Smallest TTS: {params['model']}")
        smallest_tts_settings_kwargs = {
            k: params[k] for f in dataclasses.fields(SmallestTTSService.Settings) if (k := f.name) in params
        }
        if "voice_id" in params:
            # Omit "voice" entirely when unset so pipecat's per-model default (e.g. "sophia" for
            # lightning_v3.1, "meher" for lightning_v3.1_pro) applies instead of a hardcoded EVA default.
            smallest_tts_settings_kwargs["voice"] = params["voice_id"]
        return SmallestTTSService(
            api_key=api_key,
            base_url=url or "wss://api.smallest.ai",
            sample_rate=SAMPLE_RATE,
            settings=SmallestTTSService.Settings(
                language=_to_language_enum(language_code),
                **smallest_tts_settings_kwargs,
            ),
        )

    elif model_lower == "voxtral":
        logger.info(f"Using Voxtral TTS: {params['model']}")
        voxtral_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "neutral_female"),
            base_url=url,
        )
        OpenAITTSService.run_tts = override_run_tts_voxtral
        voxtral_tts._settings.language = language_code
        return voxtral_tts

    elif model_lower == "xai":
        logger.info(f"Using xAI TTS: voice={params.get('voice', 'eve')}")
        # Lowest-latency defaults: pcm codec, optimize_streaming_latency=2,
        # text_normalization=false.  Monkey-patch _build_url to inject the
        # extra query params that pipecat's XAITTSService doesn't expose yet.
        xai_tts = XAITTSService(
            api_key=api_key,
            sample_rate=params.get("sample_rate", SAMPLE_RATE),
            codec=params.get("codec", "pcm"),
            settings=XAITTSService.Settings(
                voice=params.get("voice", "eve"),
                language=_to_language_enum(language_code),
            ),
        )
        _orig_build_url = xai_tts._build_url
        extra_strings = (
            f"&optimize_streaming_latency={params.get('optimize_streaming_latency', 2)}"
            f"&text_normalization={str(params.get('text_normalization', False)).lower()}"
        )
        speed = params.get("speed")
        if speed is not None:
            extra_strings += f"&speed={speed}"
        xai_tts._build_url = lambda: _orig_build_url() + extra_strings
        return xai_tts

    elif model_lower == "xtts":
        logger.info(f"Using XTTS TTS: {params['model']}")
        xtts_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "alloy"),
            base_url=url,
        )
        xtts_tts._eva_extra_body = {
            "streaming_quality": "fast",
            "streaming_strategy": "word",
            "streaming_chunk_size": 80,
            "streaming_buffer_size": 1,
        }
        OpenAITTSService.run_tts = override_run_tts
        xtts_tts._settings.language = language_code
        return xtts_tts

    elif model_lower in {"aws", "aws_polly", "amazon_polly", "polly"}:
        from pipecat.services.aws.tts import AWSPollyTTSService, AWSPollyTTSSettings

        logger.info("Using AWS Polly TTS")
        # Forward optional Polly tuning (engine, pitch, rate, volume, lexicon_names); voice and
        # language are passed explicitly below.
        polly_settings_kwargs = {
            k: params[k]
            for f in dataclasses.fields(AWSPollyTTSSettings)
            if (k := f.name) in params and k not in {"voice", "language", "model"}
        }
        polly_settings_kwargs.setdefault("engine", "neural")
        return AWSPollyTTSService(
            sample_rate=SAMPLE_RATE,
            settings=AWSPollyTTSSettings(
                voice=params.get("voice_id", "Joanna"),
                language=_to_language_enum(language_code),
                **polly_settings_kwargs,
            ),
            **_aws_credentials(params),
        )

    else:
        raise ValueError(
            f"Unknown TTS model: {model}. Available: aws_polly, cartesia, chatterbox, deepgram, elevenlabs, gemini, kokoro, nvidia-baseten, openai, smallest, voxtral, xai, xtts"
        )


def create_realtime_llm_service(
    model: str | None,
    params: dict[str, Any] | None = None,
    agent: AgentConfig | None = None,
    audit_log: AuditLog | None = None,
    current_date_time: str | None = None,
) -> LLMService:
    """Create realtime LLM service.

    Args:
        model: LLM model identifier (openai, gemini, groq)
        params: Model-specific parameters
        rate_limiter: Optional rate limiter for API calls
        agent: The agent config
        audit_log: AuditLog class for writing transript and tool calls
        current_date_time: Current date/time string from the evaluation record

    Returns:
        Configured LLM service
    """
    model_lower = (model or "").lower()

    # Get realtime server prompt
    prompt_manager = PromptManager()
    system_prompt = prompt_manager.get_prompt(
        "realtime_agent.system_prompt",
        agent_personality=agent.description,
        agent_instructions=agent.instructions,
        datetime=current_date_time,
    )

    openai_tools = agent.build_tools_for_realtime() if agent else None

    # Convert OpenAI format tools to pipecat format
    pipecat_tools = None
    if openai_tools:
        function_schemas = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_schemas.append(
                    FunctionSchema(
                        name=func["name"],
                        description=func["description"],
                        properties=func["properties"],
                        required=func.get("required", []),
                    )
                )
        pipecat_tools = ToolsSchema(standard_tools=function_schemas)

    if model_lower.startswith("openai"):
        session_properties = get_openai_session_properties(system_prompt, params, pipecat_tools)
        if audit_log is not None:
            logger.info(f"Using InstrumentedRealtimeLLMService for audit log interception: openai: {params['model']}")
            kwargs: dict = {
                "settings": OpenAIRealtimeLLMService.Settings(
                    model=params["model"],
                    session_properties=session_properties,
                ),
                "audit_log": audit_log,
                "api_key": params["api_key"],
            }
            if params.get("url"):
                kwargs["base_url"] = params["url"]
            return InstrumentedRealtimeLLMService(**kwargs)

        return OpenAIRealtimeLLMService(
            api_key=params["api_key"],
            settings=OpenAIRealtimeLLMService.Settings(
                model=params["model"],
                session_properties=session_properties,
            ),
        )
    elif model_lower.startswith("azure") or model_lower.startswith("gpt-realtime"):
        #
        # base_url: The full Azure WebSocket endpoint URL including api-version and deployment.
        # Example: "wss://my-project.openai.azure.com/openai/v1/realtime"
        url = params.get("url", "")
        session_properties = get_openai_session_properties(system_prompt, params, pipecat_tools)

        logger.info(f"Using Azure Realtime LLM: {model_lower}, url {url}")

        if audit_log is not None:
            logger.info("Using InstrumentedRealtimeLLMService for audit log interception")
            service = InstrumentedRealtimeLLMService(
                audit_log=audit_log,
                api_key=params["api_key"],
                base_url=url,
                session_properties=session_properties,
                settings=OpenAIRealtimeLLMService.Settings(
                    model=params["model"],
                    session_properties=session_properties,
                ),
            )
            InstrumentedRealtimeLLMService._connect = override__connect  # azure realtime connect
            return service

        return OpenAIRealtimeLLMService(
            api_key=params["api_key"],
            model=params["model"],
            base_url=url,
            session_properties=session_properties,
        )
    elif model_lower == "ultravox":
        logger.info("Using Ultravox LLM")
        return UltravoxRealtimeLLMService(
            params=OneShotInputParams(
                api_key=params["api_key"],
                system_prompt=system_prompt,
                temperature=0.3,
                max_duration=datetime.timedelta(minutes=6),
                voice=params.get("voice", "03e20d03-35e4-43c4-bb18-9b18a2cd3086"),
                model=params["model"],
            ),
            one_shot_selected_tools=pipecat_tools,
        )

    elif model_lower == "gemini-live":
        if not GEMINI_AVAILABLE:
            raise ValueError(
                "Gemini Live requested but Gemini services are unavailable. "
                "Check google-genai package installation and version compatibility."
            )

        gemini_model = params.get("model")
        logger.info(f"Using Gemini Live LLM: {gemini_model}")

        return GeminiLiveLLMService(
            api_key=params["api_key"],
            tools=pipecat_tools,
            settings=GeminiLiveLLMService.Settings(
                model=gemini_model,
                system_instruction=system_prompt,
                voice=params.get("voice", "Puck"),  # Aoede, Charon, Fenrir, Kore, Puck
                vad=GeminiVADParams(disabled=params.get("vad_disabled", True)),
            ),
        )

    else:
        raise ValueError(f"Unknown realtime model: {model}. Available: gpt-realtime, ultravox, gemini-live")


def get_openai_session_properties(system_prompt: str, params: dict, pipecat_tools) -> SessionProperties:
    """Create openai compatible session properties object.

    ``params["turn_detection_disabled"]`` (bool, default False): set True
    when the realtime endpoint does NOT implement server-side VAD (e.g. our
    vLLM-omni server). With turn_detection=False, pipecat falls back to its
    own pipeline VAD (silero) and explicitly sends
    ``input_audio_buffer.commit`` + ``response.create`` on
    UserStoppedSpeakingFrame. Without this, pipecat assumes the server will
    detect turn boundaries and never commits the audio buffer.
    """
    if params.get("turn_detection_disabled"):
        # Pipecat will drive turn detection from its own VAD signals.
        turn_detection: SemanticTurnDetection | bool = False
    else:
        # Set openai TurnDetection parameters. Not setting this at all will
        # turn it on by default.
        turn_detection = SemanticTurnDetection()
    return SessionProperties(
        instructions=system_prompt,
        audio=AudioConfiguration(
            input=AudioInput(
                transcription=InputAudioTranscription(
                    model=params.get("transcription_model", "gpt-4o-mini-transcribe")
                ),
                turn_detection=turn_detection,
            ),
            output=AudioOutput(
                voice=params.get("voice", "marin"),
            ),
        ),
        tools=pipecat_tools,
        tool_choice="auto",
    )


def create_audio_llm_client(
    model: str,
    params: dict[str, Any],
    language: str | None = None,
) -> BaseALMClient:
    """Create an audio-LLM API client.

    Audio-LLM models accept audio input + text context and return text output.
    Supports self-hosted models via vLLM's OpenAI-compatible API and Gemini
    via its OpenAI-compatibility endpoint (e.g. gemini-3-flash-preview).

    Args:
        model: Audio-LLM model identifier (e.g. "vllm", "gemini").
        params: Model-specific parameters. Required for vLLM: url (or urls).
                Required for Gemini: api_key, model.
                Optional: temperature, max_tokens, sample_rate, num_channels, sample_width.
        language: BCP 47 language tag (e.g. 'en', 'fr'). Used to build the
                  client's default transcription prompt.

    Returns:
        Configured audio-LLM client.
    """
    model_lower = model.lower()

    # Resolve URL once (supports round-robin via "base_urls" list)
    global _audio_llm_url_counter
    base_url, _audio_llm_url_counter = _resolve_url(params, _audio_llm_url_counter)

    if "gemini" in model_lower:
        gemini_model = params["model"]
        project = params.get("project")
        location = params.get("location")
        api_key = params.get("api_key")
        if not (project and location) and not api_key:
            raise ValueError(
                "Gemini audio-LLM requires either api_key (Developer API) "
                "or project+location (Vertex AI via GOOGLE_APPLICATION_CREDENTIALS)."
            )
        client = ALMGeminiClient(
            api_key=api_key,
            base_url=base_url or None,
            model=gemini_model,
            temperature=params.get("temperature", 1.0),
            max_tokens=params.get("max_tokens", 1024),
            sample_rate=params.get("sample_rate", 16000),
            num_channels=params.get("num_channels", 1),
            sample_width=params.get("sample_width", 2),
            project=project,
            location=location,
            thinking_level=params.get("thinking_level", "minimal"),
            language=language,
        )
        logger.info(f"Using Gemini audio-LLM: {gemini_model} ({'vertex' if project and location else 'api_key'})")
        return client

    if "vllm" in model_lower:
        if not base_url:
            raise ValueError("url (or urls) required in audio_llm_params for vLLM")

        client = ALMvLLMClient(
            base_url=base_url,
            api_key=params.get("api_key", "EMPTY"),
            model=params["model"],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 512),
            sample_rate=params.get("sample_rate", 16000),
            num_channels=params.get("num_channels", 1),
            sample_width=params.get("sample_width", 2),
            language=language,
            enable_thinking=params.get("enable_thinking", False),
            sampling_params=params.get("sampling_params"),
        )
        logger.info(f"Using {model} vLLM audio-LLM: {base_url}")
        return client

    raise ValueError(f"Unknown audio-LLM model: {model}. Available: vllm, gemini")


async def override_run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
    """Override OpenAITTSService.run_tts to force streaming parameters.

    Note: The only change is adding "extra_body" to the create params
    Generate speech from text using OpenAI's TTS API.

    Args:
        self: The OpenAITTSService instance.
        text: The text to synthesize into speech.
        context_id: The context ID for tracking audio frames.

    Yields:
        Frame: Audio frames containing the synthesized speech data.
    """
    logger.debug(f"{self}: Generating TTS [{text}], model {self._settings.model}")
    try:
        await self.start_ttfb_metrics()

        # Per-backend streaming knobs are attached to the service instance as
        # `_eva_extra_body` by the factory in this module (e.g. kokoro sets
        # "stream": True; chatterbox/xtts omit it).
        create_params = {
            "input": text,
            "model": self._settings.model,
            "voice": self._settings.voice,
            "response_format": "pcm",
        }
        extra_body = getattr(self, "_eva_extra_body", None)
        if extra_body:
            create_params["extra_body"] = extra_body

        if self._settings.instructions:
            create_params["instructions"] = self._settings.instructions

        if self._settings.speed:
            create_params["speed"] = self._settings.speed

        async with self._client.audio.speech.with_streaming_response.create(**create_params) as r:
            if r.status_code != 200:
                error = await r.text()
                logger.error(f"{self} error getting audio (status: {r.status_code}, error: {error})")
                yield ErrorFrame(error=f"Error getting audio (status: {r.status_code}, error: {error})")
                return

            await self.start_tts_usage_metrics(text)

            CHUNK_SIZE = self.chunk_size

            yield TTSStartedFrame(context_id=context_id)
            async for chunk in r.iter_bytes(CHUNK_SIZE):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
                    yield frame
            yield TTSStoppedFrame(context_id=context_id)
    except BadRequestError as e:
        yield ErrorFrame(error=f"Unknown error occurred: {e}")


async def override_run_tts_voxtral(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
    """Override OpenAITTSService.run_tts for Voxtral served via vLLM-omni.

    vLLM-omni's /v1/audio/speech ignores Chatterbox-style streaming knobs and
    instead uses a top-level ``stream: true`` flag (with response_format pcm/wav)
    to emit PCM chunks as they are decoded. ``speed`` is rejected when streaming.
    """
    logger.debug(f"{self}: Generating TTS [{text}], model {self._settings.model}")
    try:
        await self.start_ttfb_metrics()

        create_params = {
            "input": text,
            "model": self._settings.model,
            "voice": self._settings.voice,
            "response_format": "pcm",
            "extra_body": {
                "stream": True,
            },
        }

        if self._settings.instructions:
            create_params["instructions"] = self._settings.instructions

        async with self._client.audio.speech.with_streaming_response.create(**create_params) as r:
            if r.status_code != 200:
                error = await r.text()
                logger.error(f"{self} error getting audio (status: {r.status_code}, error: {error})")
                yield ErrorFrame(error=f"Error getting audio (status: {r.status_code}, error: {error})")
                return

            await self.start_tts_usage_metrics(text)

            CHUNK_SIZE = self.chunk_size

            yield TTSStartedFrame(context_id=context_id)
            async for chunk in r.iter_bytes(CHUNK_SIZE):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
                    yield frame
            yield TTSStoppedFrame(context_id=context_id)
    except BadRequestError as e:
        yield ErrorFrame(error=f"Unknown error occurred: {e}")


async def override__connect(self):
    # Allow connections to azure / other providers using a base_url
    try:
        if self._websocket:
            # Here we assume that if we have a websocket, we are connected. We
            # handle disconnections in the send/recv code paths.
            return

        logger.info(f"Connecting to {self.base_url}")
        self._websocket = await websocket_connect(
            uri=self.base_url,
            additional_headers={
                "api-key": self.api_key,
            },
        )
        self._receive_task = self.create_task(self._receive_task_handler())
    except Exception as e:
        await self.push_error(error_msg=f"initialization error: {e}", exception=e)
        self._websocket = None


# Unicode to ASCII replacements for TTS
_TTS_CHAR_MAP = str.maketrans(
    {
        "\u2011": "-",  # Non-breaking hyphen
        "\u2010": "-",  # Hyphen
        "\u2012": "-",  # Figure dash
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2015": "-",  # Horizontal bar
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",  # Non-breaking space
        "\u202f": " ",  # Narrow no-break space
    }
)


class ASCIITextFilter(BaseTextFilter):
    """Normalize non-ASCII characters for TTS, replacing common Unicode with ASCII equivalents."""

    async def filter(self, text: str) -> str:
        # Replace common Unicode with ASCII equivalents
        text = text.translate(_TTS_CHAR_MAP)
        # Remove any remaining non-ASCII
        return "".join(c for c in text if c.isascii())
