"""Service factories for STT, TTS, and LLM services.

Creates Pipecat services with proper configuration.
"""

import datetime
from typing import Any, AsyncGenerator, Optional

from deepgram import LiveOptions
from openai import AsyncAzureOpenAI, BadRequestError
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.assemblyai.stt import (
    AssemblyAIConnectionParams,
    AssemblyAISTTService,
)
from pipecat.services.cartesia.stt import CartesiaLiveOptions, CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
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
from pipecat.services.openai.tts import VALID_VOICES, OpenAITTSService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_filter import BaseTextFilter
from websockets.asyncio.client import connect as websocket_connect

from eva.assistant.pipeline.alm_vllm import ALMvLLMClient
from eva.assistant.pipeline.nvidia_baseten import BasetenSTTService, BasetenTTSService
from eva.assistant.pipeline.realtime_llm import InstrumentedRealtimeLLMService
from eva.models.agents import AgentConfig

# Conditional Gemini imports - may fail if google-genai package version is incompatible
try:
    from pipecat.services.google.tts import GeminiTTSService

    GEMINI_AVAILABLE = True
except ImportError:
    # Gemini services unavailable - will fail at runtime if requested
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

# Default sample rate for audio
SAMPLE_RATE = 24000

# Round-robin counters for load-balanced URLs (one per service type)
_tts_url_counter: int = 0
_stt_url_counter: int = 0
_audio_llm_url_counter: int = 0


def create_stt_service(
    model: Optional[str],
    params: Optional[dict[str, Any]] = None,
    language_code: str = "en",
) -> STTService | None:
    """Create speech-to-text service.

    Based on create_stt_service() from chatbot.py.

    Args:
        model: STT model identifier (deepgram, deepgram-flux, openai, assemblyai, cartesia, nvidia)
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
        return AssemblyAISTTService(
            api_key=api_key,
            language=language_code,
            connection_params=AssemblyAIConnectionParams(
                sample_rate=SAMPLE_RATE,
                speech_model=params["model"],
            ),
        )

    elif model_lower == "cartesia":
        logger.info(f"Using Cartesia STT: {params['model']}")
        return CartesiaSTTService(
            api_key=api_key,
            live_options=CartesiaLiveOptions(
                model=params["model"],
                language=language_code,
                sample_rate=SAMPLE_RATE,
            ),
        )

    elif model_lower.startswith("deepgram"):
        # Check if using Flux model
        if "flux" in model_lower:
            logger.info(f"Using Deepgram Flux STT: {params['model']}")
            return DeepgramFluxSTTService(
                api_key=api_key,
                model=params["model"],
                sample_rate=SAMPLE_RATE,
            )
        logger.info(f"Using Deepgram STT: {params['model']}")
        return DeepgramSTTService(
            api_key=api_key,
            live_options=LiveOptions(
                language=language_code,
                model=params["model"],
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
                interim_results=True,
            ),
            sample_rate=SAMPLE_RATE,
        )

    elif model_lower == "elevenlabs":
        logger.info("Using ElevenLabs STT")
        return ElevenLabsRealtimeSTTService(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            params=ElevenLabsRealtimeSTTService.InputParams(commit_strategy=CommitStrategy.VAD),
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
        stt_service = OpenAISTTService(
            api_key=api_key,
            base_url=url,
            model=params["model"],
            language=Language.EN,
            sample_rate=SAMPLE_RATE,
        )
        if url and "azure" in url:
            stt_service._client = AsyncAzureOpenAI(
                azure_endpoint=url,
                api_key=api_key,
                api_version=params.get("api_version", "2025-03-01-preview"),
            )
        if params.get("language"):
            stt_service._settings.language = params.get("language")
        return stt_service

    else:
        raise ValueError(
            f"Unknown STT model: {model}. Available: assemblyai, cartesia, deepgram, deepgram-flux, elevenlabs, nvidia, nvidia-baseten, openai"
        )


def create_tts_service(
    model: Optional[str],
    params: Optional[dict[str, Any]] = None,
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
        return CartesiaTTSService(
            url=url or "wss://api.cartesia.ai/tts/websocket",
            api_key=api_key,
            model=params["model"],
            voice_id=params.get("voice_id", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
            params=CartesiaTTSService.InputParams(language=language_code),
            sample_rate=SAMPLE_RATE,
        )

    elif model_lower == "chatterbox":
        logger.info(f"Using Chatterbox TTS: {params['model']}")
        chatterbox_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "alloy"),
            base_url=url,
        )
        OpenAITTSService.run_tts = override_run_tts
        chatterbox_tts._settings.language = language_code
        return chatterbox_tts

    elif model_lower == "deepgram":
        logger.info(f"Using Deepgram TTS: {params['model']}")
        return DeepgramTTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "aura-2-helena-en"),
            sample_rate=SAMPLE_RATE,
        )

    elif model_lower == "elevenlabs":
        logger.info(f"Using ElevenLabs TTS: {params['model']}")
        return ElevenLabsTTSService(
            api_key=api_key,
            model=params["model"],
            voice_id=params.get("voice_id", "hpp4J3VqNfWAUOO0d1Us"),
            sample_rate=SAMPLE_RATE,
        )

    elif model_lower == "gemini":
        if not GEMINI_AVAILABLE:
            raise ValueError(
                "Gemini TTS requested but Gemini services are unavailable. "
                "Check google-genai package installation and version compatibility."
            )

        logger.info(f"Using Gemini TTS: {params['model']}")
        return GeminiTTSService(
            api_key=api_key,
            model=params["model"],
            voice_name=params.get("voice_name", "Puck"),
        )

    elif model_lower == "kokoro":
        logger.info(f"Using Kokoro TTS: {params['model']}")
        kokoro_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "alloy"),
            base_url=url,
        )
        OpenAITTSService.run_tts = override_run_tts
        kokoro_tts._settings.language = language_code
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

    elif model_lower == "xtts":
        logger.info(f"Using XTTS TTS: {params['model']}")
        xtts_tts = OpenAITTSService(
            api_key=api_key,
            model=params["model"],
            voice=params.get("voice", "alloy"),
            base_url=url,
        )
        OpenAITTSService.run_tts = override_run_tts
        xtts_tts._settings.language = language_code
        return xtts_tts

    else:
        raise ValueError(
            f"Unknown TTS model: {model}. Available: cartesia, chatterbox, deepgram, elevenlabs, gemini, kokoro, nvidia-baseten, openai, xtts"
        )


def create_realtime_llm_service(
    model: Optional[str],
    params: Optional[dict[str, Any]] = None,
    agent: Optional[AgentConfig] = None,
    audit_log: Optional[AuditLog] = None,
    current_date_time: Optional[str] = None,
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
            logger.info(
                f"Using InstrumentedRealtimeLLMService for audit log interception: openai: {params.get('model')}"
            )
            return InstrumentedRealtimeLLMService(
                model=params.get("model"),
                audit_log=audit_log,
                api_key=params.get["api_key"],
                session_properties=session_properties,
            )

        return OpenAIRealtimeLLMService(
            api_key=params.get["api_key"],
            session_properties=session_properties,
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
                model=params.get("model"),
                audit_log=audit_log,
                api_key=params.get["api_key"],
                base_url=url,
                session_properties=session_properties,
            )
            InstrumentedRealtimeLLMService._connect = override__connect  # azure realtime connect
            return service

        return OpenAIRealtimeLLMService(
            api_key=params.get["api_key"],
            base_url=url,
            session_properties=session_properties,
        )
    elif model_lower == "ultravox":
        logger.info("Using Ultravox LLM")
        return UltravoxRealtimeLLMService(
            params=OneShotInputParams(
                api_key=params.get["api_key"],
                system_prompt=system_prompt,
                temperature=0.3,
                max_duration=datetime.timedelta(minutes=6),
                voice=params.get("voice", "03e20d03-35e4-43c4-bb18-9b18a2cd3086"),
            ),
            one_shot_selected_tools=pipecat_tools,
        )

    else:
        raise ValueError(f"Unknown realtime model: {model}. Available: gpt-realtime, ultravox")


def get_openai_session_properties(system_prompt: str, params: dict, pipecat_tools) -> SessionProperties:
    """Create openai compatible session properties object."""
    return SessionProperties(
        instructions=system_prompt,
        audio=AudioConfiguration(
            input=AudioInput(
                transcription=InputAudioTranscription(
                    model=params.get("transcription_model", "gpt-4o-mini-transcribe")
                ),
                # Set openai TurnDetection parameters. Not setting this at all will turn it on by default
                turn_detection=SemanticTurnDetection(),
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
) -> ALMvLLMClient:
    """Create an audio-LLM API client.

    Audio-LLM models accept audio input + text context and return text output.
    Currently supports self-hosted models via vLLM's OpenAI-compatible API.

    Args:
        model: Audio-LLM model identifier (e.g. "vllm").
        params: Model-specific parameters. Required: url (or urls for round-robin).
                Optional: api_key, model, temperature, max_tokens,
                sample_rate, num_channels, sample_width.

    Returns:
        Configured ALMvLLMClient.
    """
    model_lower = model.lower()

    # Resolve URL once (supports round-robin via "base_urls" list)
    global _audio_llm_url_counter
    base_url, _audio_llm_url_counter = _resolve_url(params, _audio_llm_url_counter)

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
        )
        logger.info(f"Using {model} vLLM audio-LLM: {base_url}")
        return client

    raise ValueError(f"Unknown audio-LLM model: {model}. Available: vllm")


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

        # add chatterbox streaming params to `create_params``
        # Setup API parameters
        create_params = {
            "input": text,
            "model": self._settings.model,
            "voice": VALID_VOICES[self._settings.voice],
            "response_format": "pcm",
            "extra_body": {
                "streaming_quality": "fast",
                "streaming_strategy": "word",
                "streaming_chunk_size": 80,
                "streaming_buffer_size": 1,
            },
        }

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
