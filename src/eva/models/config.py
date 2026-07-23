"""Unified run configuration: env vars, .env file, and CLI.

Priority (highest to lowest):
1. CLI arguments
2. Environment variables
3. ``.env`` file
4. Field defaults

``env_file`` and ``cli_parse_args`` are **not** in ``model_config``
so that bare ``RunConfig(...)`` in tests reads nothing but env vars
and explicit kwargs.  Scripts opt in to ``.env`` and/or CLI via
``RunConfig(_env_file=".env", _cli_parse_args=True)``.
"""

import copy
import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from litellm.types.router import DeploymentTypedDict
from pipecat.transcriptions.language import Language
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict

from eva.models.provenance import RunProvenance

logger = logging.getLogger(__name__)


_VALIDATION_METRIC_NAMES = frozenset(("conversation_valid_end", "user_behavioral_fidelity", "user_speech_fidelity"))


def _get_all_metrics() -> list[str]:
    from eva.metrics.registry import get_global_registry

    return [m for m in get_global_registry().list_metrics() if m not in _VALIDATION_METRIC_NAMES]


def _param_alias(params: dict[str, Any]) -> str:
    """Return the display alias from a params dict."""
    return params.get("alias") or params["model"]


_elevenlabs_agent_cache: dict[str, dict[str, str]] = {}


def _fetch_elevenlabs_agent_models(s2s_params: dict[str, Any]) -> dict[str, str]:
    """Fetch STT, LLM, and TTS model names from the ElevenLabs agent API.

    Results are cached per agent ID so repeated calls (e.g. run_id generation)
    don't hit the API multiple times.
    """
    agent_id = s2s_params.get("assistant_agent_id", "")
    if not agent_id:
        logger.warning("No assistant_agent_id in s2s_params, cannot fetch ElevenLabs agent models")
        return {"stt": "unknown", "llm": "unknown", "tts": "unknown"}

    if agent_id in _elevenlabs_agent_cache:
        return _elevenlabs_agent_cache[agent_id]

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=s2s_params.get("api_key"))
        agent = client.conversational_ai.agents.get(agent_id=agent_id)
        cc = agent.conversation_config

        stt = "unknown"
        if cc.asr and cc.asr.provider:
            stt = cc.asr.provider

        llm = "unknown"
        if cc.agent and cc.agent.prompt and cc.agent.prompt.llm:
            llm = cc.agent.prompt.llm

        tts = "unknown"
        if cc.tts and cc.tts.model_id:
            tts = cc.tts.model_id

        result = {"stt": stt, "llm": llm, "tts": tts}
        _elevenlabs_agent_cache[agent_id] = result
        logger.info(f"Fetched ElevenLabs agent models: {result}")
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch ElevenLabs agent models: {e}")
        return {"stt": "unknown", "llm": "unknown", "tts": "unknown"}


class ModelConfig(BaseModel):
    """Flat model configuration covering all pipeline modes.

    Exactly one mode selector (``llm``, ``s2s``, or ``audio_llm``) should be set.
    Mode exclusivity is enforced by ``RunConfig``, not here, so that
    ``max_rerun_attempts == 0 or self.aggregate_only`` can freely construct a config with mixed env vars.
    """

    model_config = ConfigDict(extra="forbid")

    # Mapping from legacy config.json field names to current names.
    _LEGACY_RENAMES: ClassVar[dict[str, str]] = {
        "llm_model": "llm",
        "stt_model": "stt",
        "tts_model": "tts",
    }
    _LEGACY_DROP: ClassVar[set[str]] = {"realtime_model", "realtime_model_params"}

    # STT models that perform their own (server-side) semantic endpointing. They drive turn
    # boundaries themselves, so they must run with the 'external' turn strategies and no local VAD.
    _SELF_ENDPOINTING_STT: ClassVar[set[str]] = {"cartesia"}

    # ── Mode selectors (exactly one group must be set for a real run) ──
    llm: str | None = Field(
        None,
        description="LLM model name matching a model_name in --model-list/EVA_MODEL_LIST",
        examples=["gpt-5.2", "gemini-3-pro"],
    )
    stt: str | None = Field(None, description="STT model", examples=["deepgram", "openai_whisper"])
    tts: str | None = Field(None, description="TTS model", examples=["cartesia", "elevenlabs"])

    s2s: str | None = Field(
        None, description="Speech-to-speech model name", examples=["gpt-realtime-mini", "gemini_live"]
    )

    audio_llm: str | None = Field(None, description="Audio-LLM model identifier", examples=["vllm"])

    # ── Params dicts ──
    stt_params: dict[str, Any] | None = Field(None, description="Additional STT model parameters (JSON)")
    tts_params: dict[str, Any] | None = Field(None, description="Additional TTS model parameters (JSON)")
    s2s_params: dict[str, Any] | None = Field(None, description="Additional speech-to-speech model parameters (JSON)")
    audio_llm_params: dict[str, Any] | None = Field(
        None,
        description=(
            "Audio-LLM parameters (JSON): base_url (required), api_key, model, temperature, "
            "max_tokens, full_audio_context (send every user turn as audio instead of only the "
            "current turn; removes reliance on transcriptions but grows context quickly)"
        ),
    )

    # Configurable turn start/stop strategies
    turn_start_strategy: str = Field(
        "vad",
        description=(
            "User turn start strategy: 'vad', 'transcription', or 'external'. "
            "Defaults to 'vad' (VADUserTurnStartStrategy). "
            "Set via EVA_MODEL__TURN_START_STRATEGY."
        ),
    )
    turn_start_strategy_params: dict[str, Any] = Field(
        {},
        description="Parameters for turn start strategy (JSON). Set via EVA_MODEL__TURN_START_STRATEGY_PARAMS.",
    )

    turn_stop_strategy: str = Field(
        "turn_analyzer",
        description=(
            "User turn stop strategy: 'speech_timeout', 'turn_analyzer', or 'external'. "
            "Defaults to 'turn_analyzer' (TurnAnalyzerUserTurnStopStrategy with LocalSmartTurnAnalyzerV3). "
            "Set via EVA_MODEL__TURN_STOP_STRATEGY."
        ),
    )
    turn_stop_strategy_params: dict[str, Any] = Field(
        {},
        description="Parameters for turn stop strategy (JSON). Set via EVA_MODEL__TURN_STOP_STRATEGY_PARAMS.",
    )

    # VAD configuration
    vad: str = Field(
        "silero",
        description=(
            "VAD analyzer type: 'silero' or 'none'. Defaults to 'silero' (SileroVADAnalyzer). Use 'none' with external turn strategies (e.g. deepgram-flux) to skip local VAD. Set via EVA_MODEL__VAD."
        ),
    )
    vad_params: dict[str, Any] = Field(
        {},
        description=(
            "VAD parameters (JSON): confidence, start_secs, stop_secs, min_volume. Set via EVA_MODEL__VAD_PARAMS."
        ),
    )

    # CASCADE-only latency controls.
    pre_tool_speech: str = Field(
        "off",
        description="Prompt a model-generated lead-in before tool calls: 'off' or 'auto'.",
    )

    @field_validator("pre_tool_speech", mode="before")
    @classmethod
    def _normalize_pre_tool_speech(cls, value: str) -> str:
        return value.lower() if isinstance(value, str) else value

    llm_streaming: bool = Field(
        False,
        description="Stream Chat Completions output to TTS sentence-by-sentence.",
    )
    parallel_tool_calls: bool | None = Field(
        None,
        description="Forward parallel_tool_calls when tools are present; None leaves provider defaults.",
    )

    @property
    def pipeline_type(self) -> "PipelineType":
        """Detected pipeline mode based on which selector is set."""
        if self.audio_llm:
            return PipelineType.AUDIO_LLM
        if self.s2s:
            return PipelineType.S2S
        if self.llm:
            return PipelineType.CASCADE

    @property
    def pipeline_parts(self) -> dict[str, str]:
        """Component names for this pipeline (used in run_id generation)."""
        match self.pipeline_type:
            case PipelineType.AUDIO_LLM:
                return {
                    "audio_llm": _param_alias(self.audio_llm_params),
                    "tts": _param_alias(self.tts_params),
                }
            case PipelineType.S2S:
                if self.s2s == "elevenlabs":
                    # hardcoded for now. Models are set on the agent UI
                    return {
                        "s2s": _param_alias(self.s2s_params) or self.s2s,
                        **_fetch_elevenlabs_agent_models(self.s2s_params),
                    }
                return {"s2s": _param_alias(self.s2s_params)}
            case PipelineType.CASCADE:
                return {
                    "stt": _param_alias(self.stt_params),
                    "llm": self.llm,
                    "tts": _param_alias(self.tts_params),
                }

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Accept old config.json field names (llm_model, stt_model, etc.)."""
        if not isinstance(data, dict):
            return data
        for old, new in cls._LEGACY_RENAMES.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
            elif old in data:
                data.pop(old)
        for key in cls._LEGACY_DROP:
            data.pop(key, None)
        return data

    @model_validator(mode="after")
    def _autowire_self_endpointing_stt(self) -> "ModelConfig":
        """Auto-configure turn-taking for self-endpointing STT models (e.g. Cartesia ink-2).

        These models drive turn boundaries themselves (the service pushes its own speech
        start/stop and transcription frames), so they require the 'external' turn strategies
        with local VAD disabled. Forcing it here lets a bare ``EVA_MODEL__STT=cartesia``
        "just work", and the forced values are reflected in the persisted config.json / run_id.
        A conflicting user-provided value is overridden with a WARNING; an untouched default is
        logged at INFO. Idempotent: a value already at the target is left untouched (clean reload).

        Only applies to the CASCADE pipeline: in AUDIO_LLM / S2S the ``stt`` field does not
        instantiate an in-pipeline STT service, so there is nothing to emit the external turn
        frames. Forcing 'external'/'none' there would disable the local VAD that those pipelines
        rely on for turn detection, hanging the conversation (no user turn ever finalizes).
        """
        if self.pipeline_type != PipelineType.CASCADE:
            return self
        if (self.stt or "").lower() not in self._SELF_ENDPOINTING_STT:
            return self

        forced = {"turn_start_strategy": "external", "turn_stop_strategy": "external", "vad": "none"}
        for field, target in forced.items():
            if getattr(self, field) == target:
                continue
            if field in self.model_fields_set:
                logger.warning(
                    f"STT '{self.stt}' performs its own endpointing; overriding "
                    f"{field}='{getattr(self, field)}' -> '{target}'."
                )
            else:
                logger.info(f"STT '{self.stt}': auto-setting {field}='{target}' (self-endpointing).")
            setattr(self, field, target)
        return self

    @model_validator(mode="after")
    def _validate_latency_optimizations(self) -> "ModelConfig":
        allowed = {"off", "auto"}
        if self.pre_tool_speech not in allowed:
            raise ValueError(f"pre_tool_speech must be one of {sorted(allowed)}, got '{self.pre_tool_speech}'")
        # pre_tool_speech is honored by both CASCADE and AUDIO_LLM; llm_streaming by both
        # (via BaseALMClient.complete_stream); only parallel_tool_calls remains CASCADE-only.
        if self.parallel_tool_calls is not None and self.pipeline_type != PipelineType.CASCADE:
            logger.warning(
                "parallel_tool_calls applies only to the CASCADE pipeline; it will be ignored "
                f"for pipeline_type={self.pipeline_type}."
            )
        return self


class PipelineType(StrEnum):
    """Type of voice pipeline."""

    CASCADE = "cascade"
    AUDIO_LLM = "audio_llm"
    S2S = "s2s"


def get_pipeline_type(model_data: dict) -> PipelineType:
    """Return the pipeline type for the given model config.

    Works with raw dicts, e.g., from config.json.
    Also handles legacy configs where ``realtime_model`` was stored alongside
    ``llm_model`` in a flat dict.
    """
    if s2s_value := model_data.get("s2s"):
        # ElevenLabs uses s2s_params for configuration but is a cascade pipeline internally
        if s2s_value == "elevenlabs":
            return PipelineType.CASCADE
        # Ultravox uses s2s_params for plumbing but is an audio-LLM (audio in, text out, separate TTS)
        if s2s_value == "ultravox":
            return PipelineType.AUDIO_LLM
        return PipelineType.S2S
    if model_data.get("audio_llm"):
        return PipelineType.AUDIO_LLM
    # Legacy: realtime_model was a sibling of llm_model before the union split
    if model_data.get("realtime_model"):
        return PipelineType.S2S
    return PipelineType.CASCADE


class BackgroundNoiseType(StrEnum):
    """Ambient noise type mixed into user audio (speech and silence)."""

    airport_gate = "airport_gate"
    baby_crying = "baby_crying"
    background_music = "background_music"
    bad_connection_static = "bad_connection_static"
    coffee_shop = "coffee_shop"
    loud_construction = "loud_construction"
    nyc_street = "nyc_street"
    road_noise = "road_noise"


class AccentType(StrEnum):
    """Accent variant — selects a different ElevenLabs agent ID for the user simulator."""

    french = "french"
    indian = "indian"
    spanish = "spanish"
    chinese = "chinese"


class BehaviorType(StrEnum):
    """User behavior variant — modifies persona prompt and selects a different agent ID."""

    aggressive_impatient = "aggressive_impatient"
    elderly_slow = "elderly_slow"
    forgetful_disorganized = "forgetful_disorganized"


# Supported languages: keys are pipecat Language codes; presence in this dict
# defines what's supported. Display names are used in prompts and logging.
LANGUAGE_DISPLAY_NAMES: dict[Language, str] = {
    Language.EN: "English",
    Language.FR: "European French",
    Language.FR_CA: "Canadian French",
    Language.ES: "European Spanish",
    Language.DE: "German",
}


class PerturbationConfig(BaseModel):
    """Perturbations applied to the simulated user during a benchmark run.

    Three independent axes:
    - background_noise: ambient audio mixed into user speech and silence
    - accent: uses accent-specific ElevenLabs agent IDs (mutually exclusive with behavior)
    - behavior: modifies persona prompt + uses behavior-specific agent IDs (mutually exclusive with accent)
    - connection_degradation: stacks codec artifacts, packet loss, and volume fluctuation on top

    Agent ID env vars follow the pattern EVA_{TYPE}_USER_F / EVA_{TYPE}_USER_M.
    Default (no accent/behavior): EVA_EN_USER_F and EVA_EN_USER_M (language defaults to English).
    """

    model_config = ConfigDict(extra="forbid")

    background_noise: BackgroundNoiseType | None = Field(
        None,
        description="Ambient noise type to mix into user audio",
    )
    snr_db: float = Field(
        15.0,
        description="Signal-to-noise ratio in dB for file-based background noise (higher = cleaner)",
    )
    accent: AccentType | None = Field(None, description="Accent variant for the user simulator voice")
    behavior: BehaviorType | None = Field(None, description="User behavior variant (modifies persona + agent ID)")
    connection_degradation: bool = Field(
        False,
        description="Apply VoIP degradation (codec artifacts, packet loss, volume fluctuation) on top of other perturbations",
    )

    @model_validator(mode="after")
    def _validate_exclusivity(self) -> "PerturbationConfig":
        if self.accent is not None and self.behavior is not None:
            raise ValueError(
                "accent and behavior cannot both be set — they each require exclusive use of the ElevenLabs agent ID"
            )
        return self


class ElevenLabsSimulatorConfig(BaseModel):
    """ElevenLabs Conversational AI settings for the user simulator."""

    provider: Literal["elevenlabs"] = "elevenlabs"

    @model_validator(mode="before")
    @classmethod
    def _warn_extra_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            extra = [k for k in data if k not in cls.model_fields and k != "provider"]
            if extra:
                logger.warning(
                    f"ElevenLabsSimulatorConfig received unrecognised fields that will be ignored: "
                    f"{', '.join(sorted(extra))}"
                )
        return data


class OpenAIRealtimeSimulatorConfig(BaseModel):
    """OpenAI Realtime-specific settings for the user simulator."""

    provider: Literal["openai_realtime"] = "openai_realtime"
    model: str = Field("gpt-realtime-1.5", description="OpenAI Realtime model.")
    female_voice: str = Field("marin", description="Voice used for female caller personas.")
    male_voice: str = Field("cedar", description="Voice used for male caller personas.")


UserSimulatorConfig = Annotated[
    ElevenLabsSimulatorConfig | OpenAIRealtimeSimulatorConfig,
    Field(discriminator="provider"),
]


class RunConfig(BaseSettings):
    """A New End-to-end Framework for Evaluating Voice Agents\033[94m

    ▁▁▁▁▁▁▁▁▁▁ ▁▁▁        ▁▁▁  ▁▁▁▁
    ▏         ▏╲  ╲      ╱  ╱ ╱    ╲
    ▏ ▕▔▔▔▔▔▔▔  ╲  ╲    ╱  ╱ ╱  ╱╲  ╲
    ▏  ▔▔▔▔▔▔▏   ╲  ╲  ╱  ╱ ╱  ╱  ╲  ╲
    ▏ ▕▔▔▔▔▔▔     ╲  ╲╱  ╱ ╱   ▔▔▔▔   ╲
    ▏  ▔▔▔▔▔▔▔▏    ╲    ╱ ╱  ╱▔▔▔▔▔▔╲  ╲
    ▔▔▔▔▔▔▔▔▔▔      ▔▔▔▔  ▔▔▔        ▔▔▔\033[m
    """

    model_config = SettingsConfigDict(
        cli_hide_none_type=True,
        cli_implicit_flags="toggle",
        cli_kebab_case=True,
        env_nested_delimiter="__",
        env_prefix="EVA_",
        extra="ignore",
        populate_by_name=True,
    )

    # Maps *_params field names to their provider field for env override logic
    _PARAMS_TO_PROVIDER: ClassVar[dict[str, str]] = {
        "stt_params": "stt",
        "tts_params": "tts",
        "s2s_params": "s2s",
        "audio_llm_params": "audio_llm",
    }
    # Keys always read from the live environment (not persisted across runs)
    _ENV_OVERRIDE_KEYS: ClassVar[set[str]] = {"url", "urls"}
    # Substrings that identify secret keys (redacted in logs and config.json)
    _SECRET_KEY_PATTERNS: ClassVar[set[str]] = {"key", "credentials", "secret"}

    class ModelDeployment(DeploymentTypedDict):
        """DeploymentTypedDict that preserves extra keys in litellm_params."""

        __pydantic_config__ = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    model_list: list[ModelDeployment] = Field(min_length=1)

    # Model to test
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Pipeline (STT + LLM + TTS), speech-to-speech, or audio-LLM model configuration",
    )

    # Framework selection
    framework: Literal["pipecat", "openai_realtime", "gemini_live", "elevenlabs", "grok_voice"] = Field(
        "pipecat",
        description=(
            "Agent framework to use for the assistant server."
            "'pipecat' (default): Pipecat pipeline."
            "'openai_realtime': OpenAI Realtime API directly."
            "'gemini_live': Gemini Live API via google-genai."
            "'elevenlabs': ElevenLabs Conversational AI API."
            "'grok_voice': xAI Grok voice realtime API."
        ),
    )

    # Run identifier
    run_id: str = Field(
        "timestamp and model name(s)",  # Overwritten by _set_default_run_id()
        description="Run identifier, auto-generated if not provided",
    )

    # Data paths
    domain: Literal["airline", "itsm", "medical_hr"] = "airline"

    # Rerun settings
    max_rerun_attempts: int = Field(3, ge=0, le=20, description="Maximum number of rerun attempts for failed records")
    force_revalidation: bool = Field(False, description="Re-validate all records even if they already have metrics")
    rerun_failed_metrics: bool = Field(
        False, description="Rerun only previously failed metric computations (requires --run-id)"
    )
    force_rerun_metrics: bool = Field(
        False,
        description="Force rerun all requested metrics, overwriting existing successful results (requires --run-id)",
    )
    tool_module_path: str | None = Field(
        None,
        description="Python module path with tool functions (e.g., 'eva.assistant.tools.airline_tools'). "
        "If not specified, will be loaded from agent config.",
    )

    provenance: CliSuppress[RunProvenance | None] = Field(
        None,
        description="Run provenance — auto-populated at runtime with git state, artifact hashes, and environment info",
        init=False,
    )

    resolved_models: CliSuppress[dict[str, Any] | None] = Field(
        None,
        description="Exact models used at runtime (provider + model + alias for STT/TTS, LLM name). "
        "Auto-populated before the run starts.",
        init=False,
    )

    validation_thresholds: dict[str, float | int] = Field(
        {
            "conversation_valid_end": 1.0,
            "user_behavioral_fidelity": 1.0,
            "max_time_limit_attempts": 1,
        },
        description="Validation metric thresholds and settings for rerun decisions (JSON). "
        "max_time_limit_attempts sets the max number of attempts that timeout before accepting a run for evaluation. Default is 1.",
    )

    # Multi-attempt (for pass@k evaluation)
    num_trials: int = Field(
        1,
        ge=1,
        le=100,
        description="Number of times to run each record (for pass@k evaluation). "
        "When > 1, each record is run num_trials times with output in "
        "{record_id}/trial_{i} directories.",
    )

    metrics: list[str] | None = Field(
        default_factory=_get_all_metrics,
        description="Metrics to run. Skip all metrics with `EVA_METRICS=` or `--metrics=`.",
    )

    # Aggregate-only mode
    aggregate_only: bool = Field(
        False,
        description="Recompute EVA aggregate scores from existing metrics.json files without re-running judges",
    )

    perturbation: PerturbationConfig | None = Field(
        None,
        description=(
            "Perturbations applied to the simulated user. "
            "Example: EVA_PERTURBATION__BACKGROUND_NOISE=coffee_shop EVA_PERTURBATION__ACCENT=french. "
            "See PerturbationConfig for all options."
        ),
    )

    user_simulator: UserSimulatorConfig = Field(
        default_factory=ElevenLabsSimulatorConfig,
        description="Configuration for the provider that simulates the caller.",
    )
    # User simulator language — picks per-language ElevenLabs agent IDs
    language: Language = Field(
        Language.EN,
        description=(
            "Language for the user simulator. When set to a non-English value, "
            "the matching EVA_{LANGUAGE}_USER_F and EVA_{LANGUAGE}_USER_M agent IDs must also be set. "
            "Mutually exclusive with accent and behavior perturbations."
        ),
    )

    # Debug and filtering
    debug: bool = Field(
        False,
        description="Debug mode: run only 1 record",
    )
    record_ids: list[str] | None = Field(
        None,
        description="Specific record IDs to run",
    )

    # Execution
    max_concurrent_conversations: int = Field(
        1,
        ge=1,
        le=100,
        description="Maximum number of concurrent conversations",
    )
    conversation_time_limit_seconds: int = Field(
        600,
        ge=30,
        le=10000,
        description="Max conversation duration in seconds",
    )

    # Output
    output_dir: Path = Field(
        Path("output"),
        description="Output directory for results",
    )

    # Port pool for parallel conversations
    base_port: int = Field(
        10000,
        ge=1024,
        le=65000,
        description="Base port for WebSocket servers",
    )
    port_pool_size: int = Field(
        150,
        ge=10,
        le=500,
        description="Number of ports in the pool",
    )

    # Script-only
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO",
        description="Logging level",
    )
    dry_run: bool = Field(False, description="Validate configuration without running")

    @computed_field
    @property
    def dataset_path(self) -> Path:
        return Path(f"data/{self.domain}_dataset.json")

    @computed_field
    @property
    def tool_mocks_path(self) -> Path:
        return Path(f"data/{self.domain}_scenarios")

    @computed_field
    @property
    def aliases_path(self) -> Path:
        return Path(f"data/{self.domain}_aliases")

    @computed_field
    @property
    def agent_config_path(self) -> Path:
        return Path(f"configs/agents/{self.domain}_agent.yaml")

    @model_validator(mode="after")
    def _check_companion_services(self) -> "RunConfig":
        """Validate pipeline mode mutual exclusivity and required companion services.

        Skipped entirely when ``max_rerun_attempts == 0 or self.aggregate_only`` where the model
        config is unused and conflicting env vars are harmless.
        """
        if (
            isinstance(self.user_simulator, OpenAIRealtimeSimulatorConfig)
            and self.perturbation is not None
            and self.perturbation.accent is not None
        ):
            raise ValueError(
                "Accent perturbations require the ElevenLabs user simulator; "
                "OpenAI Realtime supports behavior, noise, and connection perturbations."
            )

        if self.max_rerun_attempts == 0 or self.aggregate_only:
            return self

        # ── Validate pipeline mode mutual exclusivity ──
        active = [
            name
            for flag, name in [
                (self.model.llm, "EVA_MODEL__LLM"),
                (self.model.s2s, "EVA_MODEL__S2S"),
                (self.model.audio_llm, "EVA_MODEL__AUDIO_LLM"),
            ]
            if flag
        ]
        if len(active) != 1:
            raise ValueError(
                (f"Multiple pipeline modes set: {', '.join(active)}. " if active else "Model pipeline required. ")
                + "Set exactly one of: EVA_MODEL__LLM (TTS+LLM+TTS), EVA_MODEL__S2S (S2S), or EVA_MODEL__AUDIO_LLM (Audio LLM+TTS)."
            )

        # ── Validate companion services ──
        errors: list[InitErrorDetails] = []
        match self.model.pipeline_type:
            case PipelineType.CASCADE:
                errors.extend(self._validate_service_params("STT", self.model.stt, self.model.stt_params))
                errors.extend(self._validate_service_params("TTS", self.model.tts, self.model.tts_params))
            case PipelineType.AUDIO_LLM:
                errors.extend(self._validate_service_params("TTS", self.model.tts, self.model.tts_params))
                errors.extend(
                    self._validate_service_params("AUDIO_LLM", self.model.audio_llm, self.model.audio_llm_params)
                )
            case PipelineType.S2S:
                errors.extend(self._validate_service_params("S2S", self.model.s2s, self.model.s2s_params))
        if errors:
            raise ValidationError.from_exception_data(title=type(self).__name__, line_errors=errors)

        # ── Set default run_id ──
        # self.model.pipeline_parts is only available if self.model is valid, which the above asserts.
        if "run_id" not in self.model_fields_set:
            suffix = "_".join(v for v in self.model.pipeline_parts.values() if v)
            lang = self.language.value
            domain = self.domain.replace("_", "-")
            self.run_id = f"{datetime.now(UTC):%Y-%m-%d_%H-%M-%S.%f}_{domain}_{lang}_{suffix}"

        return self

    def _validate_service_params(
        self, service: str, provider: str | None, params: dict[str, Any] | None
    ) -> Iterator[InitErrorDetails]:
        """Validate that the service's name is set and its params contain the required keys."""
        if not provider:
            message = f"EVA_MODEL__{service} required in {self.model.pipeline_type} mode."
            loc = ("model", service.lower())
            yield InitErrorDetails(type=PydanticCustomError("missing_service", message), loc=loc, input=provider)

        required_keys = ["api_key", "model"]
        missing = [key for key in required_keys if key not in params] if params else required_keys
        if missing:
            missing_str = " and ".join(f'"{k}"' for k in missing)
            env_var = f"EVA_MODEL__{service}_PARAMS"
            message = (
                f"{missing_str} required in {env_var} for {provider} {service}. "
                f'Example: {env_var}=\'{{"api_key": "your_key", "model": "your_model"}}\''
            )
            loc = ("model", f"{service.lower()}_params")
            yield InitErrorDetails(type=PydanticCustomError("missing_service_params", message), loc=loc, input=params)

    @model_validator(mode="after")
    def _check_language_personas(self) -> "RunConfig":
        """When a non-English language is set, validate matching agent IDs and mutual exclusivity."""
        if self.language == Language.EN or not isinstance(self.user_simulator, ElevenLabsSimulatorConfig):
            return self

        key = self.language.value.upper().replace("-", "_")
        missing = [
            f"EVA_{key}_USER_{gender}" for gender in ("F", "M") if not os.environ.get(f"EVA_{key}_USER_{gender}")
        ]
        if missing:
            raise ValueError(
                f"EVA_LANGUAGE is set to {self.language.value!r}, but the following required env vars are missing: "
                f"{', '.join(missing)}"
            )

        if self.perturbation is not None and (
            self.perturbation.accent is not None or self.perturbation.behavior is not None
        ):
            conflicts = [
                f"EVA_PERTURBATION__{k.upper()}={v}"
                for k, v in (("accent", self.perturbation.accent), ("behavior", self.perturbation.behavior))
                if v is not None
            ]
            raise ValueError(
                f"EVA_LANGUAGE ({self.language.value!r}) cannot be combined with accent/behavior perturbations "
                f"({', '.join(conflicts)}) — they each require exclusive use of the ElevenLabs agent ID."
            )

        return self

    @model_validator(mode="after")
    def _check_openai_realtime_simulator(self) -> "RunConfig":
        """When openai_realtime user simulator is selected, OPENAI_API_KEY must be present."""
        if not isinstance(self.user_simulator, OpenAIRealtimeSimulatorConfig):
            return self
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("EVA_USER_SIMULATOR__PROVIDER=openai_realtime requires OPENAI_API_KEY to be set.")
        return self

    @model_validator(mode="before")
    @classmethod
    def _handle_all_keyword(cls, data: Any):
        """Catch EVA_METRICS=all for backward compatibility and delegate to the default_factory."""
        match data:
            case {"metrics": str() as value, **rest} | {"metrics": [str() as value], **rest} if value.lower() == "all":
                return rest
        return data

    @field_validator("metrics", "record_ids", mode="before")
    @classmethod
    def _parse_comma_separated(cls, v: Any) -> list[str] | None:
        """Accept comma-separated strings from env vars."""
        if isinstance(v, (int, float)):
            return [str(v)]
        if isinstance(v, str):
            items = [s for item in v.split(",") if (s := item.strip())]
            return items or None
        if isinstance(v, list):
            return [s for item in v if (s := str(item).strip())] or None
        return v

    @classmethod
    def _is_secret_key(cls, key: str) -> bool:
        """Return True if *key* matches any pattern in _SECRET_KEY_PATTERNS."""
        return any(pattern in key for pattern in cls._SECRET_KEY_PATTERNS)

    @classmethod
    def _redact_dict(cls, params: dict) -> dict:
        """Return a copy of *params* with secret values replaced by ``***``."""
        return {k: "***" if cls._is_secret_key(k) else v for k, v in params.items()}

    @field_serializer("model_list")
    @classmethod
    def _redact_model_list(cls, deployments: list[ModelDeployment]) -> list[dict]:
        """Redact secret values in litellm_params when serializing."""
        redacted = []
        for deployment in deployments:
            deployment = copy.deepcopy(deployment)
            if "litellm_params" in deployment:
                deployment["litellm_params"] = cls._redact_dict(deployment["litellm_params"])
            redacted.append(deployment)
        return redacted

    @field_serializer("model")
    @classmethod
    def _redact_model_params(cls, model: ModelConfig) -> dict:
        """Redact secret values in STT/TTS/S2S/AudioLLM params when serializing."""
        data = model.model_dump(mode="json", exclude_none=True)
        for field_name, value in data.items():
            if field_name.endswith("_params") and isinstance(value, dict):
                data[field_name] = cls._redact_dict(value)
        return data

    def apply_env_overrides(self, live: "RunConfig", strict_llm: bool = True) -> None:
        """Apply environment-dependent values from *live* config onto this (saved) config.

        Restores redacted secrets (``***``) and overrides dynamic fields (``url``,
        ``urls``) in ``model.*_params`` and ``model_list[].litellm_params``.

        Args:
            live: The live RunConfig with current environment values.
            strict_llm: If True (default), raise when the active LLM deployment has
                redacted secrets but is not in the current EVA_MODEL_LIST. Set to False
                for metrics-only re-runs where the LLM is not needed.

        Raises:
            ValueError: If provider or alias differs for a service with redacted secrets,
                or (when strict_llm=True) if the active LLM deployment is missing.
        """
        # ── model.*_params (STT / TTS / S2S / AudioLLM) ──
        for params_field, provider_field in self._PARAMS_TO_PROVIDER.items():
            saved = getattr(self.model, params_field, None)
            source = getattr(live.model, params_field, None)
            if not isinstance(saved, dict) or not isinstance(source, dict):
                continue

            has_redacted = any(v == "***" for v in saved.values())
            has_env_overrides = any(k in saved or k in source for k in self._ENV_OVERRIDE_KEYS)
            if not has_redacted and not has_env_overrides:
                continue

            if has_redacted:
                saved_alias = saved.get("alias")
                live_alias = source.get("alias")
                if saved_alias and live_alias and saved_alias != live_alias:
                    raise ValueError(
                        f"Cannot restore secrets: saved {params_field}[alias]={saved_alias!r} "
                        f"but current environment has {params_field}[alias]={live_alias!r}"
                    )

                saved_provider = getattr(self.model, provider_field, None)
                live_provider = getattr(live.model, provider_field, None)
                if saved_provider != live_provider:
                    logger.warning(
                        f"Provider mismatch for {params_field}: saved {saved_provider!r}, "
                        f"current environment has {live_provider!r}"
                    )

                saved_model = saved.get("model")
                live_model = source.get("model")
                if saved_model and live_model and saved_model != live_model:
                    logger.warning(
                        f"Model mismatch for {params_field}: saved {saved_model!r}, "
                        f"current environment has {live_model!r}"
                    )

                for key, value in saved.items():
                    if value == "***" and key in source:
                        saved[key] = source[key]

            # Always use url/urls from the live environment
            for key in self._ENV_OVERRIDE_KEYS:
                if key in source:
                    saved_val = saved.get(key)
                    if saved_val and saved_val != source[key]:
                        logger.warning(
                            f"{params_field}[{key}] differs: saved {saved_val!r}, "
                            f"using {source[key]!r} from current environment"
                        )
                    saved[key] = source[key]

        # ── model_list[].litellm_params (LLM deployments) ──
        live_by_name = {d["model_name"]: d for d in live.model_list if "model_name" in d}
        for deployment in self.model_list:
            name = deployment.get("model_name")
            if not name:
                continue
            saved_params = deployment.get("litellm_params", {})
            has_redacted = any(v == "***" for v in saved_params.values())
            if not has_redacted:
                continue
            if name not in live_by_name:
                active_llm = getattr(self.model, "llm", None)
                if name == active_llm and strict_llm:
                    raise ValueError(
                        f"Cannot restore secrets: deployment {name!r} not found in "
                        f"current EVA_MODEL_LIST (available: {list(live_by_name)})"
                    )
                logger.warning(
                    f"Deployment {name!r} has redacted secrets but is not in the current "
                    f"EVA_MODEL_LIST (available: {list(live_by_name)}) — skipping. "
                    f"Any metric or agent call routed to this deployment will fail."
                )
                continue
            live_params = live_by_name[name].get("litellm_params", {})
            for key, value in saved_params.items():
                if value == "***" and key in live_params:
                    saved_params[key] = live_params[key]

        # ── Log resolved configuration ──
        for params_field, provider_field in self._PARAMS_TO_PROVIDER.items():
            params = getattr(self.model, params_field, None)
            provider = getattr(self.model, provider_field, None)
            if isinstance(params, dict) and params:
                logger.info(f"Resolved {provider_field} ({provider}): {self._redact_dict(params)}")

        for deployment in self.model_list:
            name = deployment.get("model_name", "?")
            params = deployment.get("litellm_params", {})
            logger.info(f"Resolved deployment {name}: {self._redact_dict(params)}")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        # Convert to dict, handling Path objects
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
