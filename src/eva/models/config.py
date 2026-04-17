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
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from litellm.types.router import DeploymentTypedDict
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict

from eva.models.provenance import RunProvenance

logger = logging.getLogger(__name__)


def _param_alias(params: dict[str, Any]) -> str:
    """Return the display alias from a params dict."""
    return params.get("alias") or params["model"]


class PipelineConfig(BaseModel):
    """Configuration for a STT + LLM + TTS pipeline."""

    model_config = ConfigDict(extra="forbid")

    # Mapping from legacy config.json field names to current names.
    _LEGACY_RENAMES: ClassVar[dict[str, str]] = {
        "llm_model": "llm",
        "stt_model": "stt",
        "tts_model": "tts",
    }
    _LEGACY_DROP: ClassVar[set[str]] = {"realtime_model", "realtime_model_params"}

    llm: str = Field(
        description="LLM model name matching a model_name in --model-list/EVA_MODEL_LIST",
        examples=["gpt-5.2", "gemini-3-pro"],
    )
    stt: str = Field(description="STT model", examples=["deepgram", "openai_whisper"])
    tts: str = Field(description="TTS model", examples=["cartesia", "elevenlabs"])

    stt_params: dict[str, Any] = Field({}, description="Additional STT model parameters (JSON)")
    tts_params: dict[str, Any] = Field({}, description="Additional TTS model parameters (JSON)")

    turn_strategy: Literal["smart", "external"] = Field(
        "smart",
        description=(
            "User turn detection strategy. "
            "'smart' uses LocalSmartTurnAnalyzerV3 + SileroVAD (default). "
            "'external' uses ExternalUserTurnStrategies for services with built-in turn detection "
            "(e.g., deepgram-flux, Speechmatics). "
            "Set via EVA_MODEL__TURN_STRATEGY=external."
        ),
    )

    @property
    def pipeline_parts(self) -> dict[str, str]:
        """Component names for this pipeline."""
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


class SpeechToSpeechConfig(BaseModel):
    """Configuration for a speech-to-speech model."""

    model_config = ConfigDict(extra="forbid")

    s2s: str = Field(description="Speech-to-speech model name", examples=["gpt-realtime-mini", "gemini_live"])
    s2s_params: dict[str, Any] = Field({}, description="Additional speech-to-speech model parameters (JSON)")

    turn_strategy: Literal["smart", "external"] = Field(
        "smart",
        description=(
            "User turn detection strategy. "
            "'smart' uses LocalSmartTurnAnalyzerV3 + SileroVAD (default). "
            "'external' uses ExternalUserTurnStrategies for services with built-in turn detection "
            "(e.g., deepgram-flux, Speechmatics). "
            "Set via EVA_MODEL__TURN_STRATEGY=external."
        ),
    )

    @property
    def pipeline_parts(self) -> dict[str, str]:
        """Component names for this pipeline."""
        return {"s2s": _param_alias(self.s2s_params) or self.s2s}


class AudioLLMConfig(BaseModel):
    """Configuration for an Audio-LLM pipeline (audio in, text out, separate TTS).

    Used for models like self-hosted Ultravox that accept audio input + text context
    and return text output, requiring a separate TTS stage for speech synthesis.
    """

    model_config = ConfigDict(extra="forbid")

    audio_llm: str = Field(
        description="Audio-LLM model identifier",
        examples=["vllm"],
    )
    audio_llm_params: dict[str, Any] = Field(
        {},
        description="Audio-LLM parameters (JSON): base_url (required), api_key, model, temperature, max_tokens",
    )
    tts: str = Field(description="TTS model", examples=["cartesia", "elevenlabs"])
    tts_params: dict[str, Any] = Field({}, description="Additional TTS model parameters (JSON)")

    @property
    def pipeline_parts(self) -> dict[str, str]:
        """Component names for this pipeline."""
        return {
            "audio_llm": _param_alias(self.audio_llm_params) or self.audio_llm,
            "tts": _param_alias(self.tts_params) or self.tts,
        }


_PIPELINE_FIELDS = {
    "llm",
    "stt",
    "tts",
    "stt_params",
    "tts_params",
    "turn_strategy",
    *PipelineConfig._LEGACY_RENAMES,
    *PipelineConfig._LEGACY_DROP,
}
_S2S_FIELDS = {"s2s", "s2s_params", "turn_strategy"}
_AUDIO_LLM_FIELDS = {"audio_llm", "audio_llm_params", "tts", "tts_params"}


class PipelineType(StrEnum):
    """Type of voice pipeline."""

    CASCADE = "cascade"
    AUDIO_LLM = "audio_llm"
    S2S = "s2s"


def _model_config_discriminator(data: Any) -> str:
    """Discriminate which pipeline config type to use based on unique fields."""
    if isinstance(data, dict):
        if "audio_llm" in data:
            return "audio_llm"
        if "s2s" in data:
            return "s2s"
        return "pipeline"
    if isinstance(data, AudioLLMConfig):
        return "audio_llm"
    if isinstance(data, SpeechToSpeechConfig):
        return "s2s"
    return "pipeline"


def get_pipeline_type(model_data: dict | Any) -> PipelineType:
    """Return the pipeline type for the given model config.

    Works with both raw dicts (e.g. from config.json) and parsed model config objects.
    Also handles legacy configs where ``realtime_model`` was stored alongside
    ``llm_model`` in a flat dict (before the discriminated-union refactor).
    """
    mode = _model_config_discriminator(model_data)
    if mode == "s2s":
        return PipelineType.S2S
    if mode == "audio_llm":
        return PipelineType.AUDIO_LLM
    # Legacy: realtime_model was a sibling of llm_model before the union split
    if isinstance(model_data, dict) and model_data.get("realtime_model"):
        return PipelineType.S2S
    return PipelineType.CASCADE


def _strip_other_mode_fields(data: dict, strict: bool = True) -> dict:
    """Validate pipeline mode exclusivity, then strip irrelevant shared fields.

    Raises ``ValueError`` if multiple pipeline modes are specified (when strict=True).
    Then strips shared fields (e.g. ``tts`` from S2S mode) so that
    ``extra="forbid"`` on each config class doesn't reject them.

    Args:
        data: Raw config dictionary from the YAML/env input.
        strict: If False, skip the conflict error (used for metrics-only re-runs
            where the model config is not needed).
    """
    # --- Mutual exclusivity: only one pipeline mode allowed ---
    has_llm = bool(data.get("llm") or data.get("llm_model"))
    has_s2s = bool(data.get("s2s"))
    has_audio_llm = bool(data.get("audio_llm"))
    active = [
        name
        for flag, name in [
            (has_llm, "EVA_MODEL__LLM"),
            (has_s2s, "EVA_MODEL__S2S"),
            (has_audio_llm, "EVA_MODEL__AUDIO_LLM"),
        ]
        if flag
    ]
    if len(active) > 1 and strict:
        raise ValueError(
            f"Multiple pipeline modes set: {', '.join(active)}. "
            f"Set exactly one of: EVA_MODEL__LLM (ASR-LLM-TTS), "
            f"EVA_MODEL__S2S (S2S), or EVA_MODEL__AUDIO_LLM (SpeechLM-TTS)."
        )

    mode = _model_config_discriminator(data)
    if mode == "audio_llm":
        return {k: v for k, v in data.items() if k in _AUDIO_LLM_FIELDS}
    if mode == "s2s":
        return {k: v for k, v in data.items() if k in _S2S_FIELDS}
    # pipeline: keep pipeline fields + any legacy fields the model_validator handles
    return {k: v for k, v in data.items() if k in _PIPELINE_FIELDS}


# Discriminated union so Pydantic picks the right config type from env vars / CLI
ModelConfigUnion = Annotated[
    Annotated[PipelineConfig, Tag("pipeline")]
    | Annotated[SpeechToSpeechConfig, Tag("s2s")]
    | Annotated[AudioLLMConfig, Tag("audio_llm")],
    Discriminator(_model_config_discriminator),
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
        cli_shortcuts={
            # TODO: Remove deprecated aliases after a little while.
            "model.llm": "llm-model",
            "model.stt": "stt-model",
            "model.tts": "tts-model",
            "model.s2s": "realtime-model",
            "model.audio_llm": "audio-llm-model",
            "max-rerun-attempts": "max-reruns",
            "record-ids": "ids",
            "max-concurrent-conversations": ["max-concurrent", "n"],
            "conversation-timeout-seconds": "timeout",
            "output-dir": ["output", "o"],
        },
        env_nested_delimiter="__",
        env_prefix="EVA_",
        extra="ignore",
        populate_by_name=True,
    )

    _VALIDATION_METRIC_NAMES: ClassVar[set[str]] = {
        "conversation_finished",
        "user_behavioral_fidelity",
        "user_speech_fidelity",
    }

    _DEPRECATED_ENV_VARS: ClassVar[dict[str, str]] = {
        "LLM_MODEL": "EVA_MODEL__LLM",
        "STT_MODEL": "EVA_MODEL__STT",
        "TTS_MODEL": "EVA_MODEL__TTS",
        "REALTIME_MODEL": "EVA_MODEL__S2S",
        "EVA_MODEL__REALTIME_MODEL": "EVA_MODEL__S2S",
        "LLM_PARAMS": "litellm_params in EVA_MODEL_LIST",
        "LLM_MAX_TOKENS": "litellm_params in EVA_MODEL_LIST",
        "LLM_REASONING_EFFORT": "litellm_params in EVA_MODEL_LIST",
        "STT_PARAMS": "EVA_MODEL__STT_PARAMS",
        "TTS_PARAMS": "EVA_MODEL__TTS_PARAMS",
        "REALTIME_MODEL_PARAMS": "EVA_MODEL__S2S_PARAMS",
        "EVA_MODEL__REALTIME_MODEL_PARAMS": "EVA_MODEL__S2S_PARAMS",
        "EVA_MAX_CONCURRENT": "EVA_MAX_CONCURRENT_CONVERSATIONS",
        "EVA_CONVERSATION_TIMEOUT": "EVA_CONVERSATION_TIMEOUT_SECONDS",
        "EVA_METRICS_TO_RUN": "EVA_METRICS",
    }

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
    model: ModelConfigUnion = Field(
        description="Pipeline (STT + LLM + TTS), speech-to-speech, or audio-LLM model configuration",
    )

    # Run identifier
    run_id: str = Field(
        "timestamp and model name(s)",  # Overwritten by _set_default_run_id()
        description="Run identifier, auto-generated if not provided",
    )

    # Data paths
    domain: Literal["airline"] = "airline"

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

    validation_thresholds: dict[str, float] = Field(
        {
            "conversation_finished": 1.0,
            "user_behavioral_fidelity": 1.0,
        },
        description="Validation metric thresholds for rerun decisions (JSON)",
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
        None,
        description="Metrics to run with benchmark",
        validation_alias=AliasChoices("metrics", "metrics_to_run"),
    )

    # Aggregate-only mode
    aggregate_only: bool = Field(
        False,
        description="Recompute EVA aggregate scores from existing metrics.json files without re-running judges",
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
    conversation_timeout_seconds: int = Field(
        360,
        ge=30,
        le=10000,
        description="Timeout for each conversation in seconds",
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

    # Deprecated env vars
    deprecated: CliSuppress[str | None] = Field(
        None,
        validation_alias=AliasChoices(*_DEPRECATED_ENV_VARS.keys()),
        exclude=True,
    )

    @computed_field
    @property
    def dataset_path(self) -> Path:
        return Path(f"data/{self.domain}_dataset.jsonl")

    @computed_field
    @property
    def tool_mocks_path(self) -> Path:
        return Path(f"data/{self.domain}_scenarios")

    @computed_field
    @property
    def agent_config_path(self) -> Path:
        return Path(f"configs/agents/{self.domain}_agent.yaml")

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_aliases(cls, data: Any) -> Any:
        """Error out if deprecated environment variables are detected."""
        if not isinstance(data, dict):
            return data
        found = [f"  {old} -> use {new}" for old, new in cls._DEPRECATED_ENV_VARS.items() if old in data]
        if found:
            raise ValueError("Deprecated environment variables detected:\n" + "\n".join(found))

        # Strip env-var fields from other pipeline modes so extra="forbid" doesn't reject them.
        # For metrics-only re-runs, skip the strict conflict check — the model isn't used.
        if isinstance(data.get("model"), dict):
            force_rerun = bool(data.get("force_rerun_metrics"))
            data["model"] = _strip_other_mode_fields(data["model"], strict=not force_rerun)

        return data

    @model_validator(mode="after")
    def _check_companion_services(self) -> "RunConfig":
        """Ensure required companion services are set for each pipeline mode."""
        required_keys = ["api_key", "model"]
        if isinstance(self.model, PipelineConfig):
            self._validate_service_params("STT", self.model.stt, required_keys, self.model.stt_params)
            self._validate_service_params("TTS", self.model.tts, required_keys, self.model.tts_params)
        elif isinstance(self.model, AudioLLMConfig):
            self._validate_service_params("TTS", self.model.tts, required_keys, self.model.tts_params)
            self._validate_service_params("audio_llm", self.model.audio_llm, required_keys, self.model.audio_llm_params)
        elif isinstance(self.model, SpeechToSpeechConfig):
            # api_key is required, some s2s services don't require model
            self._validate_service_params("S2S", self.model.s2s, required_keys, self.model.s2s_params)
        return self

    @model_validator(mode="after")
    def _set_default_run_id(self) -> "RunConfig":
        if "run_id" not in self.model_fields_set:
            suffix = "_".join(v for v in self.model.pipeline_parts.values() if v)
            self.run_id = f"{datetime.now(UTC):%Y-%m-%d_%H-%M-%S.%f}_{suffix}"
        return self

    @classmethod
    def _validate_service_params(
        cls, service: str, provider: str, required_keys: list[str], params: dict[str, Any]
    ) -> None:
        """Validate that STT/TTS params contain required keys."""
        missing = [key for key in required_keys if key not in params]
        if missing:
            missing_str = " and ".join(f'"{k}"' for k in missing)
            env_var = f"EVA_MODEL__{service}_PARAMS"
            raise ValueError(
                f"{missing_str} required in {env_var} for {provider} {service}. "
                f'Example: {env_var}=\'{{"api_key": "your_key", "model": "your_model"}}\''
            )

    @field_validator("metrics", "record_ids", mode="before")
    @classmethod
    def _parse_comma_separated(cls, v: Any) -> list[str] | None:
        """Accept comma-separated strings from env vars."""
        if isinstance(v, str):
            items = [s.strip() for s in v.split(",") if s.strip()]
            return items or None
        return v

    @field_validator("metrics", mode="after")
    @classmethod
    def _expand_metrics_all(cls, v: list[str] | None) -> list[str] | None:
        """Expand the 'all' keyword to all non-validation metrics."""
        if v and len(v) == 1 and v[0].lower() == "all":
            # Lazy import to avoid circular: models.__init__ → config → metrics.registry → metrics.base → models
            from eva.metrics.registry import get_global_registry

            return [m for m in get_global_registry().list_metrics() if m not in cls._VALIDATION_METRIC_NAMES]
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
    def _redact_model_params(cls, model: ModelConfigUnion) -> dict:
        """Redact secret values in STT/TTS/S2S/AudioLLM params when serializing."""
        data = model.model_dump(mode="json")
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
                    f"EVA_MODEL_LIST — skipping (not used in this run)."
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
