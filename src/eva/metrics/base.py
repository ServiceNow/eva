"""Base metric class for defining evaluation metrics."""

import csv
import json
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from eva.metrics.utils import (
    aggregate_per_turn_scores,
    audio_to_base64,
    format_transcript_with_tools,
    load_audio_file,
    normalize_rating,
    parse_judge_response,
    parse_judge_response_list,
    resolve_turn_id,
    validate_rating,
)
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.llm_client import LLMClient
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import get_prompt_manager


class MetricType(StrEnum):
    """Metric computation types."""

    CODE = "code"  # Rule-based, no LLM
    TEXT_JUDGE = "text_judge"  # LLM judge with text input only
    AUDIO_JUDGE = "audio_judge"  # LLM judge with audio input


class MetricContext:
    """Context provided to metrics for computation.

    Contains all data needed to evaluate a single conversation record,
    including ground truth, processed log data, and raw files.
    """

    def __init__(
        self,
        record_id: str,
        # Ground truth from dataset
        user_goal: str,
        user_persona: str,
        # Scenario database fields (REQUIRED for deterministic metrics)
        expected_scenario_db: dict[str, Any],
        initial_scenario_db: dict[str, Any],
        final_scenario_db: dict[str, Any],
        initial_scenario_db_hash: str,
        final_scenario_db_hash: str,
        # Agent configuration (required)
        agent_role: str,
        agent_instructions: str,
        agent_tools: list[dict],
        current_date_time: str,
        # Basic stats
        num_turns: int = 0,
        num_tool_calls: int = 0,
        tools_called: list[str] = None,
        conversation_ended_reason: str | None = None,
        duration_seconds: float = 0.0,
        # Paths to files
        output_dir: str = "",
        audio_assistant_path: str | None = None,
        audio_user_path: str | None = None,
        audio_mixed_path: str | None = None,
        # Processed log data from postprocessor
        transcribed_assistant_turns: dict[int, str] | None = None,
        transcribed_user_turns: dict[int, str] | None = None,
        intended_assistant_turns: dict[int, str] | None = None,
        intended_user_turns: dict[int, str] | None = None,
        audio_timestamps_assistant_turns: dict[int, list[tuple[float, float]]] | None = None,
        audio_timestamps_user_turns: dict[int, list[tuple[float, float]]] | None = None,
        num_assistant_turns: int | None = None,
        num_user_turns: int | None = None,
        tool_params: list[dict] | None = None,
        tool_responses: list[dict] | None = None,
        conversation_trace: list[dict] | None = None,
        latency_assistant_turns: dict[int, float] | None = None,
        assistant_interrupted_turns: set[int] | None = None,
        user_interrupted_turns: set[int] | None = None,
        pipeline_type: PipelineType = PipelineType.CASCADE,
    ):
        self.record_id = record_id

        # Ground truth
        self.user_goal = user_goal
        self.user_persona = user_persona

        # Scenario database state (REQUIRED for deterministic metrics)
        self.expected_scenario_db = expected_scenario_db
        self.initial_scenario_db = initial_scenario_db
        self.final_scenario_db = final_scenario_db
        self.initial_scenario_db_hash = initial_scenario_db_hash
        self.final_scenario_db_hash = final_scenario_db_hash

        # Basic stats
        self.num_turns = num_turns
        self.num_tool_calls = num_tool_calls
        self.tools_called = tools_called or []
        self.conversation_ended_reason = conversation_ended_reason
        self.duration_seconds = duration_seconds

        # Paths
        self.output_dir = output_dir
        self.audio_assistant_path = audio_assistant_path
        self.audio_user_path = audio_user_path
        self.audio_mixed_path = audio_mixed_path

        # Agent configuration (required - will fail if not provided)
        self.agent_role = agent_role
        self.agent_instructions = agent_instructions
        self.agent_tools = agent_tools
        self.current_date_time = current_date_time

        # Processed log data
        self.transcribed_assistant_turns = transcribed_assistant_turns or {}
        self.transcribed_user_turns = transcribed_user_turns or {}
        self.intended_assistant_turns = intended_assistant_turns or {}
        self.intended_user_turns = intended_user_turns or {}
        self.audio_timestamps_assistant_turns = audio_timestamps_assistant_turns or {}
        self.audio_timestamps_user_turns = audio_timestamps_user_turns or {}
        self.num_assistant_turns = num_assistant_turns or 0
        self.num_user_turns = num_user_turns or 0
        self.tool_params = tool_params or []
        self.tool_responses = tool_responses or []
        self.conversation_trace = conversation_trace or []
        self.latency_assistant_turns = latency_assistant_turns or {}
        self.assistant_interrupted_turns = assistant_interrupted_turns or set()
        self.user_interrupted_turns = user_interrupted_turns or set()
        self.pipeline_type = pipeline_type

    @property
    def is_audio_native(self) -> bool:
        return self.pipeline_type in (PipelineType.S2S, PipelineType.AUDIO_LLM)

    def to_dict(self) -> dict[str, Any]:
        """Convert MetricContext to a serializable dictionary."""
        return {key: str(value) if isinstance(value, Path) else value for key, value in self.__dict__.items()}


class BaseMetric(ABC):
    """Base class for all metrics.

    Subclass this to create custom metrics for evaluating conversations.
    """

    # Override these in subclasses
    name: str = "base_metric"
    description: str = "Base metric class"
    category: str = "general"
    metric_type: MetricType = MetricType.CODE  # Override in subclasses
    pass_at_k_threshold: float = 0.5  # Normalized score threshold for pass@k pass/fail
    exclude_from_pass_at_k: bool = False  # Set True for metrics not suitable for pass@k
    supported_pipeline_types: frozenset[PipelineType] = frozenset(PipelineType)  # Pipeline types this metric supports
    # Direction of the displayed value (normalized_score if present, else score).
    # Override to False for lower-is-better parent metrics (e.g. latency). Sub-metric
    # direction is derived from the key suffix (see eva.metrics.utils.direction_for_sub_metric).
    higher_is_better: bool = True

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the metric.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(f"metrics.{self.name}")
        self.prompt_manager = get_prompt_manager()

    def get_judge_prompt(self, prompt_key: str = "user_prompt", **variables) -> str:
        """Get judge prompt using PromptManager."""
        prompt_path = f"judge.{self.name}.{prompt_key}"
        return self.prompt_manager.get_prompt(prompt_path, **variables)

    @abstractmethod
    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the metric for a single conversation.

        Args:
            context: MetricContext containing all data for the conversation

        Returns:
            MetricScore with the computed score and details
        """
        pass

    def _log_token_usage(
        self, context: MetricContext, model_name: str, model_params: dict, prompt: str, usage: dict | None
    ) -> None:
        """Append one row of LLM judge token usage to a per-metric CSV and update the run-level JSON summary."""
        if not context.output_dir:
            return
        # Walk up from output_dir to find the run root (directory containing config.json).
        path = Path(context.output_dir)
        run_dir = path
        while path != path.parent:
            if (path / "config.json").exists():
                run_dir = path
                break
            path = path.parent

        # Derive full record_id (including trial suffix) from path relative to records/.
        try:
            record_id = str(Path(context.output_dir).relative_to(run_dir / "records"))
        except ValueError:
            record_id = context.record_id

        csv_dir = run_dir / "judge_token_usage"
        csv_dir.mkdir(exist_ok=True)

        input_tokens = usage.get("prompt_tokens") if usage else None
        output_tokens = usage.get("completion_tokens") if usage else None
        model_id = usage.get("model_name") if usage else None

        csv_path = csv_dir / f"{self.name}.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "record_id",
                        "model_name",
                        "model_id",
                        "input_tokens",
                        "output_tokens",
                        "timestamp",
                        "model_params",
                        "input_prompt",
                    ]
                )
            writer.writerow(
                [
                    record_id,
                    model_name,
                    model_id,
                    input_tokens,
                    output_tokens,
                    datetime.now(UTC).isoformat(),
                    json.dumps(model_params) if model_params else None,
                    prompt,
                ]
            )

        # Update per-run JSON summary with running totals.
        json_path = csv_dir / "judge_token_usage.json"
        summary = json.loads(json_path.read_text()) if json_path.exists() else {}
        entry = summary.setdefault(
            self.name, {"model_id": None, "total_input_tokens": 0, "total_output_tokens": 0, "num_calls": 0}
        )
        if model_id is not None:
            entry["model_id"] = model_id
        entry["total_input_tokens"] += input_tokens or 0
        entry["total_output_tokens"] += output_tokens or 0
        entry["num_calls"] += 1
        json_path.write_text(json.dumps(summary, indent=2))

    def _handle_error(self, error: Exception, context: MetricContext) -> MetricScore:
        """Standard error handling for all metrics."""
        self.logger.exception(f"{self.name} failed for {context.record_id}: {error}")
        return MetricScore(
            name=self.name,
            score=0.0,
            normalized_score=0.0,
            error=str(error),
        )


class CodeMetric(BaseMetric):
    """Base class for rule-based metrics (no LLM)."""

    metric_type = MetricType.CODE

    @abstractmethod
    async def compute(self, context: MetricContext) -> MetricScore:
        """Implement rule-based computation logic."""
        pass


class TextJudgeMetric(BaseMetric):
    """Base class for LLM-based text judge metrics."""

    metric_type = MetricType.TEXT_JUDGE

    # Subclasses can override these
    default_model = "gpt-5.2"
    default_params: dict[str, Any] = {"max_tokens": 100000}
    rating_scale: tuple[int, int] = (1, 3)  # (min, max)

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Initialize LLM client with common defaults
        # Priority: config > env var > class default
        model = self.config.get("judge_model") or os.environ.get("JUDGE_MODEL") or self.default_model

        # Merge: class defaults < config overrides
        params = {**self.default_params}
        params.update(self.config.get("judge_params", {}))

        self.llm_client = LLMClient(model=model, params=params)

    async def call_judge(self, prompt: str, context: MetricContext) -> tuple[dict | None, str | None]:
        """Call LLM judge and parse response. Returns (parsed_dict, raw_response_text)."""
        messages = [{"role": "user", "content": prompt}]
        response_text, usage = await self.llm_client.generate_text(messages)
        self._log_token_usage(context, self.llm_client.model, self.llm_client.params, prompt, usage)
        return parse_judge_response(response_text, context.record_id, self.logger), response_text

    def validate_and_normalize_rating(self, response: dict, context: MetricContext) -> tuple[int, float]:
        """Validate rating and compute normalized score."""
        rating = validate_rating(
            response.get("rating"),
            list(range(self.rating_scale[0], self.rating_scale[1] + 1)),
            default=self.rating_scale[0],
            record_id=context.record_id,
            metric_logger=self.logger,
        )

        normalized = normalize_rating(rating, self.rating_scale[0], self.rating_scale[1])

        return rating, normalized


class ConversationTextJudgeMetric(TextJudgeMetric):
    """Base class for text judges that evaluate entire conversations."""

    async def compute(self, context: MetricContext) -> MetricScore:
        """Standard flow for conversation-level text judges."""
        try:
            # Format transcript (can be overridden)
            transcript_text = self.format_transcript(context)

            if not transcript_text:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No transcript available",
                )

            # Get prompt variables (subclass provides this)
            prompt_vars = self.get_prompt_variables(context, transcript_text)

            # Get and format prompt
            prompt = self.get_judge_prompt(**prompt_vars)

            # Call judge
            response, raw_response = await self.call_judge(prompt, context)
            if response is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Failed to parse judge response",
                )

            # Validate and normalize
            try:
                rating, normalized = self.validate_and_normalize_rating(response, context)

                # Build result
                return self.build_metric_score(rating, normalized, response, prompt, context, raw_response)
            except (KeyError, TypeError, ValueError) as e:
                self.logger.error(f"Failed to process judge response for {context.record_id}: {e}")
                self.logger.error(f"Response: {response}")
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"Failed to process judge response: {e}",
                )

        except Exception as e:
            return self._handle_error(e, context)

    @abstractmethod
    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        pass

    def format_transcript(self, context: MetricContext) -> str:
        """Format transcript. Can be overridden."""
        return format_transcript_with_tools(context.conversation_trace)

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore. Can be overridden for custom details."""
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": response.get("explanation", ""),
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
        )


class PerTurnConversationJudgeMetric(TextJudgeMetric):
    """Base class for text judges that evaluate all turns in a single call, returning per-turn ratings.

    Subclasses must implement:
        - get_expected_turn_ids(): which turn IDs the judge should rate
        - get_prompt_variables(): template variables for the judge prompt

    Subclasses may override:
        - format_transcript(): how to format conversation content (default: conversation_trace with tools)
        - process_turn_item(): extract extra per-turn fields (e.g., failure_modes)
    """

    default_aggregation: str = "mean"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.aggregation = self.config.get("aggregation", self.default_aggregation)

    @abstractmethod
    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return the ordered list of turn IDs that the judge should rate."""

    def format_transcript(self, context: MetricContext) -> str:
        """Format conversation content for the judge prompt. Can be overridden."""
        return format_transcript_with_tools(context.conversation_trace)

    @abstractmethod
    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return template variables for the judge prompt."""

    def process_turn_item(self, item: dict, turn_id: int, rating: int | None, context: MetricContext) -> dict[str, Any]:
        """Extract additional per-turn fields from a response item.

        Override to extract metric-specific fields (e.g., failure_modes for conciseness).
        Returns a dict of {field_name: value}; these are collected across turns and stored
        as per_turn_{field_name} in the result details.

        Called for every turn item, including those with null or invalid ratings.
        """
        return {}

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_extra: dict[int, dict[str, Any]],
    ) -> dict[str, MetricScore] | None:
        """Return sub-metrics derived from the per-turn data, or None.

        Override in subclasses to surface breakdowns (e.g., per-failure-mode rates).
        Default returns None so the parent metric has no sub-metrics.
        """
        return None

    async def compute(self, context: MetricContext) -> MetricScore:
        """Evaluate all turns in a single judge call and aggregate per-turn ratings."""
        try:
            transcript_text = self.format_transcript(context)
            if not transcript_text:
                return MetricScore(name=self.name, score=0.0, normalized_score=0.0, error="No turns to evaluate")

            prompt = self.get_judge_prompt(**self.get_prompt_variables(context, transcript_text))
            response_text, usage = await self.llm_client.generate_text([{"role": "user", "content": prompt}])
            self._log_token_usage(context, self.llm_client.model, self.llm_client.params, prompt, usage)
            parsed = parse_judge_response_list(response_text)

            if parsed is None:
                return MetricScore(
                    name=self.name, score=0.0, normalized_score=0.0, error="Failed to parse judge response"
                )

            turn_ids = self.get_expected_turn_ids(context)
            min_r, max_r = self.rating_scale
            valid_range = set(range(min_r, max_r + 1))

            per_turn_ratings: dict[int, int | None] = {}
            per_turn_explanations: dict[int, str] = {}
            per_turn_extra: dict[int, dict[str, Any]] = {}

            for item in parsed:
                turn_id = resolve_turn_id(item, turn_ids, self.name)
                if turn_id is None:
                    continue
                rating = item.get("rating")

                # Null rating means "not applicable" — excluded from aggregation
                if rating is None:
                    per_turn_ratings[turn_id] = None
                    per_turn_explanations[turn_id] = item.get("explanation", "")
                    per_turn_extra[turn_id] = self.process_turn_item(item, turn_id, None, context)
                    continue

                if isinstance(rating, str):
                    try:
                        rating = int(rating)
                    except ValueError:
                        pass

                if rating not in valid_range:
                    self.logger.warning(f"[{context.record_id}] Invalid rating {rating} for turn {turn_id}")
                    per_turn_ratings[turn_id] = None
                    per_turn_explanations[turn_id] = f"Invalid rating: {rating}"
                    per_turn_extra[turn_id] = self.process_turn_item(item, turn_id, None, context)
                    continue

                per_turn_ratings[turn_id] = rating
                per_turn_explanations[turn_id] = item.get("explanation", "")
                per_turn_extra[turn_id] = self.process_turn_item(item, turn_id, rating, context)

            valid_ratings = [r for r in per_turn_ratings.values() if r is not None]

            details: dict[str, Any] = {
                "per_turn_ratings": per_turn_ratings,
                "per_turn_explanations": per_turn_explanations,
                "judge_prompt": prompt,
                "judge_raw_response": response_text,
            }

            # Flatten per_turn_extra into per_turn_{field_name} in details
            extra_keys: set[str] = set()
            for extra in per_turn_extra.values():
                extra_keys.update(extra.keys())
            for key in sorted(extra_keys):
                details[f"per_turn_{key}"] = {tid: extra.get(key) for tid, extra in per_turn_extra.items()}

            if not valid_ratings:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="All turns failed to evaluate",
                    details=details,
                )

            mean_rating = aggregate_per_turn_scores(valid_ratings, self.aggregation)
            normalized_ratings = [normalize_rating(r, min_r, max_r) for r in valid_ratings]
            normalized_score = aggregate_per_turn_scores(normalized_ratings, self.aggregation)

            details.update(
                {
                    "aggregation": self.aggregation,
                    "num_turns": len(turn_ids),
                    "num_evaluated": len(valid_ratings),
                }
            )

            # Add per_turn_normalized only when the scale isn't already 0-1
            if min_r != 0 or max_r != 1:
                details["per_turn_normalized"] = {
                    tid: normalize_rating(r, min_r, max_r) for tid, r in per_turn_ratings.items() if r is not None
                }

            sub_metrics = self.build_sub_metrics(context, per_turn_ratings, per_turn_extra)

            return MetricScore(
                name=self.name,
                score=round(mean_rating, 3),
                normalized_score=round(normalized_score, 3),
                details=details,
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)


class AudioJudgeMetric(BaseMetric):
    """Base class for LLM-based audio judge metrics."""

    metric_type = MetricType.AUDIO_JUDGE

    # Subclasses can override
    default_model = "gemini-3.1-pro-preview"
    default_params: dict[str, Any] = {"temperature": 0.0, "max_tokens": 30000}
    rating_scale: tuple[int, int] = (-2, 2)  # Can vary by metric

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Initialize Gemini client
        model = self.config.get("audio_judge_model", self.default_model)

        # Merge: class defaults < config overrides
        params = {**self.default_params}
        params.update(self.config.get("judge_params", {}))

        self.llm_client = LLMClient(model=model, params=params)

    def load_audio(self, context: MetricContext) -> AudioSegment | None:
        """Load mixed audio file."""
        if not context.audio_mixed_path:
            return None

        return load_audio_file(Path(context.audio_mixed_path))

    def load_role_audio(self, context: MetricContext, role: str) -> AudioSegment | None:
        """Load role-specific audio file (assistant or user)."""
        if role == "assistant":
            audio_path = context.audio_assistant_path
        elif role == "user":
            audio_path = context.audio_user_path
        else:
            self.logger.error(f"[{context.record_id}] Invalid role for audio loading: {role}")
            return None
        return load_audio_file(Path(audio_path)) if audio_path else None

    def encode_audio_segment(self, segment: AudioSegment) -> str:
        """Encode audio segment to base64."""
        return audio_to_base64(segment)

    def create_audio_message(self, audio_b64: str, prompt: str) -> list[dict]:
        """Create properly formatted audio message for Vertex AI/Gemini via LiteLLM.

        Args:
            audio_b64: Base64-encoded audio data
            prompt: Text prompt to send with audio

        Returns:
            List of messages formatted for LiteLLM + Vertex AI
        """
        # Try using 'file' type since LiteLLM docs say it works for audio too
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_data": f"data:audio/wav;base64,{audio_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def create_audio_message_from_file_id(self, file_id: str, prompt: str) -> list[dict]:
        """Create audio message referencing an uploaded file by ID.

        Used as a fallback when inline base64 audio is too large and gets
        dropped by Gemini.

        Args:
            file_id: File ID returned by litellm.create_file()
            prompt: Text prompt to send with audio

        Returns:
            List of messages formatted for LiteLLM + Gemini
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_id": file_id, "filename": "audio.wav", "format": "audio/wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
