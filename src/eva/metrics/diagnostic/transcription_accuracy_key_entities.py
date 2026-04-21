"""Transcription accuracy key entities metric using LLM-as-judge (entire conversation).

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from typing import Any

from eva.metrics.base import MetricContext, TextJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import aggregate_per_turn_scores, parse_judge_response_list, resolve_turn_id
from eva.models.config import PipelineType
from eva.models.results import MetricScore


@register_metric
class TranscriptionAccuracyKeyEntitiesMetric(TextJudgeMetric):
    """LLM-based transcription accuracy metric for key entities only (entire conversation).

    Evaluates STT transcription accuracy by comparing key entities (names, dates,
    confirmation codes, amounts, etc.) between what the user was supposed to say
    (intended_user_turns) and what STT transcribed (transcribed_user_turns).

    Computes the ratio of correctly transcribed entities per turn:
    score = correct_entities / total_entities

    Entity types evaluated:
    - Names (people, places, organizations)
    - Dates and times
    - Confirmation codes / reference numbers
    - Flight numbers
    - Amounts and prices
    - Addresses
    - Phone numbers
    - Email addresses

    Rating scale: 0.0-1.0 (ratio of correct entities)
    Normalized: Same as raw score (already 0-1)
    Edge case: No entities → turn excluded from aggregation

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "transcription_accuracy_key_entities"
    description = "Debug metric: LLM judge evaluation of STT key entity transcription accuracy for entire conversation"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})
    rating_scale = None  # Custom scoring (not 1-3 scale)
    default_aggregation = "mean"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.aggregation = self.config.get("aggregation", self.default_aggregation)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the metric by evaluating all user turns at once.

        Args:
            context: MetricContext containing conversation data

        Returns:
            MetricScore with aggregated score and per-turn details
        """
        try:
            turns_to_evaluate = self._get_turns_to_evaluate(context)
            user_turns_text = self._format_user_turns(turns_to_evaluate, context)
            prompt = self.get_judge_prompt(user_turns=user_turns_text)

            response_text = await self._call_judge_raw(prompt, context)
            turn_evaluations = parse_judge_response_list(response_text)

            # Handle LLM wrapping array in a dict under a known key
            if (
                isinstance(turn_evaluations, list)
                and len(turn_evaluations) == 1
                and isinstance(turn_evaluations[0], dict)
            ):
                item = turn_evaluations[0]
                for key in ("turns", "evaluations", "results", "turn_evaluations"):
                    if key in item and isinstance(item[key], list):
                        turn_evaluations = item[key]
                        break

            if not turn_evaluations:
                error = "No response from judge" if not response_text else "Failed to parse judge response"
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=error,
                )

            # Compute scores for each turn, keyed by turn_id
            per_turn_ratings: dict[int, float | None] = {}
            per_turn_normalized: dict[int, float | None] = {}
            per_turn_explanations: dict[int, str] = {}
            per_turn_entity_details: dict[int, dict] = {}

            for turn_eval in turn_evaluations:
                turn_id = resolve_turn_id(turn_eval, turns_to_evaluate, self.name)
                if turn_id is None:
                    continue
                score, normalized = self._compute_turn_score(turn_eval)
                per_turn_ratings[turn_id] = score
                per_turn_normalized[turn_id] = normalized
                per_turn_explanations[turn_id] = turn_eval.get("summary", "")
                per_turn_entity_details[turn_id] = turn_eval

            # Filter out -1 (not applicable) turns before aggregation
            applicable_normalized = [v for v in per_turn_normalized.values() if v is not None and v != -1.0]
            aggregated_score = (
                aggregate_per_turn_scores(applicable_normalized, self.aggregation) if applicable_normalized else None
            )

            # Compute average raw score
            valid_ratings = [r for r in per_turn_ratings.values() if r is not None and r != -1.0]
            not_applicable = [r for r in per_turn_ratings.values() if r == -1.0]
            avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0.0

            # All turns had no entities to evaluate — not an error, just nothing to score
            skipped = not applicable_normalized

            # num_evaluated includes both scored turns and not-applicable turns
            # (judge responded for all of them — -1 means no entities, not a failure)
            num_evaluated = len(valid_ratings) + len(not_applicable)

            return MetricScore(
                name=self.name,
                score=round(avg_rating, 3),
                normalized_score=round(aggregated_score, 3) if aggregated_score is not None else None,
                details={
                    "judge_prompt": prompt,
                    "aggregation": self.aggregation,
                    "num_turns": len(turns_to_evaluate),
                    "num_evaluated": num_evaluated,
                    "num_not_applicable": len(not_applicable),
                    "skipped": skipped,
                    "skipped_reason": "No key entities found in any evaluated turn" if skipped else None,
                    "per_turn_ratings": per_turn_ratings,
                    "per_turn_normalized": per_turn_normalized,
                    "per_turn_explanations": per_turn_explanations,
                    "per_turn_entity_details": per_turn_entity_details,
                    "judge_raw_response": response_text,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)

    @staticmethod
    def _get_turns_to_evaluate(context: MetricContext) -> list[int]:
        """Return sorted turn IDs present in both tts_text_user and transcript_user."""
        return sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())

    @staticmethod
    def _format_user_turns(turn_ids: list[int], context: MetricContext) -> str:
        """Format user turns for the prompt."""
        return "\n\n".join(
            f"Turn {tid}:\n"
            f'Expected: "{context.intended_user_turns[tid]}"\n'
            f'Transcribed: "{context.transcribed_user_turns[tid]}"'
            for tid in turn_ids
        )

    async def _call_judge_raw(self, prompt: str, context: MetricContext) -> str | None:
        """Call LLM judge and return raw response text.

        Args:
            prompt: The prompt to send to the judge
            context: MetricContext for logging

        Returns:
            Raw response text or None if failed
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response_text = await self.llm_client.generate_text(
                messages,
            )
            return response_text
        except Exception as e:
            self.logger.error(f"Judge call failed for {context.record_id}: {e}")
            return None

    @staticmethod
    def _compute_turn_score(turn_eval: dict[str, Any]) -> tuple[float | None, float | None]:
        """Compute score from entity correctness ratio for a single turn.

        Args:
            turn_eval: Turn evaluation dict containing entities list

        Returns:
            Tuple of (score, normalized_score) - both are the same as score is already 0-1.
            Returns (None, None) when no entities are found so the turn is excluded
            from aggregation.

        Scoring Logic:
            - Extract entities list from turn evaluation
            - Count total entities and correct entities
            - Compute ratio: correct / total
            - Edge case: No entities → (None, None), turn excluded from aggregation
        """
        entities = turn_eval.get("entities", [])

        # No entities to evaluate — mark as not applicable (-1), excluded from aggregation
        if not entities:
            return -1.0, -1.0

        # Filter out skipped entities — they remain in details but don't affect score
        non_skipped = [e for e in entities if not e.get("skipped", False)]

        # All entities skipped — mark as not applicable (-1), excluded from aggregation
        if not non_skipped:
            return -1.0, -1.0

        # Count correct entities
        total = len(non_skipped)
        correct = sum(1 for e in non_skipped if e.get("correct", False))

        # Compute ratio (already normalized to 0-1)
        score = correct / total

        return score, score
