"""STT (Speech-to-Text) Word Error Rate metric.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

import re

import jiwer

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import aggregate_wer_errors, extract_wer_errors, make_rate_sub_metric, reverse_word_error_rate
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.wer_normalization import normalize_text

_BRACKET_PATTERN = re.compile(r"\[.*?\]")


def _build_wer_component_sub_metrics(
    parent_name: str,
    substitutions: int,
    deletions: int,
    insertions: int,
    reference_words: int,
) -> dict[str, MetricScore]:
    """Build sub-metrics for substitution, deletion, and insertion rates.

    Each rate = component count / reference word count. Returns an empty dict
    when there are no reference words (division by zero).
    """
    if reference_words <= 0:
        return {}

    components = {
        "substitution_rate": substitutions,
        "deletion_rate": deletions,
        "insertion_rate": insertions,
    }
    return {
        key: make_rate_sub_metric(
            parent_name=parent_name,
            key=key,
            numerator=count,
            denominator=reference_words,
            details={"count": count, "reference_words": reference_words},
        )
        for key, count in components.items()
    }


@register_metric
class STTWERMetric(CodeMetric):
    """Speech-to-Text Word Error Rate metric.

    Measures the accuracy of STT transcription by comparing what the user
    simulator said (tts_text_user) to what was transcribed (transcript_user).

    Lower WER is better. Converted to accuracy using reverse_word_error_rate.

    Text normalization pipeline includes:
    - Unicode character conversion
    - Digits to words conversion
    - Whisper-based text normalization
    - Apostrophe normalization
    - Single letter collapsing
    - Number suffix handling

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "stt_wer"
    description = "Debug metric: Speech-to-Text transcription accuracy using Word Error Rate"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    def __init__(self, config: dict | None = None):
        """Initialize the metric with language configuration."""
        super().__init__(config)
        # Get language from config (default: "en")
        self.language = self.config.get("language", "en")

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute STT WER for user turns."""
        try:
            # Collect reference/hypothesis pairs for turns present in both dicts
            common_turn_ids = sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())

            evaluated_turn_ids = []
            references = []
            hypotheses = []

            for turn_id in common_turn_ids:
                ref = _BRACKET_PATTERN.sub("", context.intended_user_turns[turn_id]).strip()
                hyp = _BRACKET_PATTERN.sub("", context.transcribed_user_turns[turn_id]).strip()
                if ref and hyp:
                    evaluated_turn_ids.append(turn_id)
                    references.append(ref)
                    hypotheses.append(hyp)

            references_clean = [normalize_text(r, self.language) for r in references]
            hypotheses_clean = [normalize_text(h, self.language) for h in hypotheses]

            if not references:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            # Compute WER using jiwer with normalized text
            wer = jiwer.wer(references_clean, hypotheses_clean)

            # Get detailed word-level alignment
            output = jiwer.process_words(references_clean, hypotheses_clean)

            # Convert WER to accuracy score (1 - wer, clamped to 0-1)
            accuracy = reverse_word_error_rate(wer)

            per_turn_wer: dict[int, float] = {}
            per_turn_errors: dict[int, dict] = {}

            for turn_id, ref_clean, hyp_clean in zip(evaluated_turn_ids, references_clean, hypotheses_clean):
                turn_wer = jiwer.wer(ref_clean, hyp_clean)
                per_turn_wer[turn_id] = round(turn_wer, 3)

                # Get alignment for this turn
                turn_output = jiwer.process_words(ref_clean, hyp_clean)
                turn_errors = extract_wer_errors(turn_output)
                per_turn_errors[turn_id] = turn_errors

            # Aggregate error statistics across all turns
            error_summary = aggregate_wer_errors(output)

            reference_word_count = sum(len(r.split()) for r in references_clean)
            sub_metrics = _build_wer_component_sub_metrics(
                parent_name=self.name,
                substitutions=output.substitutions,
                deletions=output.deletions,
                insertions=output.insertions,
                reference_words=reference_word_count,
            )

            return MetricScore(
                name=self.name,
                score=round(wer, 3),  # Raw WER
                normalized_score=round(accuracy, 3),  # Accuracy (1-WER)
                details={
                    "wer": round(wer, 3),
                    "accuracy": round(accuracy, 3),
                    "language": self.language,  # Include language
                    "num_turns": len(references),
                    "per_turn_wer": per_turn_wer,
                    "per_turn_errors": per_turn_errors,
                    "error_summary": error_summary,
                    # Overall error counts
                    "total_substitutions": output.substitutions,
                    "total_deletions": output.deletions,
                    "total_insertions": output.insertions,
                    "reference_words": reference_word_count,
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
