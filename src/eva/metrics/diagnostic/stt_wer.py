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
from eva.utils.wer_normalization.code_switch import ReconcileResult, reconcile_code_switch
from eva.utils.wer_normalization.wer_utils import get_language_config

_BRACKET_PATTERN = re.compile(r"\[.*?\]")

# Languages where text has no word-level whitespace segmentation — CER is the
# appropriate error-rate measure instead of WER.
_CER_LANGUAGES = frozenset({"ja", "zh", "ko"})

# CER languages without word spacing: whitespace is not part of the text, so it
# is removed after normalization and never counts as a character error. Korean
# is excluded because it separates words with spaces.
_NO_WHITESPACE_LANGUAGES = frozenset({"ja"})
_WHITESPACE_PATTERN = re.compile(r"\s+")


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


def _per_language_errors(results: list[ReconcileResult]) -> tuple[dict[str, int], dict[str, int]]:
    """Errors and reference words per language tag, over the reconciled alignments.

    Substitutions and deletions count against the reference token's tag,
    insertions against the inserted token's tag.
    """
    errors: dict[str, int] = {}
    ref_words: dict[str, int] = {}
    for r in results:
        for tag in r.ref_langs:
            ref_words[tag] = ref_words.get(tag, 0) + 1
        output = jiwer.process_words(" ".join(r.ref_tokens), " ".join(r.hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type in ("substitute", "delete"):
                tags = r.ref_langs[chunk.ref_start_idx : chunk.ref_end_idx]
            elif chunk.type == "insert":
                tags = r.hyp_langs[chunk.hyp_start_idx : chunk.hyp_end_idx]
            else:
                continue
            for tag in tags:
                errors[tag] = errors.get(tag, 0) + 1
    return errors, ref_words


def _build_code_switch_details(
    parent_name: str, primary: str, secondaries: tuple[str, ...], results: list[ReconcileResult]
) -> tuple[dict, dict[str, MetricScore]]:
    """Code-switch summary for ``details`` and its sub-metrics.

    A per-language WER is None (and its sub-metric omitted) when the run has no
    reference words in that language.
    """
    errors, ref_words = _per_language_errors(results)
    total_ref_words = sum(ref_words.values())
    rescues = [rescue for r in results for rescue in r.rescues]
    rescued_words = sum(len(rescue.expected.split()) for rescue in rescues)
    rescues_by_matcher: dict[str, int] = {}
    for rescue in rescues:
        rescues_by_matcher[rescue.matcher] = rescues_by_matcher.get(rescue.matcher, 0) + 1
    switched_words = sum(ref_words.get(lang, 0) for lang in secondaries)

    per_language = {}
    sub_metrics: dict[str, MetricScore] = {}
    for lang in (primary, *secondaries):
        n_ref, n_err = ref_words.get(lang, 0), errors.get(lang, 0)
        per_language[lang] = {"wer": round(n_err / n_ref, 3) if n_ref else None, "ref_words": n_ref, "errors": n_err}
        if n_ref:
            sub_metrics[f"wer_{lang}"] = make_rate_sub_metric(
                parent_name=parent_name,
                key=f"wer_{lang}",
                numerator=n_err,
                denominator=n_ref,
                details={"errors": n_err, "reference_words": n_ref},
            )
    if total_ref_words:
        sub_metrics["code_switch_word_rate"] = make_rate_sub_metric(
            parent_name=parent_name,
            key="code_switch_word_rate",
            numerator=switched_words,
            denominator=total_ref_words,
            details={"code_switched_words": switched_words, "reference_words": total_ref_words},
        )
        sub_metrics["code_switch_rescue_rate"] = make_rate_sub_metric(
            parent_name=parent_name,
            key="code_switch_rescue_rate",
            numerator=rescued_words,
            denominator=total_ref_words,
            details={"rescued_words": rescued_words, "reference_words": total_ref_words},
        )

    details = {
        "languages": [primary, *secondaries],
        "code_switched_words": switched_words,
        "rescued_words": rescued_words,
        "rescues_by_matcher": rescues_by_matcher,
        "per_language": per_language,
    }
    return details, sub_metrics


@register_metric
class STTWERMetric(CodeMetric):
    """Speech-to-Text Word Error Rate metric.

    Measures the accuracy of STT transcription by comparing what the user
    simulator said (tts_text_user) to what was transcribed (transcript_user).

    Lower WER is better. Converted to accuracy using reverse_word_error_rate.

    For languages with ``code_switch_languages`` in their normalization config
    (Hindi: English), error regions are re-checked with the secondary
    language's rules and correctly transcribed code-switched words are not
    counted (see ``code_switch.py``). The main score is then that reconciled
    WER; the Hindi-only WER is kept as ``details.strict_wer``.

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
    version = "v0.2"
    description = "Debug metric: Speech-to-Text transcription accuracy using Word Error Rate"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    def __init__(self, config: dict | None = None):
        """Initialize the metric with language configuration."""
        super().__init__(config)
        # Get language from config (default: "en")
        self.language = self.config.get("language", "en")
        language_config = get_language_config(self.language)
        self.code_switch_languages = tuple(language_config.code_switch_languages) if language_config else ()
        self.code_switch_phonetic = bool(language_config and language_config.code_switch_phonetic)

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

            code_switch_results: list[ReconcileResult] = []
            if self.code_switch_languages:
                code_switch_results = [
                    reconcile_code_switch(r, h, self.language, self.code_switch_languages, self.code_switch_phonetic)
                    for r, h in zip(references, hypotheses)
                ]
                references_clean = [" ".join(r.ref_tokens) for r in code_switch_results]
                hypotheses_strict = [" ".join(r.strict_output.hypotheses[0]) for r in code_switch_results]
                hypotheses_clean = [" ".join(r.hyp_tokens) for r in code_switch_results]
            else:
                references_clean = [normalize_text(r, self.language) for r in references]
                hypotheses_clean = [normalize_text(h, self.language) for h in hypotheses]
            if self.language in _NO_WHITESPACE_LANGUAGES:
                references_clean = [_WHITESPACE_PATTERN.sub("", r) for r in references_clean]
                hypotheses_clean = [_WHITESPACE_PATTERN.sub("", h) for h in hypotheses_clean]

            if not references:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            use_cer = self.language in _CER_LANGUAGES

            if use_cer:
                error_rate = jiwer.cer(references_clean, hypotheses_clean)
                output = jiwer.process_characters(references_clean, hypotheses_clean)
            else:
                error_rate = jiwer.wer(references_clean, hypotheses_clean)
                output = jiwer.process_words(references_clean, hypotheses_clean)

            accuracy = reverse_word_error_rate(error_rate)
            rate_key = "cer" if use_cer else "wer"

            per_turn_wer: dict[int, float] = {}
            per_turn_errors: dict[int, dict] = {}

            for turn_id, ref_clean, hyp_clean in zip(evaluated_turn_ids, references_clean, hypotheses_clean):
                if use_cer:
                    turn_rate = jiwer.cer(ref_clean, hyp_clean)
                    turn_output = jiwer.process_characters(ref_clean, hyp_clean)
                else:
                    turn_rate = jiwer.wer(ref_clean, hyp_clean)
                    turn_output = jiwer.process_words(ref_clean, hyp_clean)
                per_turn_wer[turn_id] = round(turn_rate, 3)
                turn_errors = extract_wer_errors(turn_output)
                per_turn_errors[turn_id] = turn_errors

            error_summary = aggregate_wer_errors(output)

            reference_unit_count = (
                len("".join(references_clean)) if use_cer else sum(len(r.split()) for r in references_clean)
            )
            sub_metrics = _build_wer_component_sub_metrics(
                parent_name=self.name,
                substitutions=output.substitutions,
                deletions=output.deletions,
                insertions=output.insertions,
                reference_words=reference_unit_count,
            )

            details = {
                rate_key: round(error_rate, 3),
                "accuracy": round(accuracy, 3),
                "language": self.language,
                "use_cer": use_cer,
                "num_turns": len(references),
                "per_turn_wer": per_turn_wer,
                "per_turn_errors": per_turn_errors,
                "error_summary": error_summary,
                "total_substitutions": output.substitutions,
                "total_deletions": output.deletions,
                "total_insertions": output.insertions,
                "reference_words": reference_unit_count,
            }
            if code_switch_results:
                details["strict_wer"] = round(jiwer.wer(references_clean, hypotheses_strict), 3)
                details["code_switch"], code_switch_sub_metrics = _build_code_switch_details(
                    self.name, self.language, self.code_switch_languages, code_switch_results
                )
                details["per_turn_code_switch_rescues"] = {
                    turn_id: [{"expected": x.expected, "actual": x.actual, "matcher": x.matcher} for x in r.rescues]
                    for turn_id, r in zip(evaluated_turn_ids, code_switch_results)
                    if r.rescues
                }
                sub_metrics.update(code_switch_sub_metrics)

            return MetricScore(
                name=self.name,
                score=round(error_rate, 3),
                normalized_score=round(accuracy, 3),
                details=details,
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
