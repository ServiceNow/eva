"""Agent speech fidelity metric using audio + LLM judge (Gemini)."""

from eva.metrics.base import MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric
from eva.metrics.utils import build_per_category_rate_sub_metrics
from eva.models.results import MetricScore

_SPEECH_FIDELITY_FAILURE_MODES = (
    "entity_error",
    "truncation",
    "garbled_hallucination",
    "insertion_hallucination",
    "wrong_language",
)


@register_metric
class AgentSpeechFidelityMetric(SpeechFidelityBaseMetric):
    """Audio-based speech fidelity metric for agent using Gemini.

    Evaluates whether the agent's spoken audio accurately represents the intended text.
    Rating scale: 0 (low fidelity) or 1 (high fidelity)
    Evaluates each agent turn for missing, added, or incorrect words.
    """

    name = "agent_speech_fidelity"
    version = "v0.3"
    description = "Audio-based evaluation of agent speech fidelity to the intended text"
    category = "accuracy"
    role = "assistant"
    rating_scale = (0, 1)
    pass_at_k_threshold = 0.95

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_failure_modes: dict[int, list[str]],
    ) -> dict[str, MetricScore] | None:
        """Surface one sub-metric per failure mode: rate = flagged turns / rated turns."""
        rated_turn_ids = [tid for tid, r in per_turn_ratings.items() if r is not None]
        return (
            build_per_category_rate_sub_metrics(
                parent_name=self.name,
                categories=_SPEECH_FIDELITY_FAILURE_MODES,
                rated_turn_ids=rated_turn_ids,
                per_turn_categories=per_turn_failure_modes,
            )
            or None
        )
