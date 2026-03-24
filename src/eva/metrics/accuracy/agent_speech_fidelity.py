"""Agent speech fidelity metric using audio + LLM judge (Gemini)."""

from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric


@register_metric
class AgentSpeechFidelityMetric(SpeechFidelityBaseMetric):
    """Audio-based speech fidelity metric for agent using Gemini.

    Evaluates whether the agent's spoken audio accurately represents the intended text.
    Rating scale: 0 (low fidelity) or 1 (high fidelity)
    Evaluates each agent turn for missing, added, or incorrect words.
    """

    name = "agent_speech_fidelity"
    description = "Audio-based evaluation of agent speech fidelity to the intended text"
    category = "accuracy"
    role = "assistant"
    rating_scale = (0, 1)
    pass_at_k_threshold = 0.95
