"""User speech fidelity validation metric using audio + LLM judge (Gemini)."""

from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric


@register_metric
class UserSpeechFidelityMetric(SpeechFidelityBaseMetric):
    """Audio-based speech fidelity validation metric for user using Gemini.

    Evaluates whether the simulated user's spoken audio accurately represents the intended text.
    Rating scale: 1 (poor fidelity), 2 (acceptable), 3 (high fidelity)
    Evaluates each user turn for missing, added, or incorrect words.
    """

    name = "user_speech_fidelity"
    description = "Audio-based validation of user speech fidelity to the intended text"
    category = "validation"
    role = "user"
    rating_scale = (1, 3)
