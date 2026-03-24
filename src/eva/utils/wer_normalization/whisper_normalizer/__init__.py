"""Whisper text normalizers from OpenAI Whisper."""

from eva.utils.wer_normalization.whisper_normalizer.basic import BasicTextNormalizer
from eva.utils.wer_normalization.whisper_normalizer.english import EnglishTextNormalizer

__all__ = ["BasicTextNormalizer", "EnglishTextNormalizer"]
