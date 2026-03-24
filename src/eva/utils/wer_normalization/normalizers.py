from jaconv import jaconv


class JapaneseTextNormalizer:
    """Normalize Japanese text for character error rate (CER) calculation."""

    def __call__(self, s: str):
        """Convert Japanese text to a normalized form.

        Convert Half-width (Hankaku) Katakana to Full-width (Zenkaku) Katakana, Full-width (Zenkaku) ASCII and
        DIGIT to Half-width (Hankaku) ASCII and DIGIT. Additionally, Full-width wave dash (〜) etc. are normalized
        """
        return jaconv.normalize(s)
