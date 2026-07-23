"""Final-turn input-characteristic flags — pure functions of the user's final utterance text.

Orthogonal to the pipeline causes: they describe the *input*, not a failure mode.
"""

import re

_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
# Acknowledgement lead — (includes negations no/nope, tolerant of leading [tags]);
# scope is "starts with a confirmation/acknowledgement".
_ACKNOWLEDGEMENT_LEAD = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)*\b(?:yes|no|nope|yep|yeah|sure|okay|ok|alright|correct|right)\b",
    re.IGNORECASE,
)
# Spelled-entity signals
# Single-letter / single-digit runs require ≥3 chars — 2 chars ("I T" = the word IT, "G A") isn't spelling.
_SEPS = r"[\s.,;:\-…]+"
_SINGLE_LETTER_RUN_RE = re.compile(rf"(?:\b[a-zA-Z]\b{_SEPS}){{2,}}\b[a-zA-Z]\b")
_SINGLE_DIGIT_RUN_RE = re.compile(rf"(?:\b\d\b{_SEPS}){{2,}}\b\d\b")
_AS_IN_RE = re.compile(r"\b[a-zA-Z]\s+as\s+in\s+\w+", re.IGNORECASE)
_CAPS_ALNUM_CODE_RE = re.compile(r"\b(?=[A-Z0-9]{3,})(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")
_CAPS_ACRONYM_RE = re.compile(r"\b[A-Z]{3,}\b")
# Single source of truth for the spoken-digit vocabulary → regex fragment + membership set.
_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
_DIGIT_WORD = r"(?:" + "|".join(_DIGIT_WORDS) + r")"
_DIGIT_WORD_SET = set(_DIGIT_WORDS)
_SPOKEN_DIGIT_RUN_RE = re.compile(rf"(?:\b{_DIGIT_WORD}\b[^a-zA-Z]+){{2,}}\b{_DIGIT_WORD}\b", re.IGNORECASE)
_NATO_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "x-ray",
    "xray",
    "yankee",
    "zulu",
)
_NATO_WORD_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _NATO_WORDS) + r")\b", re.IGNORECASE)
_NATO_SET = set(_NATO_WORDS)
# Length caps (annotation-stripped word counts). A genuine acknowledgement is short; a long sentence
# that merely starts "yeah …" isn't one. A spelled entity must *dominate* the turn — so we cap the
# surrounding non-spelled (prose) words rather than total words (real codes are long).
_ACKNOWLEDGEMENT_MAX_WORDS = 6
_SPELL_MAX_EXTRA_WORDS = 10


def _strip_annotations(text: str) -> str:
    """Remove prosody/annotation tags like [slow], [neutral], [user interrupts]."""
    return _ANNOTATION_RE.sub(" ", text or "")


def _non_spell_word_count(text: str) -> int:
    """Words that are NOT part of a spell-out (single letters, digit-words, 'dash', NATO, caps codes)."""
    n = 0
    for raw in text.split():
        tok = raw.strip(".,!?;:'\"()[]…-")
        if not tok:
            continue
        low = tok.lower()
        is_spell_tok = (
            (len(tok) == 1 and tok.isalpha())
            or low in _DIGIT_WORD_SET
            or low == "dash"
            or low in _NATO_SET
            or (tok.isupper() and len(tok) >= 3)  # caps acronym (VPN, HIPAA)
            or (len(tok) >= 3 and tok.isalnum() and any(c.isupper() for c in tok) and any(c.isdigit() for c in tok))
        )
        if not is_spell_tok:
            n += 1
    return n


def _looks_spelled(text: str) -> bool:
    """Any spelled-out signal — single-letter/digit runs, NATO, 'X as in Y', caps codes/acronyms."""
    return bool(
        _SINGLE_LETTER_RUN_RE.search(text)
        or _SINGLE_DIGIT_RUN_RE.search(text)
        or _AS_IN_RE.search(text)
        or _NATO_WORD_RE.search(text)
        or _CAPS_ALNUM_CODE_RE.search(text)
        or _CAPS_ACRONYM_RE.search(text)
        or _SPOKEN_DIGIT_RUN_RE.search(text)
    )


def final_turn_input_flags(text: str | None) -> dict[str, bool]:
    """Orthogonal input-characteristic flags for the user's final intended utterance.

    Used to test whether short / acknowledgement / spelled-out final turns correlate with failures.
    Heuristics mirror the analysis app (EVABench `apps/analysis.py`) so the two tools agree, and are
    **English-only** (the ack lead, spoken-digit words, and "X as in Y" cues are English phrasing;
    single-letter/digit runs, caps codes, and NATO are largely language-neutral). Prosody/annotation
    tags ([slow], [neutral], …) inflate word counts, so they're stripped first.

    - ``short``: 1-2 words (< 3), counted **after stripping `[annotation]` tags**.
    - ``acknowledgement``: leads with a confirmation/ack ("Yes, that is correct.", "Ok thanks",
      "[neutral] Okay.", "No.") AND is at most ``_ACKNOWLEDGEMENT_MAX_WORDS`` words — a long sentence
      that merely starts "yeah …" is not counted.
    - ``spelled_entity``: letter/digit spell-out of an ID/code/name ("E M P eight nine …", NATO,
      "V as in Victor", caps codes like EMP358) that **dominates** the turn (≤ ``_SPELL_MAX_EXTRA_WORDS``
      non-spelled words) — a long sentence with a small spelled fragment is not counted. Single-char
      runs need ≥3 chars, so 2-letter words like "IT" ("I T") don't count.
    """
    if not text or not text.strip():
        return {"short": False, "acknowledgement": False, "spelled_entity": False}
    stripped = _strip_annotations(text).strip()
    word_count = len(stripped.split())
    return {
        "short": 0 < word_count < 3,
        "acknowledgement": word_count <= _ACKNOWLEDGEMENT_MAX_WORDS and bool(_ACKNOWLEDGEMENT_LEAD.match(stripped)),
        "spelled_entity": _looks_spelled(stripped) and _non_spell_word_count(stripped) <= _SPELL_MAX_EXTRA_WORDS,
    }
