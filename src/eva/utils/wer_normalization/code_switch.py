"""Code-switch–aware WER reconciliation.

Speakers of the primary language mix in words of a secondary one ("मेरी
flight seven बजे है"), and the two transcripts often write those words
differently ("seven" vs "7", "I'm" vs "I am"). Normalizing both sides with the
primary language's rules alone counts such correct words as errors.

Two independent WER passes can't be combined, so instead:

1. Normalize both sides with the primary rules (the strict view P) and align.
2. Build a secondary view V of the raw text: primary-script runs keep the
   primary rules, the other runs get the secondary rules.
3. For each error region of the P alignment, project it onto V.
4. Rescue the region only when both V spans are identical and the reference
   is attested as secondary-language speech (it is in the secondary script, or
   the secondary rules changed it).

Rescued hypothesis regions are replaced with the reference tokens and the
alignment is recomputed, so the reconciled WER can never exceed the strict one.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import jiwer
import regex

from eva.utils.wer_normalization.wer_utils import _make_normalizer, normalize_text

# Tokens in the primary language's own script. Only languages listed here get
# the script split; the others would need a same-script approach.
_PRIMARY_SCRIPTS = {"hi": regex.compile(r"\p{Devanagari}")}

_RE_LATIN = regex.compile(r"\p{Latin}")
_RE_DIGIT_SPACE = regex.compile(r"(?<=\d) (?=\d)")

# Error regions larger than this (on either side) are never rescued.
MAX_REGION_TOKENS = 4

MATCHER_SECONDARY_VIEW = "secondary_view"


@dataclass
class Rescue:
    expected: str
    actual: str
    matcher: str
    ref_idx: int


@dataclass
class ReconcileResult:
    ref_tokens: list[str]
    hyp_tokens: list[str]
    strict_output: jiwer.WordOutput
    rescues: list[Rescue] = field(default_factory=list)
    ref_langs: list[str] = field(default_factory=list)
    hyp_langs: list[str] = field(default_factory=list)


@lru_cache(maxsize=8)
def _loanword_values(language: str) -> frozenset[str]:
    """Primary-script spellings of known secondary-language loanwords (``hi_loanwords.json`` values)."""
    return frozenset(getattr(_make_normalizer(language), "spelling_map", {}).values())


def tag_token(token: str, primary: str, secondary: str) -> str:
    """Language tag of a normalized token: ``num``, ``secondary`` or ``primary``.

    Latin-script tokens and primary-script spellings of known loanwords
    (मार्च) are secondary; other primary-script tokens are primary.
    """
    if token.isdigit():
        return "num"
    if _RE_LATIN.search(token) or token in _loanword_values(primary):
        return secondary
    return primary


def _split_runs(raw: str, primary: str) -> list[tuple[bool, str]]:
    """Split raw text into maximal whitespace-token runs: (is_primary_script, text)."""
    script = _PRIMARY_SCRIPTS.get(primary)
    runs: list[tuple[bool, list[str]]] = []
    for tok in raw.split():
        is_primary = bool(script and script.search(tok))
        if runs and runs[-1][0] == is_primary:
            runs[-1][1].append(tok)
        else:
            runs.append((is_primary, [tok]))
    return [(is_primary, " ".join(toks)) for is_primary, toks in runs]


@dataclass
class _View:
    strict: list[str]  # P tokens, identical to normalize_text(raw, primary).split()
    secondary: list[str]  # V tokens
    from_secondary_run: list[bool]  # per P token: did it come from a secondary-script run?
    cuts: dict[int, int] | None  # P index -> V index at clean cut points; None if not projectable


def _cut_points(p: list[str], v: list[str]) -> dict[int, int | None]:
    """P index -> V index where the P/V alignment can be cut cleanly (None = ambiguous)."""
    if not p:
        return {0: None if v else 0}
    if not v:
        return {0: 0, len(p): 0}
    cuts: dict[int, int | None] = {0: 0, len(p): len(v)}
    for chunk in jiwer.process_words(" ".join(p), " ".join(v)).alignments[0]:
        if chunk.type != "equal":
            continue
        for k in range(chunk.ref_end_idx - chunk.ref_start_idx + 1):
            pi, vi = chunk.ref_start_idx + k, chunk.hyp_start_idx + k
            cuts[pi] = vi if cuts.get(pi, vi) == vi else None
    return cuts


def _build_view(raw: str, primary: str, secondary: str) -> _View:
    strict = normalize_text(raw, primary).split()
    p_all: list[str] = []
    v_all: list[str] = []
    origins: list[bool] = []
    cuts: dict[int, int | None] = {}
    for is_primary, text in _split_runs(raw, primary):
        p_run = normalize_text(text, primary).split()
        v_run = p_run if is_primary else normalize_text(text, secondary).split()
        for pi, vi in _cut_points(p_run, v_run).items():
            key = len(p_all) + pi
            val = None if vi is None else len(v_all) + vi
            cuts[key] = val if cuts.get(key, val) == val else None
        p_all += p_run
        v_all += v_run
        origins += [not is_primary] * len(p_run)
    if p_all != strict:
        # Per-run normalization disagrees with the whole-sentence one: no projection.
        return _View(strict, v_all, [False] * len(strict), None)
    return _View(strict, v_all, origins, {k: v for k, v in cuts.items() if v is not None})


def _project(view: _View, start: int, end: int) -> list[str] | None:
    if view.cuts is None or start not in view.cuts or end not in view.cuts:
        return None
    return view.secondary[view.cuts[start] : view.cuts[end]]


def _same_speech(a: list[str], b: list[str]) -> bool:
    """V spans equal, ignoring spaces between digits ("seven thirty" -> 730, "7:30" -> 7 30)."""
    return bool(a) and bool(b) and _RE_DIGIT_SPACE.sub("", " ".join(a)) == _RE_DIGIT_SPACE.sub("", " ".join(b))


def _error_regions(chunks) -> list[tuple[int, int, int, int]]:
    """Maximal runs of non-equal chunks as (ref_start, ref_end, hyp_start, hyp_end)."""
    regions = []
    current = None
    for c in chunks:
        if c.type == "equal":
            if current:
                regions.append(tuple(current))
            current = None
        elif current is None:
            current = [c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx]
        else:
            current[1], current[3] = c.ref_end_idx, c.hyp_end_idx
    if current:
        regions.append(tuple(current))
    return regions


def reconcile_code_switch(ref_raw: str, hyp_raw: str, primary: str, secondaries: tuple[str, ...]) -> ReconcileResult:
    """Align one turn with code-switch rescues. See the module docstring."""
    secondary = secondaries[0]
    ref_view = _build_view(ref_raw, primary, secondary)
    hyp_view = _build_view(hyp_raw, primary, secondary)
    ref_p, hyp_p = ref_view.strict, hyp_view.strict
    strict_output = jiwer.process_words(" ".join(ref_p), " ".join(hyp_p))

    rescues: list[Rescue] = []
    replacements: list[tuple[int, int, list[str]]] = []  # hyp span -> ref tokens
    for rs, re_, hs, he in _error_regions(strict_output.alignments[0]):
        if rs == re_ or hs == he or re_ - rs > MAX_REGION_TOKENS or he - hs > MAX_REGION_TOKENS:
            continue
        ref_v, hyp_v = _project(ref_view, rs, re_), _project(hyp_view, hs, he)
        if ref_v is None or hyp_v is None or not _same_speech(ref_v, hyp_v):
            continue
        ref_span = ref_p[rs:re_]
        attested = ref_v != ref_span or any(
            ref_view.from_secondary_run[i] and tag_token(ref_p[i], primary, secondary) == secondary
            for i in range(rs, re_)
        )
        if not attested:
            continue
        rescues.append(Rescue(" ".join(ref_span), " ".join(hyp_p[hs:he]), MATCHER_SECONDARY_VIEW, rs))
        replacements.append((hs, he, ref_span))

    hyp_tokens = list(hyp_p)
    for hs, he, ref_span in reversed(replacements):
        hyp_tokens[hs:he] = ref_span

    return ReconcileResult(
        ref_tokens=ref_p,
        hyp_tokens=hyp_tokens,
        strict_output=strict_output,
        rescues=rescues,
        ref_langs=[tag_token(t, primary, secondary) for t in ref_p],
        hyp_langs=[tag_token(t, primary, secondary) for t in hyp_tokens],
    )
