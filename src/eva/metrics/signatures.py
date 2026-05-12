"""Compute drift signatures for metric classes.

A metric's "signature" captures everything we want to detect changes to:
 - `version`: the manually-bumped string on the class
 - `source_hash`: sha256[:12] of `inspect.getsource(cls)` (class body)
 - `prompt_hash`: sha256[:12] of the unrendered judge prompt template, or
                  None for non-judge metrics

The drift test compares the current signatures against a checked-in fixture
and fails if anything changed without an explicit version bump + fixture regen.
"""

import hashlib
import inspect

# Importing the metric subpackages forces all concrete metric classes to be
# registered as BaseMetric subclasses, so walking __subclasses__ finds them.
import eva.metrics.accuracy  # noqa: F401
import eva.metrics.diagnostic  # noqa: F401
import eva.metrics.experience  # noqa: F401
import eva.metrics.validation  # noqa: F401
from eva.metrics.base import AudioJudgeMetric, BaseMetric, TextJudgeMetric
from eva.models.versioning import hash_prompt_template
from eva.utils.prompt_manager import get_prompt_manager


def _all_concrete_versioned_metric_classes() -> dict[str, type[BaseMetric]]:
    """Walk BaseMetric subclasses; return concrete classes that set a version.

    Keyed on class qualname so co-named classes (e.g., the cascade vs S2S
    variants of `agent_speech_fidelity`) get distinct entries.
    """
    result: dict[str, type[BaseMetric]] = {}

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            walk(sub)
            if inspect.isabstract(sub):
                continue
            # `version` is None on BaseMetric; only concrete classes that
            # deliberately set it are participating.
            if getattr(sub, "version", None) is None:
                continue
            result[sub.__qualname__] = sub

    walk(BaseMetric)
    return result


def _source_hash(cls: type) -> str:
    """sha256[:12] of the class body source."""
    return hashlib.sha256(inspect.getsource(cls).encode("utf-8")).hexdigest()[:12]


def _prompt_hash_for_metric(cls: type[BaseMetric]) -> str | None:
    """Return the prompt template hash for judge metrics, or None.

    All judge metrics in this codebase use `judge.{name}.user_prompt`.
    A judge metric without a corresponding template raises KeyError —
    that's a configuration bug we want surfaced.
    """
    if not issubclass(cls, TextJudgeMetric | AudioJudgeMetric):
        return None
    template = get_prompt_manager().get_template(f"judge.{cls.name}.user_prompt")
    return hash_prompt_template(template)


def compute_all_metric_signatures() -> dict[str, dict[str, str | None]]:
    """Return {class_qualname: {version, source_hash, prompt_hash}} for every concrete metric."""
    out: dict[str, dict[str, str | None]] = {}
    for qualname, cls in _all_concrete_versioned_metric_classes().items():
        out[qualname] = {
            "name": cls.name,
            "version": cls.version,
            "source_hash": _source_hash(cls),
            "prompt_hash": _prompt_hash_for_metric(cls),
        }
    return out
