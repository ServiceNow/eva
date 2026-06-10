"""Per-record version + prompt-hash stamping for MetricScore.

MetricsRunner sets these contextvars around every metric.compute() call.
The MetricScore Pydantic model has a model_validator that reads them and
auto-fills the version/prompt_hash fields when unset, so all scores and
sub-scores built inside that compute() inherit the right values without
each call site having to thread them through explicitly.

Both contextvars default to None, which means "not currently inside a
metric compute() call" — that's the state during JSON deserialization
(loading metrics.json from disk), so existing on-disk values are
preserved instead of being overwritten with None.
"""

import hashlib
from contextvars import ContextVar

_CURRENT_METRIC_VERSION: ContextVar[str | None] = ContextVar("current_metric_version", default=None)
_CURRENT_PROMPT_HASH: ContextVar[str | None] = ContextVar("current_prompt_hash", default=None)


def hash_prompt_template(template: str) -> str:
    """Return sha256[:12] of an unrendered prompt template string."""
    return hashlib.sha256(template.encode()).hexdigest()[:12]
