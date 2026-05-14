"""Drift test: fail when a metric's source or prompt changes without a version bump.

Each concrete metric class has three signature fields:
  - version: manually bumped string on the class
  - source_hash: sha256[:12] of inspect.getsource(cls)
  - prompt_hash: sha256[:12] of judge.{name}.user_prompt template (None for code metrics)

The fixture at tests/fixtures/metric_signatures.json is the source of truth for
the *currently-released* state. The test compares the current signatures to it
and reports drift. Authors update the fixture by running
`python scripts/regen_metric_signatures.py` after bumping `version`.

Failure modes the test catches:
  - source_hash changed, version unchanged → "bump version then regen fixture"
  - prompt_hash changed, version unchanged → "bump version then regen fixture"
  - version bumped → "regen fixture" (caught when source/prompt also still drift)
  - new metric class with no fixture entry → "add to fixture via regen"
  - metric removed from fixture → "delete from fixture via regen"
"""

import json
from pathlib import Path

import pytest

from eva.metrics.signatures import compute_all_metric_signatures

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "metric_signatures.json"
REGEN_HINT = "Run `python scripts/regen_metric_signatures.py` after bumping `version` on affected classes."


@pytest.fixture(scope="module")
def fixture_signatures() -> dict[str, dict[str, str | None]]:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def current_signatures() -> dict[str, dict[str, str | None]]:
    return compute_all_metric_signatures()


def test_no_unannounced_metric_drift(
    fixture_signatures: dict[str, dict[str, str | None]],
    current_signatures: dict[str, dict[str, str | None]],
) -> None:
    """Fail if any metric's source/prompt changed without its version being bumped."""
    failures: list[str] = []

    for qualname, current in current_signatures.items():
        recorded = fixture_signatures.get(qualname)
        if recorded is None:
            failures.append(f"{qualname}: new metric class not in fixture. {REGEN_HINT}")
            continue

        version_changed = current["version"] != recorded["version"]
        source_changed = current["source_hash"] != recorded["source_hash"]
        prompt_changed = current["prompt_hash"] != recorded["prompt_hash"]

        if not (source_changed or prompt_changed or version_changed):
            continue  # fully in sync

        if version_changed:
            # Author bumped version; they still need to regen the fixture so
            # future drift is detected against the new baseline.
            failures.append(f"{qualname}: version bumped ({recorded['version']} → {current['version']}). {REGEN_HINT}")
            continue

        # Code or prompt changed but version is unchanged — the case the test
        # exists to catch.
        what = []
        if source_changed:
            what.append(f"source ({recorded['source_hash']} → {current['source_hash']})")
        if prompt_changed:
            what.append(f"prompt ({recorded['prompt_hash']} → {current['prompt_hash']})")
        failures.append(
            f"{qualname}: {' and '.join(what)} changed but version still {current['version']!r}. "
            f"Bump `version` on the class, then run regen."
        )

    for qualname in fixture_signatures.keys() - current_signatures.keys():
        failures.append(f"{qualname}: removed from code but still in fixture. {REGEN_HINT}")

    assert not failures, "Metric signature drift detected:\n  " + "\n  ".join(failures)
