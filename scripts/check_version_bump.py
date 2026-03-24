#!/usr/bin/env python3
"""Pre-commit hook: remind to bump simulation_version / metrics_version.

Checks staged files against path patterns. If simulation or metrics code
changed but the corresponding version in src/eva/__init__.py was NOT
also staged with a change, the hook fails with a reminder.

Exit codes:
  0 — no version bump needed, or version was already bumped
  1 — version bump needed (prints which one)
"""

import os
import subprocess
import sys

SIMULATION_PATHS = (
    "src/eva/assistant/",
    "src/eva/user_simulator/",
    "src/eva/orchestrator/",
    "configs/prompts/simulation.yaml",
    "configs/agents/",
    "src/eva/assistant/tools/",
)

METRICS_PATHS = (
    "src/eva/metrics/",
    "configs/prompts/judge.yaml",
    "src/eva/utils/pricing.py",
    "src/eva/metrics/processor.py",
)

VERSION_FILE = "src/eva/__init__.py"


def get_staged_files() -> list[str]:
    """Return list of staged file paths (relative to repo root)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
    )
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def version_file_staged() -> str:
    """Return the staged diff of __init__.py, or empty string."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--", VERSION_FILE],
        capture_output=True,
        text=True,
    )
    return result.stdout


def check_version_in_diff(diff_text: str, version_name: str) -> bool:
    """Check if version_name line was changed in the diff."""
    for line in diff_text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            if f"{version_name} =" in line:
                return True
    return False


def matches_any(filepath: str, patterns: tuple[str, ...]) -> bool:
    """Check if filepath starts with any of the given patterns."""
    return any(filepath.startswith(p) for p in patterns)


def main() -> int:
    if os.environ.get("SKIP_VERSION_CHECK"):
        return 0

    staged = get_staged_files()
    if not staged:
        return 0

    sim_changed = any(matches_any(f, SIMULATION_PATHS) for f in staged)
    met_changed = any(matches_any(f, METRICS_PATHS) for f in staged)

    if not sim_changed and not met_changed:
        return 0

    diff = version_file_staged()
    sim_bumped = check_version_in_diff(diff, "simulation_version")
    met_bumped = check_version_in_diff(diff, "metrics_version")

    missing = []
    if sim_changed and not sim_bumped:
        missing.append(("simulation_version", SIMULATION_PATHS))
    if met_changed and not met_bumped:
        missing.append(("metrics_version", METRICS_PATHS))

    if not missing:
        return 0

    print("=" * 60)
    print("VERSION BUMP REMINDER")
    print("=" * 60)
    for version_name, paths in missing:
        triggered_by = [f for f in staged if matches_any(f, paths)]
        print(f"\n  {version_name} may need a bump.")
        print("  Changed files:")
        for f in triggered_by[:5]:
            print(f"    - {f}")
        if len(triggered_by) > 5:
            print(f"    ... and {len(triggered_by) - 5} more")
    print(f"\n  Edit: {VERSION_FILE}")
    print("  Also update pyproject.toml [tool.eva] to match.")
    print()
    print("  To skip this check: git commit --no-verify")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
