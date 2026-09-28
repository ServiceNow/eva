"""Shared helper for determining whether a task was completed.

This mirrors the logic of the ``task_completion`` accuracy metric so that
efficiency diagnostics (time/turns to completion) agree with it exactly: a task
is completed only if the session is authenticated correctly and the final
scenario database state matches the expected state (SHA-256 hash comparison).
"""

from eva.metrics.base import MetricContext
from eva.metrics.diagnostic.authentication_success import compute_session_auth_mismatches
from eva.utils.hash_utils import get_dict_hash


def is_task_completed(context: MetricContext) -> tuple[bool, str]:
    """Return whether the task was completed and a human-readable reason.

    Uses the same criteria as the ``task_completion`` accuracy metric:
      1. The session must be authenticated correctly (no session mismatches).
      2. The final scenario DB hash must equal the expected scenario DB hash.

    Args:
        context: Metric context containing scenario DB states and hashes.

    Returns:
        (completed, reason) where ``completed`` is True only when both checks
        pass, and ``reason`` explains the outcome.
    """
    # Require auth success — if the session mismatches, the task cannot be complete.
    auth_mismatches = compute_session_auth_mismatches(context.expected_scenario_db, context.final_scenario_db)
    if auth_mismatches:
        return False, f"Authentication failed — session mismatch on keys: {list(auth_mismatches)}"

    expected_hash = get_dict_hash(context.expected_scenario_db)
    actual_hash = context.final_scenario_db_hash

    if expected_hash == actual_hash:
        return True, "Final database state matches expected state exactly"

    return False, "Final database state differs from expected state"
