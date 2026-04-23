"""Deterministic task completion metric via scenario database state hashing.

This metric validates task correctness by comparing SHA-256 hashes of scenario
database states. It follows tau-2 bench's approach and provides binary pass/fail
validation without LLM subjectivity.

When hashes don't match, it computes a detailed diff between expected and actual
final states to help diagnose the discrepancy.
"""

from eva.metrics.base import BaseMetric, MetricContext, MetricType
from eva.metrics.diagnostic.authentication_success import compute_session_auth_mismatches
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore
from eva.utils.hash_utils import compute_db_diff, get_dict_hash


@register_metric
class TaskCompletion(BaseMetric):
    """Deterministic task completion metric.

    Compares SHA-256 hashes of expected vs actual final scenario database states.
    Returns 1.0 (pass) if hashes match, 0.0 (fail) if they don't.

    The expected hash is computed on-the-fly from expected_scenario_db (from ground truth),
    while the actual hash is computed during execution and stored in final_scenario_db_hash.
    Both use canonical JSON serialization (sort_keys=True, separators=(',', ':')).

    When hashes don't match, computes a detailed diff showing:
    - Tables added/removed/modified
    - Records added/removed/modified within tables
    - Field-level changes within records

    This provides exact, reproducible validation without LLM variability.
    """

    name = "task_completion"
    description = "Binary task completion via scenario DB state hash comparison"
    category = "accuracy"
    metric_type = MetricType.CODE
    pass_at_k_threshold = 1.0

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compare expected vs actual scenario database hashes.

        Args:
            context: Metric context containing DB states and hashes

        Returns:
            MetricScore with:
            - score: 1.0 (match) or 0.0 (mismatch)
            - normalized_score: Same as score (already 0-1)
            - details: Match status, hashes, and diff (if mismatch)
        """
        details: dict = {"match": False, "auth_success": True}

        # Require auth success — if session mismatches, task cannot be complete
        auth_mismatches = compute_session_auth_mismatches(context.expected_scenario_db, context.final_scenario_db)
        if auth_mismatches:
            details["auth_success"] = False
            details["message"] = f"Authentication failed — session mismatch on keys: {list(auth_mismatches)}"
            details["auth_mismatches"] = auth_mismatches
            return MetricScore(name=self.name, score=0.0, normalized_score=0.0, details=details)

        # Compute expected hash from expected_scenario_db on-the-fly
        expected_hash = get_dict_hash(context.expected_scenario_db)
        actual_hash = context.final_scenario_db_hash
        details["expected_hash"] = expected_hash
        details["actual_hash"] = actual_hash

        # Compare hashes
        match = expected_hash == actual_hash
        details["match"] = match

        if match:
            details["message"] = "Final database state matches expected state exactly"
            return MetricScore(name=self.name, score=1.0, normalized_score=1.0, details=details)

        # Hashes don't match - compute diff to show what's different
        diff = compute_db_diff(expected_db=context.expected_scenario_db, actual_db=context.final_scenario_db)

        summary_parts = []
        if diff["tables_added"]:
            summary_parts.append(f"{len(diff['tables_added'])} tables added")
        if diff["tables_removed"]:
            summary_parts.append(f"{len(diff['tables_removed'])} tables removed")
        if diff["tables_modified"]:
            summary_parts.append(f"{len(diff['tables_modified'])} tables modified")
        summary = ", ".join(summary_parts) if summary_parts else "No differences found (hash collision?)"

        details["message"] = f"Final database state differs from expected: {summary}"
        details["diff"] = diff
        details["diff_summary"] = summary
        details["debugging_hints"] = [
            "Check diff.tables_modified for which tables changed",
            "For each modified table, check records_added/removed/modified",
            "For modified records, check field-level changes",
            "Expected state is from ground_truth in dataset",
            "Actual state is final_scenario_db.json from execution",
        ]
        return MetricScore(name=self.name, score=0.0, normalized_score=0.0, details=details)
