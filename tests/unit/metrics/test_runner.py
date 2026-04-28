"""Tests for MetricsRunner: all records processed, errors handled, results saved."""

import json
from pathlib import Path

import pytest
import yaml

from eva.metrics.runner import MetricsRunner
from eva.models.config import PipelineType
from eva.models.results import MetricScore, RecordMetrics
from tests.unit.conftest import make_evaluation_record

from .conftest import make_metric_score


class _FakeMetric:
    """Minimal stand-in for BaseMetric — only ``name``, ``supported_pipeline_types``, and pass@k attrs are read."""

    def __init__(self, name: str):
        self.name = name
        self.supported_pipeline_types = frozenset(PipelineType)
        self.exclude_from_pass_at_k = False
        self.pass_at_k_threshold = 0.5


def _make_record(record_id: str):
    return make_evaluation_record(
        record_id,
        ground_truth={"expected_scenario_db": {"status": "done"}},
        category="test",
    )


def _make_record_metrics(record_id: str) -> RecordMetrics:
    """Create a minimal RecordMetrics for testing."""
    return RecordMetrics(
        record_id=record_id,
        metrics={
            "test_metric": MetricScore(name="test_metric", score=1.0, normalized_score=1.0),
        },
    )


def _ms(name: str, score: float = 1.0, error: str | None = None) -> MetricScore:
    return make_metric_score(name, score=score, error=error)


def _setup_run_dir(tmp_path: Path, record_ids: list[str]) -> Path:
    """Set up a minimal run directory with records subdirs and config."""
    run_dir = tmp_path / "output" / "test_run"
    run_dir.mkdir(parents=True)

    # Create agent config
    agent_config = {
        "id": "test_agent",
        "name": "Test Agent",
        "description": "Test",
        "role": "You are a test agent.",
        "instructions": "Help the user.",
        "tools": [{"id": "test_tool", "name": "test_tool", "description": "A test tool"}],
    }
    agent_config_path = tmp_path / "agent_config.yaml"
    with open(agent_config_path, "w") as f:
        yaml.dump(agent_config, f)

    # Create config.json
    (run_dir / "config.json").write_text(json.dumps({"agent_config_path": str(agent_config_path)}))

    # Create record directories
    for rid in record_ids:
        record_dir = run_dir / "records" / rid
        record_dir.mkdir(parents=True)

    return run_dir


def _write_metrics_json(record_dir: Path, record_id: str, metrics: dict[str, MetricScore]) -> None:
    """Write a metrics.json to a record directory."""
    rm = RecordMetrics(record_id=record_id, metrics=metrics)
    (record_dir / "metrics.json").write_text(rm.model_dump_json(indent=2))


def _read_disk_scores(record_dir: Path) -> dict[str, float | None]:
    """Read metrics.json and return {name: score} for quick assertions."""
    data = json.loads((record_dir / "metrics.json").read_text())
    return {k: v["score"] for k, v in data["metrics"].items()}


def _read_disk_errors(record_dir: Path) -> dict[str, str | None]:
    """Read metrics.json and return {name: error} for quick assertions."""
    data = json.loads((record_dir / "metrics.json").read_text())
    return {k: v.get("error") for k, v in data["metrics"].items()}


def _make_runner(run_dir, records, metric_names, record_metric_filter=None):
    """Build a MetricsRunner and replace self.metrics with _FakeMetric objects."""
    runner = MetricsRunner(
        run_dir=run_dir,
        dataset=records,
        metric_names=[],  # empty — we'll set self.metrics manually
        record_metric_filter=record_metric_filter,
    )
    runner.metrics = [_FakeMetric(n) for n in metric_names]
    return runner


def _install_mock(runner, scores_by_record):
    """Install a filter-aware mock _run_record and return the call tracker.

    The mock respects ``runner.record_metric_filter`` the same way the real
    ``_run_record`` does — only returning metrics whose names are in the
    filter for the given record.

    Returns a list that is appended to on every call with::

        {"record_id": str, "metrics_requested": set | None}
    """
    calls: list[dict] = []

    async def _mock(record_id, record_dir):
        allowed = None
        if runner.record_metric_filter and record_id in runner.record_metric_filter:
            allowed = set(runner.record_metric_filter[record_id])
        calls.append({"record_id": record_id, "metrics_requested": allowed})

        all_scores = scores_by_record.get(record_id, {})
        if allowed:
            filtered = {k: v for k, v in all_scores.items() if k in allowed}
        else:
            filtered = dict(all_scores)
        return RecordMetrics(record_id=record_id, metrics=filtered)

    runner._run_record = _mock
    return calls


class TestMetricsRunner:
    """Tests for MetricsRunner.run() behavioral contract."""

    @pytest.mark.asyncio
    async def test_exception_in_one_record_does_not_cancel_others(self, tmp_path):
        """One record failing doesn't prevent other records from completing."""
        record_ids = [f"rec-{i}" for i in range(4)]
        run_dir = _setup_run_dir(tmp_path, record_ids)
        records = [_make_record(rid) for rid in record_ids]

        runner = MetricsRunner(
            run_dir=run_dir,
            dataset=records,
            metric_names=["conversation_valid_end"],
        )

        async def mock_run_record(record_id, record_dir):
            if record_id == "rec-1":
                raise RuntimeError("Boom")
            return _make_record_metrics(record_id)

        runner._run_record = mock_run_record

        result = await runner.run()

        # rec-1 failed, the other 3 succeeded
        expected_record_ids = {"rec-0", "rec-2", "rec-3"}

        assert result.all_metrics.keys() == expected_record_ids

        for rid in expected_record_ids:
            metrics_path = run_dir / "records" / rid / "metrics.json"
            assert metrics_path.exists(), f"Missing metrics.json for {rid}"

        summary_path = run_dir / "metrics_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_records"] == len(expected_record_ids)

    @pytest.mark.asyncio
    async def test_scans_nested_trial_subdirectories(self, tmp_path):
        """MetricsRunner.run() discovers rec-0/trial_0, rec-0/trial_1 when no result.json in parent."""
        run_dir = _setup_run_dir(tmp_path, [])  # no flat record dirs
        records = [_make_record("rec-0")]

        # Create nested trial dirs (no result.json in parent rec-0/)
        records_dir = run_dir / "records"
        parent_dir = records_dir / "rec-0"
        parent_dir.mkdir(parents=True)
        for i in range(2):
            trial_dir = parent_dir / f"trial_{i}"
            trial_dir.mkdir()

        # Use _make_runner so self.metrics is non-empty (avoids registry lookup)
        runner = _make_runner(run_dir, records, ["test_metric"])

        captured_ids = []

        async def mock_run_and_save(record_id, record_dir):
            captured_ids.append(record_id)
            return _make_record_metrics(record_id)

        # Mock run_and_save_record (not _run_record) to bypass caching logic —
        # this test is about directory discovery, not per-record computation.
        runner.run_and_save_record = mock_run_and_save

        await runner.run()

        assert sorted(captured_ids) == ["rec-0/trial_0", "rec-0/trial_1"]

    @pytest.mark.asyncio
    async def test_merge_adds_new_metric_preserves_old(self, tmp_path):
        """When computing a NEW metric on a record that already has other metrics.

        The new metric is added and existing metrics are preserved.
        """
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        # Pre-write existing metrics — old_metric exists, new_metric does not
        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "old_metric": _ms("old_metric", 0.5),
            },
        )

        runner = _make_runner(run_dir, records, ["new_metric"])
        calls = _install_mock(
            runner,
            {
                "rec-0": {"new_metric": _ms("new_metric", 1.0)},
            },
        )

        result = await runner.run_and_save_record("rec-0", record_dir)

        # _run_record was called (new_metric is missing)
        assert len(calls) == 1
        assert calls[0]["metrics_requested"] == {"new_metric"}

        # In-memory result has both
        assert result.metrics["old_metric"].score == 0.5
        assert result.metrics["new_metric"].score == 1.0

        # On-disk result has both
        disk = _read_disk_scores(record_dir)
        assert disk == {"old_metric": 0.5, "new_metric": 1.0}


class TestNormalModeCaching:
    """In normal mode (no --rerun-failed-metrics), existing metrics on disk are never recomputed."""

    @pytest.mark.asyncio
    async def test_all_metrics_exist_skips_computation(self, tmp_path):
        """When every requested metric already exists on disk, _run_record is NOT called."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.8),
                "m_b": _ms("m_b", 0.6),
            },
        )

        runner = _make_runner(run_dir, records, ["m_a", "m_b"])
        calls = _install_mock(runner, {"rec-0": {"m_a": _ms("m_a", 0.99), "m_b": _ms("m_b", 0.99)}})

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 0, "_run_record should not be called"
        assert result.metrics["m_a"].score == 0.8  # original value
        assert result.metrics["m_b"].score == 0.6

    @pytest.mark.asyncio
    async def test_no_metrics_on_disk_computes_all(self, tmp_path):
        """When no metrics.json exists, all requested metrics are computed."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        runner = _make_runner(run_dir, records, ["m_a", "m_b"])
        calls = _install_mock(
            runner,
            {
                "rec-0": {"m_a": _ms("m_a", 0.9), "m_b": _ms("m_b", 0.7)},
            },
        )

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 1
        assert calls[0]["metrics_requested"] == {"m_a", "m_b"}
        assert result.metrics["m_a"].score == 0.9
        assert result.metrics["m_b"].score == 0.7

        # Written to disk
        assert (record_dir / "metrics.json").exists()
        disk = _read_disk_scores(record_dir)
        assert disk == {"m_a": 0.9, "m_b": 0.7}

    @pytest.mark.asyncio
    async def test_partial_metrics_computes_only_missing(self, tmp_path):
        """When some requested metrics exist and others don't, only missing ones are computed."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        # m_a already on disk; m_b is not
        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.8),
            },
        )

        runner = _make_runner(run_dir, records, ["m_a", "m_b"])
        calls = _install_mock(
            runner,
            {
                "rec-0": {"m_b": _ms("m_b", 0.7)},
            },
        )

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 1
        assert calls[0]["metrics_requested"] == {"m_b"}
        assert result.metrics["m_a"].score == 0.8  # from disk
        assert result.metrics["m_b"].score == 0.7  # computed
        assert _read_disk_scores(record_dir) == {"m_a": 0.8, "m_b": 0.7}

    @pytest.mark.asyncio
    async def test_failed_metric_on_disk_is_not_rerun(self, tmp_path):
        """A metric with an error on disk is treated as 'present' and not recomputed."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.0, error="judge timeout"),
            },
        )

        runner = _make_runner(run_dir, records, ["m_a"])
        calls = _install_mock(runner, {"rec-0": {"m_a": _ms("m_a", 0.9)}})

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 0, "failed metric should not be rerun in normal mode"
        assert result.metrics["m_a"].error == "judge timeout"  # preserved

    @pytest.mark.asyncio
    async def test_failed_computation_does_not_overwrite_existing(self, tmp_path):
        """If a newly computed metric fails, the existing on-disk value is preserved."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        # m_a exists, m_b does not
        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.8),
            },
        )

        runner = _make_runner(run_dir, records, ["m_a", "m_b"])
        # Mock returns m_b with an error (computation failed)
        calls = _install_mock(
            runner,
            {
                "rec-0": {"m_b": _ms("m_b", 0.0, error="API error")},
            },
        )

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 1
        # m_a preserved, m_b written with error
        assert result.metrics["m_a"].score == 0.8
        assert result.metrics["m_a"].error is None
        assert result.metrics["m_b"].error == "API error"


class TestRerunMode:
    """With ``record_metric_filter`` set, only the specified failed metrics on specified records are recomputed."""

    @pytest.mark.asyncio
    async def test_one_failure_one_record(self, tmp_path):
        """Only the failed metric on the specified record is rerun; other records are read from disk."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0", "rec-1"])
        records = [_make_record("rec-0"), _make_record("rec-1")]
        rec0_dir = run_dir / "records" / "rec-0"
        rec1_dir = run_dir / "records" / "rec-1"

        # rec-0: m_a ok, m_b failed
        _write_metrics_json(
            rec0_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 1.0),
                "m_b": _ms("m_b", 0.0, error="timeout"),
            },
        )
        # rec-1: both ok
        _write_metrics_json(
            rec1_dir,
            "rec-1",
            {
                "m_a": _ms("m_a", 0.9),
                "m_b": _ms("m_b", 0.8),
            },
        )

        runner = _make_runner(
            run_dir,
            records,
            ["m_a", "m_b"],
            record_metric_filter={"rec-0": {"m_b"}},
        )
        calls = _install_mock(
            runner,
            {
                "rec-0": {"m_b": _ms("m_b", 0.75)},
            },
        )

        # Process both records
        r0 = await runner.run_and_save_record("rec-0", rec0_dir)
        r1 = await runner.run_and_save_record("rec-1", rec1_dir)

        # rec-0: m_b was rerun
        assert len(calls) == 1
        assert calls[0]["record_id"] == "rec-0"
        assert calls[0]["metrics_requested"] == {"m_b"}
        assert r0.metrics["m_a"].score == 1.0  # preserved
        assert r0.metrics["m_b"].score == 0.75  # rerun succeeded
        assert r0.metrics["m_b"].error is None

        # rec-1: untouched (not in filter)
        assert r1.metrics["m_a"].score == 0.9
        assert r1.metrics["m_b"].score == 0.8

    @pytest.mark.asyncio
    async def test_different_metrics_per_record(self, tmp_path):
        """rec-0 needs m_a rerun, rec-1 needs m_b rerun — each gets only its own metric."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0", "rec-1"])
        records = [_make_record("rec-0"), _make_record("rec-1")]
        rec0_dir = run_dir / "records" / "rec-0"
        rec1_dir = run_dir / "records" / "rec-1"

        _write_metrics_json(
            rec0_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.0, error="fail"),
                "m_b": _ms("m_b", 0.8),
            },
        )
        _write_metrics_json(
            rec1_dir,
            "rec-1",
            {
                "m_a": _ms("m_a", 0.9),
                "m_b": _ms("m_b", 0.0, error="fail"),
            },
        )

        runner = _make_runner(
            run_dir,
            records,
            ["m_a", "m_b"],
            record_metric_filter={"rec-0": {"m_a"}, "rec-1": {"m_b"}},
        )
        calls = _install_mock(
            runner,
            {
                "rec-0": {"m_a": _ms("m_a", 0.95)},
                "rec-1": {"m_b": _ms("m_b", 0.85)},
            },
        )

        r0 = await runner.run_and_save_record("rec-0", rec0_dir)
        r1 = await runner.run_and_save_record("rec-1", rec1_dir)

        # rec-0: only m_a rerun
        assert calls[0] == {"record_id": "rec-0", "metrics_requested": {"m_a"}}
        assert r0.metrics["m_a"].score == 0.95
        assert r0.metrics["m_a"].error is None
        assert r0.metrics["m_b"].score == 0.8  # untouched

        # rec-1: only m_b rerun
        assert calls[1] == {"record_id": "rec-1", "metrics_requested": {"m_b"}}
        assert r1.metrics["m_a"].score == 0.9  # untouched
        assert r1.metrics["m_b"].score == 0.85
        assert r1.metrics["m_b"].error is None

    @pytest.mark.asyncio
    async def test_already_succeeded_metric_in_filter_is_skipped(self, tmp_path):
        """If a metric in the filter has already succeeded on disk, it is NOT rerun."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        # m_a succeeded on disk (maybe fixed by a different run)
        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.9),
            },
        )

        runner = _make_runner(
            run_dir,
            records,
            ["m_a"],
            record_metric_filter={"rec-0": {"m_a"}},
        )
        calls = _install_mock(runner, {"rec-0": {"m_a": _ms("m_a", 0.5)}})

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 0, "already-succeeded metric should not be rerun"
        assert result.metrics["m_a"].score == 0.9  # disk value preserved

    @pytest.mark.asyncio
    async def test_multiple_metrics_fail_on_same_record(self, tmp_path):
        """Both failed metrics on the same record are rerun; the successful one is preserved."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.0, error="fail-a"),
                "m_b": _ms("m_b", 0.0, error="fail-b"),
                "m_c": _ms("m_c", 0.9),  # this one is fine
            },
        )

        runner = _make_runner(
            run_dir,
            records,
            ["m_a", "m_b", "m_c"],
            record_metric_filter={"rec-0": {"m_a", "m_b"}},
        )
        calls = _install_mock(
            runner,
            {
                "rec-0": {
                    "m_a": _ms("m_a", 0.85),
                    "m_b": _ms("m_b", 0.75),
                },
            },
        )

        result = await runner.run_and_save_record("rec-0", record_dir)

        assert len(calls) == 1
        assert calls[0]["metrics_requested"] == {"m_a", "m_b"}
        assert result.metrics["m_a"].score == 0.85
        assert result.metrics["m_b"].score == 0.75
        assert result.metrics["m_c"].score == 0.9  # preserved

        disk = _read_disk_scores(record_dir)
        assert disk == {"m_a": 0.85, "m_b": 0.75, "m_c": 0.9}

    @pytest.mark.asyncio
    async def test_rerun_partial_success(self, tmp_path):
        """If only one of two failed metrics succeeds on rerun, the other keeps its new error."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0"])
        records = [_make_record("rec-0")]
        record_dir = run_dir / "records" / "rec-0"

        _write_metrics_json(
            record_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.0, error="fail-a"),
                "m_b": _ms("m_b", 0.0, error="fail-b"),
            },
        )

        runner = _make_runner(
            run_dir,
            records,
            ["m_a", "m_b"],
            record_metric_filter={"rec-0": {"m_a", "m_b"}},
        )
        calls = _install_mock(
            runner,
            {
                "rec-0": {
                    "m_a": _ms("m_a", 0.85),
                    "m_b": _ms("m_b", 0.0, error="still failing"),
                },
            },
        )

        result = await runner.run_and_save_record("rec-0", record_dir)
        assert len(calls) == 1
        assert result.metrics["m_a"].score == 0.85
        assert result.metrics["m_a"].error is None
        assert result.metrics["m_b"].error == "still failing"

    @pytest.mark.asyncio
    async def test_same_metric_fails_on_two_records(self, tmp_path):
        """The same metric failing on two different records is rerun independently on each."""
        run_dir = _setup_run_dir(tmp_path, ["rec-0", "rec-1"])
        records = [_make_record("rec-0"), _make_record("rec-1")]
        rec0_dir = run_dir / "records" / "rec-0"
        rec1_dir = run_dir / "records" / "rec-1"

        _write_metrics_json(
            rec0_dir,
            "rec-0",
            {
                "m_a": _ms("m_a", 0.0, error="fail"),
            },
        )
        _write_metrics_json(
            rec1_dir,
            "rec-1",
            {
                "m_a": _ms("m_a", 0.0, error="fail"),
            },
        )

        runner = _make_runner(
            run_dir,
            records,
            ["m_a"],
            record_metric_filter={"rec-0": {"m_a"}, "rec-1": {"m_a"}},
        )
        calls = _install_mock(
            runner,
            {
                "rec-0": {"m_a": _ms("m_a", 0.7)},
                "rec-1": {"m_a": _ms("m_a", 0.8)},
            },
        )

        r0 = await runner.run_and_save_record("rec-0", rec0_dir)
        r1 = await runner.run_and_save_record("rec-1", rec1_dir)

        assert len(calls) == 2
        assert r0.metrics["m_a"].score == 0.7
        assert r1.metrics["m_a"].score == 0.8

        assert _read_disk_scores(rec0_dir) == {"m_a": 0.7}
        assert _read_disk_scores(rec1_dir) == {"m_a": 0.8}


class TestBuildPerMetricAggregates:
    """Tests for MetricsRunner._build_per_metric_aggregates splitting error_count and missing_count."""

    def test_all_successful(self):
        """All records have successful scores — no errors or missing."""
        all_metrics = {
            "r1": RecordMetrics(record_id="r1", metrics={"m": _ms("m", 0.8)}),
            "r2": RecordMetrics(record_id="r2", metrics={"m": _ms("m", 0.6)}),
        }
        result = MetricsRunner._build_per_metric_aggregates(all_metrics, ["m"])
        assert result["m"]["count"] == 2
        assert result["m"]["none_count"] == 0
        assert result["m"]["error_count"] == 0
        assert result["m"]["missing_count"] == 0
        assert result["m"]["mean"] == pytest.approx(0.7)

    def test_errors_and_missing_split(self):
        """Errors and missing metrics are tracked separately."""
        all_metrics = {
            "r1": RecordMetrics(record_id="r1", metrics={"m": _ms("m", 0.8)}),
            "r2": RecordMetrics(record_id="r2", metrics={"m": _ms("m", 0.0, error="JSON parse failed")}),
            "r3": RecordMetrics(record_id="r3", metrics={}),  # metric missing entirely
        }
        result = MetricsRunner._build_per_metric_aggregates(all_metrics, ["m"])
        assert result["m"]["count"] == 1
        assert result["m"]["error_count"] == 1
        assert result["m"]["missing_count"] == 1
        assert result["m"]["none_count"] == 2
        assert result["m"]["mean"] == pytest.approx(0.8)

    def test_only_errors(self):
        """All records have errors — mean is None."""
        all_metrics = {
            "r1": RecordMetrics(record_id="r1", metrics={"m": _ms("m", 0.0, error="fail1")}),
            "r2": RecordMetrics(record_id="r2", metrics={"m": _ms("m", 0.0, error="fail2")}),
        }
        result = MetricsRunner._build_per_metric_aggregates(all_metrics, ["m"])
        assert result["m"]["count"] == 0
        assert result["m"]["error_count"] == 2
        assert result["m"]["missing_count"] == 0
        assert result["m"]["none_count"] == 2
        assert result["m"]["mean"] is None

    def test_higher_is_better_read_from_registered_metric(self):
        """Parent direction is looked up on the metric class, not stored per-record."""
        # response_speed is registered with higher_is_better=False on its class.
        all_metrics = {
            "r1": RecordMetrics(
                record_id="r1",
                metrics={"response_speed": MetricScore(name="response_speed", score=1.2, normalized_score=None)},
            ),
        }
        result = MetricsRunner._build_per_metric_aggregates(all_metrics, ["response_speed"])
        assert result["response_speed"]["higher_is_better"] is False

    def test_higher_is_better_defaults_true_for_unknown_metric(self):
        """An unregistered metric name defaults to higher_is_better=True."""
        all_metrics = {
            "r1": RecordMetrics(record_id="r1", metrics={"m": MetricScore(name="m", score=0.3)}),
        }
        result = MetricsRunner._build_per_metric_aggregates(all_metrics, ["m"])
        assert result["m"]["higher_is_better"] is True

    def test_sub_metric_direction_derived_from_suffix(self):
        """Sub-metric direction is derived from the key suffix, not stored per-record.

        ``_rate`` suffix → lower is better, ``_accuracy`` → higher is better,
        otherwise the sub-metric inherits the parent direction.
        """
        rate_sub = MetricScore(name="faithfulness.hallucination_rate", score=1.0, normalized_score=1.0)
        accuracy_sub = MetricScore(
            name="transcription_accuracy_key_entities.name_accuracy", score=0.8, normalized_score=0.8
        )
        all_metrics = {
            "r1": RecordMetrics(
                record_id="r1",
                metrics={
                    "faithfulness": MetricScore(
                        name="faithfulness",
                        score=2.0,
                        normalized_score=0.5,
                        sub_metrics={"hallucination_rate": rate_sub},
                    ),
                    "transcription_accuracy_key_entities": MetricScore(
                        name="transcription_accuracy_key_entities",
                        score=0.8,
                        normalized_score=0.8,
                        sub_metrics={"name_accuracy": accuracy_sub},
                    ),
                },
            ),
        }
        result = MetricsRunner._build_per_metric_aggregates(
            all_metrics, ["faithfulness", "transcription_accuracy_key_entities"]
        )
        assert result["faithfulness"]["sub_metrics"]["hallucination_rate"]["higher_is_better"] is False
        assert result["transcription_accuracy_key_entities"]["sub_metrics"]["name_accuracy"]["higher_is_better"] is True


class TestBuildDataQuality:
    """Tests for MetricsRunner._build_data_quality splitting error vs missing records."""

    def test_errors_and_missing_records_split(self):
        """Records with errors and missing metrics are tracked separately."""
        all_metrics = {
            "r1": RecordMetrics(record_id="r1", metrics={"m": _ms("m", 0.8)}),
            "r2": RecordMetrics(record_id="r2", metrics={"m": _ms("m", 0.0, error="fail")}),
            "r3": RecordMetrics(record_id="r3", metrics={}),
        }
        metric_aggregates = {
            "m": {
                "count": 1,
                "none_count": 2,
                "error_count": 1,
                "missing_count": 1,
            }
        }
        dq = MetricsRunner._build_data_quality(all_metrics, metric_aggregates)

        assert dq["records_with_any_none"] == 2
        assert dq["records_with_errors"] == 1
        assert dq["records_with_missing"] == 1
        assert dq["metrics_with_none_scores"] == {"m": 2}
        assert dq["metrics_with_errors"] == {"m": 1}
        assert dq["metrics_with_missing"] == {"m": 1}

    def test_no_issues(self):
        """All records successful — no errors or missing."""
        all_metrics = {
            "r1": RecordMetrics(record_id="r1", metrics={"m": _ms("m", 0.8)}),
        }
        metric_aggregates = {"m": {"count": 1, "none_count": 0, "error_count": 0, "missing_count": 0}}
        dq = MetricsRunner._build_data_quality(all_metrics, metric_aggregates)

        assert dq["records_with_any_none"] == 0
        assert dq["records_with_errors"] == 0
        assert dq["records_with_missing"] == 0
        assert dq["metrics_with_none_scores"] == {}
        assert dq["metrics_with_errors"] == {}
        assert dq["metrics_with_missing"] == {}
