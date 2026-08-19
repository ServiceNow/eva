"""Tests for vad_turn_metrics: config gating, file parsing, turn classification, aggregation."""

import json

from eva.metrics.experience.vad_end_of_turn import (
    _classify_turns,
    _find_run_config,
    _load_stop_secs_silence_ms,
    _load_turn_metrics_events,
    _load_vad_events,
    compute_vad_turn_sub_metrics,
    vad_turn_metrics_applicable,
)

from .conftest import make_metric_context


class TestFindRunConfig:
    def test_finds_config_in_output_dir(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"framework": "pipecat"}))
        assert _find_run_config(str(tmp_path)) == {"framework": "pipecat"}

    def test_finds_config_by_walking_up_from_nested_record_dir(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"framework": "pipecat"}))
        record_dir = tmp_path / "records" / "1.2.3"
        record_dir.mkdir(parents=True)
        assert _find_run_config(str(record_dir)) == {"framework": "pipecat"}

    def test_returns_none_when_no_config_json_found(self, tmp_path):
        assert _find_run_config(str(tmp_path)) is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{not valid json")
        assert _find_run_config(str(tmp_path)) is None

    def test_returns_none_for_empty_output_dir(self):
        assert _find_run_config("") is None


class TestVadTurnMetricsApplicable:
    def test_true_for_krisp_turn_stop_strategy(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"framework": "pipecat", "model": {"turn_stop_strategy": "krisp_viva_turn"}})
        )
        assert vad_turn_metrics_applicable(str(tmp_path)) is True

    def test_false_for_non_pipecat_framework(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"framework": "elevenlabs", "model": {}}))
        assert vad_turn_metrics_applicable(str(tmp_path)) is False

    def test_false_for_external_turn_stop_strategy(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"framework": "pipecat", "model": {"turn_stop_strategy": "external", "vad": "none"}})
        )
        assert vad_turn_metrics_applicable(str(tmp_path)) is False

    def test_false_when_no_config_found(self, tmp_path):
        assert vad_turn_metrics_applicable(str(tmp_path)) is False

    def test_defaults_to_turn_analyzer_when_field_absent(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"framework": "pipecat", "model": {}}))
        assert vad_turn_metrics_applicable(str(tmp_path)) is True


def _write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


class TestLoadTurnMetricsEvents:
    def test_extracts_and_sorts_turn_metrics_data(self, tmp_path):
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 200,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 100.0},
                },
                {
                    "timestamp": 100,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": False, "e2e_processing_time_ms": 50.0},
                },
                {"timestamp": 150, "type": "TTFBMetricsData", "value": {"ttfb": 1.0}},
            ],
        )
        events = _load_turn_metrics_events(str(tmp_path))
        assert events == [
            {"timestamp": 100, "is_complete": False, "e2e_processing_time_ms": 50.0},
            {"timestamp": 200, "is_complete": True, "e2e_processing_time_ms": 100.0},
        ]


class TestLoadVadEvents:
    def test_extracts_and_sorts_start_and_stop_events(self, tmp_path):
        _write_jsonl(
            tmp_path / "pipecat_logs.jsonl",
            [
                {"timestamp": 300, "type": "user_started_speaking"},
                {"timestamp": 100, "type": "user_started_speaking"},
                {"timestamp": 150, "type": "user_stopped_speaking"},
                {"timestamp": 999, "type": "turn_start"},
            ],
        )
        starts, stops = _load_vad_events(str(tmp_path))
        assert starts == [100, 300]
        assert stops == [150]

    def test_empty_when_file_missing(self, tmp_path):
        assert _load_vad_events(str(tmp_path)) == ([], [])


class TestLoadStopSecsSilenceMs:
    def test_extracts_silence_values_in_file_order(self, tmp_path):
        (tmp_path / "logs.log").write_text(
            "2026-08-04 19:31:15,912 | DEBUG    | pipecat:133 | End of Turn complete due to stop_secs. "
            "Silence in ms: 3032.0\n"
            "2026-08-04 19:32:51,140 | DEBUG    | pipecat:133 | End of Turn complete due to stop_secs. "
            "Silence in ms: 3105.5\n"
        )
        assert _load_stop_secs_silence_ms(str(tmp_path)) == [3032.0, 3105.5]

    def test_empty_when_no_stop_secs_lines(self, tmp_path):
        (tmp_path / "logs.log").write_text(
            "2026-08-04 19:31:15,912 | DEBUG | pipecat:166 | End of Turn result: EndOfTurnState.COMPLETE\n"
        )
        assert _load_stop_secs_silence_ms(str(tmp_path)) == []


class TestClassifyTurns:
    def test_single_natural_completion(self):
        # Window: [1000, None). TurnMetricsData at 1103 (is_complete=True) closes it naturally.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[1050],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.2}],
            stop_secs_silence_ms=[],
        )
        assert result == [
            {"turn_index": 0, "open_duration_ms": 103, "time_to_complete_ms": 103.2, "completion": "natural"}
        ]

    def test_incomplete_evaluations_before_natural_completion_are_ignored(self):
        # Matches real Smart Turn data: several is_complete=False evaluations, then True.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[1050],
            turn_metrics_events=[
                {"timestamp": 1100, "is_complete": False, "e2e_processing_time_ms": 105.0},
                {"timestamp": 1200, "is_complete": False, "e2e_processing_time_ms": 110.0},
                {"timestamp": 1300, "is_complete": True, "e2e_processing_time_ms": 108.0},
            ],
            stop_secs_silence_ms=[],
        )
        assert result == [
            {"turn_index": 0, "open_duration_ms": 300, "time_to_complete_ms": 108.0, "completion": "natural"}
        ]

    def test_forced_completion_via_stop_secs_ordinal_match(self):
        # No natural completion in the window -> consumes the one stop_secs silence value.
        # close_time = last vad_stop in window (1050) + silence_ms (3032.0) = 4082.0
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[1050],
            turn_metrics_events=[{"timestamp": 1100, "is_complete": False, "e2e_processing_time_ms": 105.0}],
            stop_secs_silence_ms=[3032.0],
        )
        assert result == [
            {"turn_index": 0, "open_duration_ms": 3082.0, "time_to_complete_ms": None, "completion": "forced"}
        ]

    def test_stuck_when_no_natural_or_forced_signal(self):
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[],
            turn_metrics_events=[],
            stop_secs_silence_ms=[],
        )
        assert result == [{"turn_index": 0, "open_duration_ms": 0, "time_to_complete_ms": None, "completion": "stuck"}]

    def test_multiple_turns_ordinal_stop_secs_matching(self):
        # Turn 0: natural completion. Turn 1: no natural completion -> gets the stop_secs value.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[1050, 5100],
            turn_metrics_events=[
                {"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0},
            ],
            stop_secs_silence_ms=[3000.0],
        )
        assert result == [
            {"turn_index": 0, "open_duration_ms": 103, "time_to_complete_ms": 103.0, "completion": "natural"},
            {"turn_index": 1, "open_duration_ms": 3100.0, "time_to_complete_ms": None, "completion": "forced"},
        ]

    def test_stuck_last_window_open_duration_uses_last_observed_epoch(self):
        # Turn 1 (last, open-ended window) is stuck; open_duration_ms measured to the
        # latest timestamp seen anywhere (a proxy for conversation end), since there's
        # no next VAD start to bound it.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[1050, 5100],
            turn_metrics_events=[
                {"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0},
            ],
            stop_secs_silence_ms=[],
        )
        assert result[1] == {
            "turn_index": 1,
            "open_duration_ms": 5100 - 5000,
            "time_to_complete_ms": None,
            "completion": "stuck",
        }

    def test_stuck_middle_window_bounded_by_next_vad_start_not_global_last_epoch(self):
        # Turn 0 is stuck (no natural/forced signal) but has a KNOWN window_end (turn 1's
        # start) - its open_duration_ms must use that bound, not the conversation's global
        # last-observed timestamp (which is much later here, at turn 2's natural completion).
        # A real Krisp example fixture showed this exact shape: a middle window with no
        # natural completion, followed much later by other turns finishing normally.
        result = _classify_turns(
            vad_starts=[1000, 1050, 50000],
            vad_stops=[1040],
            turn_metrics_events=[
                {"timestamp": 50103, "is_complete": True, "e2e_processing_time_ms": 103.0},
            ],
            stop_secs_silence_ms=[],
        )
        assert result[0] == {
            "turn_index": 0,
            "open_duration_ms": 1050 - 1000,
            "time_to_complete_ms": None,
            "completion": "stuck",
        }

    def test_false_start_does_not_steal_stop_secs_from_later_genuinely_forced_window(self):
        # Window 0: brief VAD false-start - a start immediately superseded by the next
        # start, no user_stopped_speaking ever registered, no natural completion. Since
        # Smart Turn's stop_secs mechanism only ever fires after VAD registers a stop, this
        # window cannot legitimately be the source of a forced completion.
        # Window 1 (last, open-ended): has a real user_stopped_speaking at 5100 and no
        # natural completion - this is the genuinely forced window and should get the
        # single available stop_secs silence value.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[5100],
            turn_metrics_events=[],
            stop_secs_silence_ms=[3032.0],
        )
        assert result[0]["completion"] == "stuck"
        assert result[1]["completion"] == "forced"
        assert result[1]["open_duration_ms"] == 5100 + 3032.0 - 5000
        assert result[1]["time_to_complete_ms"] is None


def _write_config(tmp_path, framework="pipecat", turn_stop_strategy="turn_analyzer"):
    (tmp_path / "config.json").write_text(
        json.dumps({"framework": framework, "model": {"turn_stop_strategy": turn_stop_strategy}})
    )


class TestComputeVadTurnSubMetrics:
    def test_none_when_not_applicable(self, tmp_path):
        _write_config(tmp_path, framework="elevenlabs")
        ctx = make_metric_context(output_dir=str(tmp_path))
        assert compute_vad_turn_sub_metrics(ctx) is None

    def test_none_when_no_vad_starts_recorded(self, tmp_path):
        _write_config(tmp_path)
        ctx = make_metric_context(output_dir=str(tmp_path))
        assert compute_vad_turn_sub_metrics(ctx) is None

    def test_krisp_run_all_natural_omits_forced_completion_rate(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="krisp_viva_turn")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 1100,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 100.0},
                }
            ],
        )
        ctx = make_metric_context(output_dir=str(tmp_path))
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert "forced_completion_rate" not in sub_metrics
        assert sub_metrics["mean_time_to_complete_ms"].score == 100.0
        assert per_turn == [
            {"turn_index": 0, "open_duration_ms": 100, "time_to_complete_ms": 100.0, "completion": "natural"}
        ]

    def test_smart_turn_run_all_natural_reports_zero_forced_completion_rate(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="turn_analyzer")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 1100,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 100.0},
                }
            ],
        )
        ctx = make_metric_context(output_dir=str(tmp_path))
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert "forced_completion_rate" in sub_metrics
        assert sub_metrics["forced_completion_rate"].score == 0.0
        assert sub_metrics["mean_time_to_complete_ms"].score == 100.0
        assert per_turn == [
            {"turn_index": 0, "open_duration_ms": 100, "time_to_complete_ms": 100.0, "completion": "natural"}
        ]

    def test_smart_turn_run_with_forced_completion(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="turn_analyzer")
        _write_jsonl(
            tmp_path / "pipecat_logs.jsonl",
            [
                {"timestamp": 1000, "type": "user_started_speaking"},
                {"timestamp": 1050, "type": "user_stopped_speaking"},
                {"timestamp": 5000, "type": "user_started_speaking"},
                {"timestamp": 5103, "type": "user_stopped_speaking"},
            ],
        )
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 5203,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 100.0},
                }
            ],
        )
        (tmp_path / "logs.log").write_text(
            "2026-08-04 19:31:15,912 | DEBUG | pipecat:133 | End of Turn complete due to stop_secs. Silence in ms: 3032.0\n"
        )
        ctx = make_metric_context(output_dir=str(tmp_path))
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert sub_metrics["forced_completion_rate"].score == 0.5
        assert per_turn[0]["completion"] == "forced"
        assert per_turn[1]["completion"] == "natural"
        # Only the natural completion feeds mean_time_to_complete_ms.
        assert sub_metrics["mean_time_to_complete_ms"].score == 100.0

    def test_eot_vad_not_fired_rate_true_on_stuck_plus_inactivity_timeout(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="krisp_viva_turn")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        ctx = make_metric_context(
            output_dir=str(tmp_path),
            conversation_ended_reason="inactivity_timeout",
            audio_timestamps_user_turns={1: [(0.0, 5.0)]},
            audio_timestamps_assistant_turns={},
        )
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert per_turn[0]["completion"] == "stuck"
        assert sub_metrics["eot_vad_not_fired_rate"].score == 1.0

    def test_eot_vad_not_fired_rate_false_on_clean_ending(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="krisp_viva_turn")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 1100,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 100.0},
                }
            ],
        )
        ctx = make_metric_context(output_dir=str(tmp_path), conversation_ended_reason="goodbye")
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, _ = result
        assert sub_metrics["eot_vad_not_fired_rate"].score == 0.0
