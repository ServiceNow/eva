"""Tests for vad_turn_metrics: config gating, file parsing, turn classification, aggregation."""

import json

import pytest

from eva.metrics.experience.vad_end_of_turn import (
    _classify_turns,
    _find_run_config,
    _load_dispatched_user_turns,
    _load_stop_secs_silence_ms,
    _load_turn_fallback_timestamps,
    _load_turn_metrics_events,
    _load_turn_start_timestamps,
    _load_vad_events,
    _transcript_for_vad_stop,
    compute_vad_turn_sub_metrics,
    vad_turn_metrics_applicable,
)
from eva.utils.conversation_correctly_finished.final_turn import final_turn_input_flags

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


class TestLoadTurnStartTimestamps:
    def test_extracts_turn_start_timestamps(self, tmp_path):
        _write_jsonl(
            tmp_path / "pipecat_logs.jsonl",
            [
                {"timestamp": 1000, "type": "turn_start"},
                {"timestamp": 1050, "type": "turn_end"},
                {"timestamp": 5000, "type": "turn_start"},
                {"timestamp": 5000, "type": "user_started_speaking"},
            ],
        )
        assert _load_turn_start_timestamps(str(tmp_path)) == {1000, 5000}

    def test_empty_when_file_missing(self, tmp_path):
        assert _load_turn_start_timestamps(str(tmp_path)) == set()


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


def _write_audit_log(path, user_turns):
    """Write an audit_log.json whose transcript holds the given (timestamp, text) user turns."""
    path.write_text(
        json.dumps(
            {
                "transcript": [
                    {"value": text, "message_type": "user", "timestamp": str(ts), "isBotMessage": False}
                    for ts, text in user_turns
                ]
            }
        )
    )


class TestLoadDispatchedUserTurns:
    def test_extracts_and_sorts_user_turns_with_string_timestamps(self, tmp_path):
        _write_audit_log(tmp_path / "audit_log.json", [(5000, "second"), (1000, "first")])
        assert _load_dispatched_user_turns(str(tmp_path)) == [(1000, "first"), (5000, "second")]

    def test_ignores_assistant_entries(self, tmp_path):
        (tmp_path / "audit_log.json").write_text(
            json.dumps(
                {
                    "transcript": [
                        {"value": "user text", "message_type": "user", "timestamp": "1000"},
                        {"value": "bot text", "message_type": "assistant", "timestamp": "2000"},
                    ]
                }
            )
        )
        assert _load_dispatched_user_turns(str(tmp_path)) == [(1000, "user text")]

    def test_empty_when_audit_log_missing(self, tmp_path):
        assert _load_dispatched_user_turns(str(tmp_path)) == []

    def test_empty_when_audit_log_malformed(self, tmp_path):
        (tmp_path / "audit_log.json").write_text("{not json")
        assert _load_dispatched_user_turns(str(tmp_path)) == []

    def test_skips_entries_with_unusable_timestamp_or_empty_text(self, tmp_path):
        (tmp_path / "audit_log.json").write_text(
            json.dumps(
                {
                    "transcript": [
                        {"value": "", "message_type": "user", "timestamp": "1000"},
                        {"value": "no timestamp", "message_type": "user"},
                        {"value": "bad timestamp", "message_type": "user", "timestamp": "not-a-number"},
                        {"value": "good", "message_type": "user", "timestamp": "4000"},
                    ]
                }
            )
        )
        assert _load_dispatched_user_turns(str(tmp_path)) == [(4000, "good")]


class TestLoadTurnFallbackTimestamps:
    def test_extracts_and_sorts_fallback_timestamps(self, tmp_path):
        (tmp_path / "audit_log.json").write_text(
            json.dumps(
                {
                    "transcript": [
                        {"value": "second fallback", "message_type": "turn_fallback", "timestamp": "5000"},
                        {"value": "first fallback", "message_type": "turn_fallback", "timestamp": "1000"},
                        {"value": "user text", "message_type": "user", "timestamp": "2000"},
                    ]
                }
            )
        )
        assert _load_turn_fallback_timestamps(str(tmp_path)) == [1000, 5000]

    def test_empty_when_audit_log_missing(self, tmp_path):
        assert _load_turn_fallback_timestamps(str(tmp_path)) == []

    def test_skips_entries_with_unusable_timestamp(self, tmp_path):
        (tmp_path / "audit_log.json").write_text(
            json.dumps(
                {
                    "transcript": [
                        {"value": "bad", "message_type": "turn_fallback", "timestamp": "not-a-number"},
                        {"value": "good", "message_type": "turn_fallback", "timestamp": "3000"},
                    ]
                }
            )
        )
        assert _load_turn_fallback_timestamps(str(tmp_path)) == [3000]


class TestTranscriptForVadStop:
    def test_matches_turn_dispatched_just_after_the_stop(self):
        # Measured against real fixtures: dispatch lands 3-8ms after the closing vad_stop.
        turns = [(1056, "first turn"), (5104, "second turn")]
        assert _transcript_for_vad_stop(turns, 1050) == "first turn"
        assert _transcript_for_vad_stop(turns, 5100) == "second turn"

    def test_returns_nearest_when_several_are_within_tolerance(self):
        assert _transcript_for_vad_stop([(1400, "far"), (1005, "near")], 1000) == "near"

    def test_none_when_no_turn_is_close_enough(self):
        # A window whose utterance was never dispatched must not borrow a distant turn's text.
        assert _transcript_for_vad_stop([(30000, "much later turn")], 1050) is None

    def test_none_when_no_dispatched_turns(self):
        assert _transcript_for_vad_stop([], 1050) is None


class TestClassifyTurns:
    def test_forced_turn_text_is_the_full_dispatched_turn_not_the_last_stt_fragment(self):
        # Reproduces the nonkrisp_2 shape. Window 0 is [1000, 26000) and window 1 is [26000, None).
        # The user's second utterance began before VAD registered it, so its STT fragments
        # finalized at 24000/25000 -- inside window 0, whose own utterance ended back at 6000.
        # Attribution must follow the vad_stop that closed each turn, not the fragment timestamps,
        # and must report the whole dispatched turn rather than only its trailing fragment.
        dispatched = [
            (6005, "Hi. I need help resetting my VPN password."),
            (32004, "My employee ID is e n p zero six six. The last four of my phone are four four two five."),
        ]
        result = _classify_turns(
            vad_starts=[1000, 26000],
            vad_stops=[6000, 32000],
            turn_metrics_events=[],
            stop_secs_silence_ms=[3000.0, 3000.0],
            dispatched_user_turns=dispatched,
        )
        assert [r["completion"] for r in result] == ["forced", "forced"]
        assert result[0]["transcript_text"] == "Hi. I need help resetting my VPN password."
        assert result[1]["transcript_text"] == (
            "My employee ID is e n p zero six six. The last four of my phone are four four two five."
        )

    def test_forced_turn_omits_text_when_no_dispatched_turn_matches(self):
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[1050],
            turn_metrics_events=[],
            stop_secs_silence_ms=[3000.0],
            dispatched_user_turns=[(90000, "unrelated later turn")],
        )
        assert result[0]["completion"] == "forced"
        assert "transcript_text" not in result[0]
        assert "final_turn_flags" not in result[0]

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
        # No natural completion in the window -> consumes the one stop_secs silence value
        # to confirm the fallback fired, but close_time = last vad_stop in window (1050)
        # directly - the silence value is NOT added again (it's already baked into when
        # last_stop itself was emitted).
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[1050],
            turn_metrics_events=[{"timestamp": 1100, "is_complete": False, "e2e_processing_time_ms": 105.0}],
            stop_secs_silence_ms=[3032.0],
        )
        assert result == [
            {"turn_index": 0, "open_duration_ms": 50, "time_to_complete_ms": None, "completion": "forced"}
        ]

    def test_stuck_when_no_natural_or_forced_signal(self):
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[],
            turn_metrics_events=[],
            stop_secs_silence_ms=[],
        )
        assert result == [{"turn_index": 0, "open_duration_ms": 0, "time_to_complete_ms": None, "completion": "stuck"}]

    def test_dispatched_when_a_real_llm_turn_landed_in_an_otherwise_signal_less_window(self):
        # Reproduces example: the user-simulator's audio bridge ended the
        # session mid-turn, so pipecat's VADController force-stopped speech itself ("no
        # audio received while speaking") instead of the turn analyzer's is_complete or
        # stop_secs mechanisms ever firing. Without checking audit_log.json this window
        # would wrongly read as "stuck" even though the LLM genuinely got the turn and
        # responded.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[5908],
            turn_metrics_events=[{"timestamp": 4892, "is_complete": False, "e2e_processing_time_ms": 104.0}],
            stop_secs_silence_ms=[],
            dispatched_user_turns=[(5912, "I do not need anything else today. Bye.")],
        )
        assert result[0]["completion"] == "dispatched"
        assert result[0]["open_duration_ms"] == 5912 - 1000
        assert result[0]["time_to_complete_ms"] is None
        assert result[0]["transcript_text"] == "I do not need anything else today. Bye."
        assert result[0]["final_turn_flags"] == final_turn_input_flags("I do not need anything else today. Bye.")

    def test_dispatched_not_used_when_forced_stop_secs_already_matched(self):
        # A window that legitimately earns the stop_secs fallback must stay "forced", not
        # get reclassified just because a dispatched turn also happens to land inside it.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[1050],
            turn_metrics_events=[],
            stop_secs_silence_ms=[3032.0],
            dispatched_user_turns=[(1056, "some turn")],
        )
        assert result[0]["completion"] == "forced"

    def test_stuck_when_no_dispatched_turn_falls_inside_the_window_either(self):
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[],
            stop_secs_silence_ms=[],
            dispatched_user_turns=[(90000, "unrelated later turn")],
        )
        assert result[0]["completion"] == "stuck"

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
            {"turn_index": 1, "open_duration_ms": 100.0, "time_to_complete_ms": None, "completion": "forced"},
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

    def test_window_with_two_naturals_splits_into_two_turns_instead_of_dropping_the_second(self):
        # Real Krisp data shape: a single raw VAD-start window contains two genuine
        # completions for two separate real user turns (Krisp's own start event for the
        # second turn fired late/not at all, so both completions landed in the first
        # window). Both must survive as their own "natural" entries, not just the first.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[],
            turn_metrics_events=[
                {"timestamp": 1001, "is_complete": True, "e2e_processing_time_ms": 188.9},
                {"timestamp": 4680, "is_complete": True, "e2e_processing_time_ms": 268.7},
            ],
            stop_secs_silence_ms=[],
        )
        assert result == [
            {"turn_index": 0, "open_duration_ms": 1, "time_to_complete_ms": 188.9, "completion": "natural"},
            {"turn_index": 1, "open_duration_ms": 3679, "time_to_complete_ms": 268.7, "completion": "natural"},
        ]

    def test_turn_index_accounts_for_earlier_split_windows(self):
        # A split in an earlier window must push every later window's turn_index forward.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[
                {"timestamp": 1001, "is_complete": True, "e2e_processing_time_ms": 100.0},
                {"timestamp": 2000, "is_complete": True, "e2e_processing_time_ms": 100.0},
                {"timestamp": 5100, "is_complete": True, "e2e_processing_time_ms": 100.0},
            ],
            stop_secs_silence_ms=[],
        )
        assert [r["turn_index"] for r in result] == [0, 1, 2]
        assert result[2]["open_duration_ms"] == 100

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
        assert result[1]["open_duration_ms"] == 5100 - 5000
        assert result[1]["time_to_complete_ms"] is None

    def test_forced_completion_does_not_double_count_silence_ms(self):
        # Regression guard, values taken from an example (real Smart Turn run):
        # vad_start=1787171562461, vad_stop=1787171568321, stop_secs silence=3032.0,
        # and the LLM-dispatched turn in audit_log.json landed at 1787171568327 - only
        # 6ms after vad_stop. That means vad_stop (user_stopped_speaking) already fires
        # only once the analyzer's internal silence_ms wait is over, so close_time must
        # be ~vad_stop, not ~vad_stop + silence_ms (which would overshoot the real
        # dispatch time by ~3032ms - one whole extra stop_secs timeout).
        result = _classify_turns(
            vad_starts=[1787171562461],
            vad_stops=[1787171568321],
            turn_metrics_events=[],
            stop_secs_silence_ms=[3032.0],
        )
        assert result[0]["completion"] == "forced"
        assert result[0]["open_duration_ms"] == 1787171568321 - 1787171562461

    def test_natural_completion_reclassified_early_when_next_turn_never_started(self):
        # Reproduces example: the turn analyzer reported is_complete=True on
        # "Sure. My employee ID is emp zero four eight two seven one." but the user was still
        # mid-utterance. Real evidence of that: pipecat's own TurnTrackingObserver never
        # logged a turn_start for the very next VAD-start window (5000) - it stayed inside
        # the same ongoing turn rather than starting a fresh one - and that next window went
        # on to complete naturally itself ("Last four of my phone number are seven two nine
        # four."), confirming the pair is really one continued utterance.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[
                {"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0},
                {"timestamp": 5103, "is_complete": True, "e2e_processing_time_ms": 100.0},
            ],
            stop_secs_silence_ms=[],
            turn_start_timestamps={1000},
        )
        assert result[0]["completion"] == "early"
        assert result[1]["completion"] == "natural"

    def test_natural_completion_stays_natural_when_next_turn_started(self):
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[
                {"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0},
                {"timestamp": 5103, "is_complete": True, "e2e_processing_time_ms": 100.0},
            ],
            stop_secs_silence_ms=[],
            turn_start_timestamps={1000, 5000},
        )
        assert result[0]["completion"] == "natural"

    def test_natural_completion_stays_natural_when_next_window_is_stuck(self):
        # A missing turn_start alone is not enough - it also shows up ahead of an ordinary
        # Krisp VAD false-start/blip window that never resolves at all ("stuck"), which is
        # already covered by stuck_rate and must not be conflated with a early split.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0}],
            stop_secs_silence_ms=[],
            turn_start_timestamps={1000},
        )
        assert result[0]["completion"] == "natural"
        assert result[1]["completion"] == "stuck"

    def test_forced_completion_reclassified_early_when_next_turn_never_started(self):
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[1050, 5100],
            turn_metrics_events=[],
            stop_secs_silence_ms=[3032.0, 3032.0],
            turn_start_timestamps={1000},
        )
        assert result[0]["completion"] == "early"
        assert result[1]["completion"] == "forced"

    def test_last_window_never_reclassified_early_with_no_following_window(self):
        # window_end is None for the final, open-ended window - there is no "next" VAD-start
        # to check turn_start against, so the natural completion stands.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0}],
            stop_secs_silence_ms=[],
            turn_start_timestamps={1000},
        )
        assert result[0]["completion"] == "natural"

    def test_empty_turn_start_timestamps_disables_early_detection(self):
        # An empty set means this run's pipeline never emits turn_start at all - that must
        # read as "signal unavailable", not "every transition is early".
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0}],
            stop_secs_silence_ms=[],
            turn_start_timestamps=set(),
        )
        assert result[0]["completion"] == "natural"

    def test_natural_completion_followed_by_stuck_fallback_is_not_dropped(self):
        # Reproduces example: window 0 closes an earlier, unrelated turn
        # naturally at 1103, but the turn analyzer's VAD never fires again until window 1's
        # own vad_start at 60000 - a genuinely separate turn got stuck in between and only
        # got resolved via the 20s turn-end fallback nudge (audit_log's turn_fallback marker,
        # timestamped mid-gap at 45000). That stuck turn must not be silently swallowed just
        # because the window already had an earlier, unrelated natural completion.
        result = _classify_turns(
            vad_starts=[1000, 60000],
            vad_stops=[],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0}],
            stop_secs_silence_ms=[],
            turn_fallback_timestamps=[45000],
        )
        assert result[0]["completion"] == "natural"
        assert result[1] == {
            "turn_index": 1,
            "open_duration_ms": 60000 - 1103,
            "time_to_complete_ms": None,
            "completion": "stuck",
        }

    def test_natural_completion_with_no_trailing_fallback_marker_stays_a_single_entry(self):
        # Ordinary case: a natural completion followed only by normal inter-turn silence
        # (assistant response + user think time) before the next window's vad_start - with
        # no turn_fallback marker as evidence a turn got stuck in the gap, no trailing entry
        # should be fabricated for window 0 itself. Otherwise every ordinary turn transition
        # would be misread as containing a stuck turn. (Window 1, the next raw VAD-start
        # window, legitimately resolves "stuck" on its own since it has no signal at all -
        # that's unrelated to this fix.)
        result = _classify_turns(
            vad_starts=[1000, 60000],
            vad_stops=[],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0}],
            stop_secs_silence_ms=[],
            turn_fallback_timestamps=[],
        )
        assert result[0] == {
            "turn_index": 0,
            "open_duration_ms": 103,
            "time_to_complete_ms": 103.0,
            "completion": "natural",
        }
        assert len(result) == 2
        assert result[1]["completion"] == "stuck"

    def test_natural_completion_followed_by_stuck_fallback_in_open_ended_window(self):
        # Same shape as above but in the run's final, open-ended window (window_end=None):
        # close_time must fall back to the fallback marker's own timestamp rather than a
        # nonexistent next-window bound.
        result = _classify_turns(
            vad_starts=[1000],
            vad_stops=[],
            turn_metrics_events=[{"timestamp": 1103, "is_complete": True, "e2e_processing_time_ms": 103.0}],
            stop_secs_silence_ms=[],
            turn_fallback_timestamps=[45000],
        )
        assert result[1] == {
            "turn_index": 1,
            "open_duration_ms": 45000 - 1103,
            "time_to_complete_ms": None,
            "completion": "stuck",
        }

    def test_only_last_split_segment_in_a_multi_natural_window_can_be_early(self):
        # Two naturals split out of one raw window. Only the second (the one bordering the
        # next VAD-start window) is eligible for reclassification - the first is followed by
        # another natural completion in the same window, not a missing turn_start.
        result = _classify_turns(
            vad_starts=[1000, 5000],
            vad_stops=[],
            turn_metrics_events=[
                {"timestamp": 1001, "is_complete": True, "e2e_processing_time_ms": 100.0},
                {"timestamp": 2000, "is_complete": True, "e2e_processing_time_ms": 100.0},
                {"timestamp": 5103, "is_complete": True, "e2e_processing_time_ms": 100.0},
            ],
            stop_secs_silence_ms=[],
            turn_start_timestamps={1000},
        )
        assert result[0]["completion"] == "natural"
        assert result[1]["completion"] == "early"
        assert result[2]["completion"] == "natural"


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

    def test_dispatched_turn_excluded_from_stuck_rate(self, tmp_path):
        # Reproduces example: no TurnMetricsData is_complete=True and no
        # stop_secs line for the final window, but audit_log.json proves the LLM was
        # dispatched and responded - so this must not count toward stuck_rate.
        _write_config(tmp_path, turn_stop_strategy="turn_analyzer")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 4892,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": False, "e2e_processing_time_ms": 104.0},
                }
            ],
        )
        _write_audit_log(tmp_path / "audit_log.json", [(5912, "I do not need anything else today. Bye.")])
        ctx = make_metric_context(output_dir=str(tmp_path))
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert per_turn[0]["completion"] == "dispatched"
        assert sub_metrics["stuck_rate"].score == 0.0

    def test_stuck_rate_counts_mid_conversation_hangs_regardless_of_outcome(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="krisp_viva_turn")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        ctx = make_metric_context(
            output_dir=str(tmp_path),
            conversation_ended_reason="goodbye",  # the call still ended cleanly
            audio_timestamps_user_turns={1: [(0.0, 5.0)]},
            audio_timestamps_assistant_turns={},
        )
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert per_turn[0]["completion"] == "stuck"
        assert sub_metrics["stuck_rate"].score == 1.0

    def test_stuck_rate_includes_final_turn_with_no_vad_start_at_all(self, tmp_path):
        # Real Krisp shape: 5 real turns all completed naturally, but the 6th and final
        # real user turn never got a user_started_speaking event from the turn analyzer at
        # all - no window exists for it in per_turn, so it must be folded in explicitly to
        # be counted, rather than silently vanishing from stuck_rate's denominator.
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
        ctx = make_metric_context(
            output_dir=str(tmp_path),
            conversation_ended_reason="inactivity_timeout",
            # Last real user turn started at 30s (30000ms) - over 28s away from the only
            # vad_start (1000ms), far past legitimate clock skew.
            audio_timestamps_user_turns={1: [(0.0, 0.5)], 2: [(30.0, 30.5)]},
            audio_timestamps_assistant_turns={0: [(0.6, 1.0)]},
        )
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert per_turn[0]["completion"] == "natural"
        assert per_turn[-1] == {
            "turn_index": 1,
            "open_duration_ms": None,
            "time_to_complete_ms": None,
            "completion": "stuck",
            "vad_never_started": True,
        }
        # 1 stuck (the synthetic missing-turn entry) out of 2 total turns.
        assert sub_metrics["stuck_rate"].score == 0.5

    def test_stuck_rate_does_not_add_synthetic_entry_within_clock_skew_tolerance(self, tmp_path):
        # The only vad_start is within clock-skew tolerance of the last real user turn, so
        # the synthetic missing-turn entry is NOT added - this call had exactly one real
        # turn, and it's classified stuck for an unrelated reason (no completion signal),
        # not because VAD never started at all.
        _write_config(tmp_path, turn_stop_strategy="krisp_viva_turn")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        ctx = make_metric_context(
            output_dir=str(tmp_path),
            conversation_ended_reason="inactivity_timeout",
            audio_timestamps_user_turns={1: [(0.0, 0.5)]},
            audio_timestamps_assistant_turns={},
        )
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert per_turn == [
            {"turn_index": 0, "open_duration_ms": 0, "time_to_complete_ms": None, "completion": "stuck"}
        ]
        assert sub_metrics["stuck_rate"].score == 1.0

    def test_early_detection_rate_reported_when_a_natural_completion_is_undone(self, tmp_path):
        _write_config(tmp_path, turn_stop_strategy="turn_analyzer")
        _write_jsonl(
            tmp_path / "pipecat_logs.jsonl",
            [
                {"timestamp": 1000, "type": "turn_start"},
                {"timestamp": 1000, "type": "user_started_speaking"},
                {"timestamp": 5000, "type": "user_started_speaking"},
                # No turn_start at 5000: pipecat kept the turn open through the split.
                {"timestamp": 9000, "type": "turn_start"},
                {"timestamp": 9000, "type": "user_started_speaking"},
            ],
        )
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 1103,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 103.0},
                },
                {
                    "timestamp": 5103,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 103.0},
                },
                {
                    "timestamp": 9103,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 103.0},
                },
            ],
        )
        ctx = make_metric_context(output_dir=str(tmp_path))
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert [t["completion"] for t in per_turn] == ["early", "natural", "natural"]
        assert sub_metrics["early_detection_rate"].score == pytest.approx(1 / 3, abs=1e-4)

    def test_early_detection_rate_absent_when_no_turn_start_signal_available(self, tmp_path):
        # No turn_start events recorded at all - early detection has no signal to work
        # with, so it must not fabricate a rate off an empty turn_start set.
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
        assert per_turn[0]["completion"] == "natural"
        assert sub_metrics["early_detection_rate"].score == 0.0

    def test_stuck_rate_counts_fallback_stuck_turn_hidden_behind_earlier_natural_completion(self, tmp_path):
        # Reproduces example: window 0 has an earlier,
        # unrelated natural completion, but the turn analyzer's VAD then went stuck on a
        # separate turn that only got resolved via the 20s turn-end fallback nudge -
        # recorded in audit_log.json as a turn_fallback entry, not a real "user" dispatch.
        # Must count toward stuck_rate rather than vanishing because the window already
        # resolved once.
        _write_config(tmp_path, turn_stop_strategy="turn_analyzer")
        _write_jsonl(tmp_path / "pipecat_logs.jsonl", [{"timestamp": 1000, "type": "user_started_speaking"}])
        _write_jsonl(
            tmp_path / "pipecat_metrics.jsonl",
            [
                {
                    "timestamp": 1103,
                    "type": "TurnMetricsData",
                    "value": {"is_complete": True, "e2e_processing_time_ms": 103.0},
                }
            ],
        )
        (tmp_path / "audit_log.json").write_text(
            json.dumps(
                {
                    "transcript": [
                        {
                            "value": "[TURN-END FALLBACK after 20s] partial user speech: 'Mac fourteen inch.'",
                            "message_type": "turn_fallback",
                            "timestamp": "45000",
                        }
                    ]
                }
            )
        )
        ctx = make_metric_context(output_dir=str(tmp_path))
        result = compute_vad_turn_sub_metrics(ctx)
        assert result is not None
        sub_metrics, per_turn = result
        assert [t["completion"] for t in per_turn] == ["natural", "stuck"]
        assert sub_metrics["stuck_rate"].score == 0.5

    def test_stuck_rate_zero_on_clean_all_natural_ending(self, tmp_path):
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
        assert sub_metrics["stuck_rate"].score == 0.0
