"""Tests for reconstructing STT latency from ink-2 committed-turn times."""

import json

import pytest

from eva.utils.stt_turn_latency import compute_turn_commit_latencies


def _write_events(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _audio(kind, who, ts):
    return {"event_type": kind, "user": who, "audio_timestamp": ts}


class TestComputeTurnCommitLatencies:
    def test_latency_is_commit_minus_real_end_of_speech(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 103.0),
                _audio("audio_start", "assistant", 104.5),
            ],
        )
        assert compute_turn_commit_latencies([103.7], events) == [pytest.approx(0.7)]

    def test_split_utterance_uses_last_turn_end_in_window(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 106.0),
                _audio("audio_start", "assistant", 108.0),
            ],
        )
        # ink-2 committed once mid-utterance (102.0) and again after the user finished (106.6).
        assert compute_turn_commit_latencies([102.0, 106.6], events) == [pytest.approx(0.6)]

    def test_utterance_without_turn_end_after_speech_is_skipped(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 106.0),
                _audio("audio_start", "assistant", 108.0),
            ],
        )
        assert compute_turn_commit_latencies([102.0], events) == []

    def test_turn_end_in_next_assistant_response_is_not_attributed(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 103.0),
                _audio("audio_start", "assistant", 104.0),
                _audio("audio_start", "simulated_user", 110.0),
                _audio("audio_end", "simulated_user", 112.0),
                _audio("audio_start", "assistant", 113.0),
            ],
        )
        result = compute_turn_commit_latencies([103.6, 112.5], events)
        assert result == [pytest.approx(0.6), pytest.approx(0.5)]

    def test_ignores_malformed_lines(self, tmp_path):
        events = tmp_path / "events.jsonl"
        events.write_text(
            "not json\n"
            + json.dumps(_audio("audio_start", "simulated_user", 100.0))
            + "\n"
            + json.dumps(_audio("audio_end", "simulated_user", 101.0))
            + "\n"
        )
        assert compute_turn_commit_latencies([101.5], events) == [pytest.approx(0.5)]

    def test_unmatched_audio_start_does_not_shift_later_pairs(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                # Interrupted utterance: start with no end.
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_start", "simulated_user", 110.0),
                _audio("audio_end", "simulated_user", 112.0),
                _audio("audio_start", "assistant", 113.0),
            ],
        )
        assert compute_turn_commit_latencies([112.4], events) == [pytest.approx(0.4)]

    def test_audio_end_without_start_is_ignored(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_end", "simulated_user", 99.0),
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 101.0),
            ],
        )
        assert compute_turn_commit_latencies([101.5], events) == [pytest.approx(0.5)]

    def test_turn_end_is_not_counted_for_two_utterances_when_assistant_is_silent(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 102.0),
                # No assistant audio between the two utterances.
                _audio("audio_start", "simulated_user", 105.0),
                _audio("audio_end", "simulated_user", 107.0),
            ],
        )
        # First utterance has no turn end of its own; 107.5 belongs only to the second one.
        assert compute_turn_commit_latencies([107.5], events) == [pytest.approx(0.5)]

    def test_turn_end_beyond_max_window_is_not_attributed(self, tmp_path):
        events = tmp_path / "events.jsonl"
        _write_events(
            events,
            [
                _audio("audio_start", "simulated_user", 100.0),
                _audio("audio_end", "simulated_user", 102.0),
            ],
        )
        assert compute_turn_commit_latencies([140.0], events) == []
