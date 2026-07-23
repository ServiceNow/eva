"""Tests for extract_conv_finish_signals (reads a record dir → ConvFinishSignals)."""

import json

from eva.metrics.diagnostic.conv_finish_classifier import (
    classify_conv_finish_failure,
    extract_conv_finish_signals,
)

from .conftest import make_metric_context

TS = "2026-07-20 10:00:00,000 | INFO     | {src} | {msg}"


def _write(d, **files):
    for name, content in files.items():
        (d / name).write_text(content)


def _ctx(tmp_path, reason="inactivity_timeout", user_last=True, **overrides):
    # user_last → user audio ends after assistant → is_agent_timeout_on_user_turn True
    user_ts = {0: [(0.0, 5.0)]} if user_last else {0: [(0.0, 2.0)]}
    asst_ts = {0: [(1.0, 2.0)]} if user_last else {0: [(1.0, 5.0)]}
    return make_metric_context(
        output_dir=str(tmp_path),
        conversation_ended_reason=reason,
        audio_timestamps_user_turns=user_ts,
        audio_timestamps_assistant_turns=asst_ts,
        **overrides,
    )


def test_parent_gate_false_on_goodbye(tmp_path):
    _write(tmp_path, **{"audit_log.json": json.dumps({"conversation_messages": []})})
    s = extract_conv_finish_signals(_ctx(tmp_path, reason="goodbye"))
    assert s.is_parent_failure is False
    assert classify_conv_finish_failure(s).category is None


def test_reasoning_only_from_perf_csv(tmp_path):
    csv = 'response,tool_calls,reasoning\n,,"Great. I booked room 201 for you."\n'
    _write(
        tmp_path,
        **{
            "agent_perf_stats.csv": csv,
            "audit_log.json": json.dumps({"conversation_messages": [{"role": "assistant", "content": ""}]}),
        },
    )
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.last_perf_response_empty is True
    assert s.last_perf_has_tool_call is False
    assert "booked room 201" in s.last_perf_reasoning.lower()
    assert classify_conv_finish_failure(s).category == "reasoning_only"


def _write_perf_csv(tmp_path, reasoning_value, reasoning_tokens="0", stop_reason="stop"):
    # Mirrors the writer (system.py) which stores reasoning as f'"{content}"' — so empty reasoning
    # content lands in the CSV as the literal 2-char string '""'. Use csv.writer so the parsed cell
    # equals reasoning_value exactly.
    import csv as _csv

    with (tmp_path / "agent_perf_stats.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["response", "tool_calls", "reasoning", "reasoning_tokens", "stop_reason"])
        w.writerow(["", "", reasoning_value, reasoning_tokens, stop_reason])
    (tmp_path / "audit_log.json").write_text(
        json.dumps({"conversation_messages": [{"role": "assistant", "content": ""}]})
    )


def test_reasoning_too_long_when_stop_reason_length(tmp_path):
    # gemma-4 rec 40: huge reasoning, hit the token cap (stop_reason=length), empty response.
    _write_perf_csv(tmp_path, '"lots of thinking…"', reasoning_tokens="2047", stop_reason="length")
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.last_perf_stop_reason == "length"
    assert classify_conv_finish_failure(s).category == "reasoning_too_long"


def test_hidden_reasoning_tokens_still_reasoning_only(tmp_path):
    # gemini-3.5-flash-lite / gpt-5: reasoning_tokens>0 but NO reasoning text (writer stores '""').
    # The model DID reason, the answer just never reached TTS → still reasoning_only.
    _write_perf_csv(tmp_path, '""', reasoning_tokens="26")
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.last_perf_reasoning == ""  # text unwrapped to empty
    assert s.last_perf_reasoning_tokens == 26
    assert classify_conv_finish_failure(s).category == "reasoning_only"


def test_truly_empty_generation_is_not_reasoning_only(tmp_path):
    # No reasoning text AND 0 reasoning tokens → the model did not reason; not reasoning_only.
    _write_perf_csv(tmp_path, '""', reasoning_tokens="0")
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert classify_conv_finish_failure(s).category != "reasoning_only"


def test_quote_wrapped_real_reasoning_still_fires(tmp_path):
    _write_perf_csv(tmp_path, '"Booked room 201."')  # writer-wrapped real content, 0 tokens
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.last_perf_reasoning == "Booked room 201."  # outer wrapper stripped
    r = classify_conv_finish_failure(s)
    assert r.category == "reasoning_only"
    assert r.details["reasoning_preview"] == "Booked room 201."


def test_tts_service_error_from_pipecat_frames(tmp_path):
    pl = "\n".join(
        json.dumps(x)
        for x in [
            {
                "type": "error",
                "data": {
                    "frame": "DeepgramTTSService#1 error: server rejected WebSocket connection: HTTP 402, fatal: False"
                },
            },
            {
                "type": "error",
                "data": {
                    "frame": "DeepgramTTSService#1 error: server rejected WebSocket connection: HTTP 402, fatal: False"
                },
            },
        ]
    )
    _write(
        tmp_path,
        **{
            "pipecat_logs.jsonl": pl,
            "user_simulator_events.jsonl": "",  # no assistant audio events
            "audit_log.json": json.dumps({"conversation_messages": [{"role": "assistant", "content": ""}]}),
        },
    )
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.tts_service_error is True
    assert s.assistant_audio_events == 0
    r = classify_conv_finish_failure(s)
    assert r.category == "tts_api_error"
    assert r.details["invalid_run"] is True


def test_llm_api_error_terminal_from_logs(tmp_path):
    log = "\n".join(
        [
            TS.format(
                src="eva.assistant.pipecat_server:738", msg="Assistant turn stopped - complete response: 'What floor?'"
            ),
            TS.format(src="eva.assistant.pipeline.agent_processor:196", msg="Processing complete user turn: floor two"),
            TS.format(
                src="eva.assistant.services.llm:354",
                msg="Retryable streaming error on attempt 1/6: litellm.MidStreamFallbackError: APIConnectionError: XaiException - Timeout",
            ),
        ]
    )
    _write(
        tmp_path,
        **{
            "logs.log": log,
            "audit_log.json": json.dumps({"conversation_messages": [{"role": "tool", "content": "{}"}]}),
        },
    )
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.llm_api_error_terminal is True
    assert classify_conv_finish_failure(s).category == "llm_api_error"


def test_stt_empty_transcription_from_logs(tmp_path):
    log = "\n".join(
        [
            TS.format(
                src="eva.assistant.pipecat_server:738",
                msg="Assistant turn stopped - complete response: 'Is that correct?'",
            ),
            TS.format(
                src="eva.assistant.pipeline.agent_processor:380",
                msg="VAD fired but no previous transcription timestamp found",
            ),
            TS.format(src="eva.assistant.pipecat_server:701", msg="User turn stopped - complete transcript: ''"),
        ]
    )
    _write(
        tmp_path,
        **{
            "logs.log": log,
            "audit_log.json": json.dumps(
                {"conversation_messages": [{"role": "assistant", "content": "Is that correct?"}]}
            ),
        },
    )
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.vad_no_tx_warning_after_last_response is True
    assert s.empty_transcript_after_last_response is True
    assert classify_conv_finish_failure(s).category == "stt_empty_transcription"


def test_vad_no_turn_detected_from_events(tmp_path):
    events = "\n".join(
        json.dumps(x)
        for x in [
            {"event_type": "audio_start", "user": "assistant", "audio_timestamp": 100.0},
            {"event_type": "audio_end", "user": "assistant", "audio_timestamp": 108.0},
            {"type": "user_speech", "data": {"text": "It is for end of life."}},
            {"event_type": "audio_start", "user": "simulated_user", "audio_timestamp": 112.0},
            {"event_type": "audio_end", "user": "simulated_user", "audio_timestamp": 113.5},
        ]
    )
    # pipecat: only an EARLIER user_started_speaking, none in the final window (112-113.5)
    pl = json.dumps({"type": "user_started_speaking", "timestamp": 90000})
    _write(
        tmp_path,
        **{
            "user_simulator_events.jsonl": events,
            "pipecat_logs.jsonl": pl,
            "audit_log.json": json.dumps(
                {"conversation_messages": [{"role": "assistant", "content": "Just to confirm..."}]}
            ),
        },
    )
    s = extract_conv_finish_signals(_ctx(tmp_path))
    assert s.user_final_utterance_after_agent is True
    assert s.vad_onset_in_final_window is False
    assert s.user_final_words == "It is for end of life."
    assert classify_conv_finish_failure(s).category == "vad_no_turn_detected"
