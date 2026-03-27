"""Focused tests for conversation trace validation."""

from eva.metrics.processor import MetricsContextProcessor, _ProcessorContext, _validate_conversation_trace


def test_validate_conversation_trace_filters_without_pipecat_text_for_continuous_stream():
    ctx = _ProcessorContext()
    ctx.record_id = "telnyx-empty-pipecat"
    ctx.is_continuous_assistant_stream = True

    trace = [
        {
            "role": "assistant",
            "content": "Sure, let me help with that.",
            "type": "intended",
            "turn_id": 1,
            "_audit_source": True,
        }
    ]

    assert _validate_conversation_trace(trace, ctx) == []


def test_validate_conversation_trace_still_filters_without_pipecat_text_for_non_continuous_stream():
    ctx = _ProcessorContext()
    ctx.record_id = "pipecat-missing-segments"

    trace = [
        {
            "role": "assistant",
            "content": "Sure, let me help with that.",
            "type": "intended",
            "turn_id": 1,
            "_audit_source": True,
        }
    ]

    assert _validate_conversation_trace(trace, ctx) == []


def test_validate_conversation_trace_still_truncates_when_pipecat_text_exists():
    ctx = _ProcessorContext()
    ctx.record_id = "pipecat-truncation"
    ctx.is_continuous_assistant_stream = True
    ctx._intended_assistant_segments = {
        1: ["Sure, I can help you with that."]
    }

    trace = [
        {
            "role": "assistant",
            "content": "Sure, I can help you with that. Could you please provide your confirmation number?",
            "type": "intended",
            "turn_id": 1,
            "_audit_source": True,
        }
    ]

    assert _validate_conversation_trace(trace, ctx) == [
        {
            "role": "assistant",
            "content": "Sure, I can help you with that.",
            "type": "intended",
            "turn_id": 1,
        }
    ]


def test_extract_turns_populates_message_trace_for_continuous_stream_without_pipecat_text():
    ctx = _ProcessorContext()
    ctx.record_id = "telnyx-native-message-trace"
    ctx.history = [
        {"timestamp_ms": 1000, "source": "audit_log", "event_type": "assistant", "data": "Welcome to EVA."},
        {
            "timestamp_ms": 1100,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "pipecat_agent", "audio_timestamp": 1.1},
        },
        {"timestamp_ms": 5000, "source": "audit_log", "event_type": "user", "data": "I need to change my flight"},
        {
            "timestamp_ms": 5100,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 5.1},
        },
        {
            "timestamp_ms": 5400,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 5.4},
        },
        {
            "timestamp_ms": 5600,
            "source": "audit_log",
            "event_type": "tool_call",
            "data": {"tool": "lookup_booking", "parameters": {"confirmation_number": "ABC123"}},
        },
        {
            "timestamp_ms": 5700,
            "source": "audit_log",
            "event_type": "tool_response",
            "data": {"tool": "lookup_booking", "response": {"status": "success"}},
        },
        {"timestamp_ms": 6000, "source": "audit_log", "event_type": "assistant", "data": "I found your reservation."},
        {
            "timestamp_ms": 9000,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "pipecat_agent", "audio_timestamp": 9.0},
        },
        {
            "timestamp_ms": 9100,
            "source": "elevenlabs",
            "event_type": "connection_state",
            "data": {"data": {"state": "session_ended"}},
        },
    ]

    MetricsContextProcessor._extract_turns_from_history(ctx)
    MetricsContextProcessor._reconcile_transcript_with_tools(ctx)

    assert ctx.is_continuous_assistant_stream is True
    assert ctx.conversation_trace == [
        {
            "role": "user",
            "content": "[user interrupts] I need to change my flight",
            "timestamp": 5000,
            "type": "transcribed",
            "turn_id": 1,
        },
        {
            "tool_name": "lookup_booking",
            "parameters": {"confirmation_number": "ABC123"},
            "timestamp": 5600,
            "type": "tool_call",
            "turn_id": 0,
        },
        {
            "tool_name": "lookup_booking",
            "tool_response": {"status": "success"},
            "timestamp": 5700,
            "type": "tool_response",
            "turn_id": 0,
        },
    ]
    assert ctx.message_trace == [
        {"role": "assistant", "content": "Welcome to EVA.", "timestamp": 1000, "type": "intended", "turn_id": 0},
        {
            "role": "user",
            "content": "[user interrupts] I need to change my flight",
            "timestamp": 5000,
            "type": "transcribed",
            "turn_id": 1,
        },
        {
            "tool_name": "lookup_booking",
            "parameters": {"confirmation_number": "ABC123"},
            "timestamp": 5600,
            "type": "tool_call",
            "turn_id": 0,
        },
        {
            "tool_name": "lookup_booking",
            "tool_response": {"status": "success"},
            "timestamp": 5700,
            "type": "tool_response",
            "turn_id": 0,
        },
        {"role": "assistant", "content": "I found your reservation.", "timestamp": 6000, "type": "intended", "turn_id": 0},
    ]


def test_extract_turns_aligns_telnyx_assistant_speech_to_previous_turn_on_same_timestamp():
    ctx = _ProcessorContext()
    ctx.record_id = "telnyx-assistant-speech-alignment"
    ctx.history = [
        {
            "timestamp_ms": 1000,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "pipecat_agent", "audio_timestamp": 1.0},
        },
        {
            "timestamp_ms": 1800,
            "source": "elevenlabs",
            "event_type": "assistant_speech",
            "data": {"data": {"text": "Hello there."}},
        },
        {
            "timestamp_ms": 1800,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 1.8},
        },
        {
            "timestamp_ms": 1900,
            "source": "elevenlabs",
            "event_type": "user_speech",
            "data": {"data": {"text": "I need help"}},
        },
        {
            "timestamp_ms": 2400,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 2.4},
        },
        {
            "timestamp_ms": 3200,
            "source": "elevenlabs",
            "event_type": "assistant_speech",
            "data": {"data": {"text": "Sure, what do you need?"}},
        },
        {
            "timestamp_ms": 3200,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 3.2},
        },
        {
            "timestamp_ms": 3300,
            "source": "elevenlabs",
            "event_type": "user_speech",
            "data": {"data": {"text": "Thanks, goodbye"}},
        },
        {
            "timestamp_ms": 3600,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 3.6},
        },
        {
            "timestamp_ms": 4200,
            "source": "elevenlabs",
            "event_type": "assistant_speech",
            "data": {"data": {"text": "Goodbye."}},
        },
        {
            "timestamp_ms": 5000,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "pipecat_agent", "audio_timestamp": 5.0},
        },
        {
            "timestamp_ms": 5100,
            "source": "elevenlabs",
            "event_type": "connection_state",
            "data": {"data": {"state": "session_ended"}},
        },
    ]

    MetricsContextProcessor._extract_turns_from_history(ctx)

    assert ctx.is_continuous_assistant_stream is True
    assert ctx.transcribed_assistant_turns[0].startswith("Hello there.")
    assert ctx.transcribed_assistant_turns[1].startswith("Sure, what do you need?")
    assert ctx.transcribed_assistant_turns[2] == "Goodbye."


def test_synthesize_telnyx_message_native_stream_replaces_collapsed_assistant_spans():
    ctx = _ProcessorContext()
    ctx.record_id = "telnyx-synthesis"
    ctx.history = [
        {
            "timestamp_ms": 1000,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "pipecat_agent", "audio_timestamp": 1.0},
        },
        {
            "timestamp_ms": 20000,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "pipecat_agent", "audio_timestamp": 20.0},
        },
    ]
    ctx.is_continuous_assistant_stream = True
    ctx.response_speed_latencies = [0.5, 0.6]
    ctx.intended_assistant_turns = {0: "Hello.", 1: "Sure.", 2: "Goodbye."}
    ctx.audio_timestamps_assistant_turns = {
        0: [(1.0, 20.0)],
        1: [(10.0, 10.0)],
        2: None,
    }
    ctx.audio_timestamps_user_turns = {
        1: [(3.0, 4.0)],
        2: [(8.0, 9.0)],
        3: [(12.0, 13.0)],
    }

    MetricsContextProcessor._synthesize_continuous_stream_timestamps(ctx)

    assert ctx.audio_timestamps_assistant_turns == {
        0: [(1.0, 3.0)],
        1: [(4.5, 8.0)],
        2: [(9.6, 12.0)],
    }


def test_synthesize_continuous_stream_preserves_existing_pipecat_turns():
    ctx = _ProcessorContext()
    ctx.record_id = "pipecat-continuous-guard"
    ctx.history = [
        {"timestamp_ms": 1000, "source": "pipecat", "event_type": "tts_text", "data": {"frame": "Hello."}},
        {
            "timestamp_ms": 1100,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "pipecat_agent", "audio_timestamp": 1.1},
        },
        {
            "timestamp_ms": 5000,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "pipecat_agent", "audio_timestamp": 5.0},
        },
    ]
    ctx.is_continuous_assistant_stream = True
    ctx.response_speed_latencies = [0.5, 0.6]
    ctx.intended_assistant_turns = {0: "Hello.", 1: "Sure.", 2: "Goodbye."}
    ctx.audio_timestamps_assistant_turns = {
        0: [(1.1, 2.5)],
        1: [(4.6, 5.2)],
        2: None,
    }
    ctx.audio_timestamps_user_turns = {
        1: [(3.0, 4.0)],
        2: [(8.0, 9.0)],
    }

    MetricsContextProcessor._synthesize_continuous_stream_timestamps(ctx)

    assert ctx.audio_timestamps_assistant_turns == {
        0: [(1.1, 2.5)],
        1: [(4.6, 5.2)],
        2: [(9.6, 14.6)],
    }
