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
