"""Tests for aai wire-event parsing."""

from eva.assistant.aai_events import (
    AAIAgentTranscriptEvent,
    AAIErrorEvent,
    AAIIdleTimeoutEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAIToolCallEvent,
    AAIUnknownEvent,
    AAIUserTranscriptEvent,
    parse_aai_event,
)


class TestParseAaiEvent:
    def test_parses_tool_call_camel_case_aliases(self):
        event = parse_aai_event(
            {
                "type": "tool_call",
                "toolCallId": "call_abc",
                "toolName": "get_reservation",
                "args": {"confirmation_number": "DJ3LPO"},
            }
        )
        assert isinstance(event, AAIToolCallEvent)
        assert event.tool_call_id == "call_abc"
        assert event.tool_name == "get_reservation"
        assert event.args == {"confirmation_number": "DJ3LPO"}

    def test_tool_call_args_default_to_empty_dict(self):
        event = parse_aai_event({"type": "tool_call", "toolCallId": "c1", "toolName": "list_flights"})
        assert isinstance(event, AAIToolCallEvent)
        assert event.args == {}

    def test_parses_agent_transcript(self):
        event = parse_aai_event({"type": "agent_transcript", "text": "How can I help?"})
        assert isinstance(event, AAIAgentTranscriptEvent)
        assert event.text == "How can I help?"

    def test_parses_user_transcript_with_turn_order(self):
        event = parse_aai_event({"type": "user_transcript", "text": "hello", "turnOrder": 3})
        assert isinstance(event, AAIUserTranscriptEvent)
        assert event.text == "hello"
        assert event.turn_order == 3

    def test_parses_zero_field_events(self):
        assert isinstance(parse_aai_event({"type": "speech_started"}), AAISpeechStartedEvent)
        assert isinstance(parse_aai_event({"type": "reply_done"}), AAIReplyDoneEvent)
        assert isinstance(parse_aai_event({"type": "idle_timeout"}), AAIIdleTimeoutEvent)

    def test_parses_error_with_code_and_message(self):
        event = parse_aai_event({"type": "error", "code": "host_disabled", "message": "AAI_ALLOW_HOST"})
        assert isinstance(event, AAIErrorEvent)
        assert event.code == "host_disabled"
        assert event.message == "AAI_ALLOW_HOST"

    def test_ignores_unknown_extra_fields(self):
        event = parse_aai_event({"type": "agent_transcript", "text": "hi", "futureField": 1})
        assert isinstance(event, AAIAgentTranscriptEvent)

    def test_unrecognized_type_returns_unknown_event(self):
        event = parse_aai_event({"type": "brand_new_thing", "x": 1})
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "brand_new_thing"
        assert event.raw == {"type": "brand_new_thing", "x": 1}

    def test_missing_type_returns_unknown_event(self):
        event = parse_aai_event({"text": "orphan"})
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "unknown"

    def test_malformed_known_event_returns_unknown_rather_than_raising(self):
        # `text` is required on agent_transcript; a malformed frame must never abort a conversation.
        event = parse_aai_event({"type": "agent_transcript"})
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "agent_transcript"
