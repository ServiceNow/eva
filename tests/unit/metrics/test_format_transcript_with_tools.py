"""Tests for format_transcript_with_tools."""

from eva.metrics.utils import format_transcript_with_tools


def test_empty_turns():
    assert format_transcript_with_tools([]) == ""


def test_user_and_assistant_only():
    turns = [
        {"turn_id": 1, "role": "user", "content": "Hi"},
        {"turn_id": 1, "role": "assistant", "content": "Hello!"},
        {"turn_id": 2, "role": "user", "content": "Help me"},
        {"turn_id": 2, "role": "assistant", "content": "Sure"},
    ]
    result = format_transcript_with_tools(turns)
    assert result == ("Turn 1:\n  user: Hi\n  assistant: Hello!\n\nTurn 2:\n  user: Help me\n  assistant: Sure")


def test_with_tool_call_and_response():
    turns = [
        {"turn_id": 1, "role": "user", "content": "Book flight ABC123"},
        {"turn_id": 1, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "ABC123"}},
        {"turn_id": 1, "type": "tool_response", "tool_name": "get_reservation", "tool_response": {"status": "ok"}},
        {"turn_id": 1, "role": "assistant", "content": "Found your reservation."},
    ]
    result = format_transcript_with_tools(turns)
    assert result == (
        "Turn 1:\n"
        "  user: Book flight ABC123\n"
        "  tool_call: get_reservation({'code': 'ABC123'})\n"
        "  tool_response: get_reservation: {'status': 'ok'}\n"
        "  assistant: Found your reservation."
    )


def test_tool_error_response():
    """Tool errors appear as tool_response with error info in the response dict."""
    turns = [
        {"turn_id": 1, "role": "user", "content": "Look up reservation XYZ"},
        {"turn_id": 1, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "XYZ"}},
        {
            "turn_id": 1,
            "type": "tool_response",
            "tool_name": "get_reservation",
            "tool_response": {"status": "error", "error_type": "not_found", "message": "Tool execution failed"},
        },
        {"turn_id": 1, "role": "assistant", "content": "Sorry, I couldn't find that reservation."},
    ]
    result = format_transcript_with_tools(turns)
    assert result == (
        "Turn 1:\n"
        "  user: Look up reservation XYZ\n"
        "  tool_call: get_reservation({'code': 'XYZ'})\n"
        "  tool_response: get_reservation: {'status': 'error', 'error_type': 'not_found', 'message': 'Tool execution failed'}\n"
        "  assistant: Sorry, I couldn't find that reservation."
    )


def test_multiple_tool_calls_in_one_turn():
    turns = [
        {"turn_id": 2, "role": "user", "content": "Rebook me"},
        {"turn_id": 2, "type": "tool_call", "tool_name": "search_flights", "parameters": {"date": "2026-03-25"}},
        {"turn_id": 2, "type": "tool_response", "tool_name": "search_flights", "tool_response": {"flights": []}},
        {"turn_id": 2, "type": "tool_call", "tool_name": "search_flights", "parameters": {"date": "2026-03-26"}},
        {"turn_id": 2, "type": "tool_response", "tool_name": "search_flights", "tool_response": {"flights": ["FL1"]}},
        {"turn_id": 2, "role": "assistant", "content": "Found a flight on March 26."},
    ]
    result = format_transcript_with_tools(turns)
    lines = result.split("\n")
    tool_call_lines = [line for line in lines if "tool_call:" in line]
    tool_response_lines = [line for line in lines if "tool_response:" in line]
    assert len(tool_call_lines) == 2
    assert len(tool_response_lines) == 2


def test_missing_content_fields():
    """Entries with missing content/tool fields should not crash."""
    turns = [
        {"turn_id": 1, "role": "user"},
        {"turn_id": 1, "role": "assistant"},
        {"turn_id": 1, "type": "tool_call"},
        {"turn_id": 1, "type": "tool_response"},
    ]
    result = format_transcript_with_tools(turns)
    assert "Turn 1:" in result
    assert "user: " in result
    assert "tool_call: (" in result
    assert "tool_response: :" in result


def test_unknown_entry_type_skipped():
    """Entries that are neither user/assistant nor tool_call/tool_response are skipped."""
    turns = [
        {"turn_id": 1, "role": "user", "content": "Hi"},
        {"turn_id": 1, "type": "system_event", "data": "something"},
        {"turn_id": 1, "role": "assistant", "content": "Hello"},
    ]
    result = format_transcript_with_tools(turns)
    assert "system_event" not in result
    assert "user: Hi" in result
    assert "assistant: Hello" in result


def test_turn_id_grouping_preserves_order():
    """Turns are grouped by turn_id in order of appearance."""
    turns = [
        {"turn_id": 3, "role": "user", "content": "Third"},
        {"turn_id": 3, "role": "assistant", "content": "Three"},
        {"turn_id": 1, "role": "user", "content": "First"},
        {"turn_id": 1, "role": "assistant", "content": "One"},
    ]
    result = format_transcript_with_tools(turns)
    # groupby preserves input order, so turn 3 comes before turn 1
    assert result.index("Turn 3:") < result.index("Turn 1:")
