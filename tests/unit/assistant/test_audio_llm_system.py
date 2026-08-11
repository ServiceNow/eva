"""Tests for AudioLLMAgenticSystem turn-history construction.

The user can only submit audio, but a turn may carry no audio at all (genuine silence hitting
the turn-end fallback, which reprompts via text). full_audio_context must still map each audio
turn's audio to the correct user message: _turn_audio_history is kept aligned 1:1 with user
messages, with None marking a no-audio turn so audio never shifts onto the wrong turn.
"""

from unittest.mock import MagicMock

from eva.assistant.agentic.audio_llm_system import AudioLLMAgenticSystem
from eva.assistant.agentic.audit_log import AuditLog


def _make_system(full_audio_context: bool):
    """Build a bare AudioLLMAgenticSystem with just what _execute_agent_with_audio needs."""
    sys = object.__new__(AudioLLMAgenticSystem)
    sys.system_prompt = "SYS"
    sys.full_audio_context = full_audio_context
    sys.audit_log = AuditLog()
    sys._turn_text_hint = ""
    sys._turn_audio_history = []
    sys.agent = MagicMock()

    alm = MagicMock()
    alm.build_audio_user_message = lambda audio_bytes, source_sample_rate, text_hint="": {
        "role": "user",
        "content": [{"type": "audio", "_bytes": audio_bytes, "hint": text_hint}],
    }
    sys.alm_client = alm

    captured = {}

    async def fake_run_tool_loop(messages, agent):
        captured["messages"] = messages
        return
        yield  # make it an async generator

    sys._run_tool_loop = fake_run_tool_loop
    return sys, captured


def _user_msgs(messages):
    return [m for m in messages if m.get("role") == "user"]


async def test_full_audio_context_aligns_audio_across_silence_fallback():
    """A no-audio (silence) fallback in the middle must not shift later audio onto the wrong turn."""
    sys, captured = _make_system(full_audio_context=True)
    sys.audit_log.append_user_input("[user audio]")  # turn 1 (audio)
    sys.audit_log.append_assistant_output("resp 1")
    sys.audit_log.append_user_input(
        "[TURN-END FALLBACK after 10s] no user speech captured",
        message_type="turn_fallback",
        llm_content="[nudge: please repeat]",
    )  # genuine-silence fallback (no audio)
    sys.audit_log.append_assistant_output("resp 2")
    sys.audit_log.append_user_input("[user audio]")  # turn 3 (audio, current)

    # Aligned 1:1 with the three user messages: audio, None (silence fallback), audio.
    sys._turn_audio_history = [(b"A1", 16000), None, (b"A2", 16000)]
    sys._turn_text_hint = "HINT"

    async for _ in sys._execute_agent_with_audio(sys.agent):
        pass

    users = _user_msgs(captured["messages"])
    assert len(users) == 3
    assert users[0]["content"][0]["_bytes"] == b"A1"
    assert users[0]["content"][0]["hint"] == ""
    assert users[1]["content"] == "[nudge: please repeat]"  # no-audio turn stays text, NOT audio
    assert users[2]["content"][0]["_bytes"] == b"A2"
    assert users[2]["content"][0]["hint"] == "HINT"


async def test_default_mode_replaces_only_current_turn():
    """Default (non-full) mode replaces just the last user message with the current audio."""
    sys, captured = _make_system(full_audio_context=False)
    sys.audit_log.append_user_input("[user audio]")
    sys.audit_log.append_assistant_output("resp 1")
    sys.audit_log.append_user_input("[user audio]")  # current
    sys._turn_audio_history = [(b"A1", 16000), (b"A2", 16000)]
    sys._turn_text_hint = "HINT"

    async for _ in sys._execute_agent_with_audio(sys.agent):
        pass

    users = _user_msgs(captured["messages"])
    assert users[0]["content"] == "[user audio]"
    assert users[1]["content"][0]["_bytes"] == b"A2"
    assert users[1]["content"][0]["hint"] == "HINT"


def test_note_non_audio_turn_appends_none():
    sys, _ = _make_system(full_audio_context=True)
    sys.set_turn_audio(b"A1", 16000)
    sys.note_non_audio_turn()
    sys.set_turn_audio(b"A2", 16000)
    assert sys._turn_audio_history == [(b"A1", 16000), None, (b"A2", 16000)]
