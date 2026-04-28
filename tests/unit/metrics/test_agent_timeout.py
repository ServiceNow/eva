"""Unit tests for agent-timeout-on-user-turn derivation helpers."""

from eva.metrics.processor import is_agent_timeout_on_user_turn, last_audio_speaker


class TestLastAudioSpeaker:
    def test_user_later_than_assistant(self):
        assert last_audio_speaker({0: [(0.0, 5.0)]}, {0: [(2.0, 3.0)]}) == "user"

    def test_assistant_later_than_user(self):
        assert last_audio_speaker({0: [(0.0, 2.0)]}, {0: [(1.0, 5.0)]}) == "assistant"

    def test_only_user(self):
        assert last_audio_speaker({0: [(0.0, 1.0)]}, {}) == "user"

    def test_only_assistant(self):
        assert last_audio_speaker({}, {0: [(0.0, 1.0)]}) == "assistant"

    def test_neither(self):
        assert last_audio_speaker({}, {}) is None


class TestIsAgentTimeoutOnUserTurn:
    def test_goodbye_returns_false(self):
        assert is_agent_timeout_on_user_turn("goodbye", {0: [(0.0, 5.0)]}, {0: [(2.0, 3.0)]}) is False

    def test_error_returns_false(self):
        assert is_agent_timeout_on_user_turn("error", {0: [(0.0, 5.0)]}, {0: [(2.0, 3.0)]}) is False

    def test_none_returns_false(self):
        assert is_agent_timeout_on_user_turn(None, {0: [(0.0, 5.0)]}, {0: [(2.0, 3.0)]}) is False

    def test_inactivity_timeout_user_last(self):
        assert is_agent_timeout_on_user_turn("inactivity_timeout", {0: [(0.0, 5.0)]}, {0: [(2.0, 3.0)]}) is True

    def test_inactivity_timeout_assistant_last(self):
        assert is_agent_timeout_on_user_turn("inactivity_timeout", {0: [(0.0, 2.0)]}, {0: [(1.0, 5.0)]}) is False

    def test_inactivity_timeout_no_audio(self):
        assert is_agent_timeout_on_user_turn("inactivity_timeout", {}, {}) is False
