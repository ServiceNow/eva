"""Unit tests for record models."""

from eva.models.record import (
    AgentOverride,
    GroundTruth,
    ToolMock,
    ToolMockDatabase,
    ToolMockMatch,
)


class TestToolMockMatch:
    def test_match_any_mode(self):
        """Test matching with 'any' mode - just matches tool name."""
        match = ToolMockMatch(tool_name="get_balance", match_mode="any")

        assert match.matches("get_balance", {}) is True
        assert match.matches("get_balance", {"foo": "bar"}) is True
        assert match.matches("other_tool", {}) is False

    def test_match_exact_mode(self):
        """Test matching with 'exact' mode - all params must match exactly."""
        match = ToolMockMatch(
            tool_name="search",
            match_params={"query": "sick leave"},
            match_mode="exact",
        )

        assert match.matches("search", {"query": "sick leave"}) is True
        assert match.matches("search", {"query": "vacation"}) is False
        assert match.matches("search", {"query": "sick leave", "extra": "param"}) is False
        assert match.matches("search", {}) is False

    def test_match_contains_mode(self):
        """Test matching with 'contains' mode - subset match."""
        match = ToolMockMatch(
            tool_name="search",
            match_params={"query": "sick"},
            match_mode="contains",
        )

        # Substring match for strings
        assert match.matches("search", {"query": "sick leave policy"}) is True
        assert match.matches("search", {"query": "vacation"}) is False
        # Extra params are OK
        assert match.matches("search", {"query": "sick", "limit": 10}) is True

    def test_match_wrong_tool_name(self):
        """Test that wrong tool name never matches."""
        match = ToolMockMatch(tool_name="tool_a", match_mode="any")
        assert match.matches("tool_b", {}) is False


class TestToolMock:
    def test_create_tool_mock(self):
        """Test creating a ToolMock."""
        mock = ToolMock(
            match=ToolMockMatch(tool_name="get_balance", match_mode="any"),
            response={"balance": "15 days"},
        )

        assert mock.match.tool_name == "get_balance"
        assert mock.response == {"balance": "15 days"}

    def test_tool_mock_with_match_params(self):
        """Test creating a ToolMock with match params."""
        mock = ToolMock(
            match=ToolMockMatch(
                tool_name="search",
                match_params={"query": "test"},
                match_mode="contains",
            ),
            response={"results": ["item1"]},
        )

        assert mock.match.match_params == {"query": "test"}
        assert mock.match.match_mode == "contains"


class TestToolMockDatabase:
    def test_load_from_file(self, tool_mocks_file, sample_tool_mock_data):
        """Test loading tool mocks from JSON file."""
        db = ToolMockDatabase.load(tool_mocks_file)

        assert "record_001" in db.mocks
        assert len(db.mocks["record_001"]) == 3
        assert db.mocks["record_001"][0].match.tool_name == "get_time_off_balance"

    def test_save_and_load_roundtrip(self, temp_dir):
        """Test saving and loading tool mocks."""
        original_db = ToolMockDatabase(
            mocks={
                "rec_1": [
                    ToolMock(
                        match=ToolMockMatch(tool_name="tool_a", match_mode="any"),
                        response={"result": "success"},
                    )
                ]
            }
        )

        path = temp_dir / "mocks.json"
        original_db.save(path)

        loaded_db = ToolMockDatabase.load(path)
        assert len(loaded_db.mocks["rec_1"]) == 1
        assert loaded_db.mocks["rec_1"][0].response == {"result": "success"}

    def test_get_mocks_for_record(self, tool_mocks_file):
        """Test getting mocks by record ID."""
        db = ToolMockDatabase.load(tool_mocks_file)

        mocks = db.get_mocks_for_record("record_001")
        assert len(mocks) == 3

        # Test with non-existent record ID
        mocks = db.get_mocks_for_record("nonexistent")
        assert len(mocks) == 0


class TestGroundTruth:
    def test_ground_truth_with_required_fields(self):
        """Test creating GroundTruth with required fields."""
        gt = GroundTruth(
            expected_scenario_db={"booking": {"status": "confirmed"}},
        )

        assert gt.expected_scenario_db == {"booking": {"status": "confirmed"}}


class TestAgentOverride:
    def test_create_override(self):
        """Test creating an AgentOverride."""
        override = AgentOverride(
            instructions="Custom instructions",
            tools_enabled=["tool_a", "tool_b"],
        )

        assert override.instructions == "Custom instructions"
        assert override.tools_enabled == ["tool_a", "tool_b"]
        assert override.personality is None
