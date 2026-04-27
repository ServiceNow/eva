"""Evaluation record and tool mock data models."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolMockMatch(BaseModel):
    """Defines how to match a tool call for mocking."""

    tool_name: str = Field(..., description="Name of the tool to match")
    match_params: dict[str, Any] = Field(default_factory=dict, description="Parameters to match on")
    match_mode: str = Field(
        "exact",
        description="Match mode: 'exact' (all params match), 'contains' (subset match), 'any' (just tool name)",
    )

    def matches(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Check if this matcher matches the given tool call."""
        if self.tool_name != tool_name:
            return False

        if self.match_mode == "any":
            return True

        if self.match_mode == "exact":
            return self.match_params == params

        if self.match_mode == "contains":
            # Check if all match_params are present in params with same values
            for key, value in self.match_params.items():
                if key not in params:
                    return False
                # For string values, do substring match
                if isinstance(value, str) and isinstance(params[key], str):
                    if value.lower() not in params[key].lower():
                        return False
                elif params[key] != value:
                    return False
            return True

        return False


class ToolMock(BaseModel):
    """A mocked tool response."""

    match: ToolMockMatch = Field(..., description="Matching criteria for this mock")
    response: dict[str, Any] = Field(..., description="Mocked response to return")


class ToolMockDatabase(BaseModel):
    """Collection of tool mocks, stored in a separate file."""

    mocks: dict[str, list[ToolMock]] = Field(default_factory=dict, description="Tool mocks keyed by record ID")

    @classmethod
    def load(cls, path: Path | str) -> "ToolMockDatabase":
        """Load from JSON file where keys are record IDs and values are lists of tool mocks."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate and parse mocks for each record
        mocks = {}
        for record_id, mock_list in data.items():
            mocks[record_id] = [ToolMock.model_validate(mock) for mock in mock_list]

        return cls(mocks=mocks)

    def save(self, path: Path | str) -> None:
        """Save to JSON file."""
        path = Path(path)
        # Convert to dict of lists for JSON serialization
        data = {
            record_id: [mock.model_dump(mode="json") for mock in mock_list]
            for record_id, mock_list in self.mocks.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_mocks_for_record(self, record_id: str) -> list[ToolMock]:
        """Get mocks for a specific record ID."""
        return self.mocks.get(record_id, [])


class GroundTruth(BaseModel):
    """Expected outcomes for evaluation."""

    model_config = ConfigDict(extra="ignore")

    expected_scenario_db: dict[str, Any] = Field(
        ..., description="Expected final scenario database state for diff computation"
    )


class AgentOverride(BaseModel):
    """Override agent configuration for a specific record."""

    instructions: str | None = Field(None, description="Override agent instructions")
    tools_enabled: list[str] | None = Field(None, description="Subset of tools to enable")
    personality: str | None = Field(None, description="Override agent personality")


class EvaluationRecord(BaseModel):
    """A single test case for voice agent evaluation."""

    id: str = Field(..., description="Unique identifier for this record")

    # User simulation
    user_goal: Any = Field(..., description="Natural language description of user's goal")
    user_config: dict = Field(..., description="User persona config for user simulator")

    current_date_time: str = Field(..., description="Current date and time for the record")

    scenario_context: dict = Field(..., description="Scenario context for the record")

    ground_truth: GroundTruth = Field(default_factory=GroundTruth, description="Expected outcomes for evaluation")

    agent_override: AgentOverride | None = Field(None, description="Override agent configuration for this record")

    # Metadata
    category: str | None = Field(None, description="Category for grouping, e.g., 'hr_pto', 'it_support'")

    @classmethod
    def load_dataset(cls, path: Path | str) -> list["EvaluationRecord"]:
        """Load records from JSONL file."""
        path = Path(path)
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(cls.model_validate_json(line))
        return records

    @classmethod
    def save_dataset(cls, records: list["EvaluationRecord"], path: Path | str) -> None:
        """Save records to JSONL file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(record.model_dump_json() + "\n" for record in records)
