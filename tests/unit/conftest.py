"""Test fixtures for model tests."""

import json
import tempfile
from pathlib import Path

import pytest

from eva.models.record import EvaluationRecord, GroundTruth


def make_evaluation_record(record_id: str = "test_record", **overrides) -> EvaluationRecord:
    """Create a minimal EvaluationRecord with sensible defaults, overridable via kwargs."""
    defaults = {
        "id": record_id,
        "user_goal": "Test goal",
        "user_config": {
            "name": "Robert White",
            "gender": "man",
            "user_persona_id": 2,
            "user_persona": "You're direct and to the point.",
        },
        "current_date_time": "2026-01-01T00:00:00",
        "scenario_context": {},
        "ground_truth": GroundTruth(expected_scenario_db={}),
    }
    defaults.update(overrides)
    return EvaluationRecord(**defaults)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tool_mock_data():
    """Sample tool mock data for testing (JSON format keyed by record_id)."""
    return {
        "record_001": [
            {
                "match": {"tool_name": "get_time_off_balance", "match_mode": "any"},
                "response": {"total_balance": "15 days", "used": "5 days", "remaining": "10 days"},
            },
            {
                "match": {"tool_name": "submit_leave_request", "match_mode": "any"},
                "response": {"status": "success", "message": "Leave request submitted"},
            },
            {
                "match": {
                    "tool_name": "pto_policy_search",
                    "match_params": {"query": "sick"},
                    "match_mode": "contains",
                },
                "response": {"policy": "Employees receive 10 sick days per year."},
            },
        ]
    }


@pytest.fixture
def tool_mocks_file(temp_dir, sample_tool_mock_data):
    """Create a temporary tool mocks JSON file."""
    path = temp_dir / "tool_mocks.json"
    with open(path, "w") as f:
        json.dump(sample_tool_mock_data, f)
    return path


@pytest.fixture
def sample_record_data():
    """Sample evaluation record data for testing."""
    return [
        {
            "id": "hr_pto_001",
            "user_goal": "Check my PTO balance and request 3 days off",
            "user_persona": "Impatient Employee who wants to check PTO balance",
            "current_date_time": "2026-01-15T10:00:00Z",
            "scenario_context": {"steps": ["check_balance", "submit_request"]},
            "ground_truth": {
                "expected_scenario_db": {"pto_balance": {"remaining": "7 days"}},
            },
            "category": "hr_pto",
        },
        {
            "id": "hr_pto_002",
            "user_goal": "Ask about sick leave policy",
            "user_persona": "New Employee who wants to learn about sick leave",
            "current_date_time": "2026-01-15T10:00:00Z",
            "scenario_context": {"steps": ["search_policy"]},
            "ground_truth": {
                "expected_scenario_db": {},
            },
            "category": "hr_pto",
        },
    ]


@pytest.fixture
def dataset_file(temp_dir, sample_record_data):
    """Create a temporary dataset JSONL file."""
    path = temp_dir / "dataset.jsonl"
    with open(path, "w") as f:
        f.writelines(json.dumps(record) + "\n" for record in sample_record_data)
    return path


@pytest.fixture
def sample_agent_data():
    """Sample agent configuration data for testing."""
    return {
        "agents": [
            {
                "id": "agent_hr_pto",
                "name": "HR PTO Agent",
                "description": "Handles PTO and leave requests",
                "role": "You are an HR assistant helping with time off requests.",
                "instructions": "Help users with their PTO balance and leave requests.",
                "tool_module_path": "eva.assistant.tools.test_tools",
                "tools": [
                    {
                        "id": "tool_get_balance",
                        "name": "Get time off balance",
                        "description": "Get the user's current PTO balance",
                        "required_parameters": [],
                        "optional_parameters": [{"name": "type", "type": "string", "enum": ["sick", "personal"]}],
                    },
                    {
                        "id": "tool_submit_leave",
                        "name": "Submit leave request",
                        "description": "Submit a new leave request",
                        "required_parameters": [
                            {"name": "start_date", "type": "string", "description": "Start date"},
                            {"name": "end_date", "type": "string", "description": "End date"},
                            {
                                "name": "leave_type",
                                "type": "string",
                                "enum": ["sick", "personal"],
                            },
                        ],
                    },
                ],
            }
        ]
    }


@pytest.fixture
def agents_config_file(temp_dir, sample_agent_data):
    """Create a temporary agents config YAML file."""
    import yaml

    path = temp_dir / "agents.yaml"
    with open(path, "w") as f:
        yaml.dump(sample_agent_data, f)
    return path
