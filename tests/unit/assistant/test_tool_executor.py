# Inline tool module source used by ToolExecutor tests.
import json
from pathlib import Path

import pytest
import yaml

from eva.assistant.tools.tool_executor import ToolExecutor

_TOOL_MODULE_SRC = """\
def get_reservation(params, db, call_index):
    conf = params.get("confirmation_number", "").upper()
    res = db.get("reservations", {}).get(conf)
    if not res:
        return {"status": "error", "error_type": "not_found", "message": "Not found"}
    return {"status": "success", "reservation": res}

def update_reservation(params, db, call_index):
    conf = params.get("confirmation_number", "").upper()
    res = db.get("reservations", {}).get(conf)
    if not res:
        return {"status": "error", "error_type": "not_found", "message": "Not found"}
    new_status = params.get("new_status", "modified")
    res["status"] = new_status
    return {"status": "success", "updated": res}

def failing_tool(params, db, call_index):
    raise RuntimeError("Something broke")
"""


@pytest.fixture()
def tool_executor_env(tmp_path: Path):
    """Set up temp files for a ToolExecutor: agent YAML, scenario DB, and tool module."""
    # Agent config YAML with tool schemas
    agent_config = {
        "id": "test-agent",
        "name": "Test Agent",
        "description": "Agent for tests",
        "role": "test",
        "instructions": "test",
        "tools": [
            {
                "id": "get_reservation",
                "name": "get_reservation",
                "description": "Get reservation",
                "required_parameters": [
                    {"name": "confirmation_number", "type": "string"},
                ],
            },
            {
                "id": "update_reservation",
                "name": "update_reservation",
                "description": "Update reservation",
                "required_parameters": [
                    {"name": "confirmation_number", "type": "string"},
                    {"name": "new_status", "type": "string"},
                ],
            },
            {
                "id": "failing_tool",
                "name": "failing_tool",
                "description": "A tool that always fails",
            },
        ],
    }
    config_path = tmp_path / "agent.yaml"
    with open(config_path, "w") as f:
        yaml.dump(agent_config, f)

    # Scenario database JSON
    scenario_db = {
        "reservations": {
            "ABC123": {
                "confirmation_number": "ABC123",
                "last_name": "Smith",
                "status": "confirmed",
            }
        }
    }
    db_path = tmp_path / "scenario.json"
    with open(db_path, "w") as f:
        json.dump(scenario_db, f)

    # Tool module file
    module_path = tmp_path / "test_tools.py"
    module_path.write_text(_TOOL_MODULE_SRC)

    return config_path, db_path, module_path, scenario_db


def _make_executor(tool_executor_env) -> ToolExecutor:
    """Build a ToolExecutor from the fixture, bypassing normal module loading."""
    config_path, db_path, module_path, _ = tool_executor_env

    # We can't use the normal module_path import mechanism in tests because it
    # resolves relative to the project `src/` directory.  Instead, we instantiate
    # partially and patch the tool functions dict.
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_tools", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Build executor, monkey-patching _load_tool_module to return our module
    original_load = ToolExecutor._load_tool_module
    ToolExecutor._load_tool_module = lambda self: {
        name: getattr(mod, name) for name in dir(mod) if not name.startswith("_") and callable(getattr(mod, name))
    }
    try:
        executor = ToolExecutor(
            tool_config_path=str(config_path),
            scenario_db_path=str(db_path),
            tool_module_path="test_tools",
            current_date_time="2026-02-25 10:00:00",
        )
    finally:
        ToolExecutor._load_tool_module = original_load

    return executor


class TestToolExecutorExecute:
    """Tests for ToolExecutor.execute() return values."""

    @pytest.mark.asyncio
    async def test_successful_call_returns_result(self, tool_executor_env):
        executor = _make_executor(tool_executor_env)
        result = await executor.execute("get_reservation", {"confirmation_number": "ABC123"})
        assert result["status"] == "success"
        assert result["reservation"]["confirmation_number"] == "ABC123"

    @pytest.mark.asyncio
    async def test_tool_not_found_in_config(self, tool_executor_env):
        executor = _make_executor(tool_executor_env)
        result = await executor.execute("nonexistent_tool", {})
        assert result["status"] == "error"
        assert result["error_type"] == "tool_not_found"

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error(self, tool_executor_env):
        executor = _make_executor(tool_executor_env)
        result = await executor.execute("failing_tool", {})
        assert result["status"] == "error"
        assert result["error_type"] == "execution_error"
        assert "Something broke" in result["message"]

    @pytest.mark.asyncio
    async def test_tool_in_config_but_missing_from_module(self, tool_executor_env):
        """A tool defined in the YAML config but absent from the module returns function_not_found."""
        executor = _make_executor(tool_executor_env)
        # Remove a tool function that exists in the config
        del executor.tool_functions["get_reservation"]
        result = await executor.execute("get_reservation", {"confirmation_number": "ABC123"})
        assert result["status"] == "error"
        assert result["error_type"] == "function_not_found"

    @pytest.mark.asyncio
    async def test_call_index_increments(self, tool_executor_env):
        """call_index passed to the tool increments on repeated calls."""
        executor = _make_executor(tool_executor_env)
        await executor.execute("get_reservation", {"confirmation_number": "ABC123"})
        await executor.execute("get_reservation", {"confirmation_number": "ABC123"})
        assert executor._tool_call_counts["get_reservation"] == 2


class TestToolExecutorDbMutation:
    """Tests for ToolExecutor database state changes and reset."""

    @pytest.mark.asyncio
    async def test_tool_mutates_db(self, tool_executor_env):
        """A tool that writes to the db persists changes for subsequent calls."""
        executor = _make_executor(tool_executor_env)

        # Verify initial state
        assert executor.db["reservations"]["ABC123"]["status"] == "confirmed"

        # Mutate via update_reservation
        result = await executor.execute(
            "update_reservation",
            {"confirmation_number": "ABC123", "new_status": "cancelled"},
        )
        assert result["status"] == "success"
        assert result["updated"]["status"] == "cancelled"

        # DB should reflect the mutation
        assert executor.db["reservations"]["ABC123"]["status"] == "cancelled"

        # Subsequent read should see the mutated state
        result2 = await executor.execute("get_reservation", {"confirmation_number": "ABC123"})
        assert result2["reservation"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_reset_restores_original_db(self, tool_executor_env):
        """reset() restores the database to its original state and clears logs."""
        executor = _make_executor(tool_executor_env)

        # Mutate the db
        await executor.execute(
            "update_reservation",
            {"confirmation_number": "ABC123", "new_status": "cancelled"},
        )
        assert executor.db["reservations"]["ABC123"]["status"] == "cancelled"

        # Reset
        executor.reset()

        # DB should be back to original
        assert executor.db["reservations"]["ABC123"]["status"] == "confirmed"
        assert executor._tool_call_counts == {}

    @pytest.mark.asyncio
    async def test_original_db_not_mutated(self, tool_executor_env):
        """Mutations to db do not leak into original_db."""
        executor = _make_executor(tool_executor_env)

        await executor.execute(
            "update_reservation",
            {"confirmation_number": "ABC123", "new_status": "cancelled"},
        )

        # original_db should be untouched
        assert executor.original_db["reservations"]["ABC123"]["status"] == "confirmed"
