"""Python function-based tool executor that replaces YAML-based declarative mocking."""

import copy
import importlib
import json
from collections.abc import Callable
from pathlib import Path

import yaml
from pipecat.services.llm_service import FunctionCallParams

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """Python function-based tool executor.

    Executes tools as Python functions with explicit logic.
    Each tool is implemented as a regular Python function.

    Architecture:
    - One Python module per agent (e.g., airline_tools.py)
    - Scenario database JSON files contain test data
    - Tool functions query/mutate database directly
    - Supports stateful execution with enable_state=True
    """

    def __init__(
        self,
        tool_config_path: str,
        scenario_db_path: str,
        tool_module_path: str,
        current_date_time: str,
    ):
        """Initialize the tool executor.

        Args:
            tool_config_path: Path to agent YAML (for schema validation)
            scenario_db_path: Path to scenario JSON database
            tool_module_path: Python module path with tool functions (e.g., "eva.assistant.tools.airline_tools")
            current_date_time: Current date/time string from the evaluation record (e.g. "2026-06-23 10:45 EST")
        """
        self.tool_config_path = Path(tool_config_path)
        self.scenario_db_path = Path(scenario_db_path)
        self.tool_module_path = tool_module_path
        self._current_date: str = current_date_time.split(" ")[0]

        # Load configurations
        self.tool_configs = self._load_tool_configs()
        self.db = self._load_scenario_db()
        self.db["_current_date"] = self._current_date
        self.original_db = copy.deepcopy(self.db)

        # Load tool functions module
        self.tool_functions = self._load_tool_module()

        # Call tracking
        self._tool_call_counts: dict[str, int] = {}

    def _load_tool_configs(self) -> dict[str, dict]:
        """Load tool configurations from YAML for schema validation."""
        with open(self.tool_config_path) as f:
            config = yaml.safe_load(f)

        # Extract tools from agent config
        tools = {}
        if isinstance(config, dict) and "tools" in config:
            for tool in config["tools"]:
                tools[tool["id"]] = tool
        elif isinstance(config, list):
            # Multiple agents - combine all tools
            for agent in config:
                if "tools" in agent:
                    for tool in agent["tools"]:
                        tools[tool["id"]] = tool

        return tools

    def _load_scenario_db(self) -> dict:
        """Load scenario database from JSON."""
        with open(self.scenario_db_path) as f:
            return json.load(f)

    def _load_tool_module(self) -> dict[str, Callable]:
        """Dynamically load tool functions from module.

        Uses importlib.util.spec_from_file_location to avoid triggering
        package __init__.py files which may have heavy imports.

        Returns:
            Dictionary mapping tool_name -> function
        """
        try:
            # Convert module path to file path
            # e.g., "eva.assistant.tools.airline_tools" -> "src/eva/assistant/tools/airline_tools.py"
            module_parts = self.tool_module_path.split(".")
            module_file = Path("src") / "/".join(module_parts) / "__init__.py"

            # Try with .py extension if __init__.py doesn't exist
            if not module_file.exists():
                module_file = Path("src") / "/".join(module_parts[:-1]) / f"{module_parts[-1]}.py"

            if not module_file.exists():
                raise FileNotFoundError(f"Could not find module file for {self.tool_module_path}")

            # Load module directly from file to avoid __init__.py chain
            spec = importlib.util.spec_from_file_location(self.tool_module_path, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract all functions from module that don't start with underscore
            tool_functions = {}
            for name in dir(module):
                if not name.startswith("_"):
                    obj = getattr(module, name)
                    if callable(obj):
                        tool_functions[name] = obj

            logger.info(f"Loaded {len(tool_functions)} tool functions from {module_file}")
            return tool_functions
        except Exception as e:
            logger.error(f"Failed to load tool module {self.tool_module_path}: {e}")
            raise

    async def execute_realtime_tool(self, params: FunctionCallParams):
        logger.info(f"Executing realtime tool: {params.function_name}, params {params.arguments}")
        result = await self.execute(params.function_name, params.arguments)
        await params.result_callback(result)

    async def execute(self, tool_name: str, params: dict) -> dict:
        """Execute a tool call and return the response.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters

        Returns:
            Response dictionary (success or error)
        """
        # Track call count
        self._tool_call_counts[tool_name] = self._tool_call_counts.get(tool_name, 0) + 1
        call_index = self._tool_call_counts[tool_name]

        # Get tool config for validation
        if tool_name not in self.tool_configs:
            return {
                "status": "error",
                "error_type": "tool_not_found",
                "message": f"Tool {tool_name} not found in configuration",
            }

        # Get tool function
        tool_function = self._get_tool_function(tool_name)
        if not tool_function:
            return {
                "status": "error",
                "error_type": "function_not_found",
                "message": f"No Python function found for tool {tool_name}",
            }

        # Execute tool function
        try:
            result = tool_function(params=params, db=self.db, call_index=call_index)
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "execution_error",
                "message": f"Tool execution failed: {str(e)}",
            }

        if "status" not in result:
            raise ValueError(f"Tool {tool_name} returned response without 'status' field: {result}")
        return result

    def _get_tool_function(self, tool_name: str) -> Callable | None:
        """Get tool function by name."""
        return self.tool_functions.get(tool_name)

    def reset(self):
        """Reset database state and call counts."""
        self.db = copy.deepcopy(self.original_db)
        self._tool_call_counts = {}
