"""Agent configuration models."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AgentToolParameter(BaseModel):
    """Agent tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field("string", description="Parameter type")
    enum: list[str] | None = Field(None, description="Allowed values for enum types")
    description: str = Field("", description="Parameter description")
    items: dict[str, Any] | None = Field(None, description="Items schema for array types")
    properties: dict[str, Any] | None = Field(None, description="Properties schema for object types")
    additionalProperties: bool | dict[str, Any] | None = Field(
        None, description="Additional properties for object types"
    )


class AgentTool(BaseModel):
    """Agent tool definition."""

    id: str = Field(..., description="Unique tool identifier")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    required_parameters: list[str | AgentToolParameter] = Field(default_factory=list, description="Required parameters")
    optional_parameters: list[str | AgentToolParameter] = Field(default_factory=list, description="Optional parameters")
    invoke_cache_flush: bool = Field(False, description="Whether to flush cache on invocation")
    tool_type: str | None = Field(None, description="Type of tool: 'read' or 'write'")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(extra="allow")

    @property
    def function_name(self) -> str:
        """Generate a valid function name for LLM tool calling."""
        first_name = self.name.split(".")[0].lower().replace(" ", "_")
        cleaned = "".join([c for c in first_name if c.isalnum() or c == "_"])
        if cleaned and cleaned[0].isdigit():
            cleaned = "_" + cleaned
        return cleaned or "unnamed_tool"

    def get_required_param_names(self) -> list[str]:
        """Get list of required parameter names."""
        return [p if isinstance(p, str) else p.name for p in self.required_parameters]

    def get_parameter_properties(self) -> dict[str, dict[str, Any]]:
        """Build parameter properties dict for OpenAI function calling format."""
        # Map Python types to JSON Schema types
        type_mapping = {
            "list": "array",
            "dict": "object",
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }

        properties: dict[str, dict[str, Any]] = {}

        for param in self.required_parameters + self.optional_parameters:
            if isinstance(param, str):
                properties[param] = {"type": "string"}
            else:
                # Convert Python type to JSON Schema type
                json_schema_type = type_mapping.get(param.type, param.type)
                param_def: dict[str, Any] = {"type": json_schema_type}

                if param.description:
                    param_def["description"] = param.description
                if param.enum:
                    param_def["enum"] = param.enum

                # Handle array types - must have items
                if json_schema_type == "array":
                    if param.items:
                        param_def["items"] = param.items
                    else:
                        # Default to object items if not specified
                        param_def["items"] = {"type": "object"}

                # Handle object types
                if json_schema_type == "object":
                    if param.properties:
                        param_def["properties"] = param.properties
                    if param.additionalProperties is not None:
                        param_def["additionalProperties"] = param.additionalProperties

                properties[param.name] = param_def

        return properties


class AgentConfig(BaseModel):
    """Agent configuration."""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    role: str = Field(..., description="Agent role description")
    instructions: str = Field(..., description="Agent instructions/prompt")
    tools: list[AgentTool] = Field(default_factory=list, description="Tools available to this agent")
    personality: str | None = Field(None, description="Agent personality description")
    tool_module_path: str = Field(
        description="Python module path for tool implementations (e.g., 'eva.assistant.tools.airline_tools')",
    )
    user_simulator_context: str = Field(
        "",
        description=(
            "Domain-specific context line prepended to the user simulator system prompt. "
            "Describes who the user is in this domain, e.g. "
            "'You are a passenger of SkyWay Airlines calling customer service.' "
            "Required when adding a new domain."
        ),
    )

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AgentConfig":
        """Load a single agent configuration from YAML file.

        The YAML file must contain exactly one agent definition.
        Use the format: single agent dict (not wrapped in 'agents' list).

        Args:
            path: Path to agent YAML file

        Returns:
            AgentConfig instance

        Raises:
            ValueError: If file doesn't contain exactly one agent
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Check if this is a single agent config (dict) or legacy format (with 'agents' list)
        if isinstance(data, dict):
            # Check if it has 'agents' key (legacy multi-agent format)
            if "agents" in data:
                agents_list = data["agents"]
                if not isinstance(agents_list, list):
                    raise ValueError(
                        f"Invalid agent config format in {path}. "
                        f"Expected a list under 'agents' key or a single agent dict."
                    )
                if len(agents_list) == 0:
                    raise ValueError(f"Agent config file {path} contains no agents. Must contain exactly one agent.")
                if len(agents_list) > 1:
                    raise ValueError(
                        f"Agent config file {path} contains {len(agents_list)} agents. "
                        f"Must contain exactly one agent. "
                        f"Please use separate files for each agent."
                    )
                # Extract the single agent
                agent_data = agents_list[0]
            else:
                # Direct agent config (new format)
                agent_data = data
        else:
            raise ValueError(f"Invalid agent config format in {path}. Expected a dictionary with agent configuration.")

        return cls.model_validate(agent_data)

    def build_tools_for_agent(self) -> list[dict] | None:
        """Build the tools list in OpenAI format for an agent."""
        if not self.tools:
            return None
        tools = []
        for tool in self.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.function_name,
                        "description": f"{tool.name}: {tool.description}",
                        "parameters": {
                            "type": "object",
                            "properties": tool.get_parameter_properties(),
                            "required": tool.get_required_param_names(),
                        },
                    },
                }
            )
        return tools

    def build_tools_for_realtime(self) -> list[dict] | None:
        """Build the tools list in OpenAI format for an agent."""
        if not self.tools:
            return None
        tools = []
        for tool in self.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.function_name,
                        "description": f"{tool.name}: {tool.description}",
                        "properties": tool.get_parameter_properties(),
                        "required": tool.get_required_param_names(),
                    },
                }
            )
        return tools


class AgentsConfig(BaseModel):
    """Collection of agents for a benchmark run."""

    agents: list[AgentConfig] = Field(default_factory=list, description="List of available agents")

    def get_agent_by_id(self, agent_id: str) -> AgentConfig | None:
        """Get an agent by its ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_agent_by_name(self, name: str) -> AgentConfig | None:
        """Get an agent by its name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AgentsConfig":
        """Load agents configuration from YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save agents configuration to YAML file."""
        path = Path(path)
        data = self.model_dump(mode="json", exclude_none=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
