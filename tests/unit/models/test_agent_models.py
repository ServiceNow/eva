"""Unit tests for agent models."""

from eva.models.agents import (
    AgentConfig,
    AgentsConfig,
    AgentTool,
    AgentToolParameter,
)


class TestAgentToolParameter:
    def test_create_simple_parameter(self):
        """Test creating a simple string parameter."""
        param = AgentToolParameter(name="query")

        assert param.name == "query"
        assert param.type == "string"
        assert param.enum is None
        assert param.description == ""

    def test_create_enum_parameter(self):
        """Test creating an enum parameter."""
        param = AgentToolParameter(
            name="leave_type",
            type="string",
            enum=["sick", "personal", "vacation"],
            description="Type of leave",
        )

        assert param.name == "leave_type"
        assert param.enum == ["sick", "personal", "vacation"]
        assert param.description == "Type of leave"


class TestAgentTool:
    def test_create_minimal_tool(self):
        """Test creating a minimal AgentTool."""
        tool = AgentTool(
            id="tool_001",
            name="Get balance",
            description="Gets user balance",
            type="script",
        )

        assert tool.id == "tool_001"
        assert tool.name == "Get balance"
        assert tool.type == "script"
        assert tool.required_parameters == []
        assert tool.optional_parameters == []

    def test_function_name_generation(self):
        """Test function name generation from tool name."""
        # Normal name - splits on ".", takes first part, replaces spaces with "_"
        tool = AgentTool(
            id="t1",
            name="Get time off balance",
            description="desc",
        )
        assert tool.function_name == "get_time_off_balance"

        # Name with dots - splits on ".", takes first part
        tool = AgentTool(
            id="t2",
            name="hr.pto.balance",
            description="desc",
        )
        assert tool.function_name == "hr"

        # Name starting with number
        tool = AgentTool(
            id="t3",
            name="123 Tool",
            description="desc",
        )
        assert tool.function_name == "_123_tool"

    def test_get_required_param_names(self):
        """Test getting required parameter names."""
        tool = AgentTool(
            id="t1",
            name="Submit",
            description="desc",
            type="subflow",
            required_parameters=[
                "start_date",
                AgentToolParameter(name="end_date", description="End date"),
            ],
        )

        names = tool.get_required_param_names()
        assert names == ["start_date", "end_date"]

    def test_get_parameter_properties(self):
        """Test getting parameter properties for OpenAI format."""
        tool = AgentTool(
            id="t1",
            name="Search",
            description="Search tool",
            type="action",
            required_parameters=[
                AgentToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                ),
            ],
            optional_parameters=[
                AgentToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                ),
                "simple_param",
            ],
        )

        props = tool.get_parameter_properties()

        assert "query" in props
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "Search query"

        assert "limit" in props
        assert props["limit"]["type"] == "integer"

        assert "simple_param" in props
        assert props["simple_param"]["type"] == "string"


class TestAgentConfig:
    def test_create_minimal_agent(self):
        """Test creating a minimal AgentConfig."""
        agent = AgentConfig(
            id="agent_001",
            name="Test Agent",
            description="A test agent",
            role="You are a test agent",
            instructions="Help users with tests",
            tool_module_path="eva.assistant.tools.test_tools",
        )

        assert agent.id == "agent_001"
        assert agent.name == "Test Agent"
        assert agent.tools == []

    def test_get_tool_by_name(self):
        """Test getting a tool by name."""
        agent = AgentConfig(
            id="agent",
            name="Agent",
            description="desc",
            role="role",
            instructions="instr",
            tool_module_path="eva.assistant.tools.test_tools",
            tools=[
                AgentTool(id="t1", name="Tool A", description="A", type="script"),
                AgentTool(id="t2", name="Tool B", description="B", type="action"),
            ],
        )

        tool = next((t for t in agent.tools if t.name == "Tool A"), None)
        assert tool is not None
        assert tool.id == "t1"

        tool = next((t for t in agent.tools if t.name == "Nonexistent"), None)
        assert tool is None


class TestAgentsConfig:
    def test_create_agents_config(self):
        """Test creating an AgentsConfig."""
        config = AgentsConfig(
            agents=[
                AgentConfig(
                    id="a1",
                    name="Agent 1",
                    description="d1",
                    role="r1",
                    instructions="i1",
                    tool_module_path="eva.assistant.tools.test_tools",
                ),
                AgentConfig(
                    id="a2",
                    name="Agent 2",
                    description="d2",
                    role="r2",
                    instructions="i2",
                    tool_module_path="eva.assistant.tools.test_tools",
                ),
            ]
        )

        assert len(config.agents) == 2

    def test_get_agent_by_id(self):
        """Test getting an agent by ID."""
        config = AgentsConfig(
            agents=[
                AgentConfig(
                    id="agent_1",
                    name="Agent 1",
                    description="d",
                    role="r",
                    instructions="i",
                    tool_module_path="eva.assistant.tools.test_tools",
                ),
            ]
        )

        agent = config.get_agent_by_id("agent_1")
        assert agent is not None
        assert agent.name == "Agent 1"

        agent = config.get_agent_by_id("nonexistent")
        assert agent is None

    def test_yaml_roundtrip(self, temp_dir, sample_agent_data):
        """Test saving and loading from YAML."""
        config = AgentsConfig.model_validate(sample_agent_data)

        yaml_path = temp_dir / "agents.yaml"
        config.to_yaml(yaml_path)

        loaded = AgentsConfig.from_yaml(yaml_path)
        assert len(loaded.agents) == 1
        assert loaded.agents[0].name == "HR PTO Agent"
        assert len(loaded.agents[0].tools) == 2
