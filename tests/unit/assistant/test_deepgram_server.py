"""Tests for DeepgramAssistantServer settings + tool conversion helpers."""

from unittest.mock import MagicMock

from deepgram.agent.v1.types.agent_v1settings import AgentV1Settings

from eva.assistant.deepgram_server import (
    INITIAL_MESSAGE,
    DeepgramAssistantServer,
    _agent_tools_to_deepgram,
)
from eva.models.agents import AgentConfig, AgentTool, AgentToolParameter


def _agent_with_tools() -> AgentConfig:
    return AgentConfig(
        id="a1",
        name="Test Agent",
        description="desc",
        role="role",
        instructions="be helpful",
        tool_module_path="eva.assistant.tools.airline_tools",
        tools=[
            AgentTool(
                id="t1",
                name="Lookup Booking",
                description="Look up a booking",
                required_parameters=[AgentToolParameter(name="booking_id", type="str", description="The booking id")],
                optional_parameters=[AgentToolParameter(name="verbose", type="bool")],
            )
        ],
    )


def _bare_server() -> DeepgramAssistantServer:
    """Build a server without running __init__ (which needs file-backed tool config)."""
    srv = object.__new__(DeepgramAssistantServer)
    srv._audio_sample_rate = 24000
    srv._language = "en"
    srv._listen_model = "nova-3"
    srv._think_provider = "open_ai"
    srv._think_model = "gpt-4o-mini"
    srv._model = "gpt-4o-mini"
    srv._speak_model = "aura-2-thalia-en"
    srv._system_prompt = "you are a helpful assistant"
    srv._functions = None
    return srv


class TestToolConversion:
    def test_no_tools_returns_none(self):
        agent = MagicMock()
        agent.tools = []
        assert _agent_tools_to_deepgram(agent) is None

    def test_tool_converted_to_client_side_function(self):
        functions = _agent_tools_to_deepgram(_agent_with_tools())
        assert functions is not None
        assert len(functions) == 1
        fn = functions[0]
        # Client-side functions have no "endpoint" key.
        assert "endpoint" not in fn
        assert fn["name"]  # function_name derived from the tool
        assert "Lookup Booking" in fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "booking_id" in params["properties"]
        assert params["required"] == ["booking_id"]


class TestBuildSettings:
    def test_audio_encoding_and_sample_rate(self):
        settings = _bare_server()._build_settings()
        assert isinstance(settings, AgentV1Settings)
        wire = settings.dict()
        assert wire["audio"]["input"] == {"encoding": "linear16", "sample_rate": 24000}
        assert wire["audio"]["output"]["encoding"] == "linear16"
        assert wire["audio"]["output"]["sample_rate"] == 24000
        # Raw PCM output (no WAV header) so the pacer can stream it directly.
        assert wire["audio"]["output"]["container"] == "none"

    def test_providers_and_greeting(self):
        wire = _bare_server()._build_settings().dict()
        agent = wire["agent"]
        assert agent["greeting"] == INITIAL_MESSAGE
        assert agent["language"] == "en"
        assert agent["listen"]["provider"]["model"] == "nova-3"
        assert agent["think"]["provider"]["type"] == "open_ai"
        assert agent["think"]["provider"]["model"] == "gpt-4o-mini"
        assert agent["think"]["prompt"] == "you are a helpful assistant"
        assert agent["speak"]["provider"]["model"] == "aura-2-thalia-en"

    def test_functions_omitted_when_no_tools(self):
        wire = _bare_server()._build_settings().dict()
        assert "functions" not in wire["agent"]["think"]

    def test_functions_included_when_present(self):
        srv = _bare_server()
        srv._functions = _agent_tools_to_deepgram(_agent_with_tools())
        wire = srv._build_settings().dict()
        functions = wire["agent"]["think"]["functions"]
        assert len(functions) == 1
        assert functions[0]["parameters"]["required"] == ["booking_id"]
