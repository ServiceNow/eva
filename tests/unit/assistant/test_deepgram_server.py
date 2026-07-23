"""Tests for DeepgramAssistantServer settings + tool conversion helpers."""

from unittest.mock import MagicMock

from deepgram.agent.v1.types.agent_v1settings import AgentV1Settings

from eva.assistant.deepgram_server import (
    DeepgramAssistantServer,
    _agent_tools_to_deepgram,
)
from eva.models.agents import AgentConfig, AgentTool, AgentToolParameter

INITIAL_MESSAGE = "Hello! How can I help you today?"


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
    srv.language = "en"
    srv._listen_model = "nova-3"
    srv._think_provider = "open_ai"
    srv._think_model = "gpt-4o-mini"
    srv._model = "gpt-4o-mini"
    srv._speak_model = "aura-2-thalia-en"
    srv._system_prompt = "you are a helpful assistant"
    srv._functions = None
    srv.initial_message = INITIAL_MESSAGE
    # BYO think config (all default to "off"; managed provider path).
    srv._think_credentials = None
    srv._think_endpoint = None
    srv._think_params = {}
    srv._context_length = None
    srv._think_region = None
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


class TestBuildSettingsByo:
    """BYO ('bring your own') think configuration is opt-in and additive."""

    def test_managed_default_has_no_credentials_or_endpoint(self):
        # With no BYO params set, the managed path is unchanged.
        think = _bare_server()._build_settings().dict()["agent"]["think"]
        assert "credentials" not in think["provider"]
        assert think.get("endpoint") is None
        assert think.get("context_length") is None

    def test_bedrock_explicit_credentials_passthrough(self):
        srv = _bare_server()
        srv._think_provider = "aws_bedrock"
        srv._think_model = "anthropic/claude-3-5-haiku-20240307-v1:0"
        srv._think_credentials = {
            "type": "iam",
            "region": "us-east-1",
            "access_key_id": "AK",
            "secret_access_key": "SK",
        }
        provider = srv._build_settings().dict()["agent"]["think"]["provider"]
        assert provider["type"] == "aws_bedrock"
        assert provider["credentials"]["region"] == "us-east-1"
        assert provider["credentials"]["access_key_id"] == "AK"

    def test_endpoint_params_and_context_length_passthrough(self):
        srv = _bare_server()
        srv._think_params = {"temperature": 0.5}
        srv._think_endpoint = {"url": "https://example.test/v1", "headers": {"x-api-key": "KEY"}}
        srv._context_length = "max"
        think = srv._build_settings().dict()["agent"]["think"]
        assert think["provider"]["temperature"] == 0.5
        assert think["endpoint"]["url"] == "https://example.test/v1"
        assert think["endpoint"]["headers"]["x-api-key"] == "KEY"
        assert think["context_length"] == "max"

    def test_bedrock_credentials_built_from_env(self, monkeypatch):
        srv = _bare_server()
        srv._think_provider = "aws_bedrock"
        srv._think_model = "anthropic/claude-3-5-haiku-20240307-v1:0"
        srv._think_region = "us-west-2"
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "envAK")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envSK")
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
        creds = srv._build_settings().dict()["agent"]["think"]["provider"]["credentials"]
        assert creds["type"] == "iam"
        assert creds["region"] == "us-west-2"
        assert creds["access_key_id"] == "envAK"

    def test_google_custom_endpoint_omits_model(self):
        # Deepgram custom Google endpoints want the model in the URL, not in settings.
        srv = _bare_server()
        srv._think_provider = "google"
        srv._think_model = "gemini-3.5-flash"
        srv._think_endpoint = {
            "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
            "headers": {"x-goog-api-key": "KEY"},
        }
        assert srv._omit_think_model() is True
        think = srv._build_settings_dict()["agent"]["think"]
        assert think["provider"] == {"type": "google"}  # no model
        assert "streamGenerateContent" in think["endpoint"]["url"]

    def test_managed_google_keeps_model(self):
        # google WITHOUT a custom endpoint is managed -> model stays required/included.
        srv = _bare_server()
        srv._think_provider = "google"
        srv._think_model = "gemini-2.5-flash"
        assert srv._omit_think_model() is False
        think = srv._build_settings().dict()["agent"]["think"]
        assert think["provider"]["model"] == "gemini-2.5-flash"
