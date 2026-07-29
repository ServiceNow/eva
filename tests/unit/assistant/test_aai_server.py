"""Tests for AAIAssistantServer configuration, tool schemas, and registration."""

from unittest.mock import MagicMock

from eva.assistant.aai_server import AAIAssistantServer
from eva.assistant.aai_session import DEFAULT_AAI_MODEL, DEFAULT_AAI_WS_URL


def _bare_server(s2s_params: dict | None = None) -> AAIAssistantServer:
    """Construct without __init__ (skips PromptManager and ToolExecutor setup)."""
    srv = object.__new__(AAIAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = s2s_params if s2s_params is not None else {}
    srv.agent = MagicMock()
    srv.agent.tools = []
    srv._model = (s2s_params or {}).get("model", DEFAULT_AAI_MODEL)
    srv._ws_url = (s2s_params or {}).get("ws_url", DEFAULT_AAI_WS_URL)
    return srv


def _tool(function_name: str, name: str, description: str, properties: dict, required: list[str]) -> MagicMock:
    tool = MagicMock()
    tool.function_name = function_name
    tool.name = name
    tool.description = description
    tool.get_parameter_properties.return_value = properties
    tool.get_required_param_names.return_value = required
    return tool


class TestModelName:
    def test_defaults_to_aai_host(self):
        assert _bare_server().model_name == DEFAULT_AAI_MODEL

    def test_honors_configured_model(self):
        assert _bare_server({"model": "aai-host-custom"}).model_name == "aai-host-custom"


class TestBuildAaiTools:
    def test_returns_empty_list_when_agent_has_no_tools(self):
        assert _bare_server()._build_aai_tools() == []

    def test_builds_flat_function_schema(self):
        srv = _bare_server()
        srv.agent.tools = [
            _tool(
                "get_reservation",
                "Get Reservation",
                "Look up a booking",
                {"confirmation_number": {"type": "string"}},
                ["confirmation_number"],
            )
        ]

        tools = srv._build_aai_tools()

        assert tools == [
            {
                "type": "function",
                "name": "get_reservation",
                "description": "Get Reservation: Look up a booking",
                "parameters": {
                    "type": "object",
                    "properties": {"confirmation_number": {"type": "string"}},
                    "required": ["confirmation_number"],
                },
            }
        ]

    def test_description_is_never_empty(self):
        """The aai ToolSchema requires a description of at least one character."""
        srv = _bare_server()
        srv.agent.tools = [_tool("f", "", "", {}, [])]

        description = srv._build_aai_tools()[0]["description"]

        assert len(description) >= 1


class TestWebSocketUrl:
    def test_defaults_to_localhost(self):
        assert _bare_server()._ws_url == DEFAULT_AAI_WS_URL

    def test_honors_configured_url(self):
        assert _bare_server({"ws_url": "ws://aai.internal/websocket"})._ws_url == "ws://aai.internal/websocket"


class TestRegistration:
    def test_framework_literal_accepts_aai(self):
        from eva.models.config import RunConfig

        assert "aai" in RunConfig.model_fields["framework"].annotation.__args__

    def test_worker_resolves_the_aai_server_class(self):
        from eva.orchestrator.worker import _get_server_class

        assert _get_server_class("aai") is AAIAssistantServer

    def test_unknown_framework_error_lists_aai(self):
        from eva.orchestrator.worker import _get_server_class

        try:
            _get_server_class("nope")
        except ValueError as e:
            assert "aai" in str(e)
        else:
            raise AssertionError("expected ValueError")
