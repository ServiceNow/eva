"""Tests for GeminiLiveAssistantServer._create_genai_client routing.

Vertex-only Live/S2S preview models must route through Vertex AI
(aiplatform.googleapis.com with vertexai=True). A Developer API key would send
them to generativelanguage.googleapis.com/v1beta where they 404, so Vertex must
win whenever a project is resolvable (or GOOGLE_GENAI_USE_VERTEXAI is set) and
the Developer key must be ignored in that mode.
"""

from unittest.mock import patch

import pytest

from eva.assistant.gemini_live_server import GeminiLiveAssistantServer


def _make_server(**attrs) -> GeminiLiveAssistantServer:
    """Build a bare server instance with only the client-factory inputs set.

    Bypasses __init__ (which needs a full pipeline config) and injects the
    attributes _create_genai_client reads.
    """
    server = object.__new__(GeminiLiveAssistantServer)
    defaults = {
        "_api_key": "",
        "_endpoint": None,
        "_api_version": None,
        "_vertex_project": None,
        "_vertex_location": "us-central1",
    }
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(server, k, v)
    return server


@pytest.fixture
def mock_client():
    with patch("eva.assistant.gemini_live_server.genai.Client") as mock:
        yield mock


class TestCreateGenaiClient:
    def test_vertex_when_project_resolvable_ignores_dev_key(self, mock_client, monkeypatch):
        """A resolvable project routes to Vertex even when a Developer key is present."""
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        server = _make_server(_vertex_project="proj-x", _api_key="AIzaSyDEVKEY")

        server._create_genai_client()

        _, kwargs = mock_client.call_args
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "proj-x"
        assert kwargs["location"] == "us-central1"
        # The Developer API key must NOT be forwarded in Vertex mode.
        assert "api_key" not in kwargs

    def test_use_vertexai_flag_forces_vertex(self, mock_client, monkeypatch):
        """GOOGLE_GENAI_USE_VERTEXAI=1 forces Vertex when a project is present."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
        server = _make_server(_vertex_project="proj-x", _api_key="AIzaSyDEVKEY")

        server._create_genai_client()

        _, kwargs = mock_client.call_args
        assert kwargs["vertexai"] is True

    def test_flag_zero_forces_dev_api_when_key_present(self, mock_client, monkeypatch):
        """GOOGLE_GENAI_USE_VERTEXAI=0 opts out of Vertex even if a project exists."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "0")
        server = _make_server(_vertex_project="proj-x", _api_key="AIzaSyDEVKEY")

        server._create_genai_client()

        _, kwargs = mock_client.call_args
        assert kwargs.get("api_key") == "AIzaSyDEVKEY"
        assert "vertexai" not in kwargs

    def test_vertex_flag_without_project_raises(self, mock_client, monkeypatch):
        """Forcing Vertex without a project is a hard error, not a silent dev-API fallback."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
        server = _make_server(_vertex_project=None)

        with pytest.raises(ValueError, match="no project found"):
            server._create_genai_client()

    def test_endpoint_and_api_version_passed_via_http_options(self, mock_client, monkeypatch):
        """Endpoint and api_version overrides flow into http_options."""
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        server = _make_server(
            _vertex_project="proj-x",
            _endpoint="us-central1-aiplatform.googleapis.com",
            _api_version="v1beta1",
        )

        server._create_genai_client()

        _, kwargs = mock_client.call_args
        http_options = kwargs["http_options"]
        assert http_options.base_url == "wss://us-central1-aiplatform.googleapis.com"
        assert http_options.api_version == "v1beta1"

    def test_dev_api_when_only_key_and_no_project(self, mock_client, monkeypatch):
        """With no project and no flag, a Developer key uses the Developer API."""
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        server = _make_server(_api_key="AIzaSyDEVKEY")

        server._create_genai_client()

        _, kwargs = mock_client.call_args
        assert kwargs.get("api_key") == "AIzaSyDEVKEY"
        assert "vertexai" not in kwargs
