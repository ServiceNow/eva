"""Tests for eva.utils.error_handler module."""

import httpx
import pytest
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
)

from eva.utils.error_handler import (
    categorize_error,
    create_error_details,
    get_error_source,
    is_retryable_error,
)

# LiteLLM exceptions have inconsistent constructors. Build a registry of factories.
_MOCK_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
_MOCK_RESPONSE = httpx.Response(status_code=403, request=_MOCK_REQUEST)

_EXCEPTION_FACTORIES = {
    Timeout: lambda msg: Timeout(message=msg, model="gpt-5.2", llm_provider="openai"),
    APIConnectionError: lambda msg: APIConnectionError(message=msg, model="gpt-5.2", llm_provider="openai"),
    RateLimitError: lambda msg: RateLimitError(message=msg, model="gpt-5.2", llm_provider="openai"),
    ServiceUnavailableError: lambda msg: ServiceUnavailableError(message=msg, model="gpt-5.2", llm_provider="openai"),
    InternalServerError: lambda msg: InternalServerError(message=msg, model="gpt-5.2", llm_provider="openai"),
    AuthenticationError: lambda msg: AuthenticationError(message=msg, model="gpt-5.2", llm_provider="openai"),
    BadRequestError: lambda msg: BadRequestError(message=msg, model="gpt-5.2", llm_provider="openai"),
    ContextWindowExceededError: lambda msg: ContextWindowExceededError(
        message=msg, model="gpt-5.2", llm_provider="openai"
    ),
    ContentPolicyViolationError: lambda msg: ContentPolicyViolationError(
        message=msg, model="gpt-5.2", llm_provider="openai"
    ),
    InvalidRequestError: lambda msg: InvalidRequestError(message=msg, model="gpt-5.2", llm_provider="openai"),
    UnprocessableEntityError: lambda msg: UnprocessableEntityError(message=msg, model="gpt-5.2", llm_provider="openai"),
    NotFoundError: lambda msg: NotFoundError(message=msg, model="gpt-5.2", llm_provider="openai"),
    PermissionDeniedError: lambda msg: PermissionDeniedError(
        message=msg, model="gpt-5.2", llm_provider="openai", response=_MOCK_RESPONSE
    ),
    BudgetExceededError: lambda msg: BudgetExceededError(current_cost=10.0, max_budget=5.0, message=msg),
    APIError: lambda msg: APIError(status_code=502, message=msg, model="gpt-5.2", llm_provider="openai"),
}


def _make_litellm_error(cls, message="test error"):
    """Create a LiteLLM exception using the factory registry."""
    factory = _EXCEPTION_FACTORIES.get(cls)
    if factory:
        return factory(message)
    raise ValueError(f"No factory registered for {cls}")


class TestCategorizeError:
    def test_timeout(self):
        err = _make_litellm_error(Timeout)
        info = categorize_error(err)
        assert info.error_type == "timeout_error"
        assert info.is_retryable is True

    def test_api_connection_error(self):
        err = _make_litellm_error(APIConnectionError)
        info = categorize_error(err)
        assert info.error_type == "network_error"
        assert info.is_retryable is True

    def test_rate_limit_error(self):
        err = _make_litellm_error(RateLimitError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is True

    def test_service_unavailable(self):
        err = _make_litellm_error(ServiceUnavailableError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is True

    def test_internal_server_error(self):
        err = _make_litellm_error(InternalServerError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is True

    def test_authentication_error(self):
        err = _make_litellm_error(AuthenticationError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is False

    def test_permission_denied(self):
        err = _make_litellm_error(PermissionDeniedError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is False

    def test_bad_request(self):
        err = _make_litellm_error(BadRequestError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is False

    def test_context_window_exceeded(self):
        err = _make_litellm_error(ContextWindowExceededError)
        info = categorize_error(err)
        assert info.is_retryable is False

    def test_not_found(self):
        err = _make_litellm_error(NotFoundError)
        info = categorize_error(err)
        assert info.is_retryable is False

    def test_budget_exceeded(self):
        err = _make_litellm_error(BudgetExceededError)
        info = categorize_error(err)
        assert info.is_retryable is False

    def test_generic_api_error(self):
        err = _make_litellm_error(APIError)
        info = categorize_error(err)
        assert info.error_type == "llm_error"
        assert info.is_retryable is True

    def test_asyncio_timeout(self):
        err = TimeoutError()
        info = categorize_error(err)
        assert info.error_type == "timeout_error"
        assert info.error_source == "system"
        assert info.is_retryable is True

    def test_tts_cartesia(self):
        err = Exception("Cartesia connection failed")
        info = categorize_error(err)
        assert info.error_type == "tts_error"
        assert info.error_source == "cartesia"
        assert info.is_retryable is True

    def test_tts_elevenlabs(self):
        err = Exception("ElevenLabs API rate limited")
        info = categorize_error(err)
        assert info.error_type == "tts_error"
        assert info.error_source == "elevenlabs"

    def test_stt_deepgram(self):
        err = Exception("Deepgram transcription error")
        info = categorize_error(err)
        assert info.error_type == "stt_error"
        assert info.error_source == "deepgram"

    def test_stt_assemblyai(self):
        err = Exception("AssemblyAI connection refused")
        info = categorize_error(err)
        assert info.error_type == "stt_error"
        assert info.error_source == "assemblyai"

    def test_tool_error(self):
        err = Exception("Tool execution failed")
        info = categorize_error(err)
        assert info.error_type == "tool_error"
        assert info.error_source == "tool_executor"
        assert info.is_retryable is False

    def test_function_error(self):
        err = Exception("Function call error")
        info = categorize_error(err)
        assert info.error_type == "tool_error"

    def test_port_error(self):
        err = Exception("Port 8080 already in use")
        info = categorize_error(err)
        assert info.error_type == "system_error"
        assert info.error_source == "port_pool"
        assert info.is_retryable is False

    def test_default_fallback(self):
        err = Exception("Something completely unexpected")
        info = categorize_error(err)
        assert info.error_type == "system_error"
        assert info.error_source == "unknown"
        assert info.is_retryable is False


class TestGetErrorSource:
    def test_llm_provider_attribute(self):
        err = _make_litellm_error(Timeout)
        assert get_error_source(err) == "openai"

    def test_asyncio_timeout(self):
        assert get_error_source(TimeoutError()) == "system"

    def test_tts_providers(self):
        assert get_error_source(Exception("cartesia error")) == "cartesia"
        assert get_error_source(Exception("elevenlabs error")) == "elevenlabs"

    def test_stt_providers(self):
        assert get_error_source(Exception("deepgram error")) == "deepgram"
        assert get_error_source(Exception("assemblyai error")) == "assemblyai"

    def test_provider_in_error_string(self):
        err = Exception("Error from anthropic API")
        assert get_error_source(err) == "anthropic"

    def test_unknown(self):
        assert get_error_source(Exception("random error")) == "unknown"


class TestIsRetryableError:
    @pytest.mark.parametrize(
        "cls",
        [Timeout, APIConnectionError, RateLimitError, ServiceUnavailableError, InternalServerError],
    )
    def test_retryable_litellm_errors(self, cls):
        err = _make_litellm_error(cls)
        assert is_retryable_error(err) is True

    def test_asyncio_timeout_retryable(self):
        assert is_retryable_error(TimeoutError()) is True

    def test_tts_stt_retryable(self):
        assert is_retryable_error(Exception("cartesia error")) is True
        assert is_retryable_error(Exception("deepgram error")) is True

    def test_non_retryable(self):
        err = _make_litellm_error(AuthenticationError)
        assert is_retryable_error(err) is False

    def test_generic_exception_not_retryable(self):
        assert is_retryable_error(Exception("random")) is False


class TestCreateErrorDetails:
    def test_creates_error_details(self):
        err = _make_litellm_error(Timeout)
        details = create_error_details(err, retry_count=2, retry_succeeded=True)
        assert details.error_type == "timeout_error"
        assert details.error_source == "openai"
        assert details.is_retryable is True
        assert details.retry_count == 2
        assert details.retry_succeeded is True
        assert details.original_error == str(err)
        assert len(details.timestamps) == 1

    def test_stack_trace_included(self):
        try:
            raise ValueError("test")
        except ValueError as e:
            details = create_error_details(e)
        assert "ValueError" in details.stack_trace
        assert "test" in details.stack_trace
