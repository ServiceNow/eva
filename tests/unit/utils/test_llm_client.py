"""Tests for LLMClient: generate_text, retry logic, and Router concurrency."""

import asyncio
import logging
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm import Router
from litellm.exceptions import AuthenticationError, RateLimitError

from eva.utils import router
from eva.utils.llm_client import LLMClient

LLM_LOGGER = "eva.utils.llm_client"


def _make_client(**kwargs) -> tuple[LLMClient, MagicMock]:
    """Create an LLMClient with a mock router and fast retry settings for testing.

    Returns:
        Tuple of (client, mock_router)
    """
    mock_router = MagicMock(spec=Router)
    router._router = mock_router  # inject mock directly
    defaults = {
        "model": "test-model",
        "max_retries": 2,
        "retry_min_wait": 0.001,
        "retry_max_wait": 0.01,
        "retry_multiplier": 2.0,
    }
    defaults.update(kwargs)
    return LLMClient(**defaults), mock_router


def _mock_response(content: str) -> MagicMock:
    """Create a mock acompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _assert_log_messages(caplog: pytest.LogCaptureFixture, expected_messages: tuple[str, ...]):
    """Check that the log messages match the expected patterns."""
    expected = "\n".join(expected_messages)
    actual = "\n".join(record.message.partition(" for ")[0] for record in caplog.records)
    assert re.fullmatch(expected, actual) is not None, f"Expected:\n{expected}\nActual:\n{actual}"


class TestGenerateText:
    """Tests for generate_text behavioral contract."""

    @pytest.mark.asyncio
    async def test_returns_content_on_success(self, caplog: pytest.LogCaptureFixture):
        """Successful call returns the message content."""
        client, mock_router = _make_client()
        caplog.set_level(logging.DEBUG, logger=LLM_LOGGER)

        mock_router.acompletion = AsyncMock(return_value=_mock_response("Hello world"))
        result = await client.generate_text([{"role": "user", "content": "Hi"}])

        assert result == "Hello world"

        _assert_log_messages(
            caplog,
            (
                r"LLM call (\d+) \(attempt 1/3\) started",
                r"LLM call \1 \(attempt 1/3\) succeeded",
            ),
        )

    @pytest.mark.asyncio
    async def test_passes_model(self):
        """Model name is passed directly to the Router."""
        client, mock_router = _make_client(model="gpt-5")

        mock_router.acompletion = AsyncMock(return_value=_mock_response("ok"))
        await client.generate_text([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_router.acompletion.call_args[1]
        assert call_kwargs["model"] == "gpt-5"

    @pytest.mark.asyncio
    async def test_passes_response_format(self):
        """response_format is forwarded to acompletion."""
        client, mock_router = _make_client()
        fmt = {"type": "json_object"}

        mock_router.acompletion = AsyncMock(return_value=_mock_response("{}"))
        await client.generate_text(
            [{"role": "user", "content": "Hi"}],
            response_format=fmt,
        )

        call_kwargs = mock_router.acompletion.call_args[1]
        assert call_kwargs["response_format"] == fmt


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self, caplog: pytest.LogCaptureFixture):
        """Retries on RateLimitError and succeeds on second attempt."""
        client, mock_router = _make_client(max_retries=2)
        caplog.set_level(logging.DEBUG, logger=LLM_LOGGER)

        mock_router.acompletion = AsyncMock(
            side_effect=[
                RateLimitError(
                    "rate limited", llm_provider="openai", model="test", response=MagicMock(status_code=429)
                ),
                _mock_response("ok"),
            ]
        )
        result = await client.generate_text([{"role": "user", "content": "Hi"}])

        assert result == "ok"
        assert mock_router.acompletion.await_count == 2

        _assert_log_messages(
            caplog,
            (
                r"LLM call (\d+) \(attempt 1/3\) started",
                r"LLM call \1 \(attempt 1/3\) failed",
                r"LLM call \1 \(attempt 2/3\) started",
                r"LLM call \1 \(attempt 2/3\) succeeded",
            ),
        )

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retryable_error(self, caplog: pytest.LogCaptureFixture):
        """AuthenticationError is not retried."""
        client, mock_router = _make_client(max_retries=3)
        caplog.set_level(logging.DEBUG, logger=LLM_LOGGER)

        mock_router.acompletion = AsyncMock(
            side_effect=AuthenticationError(
                "invalid key", llm_provider="openai", model="test", response=MagicMock(status_code=401)
            )
        )
        with pytest.raises(AuthenticationError):
            await client.generate_text([{"role": "user", "content": "Hi"}])

        # Should NOT have retried
        assert mock_router.acompletion.await_count == 1

        _assert_log_messages(
            caplog,
            (
                r"LLM call (\d+) \(attempt 1/4\) started",
                r"LLM call \1 \(attempt 1/4\) failed with non-retryable error",
            ),
        )

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, caplog: pytest.LogCaptureFixture):
        """Raises after exhausting all retry attempts."""
        client, mock_router = _make_client(max_retries=2)
        caplog.set_level(logging.DEBUG, logger=LLM_LOGGER)

        mock_router.acompletion = AsyncMock(
            side_effect=RateLimitError(
                "rate limited", llm_provider="openai", model="test", response=MagicMock(status_code=429)
            )
        )
        with pytest.raises(RateLimitError):
            await client.generate_text([{"role": "user", "content": "Hi"}])

        # 1 initial + 2 retries = 3 total
        assert mock_router.acompletion.await_count == 3

        _assert_log_messages(
            caplog,
            (
                r"LLM call (\d+) \(attempt 1/3\) started",
                r"LLM call \1 \(attempt 1/3\) failed",
                r"LLM call \1 \(attempt 2/3\) started",
                r"LLM call \1 \(attempt 2/3\) failed",
                r"LLM call \1 \(attempt 3/3\) started",
                r"LLM call \1 \(attempt 3/3\) failed last retry",
            ),
        )


class TestRouterConcurrency:
    """Verify that the LiteLLM Router enforces per-deployment concurrency limits."""

    @pytest.mark.asyncio
    async def test_max_parallel_requests_bounds_concurrency(self):
        """Peak concurrent calls never exceeds max_parallel_requests."""
        test_router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_key": "test-key",
                        "max_parallel_requests": 2,
                    },
                }
            ],
            num_retries=0,
        )

        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_acompletion(*args, **kwargs):
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return _mock_response("ok")

        with patch("litellm.acompletion", new=mock_acompletion):
            tasks = [
                test_router.acompletion(
                    model="test-model",
                    messages=[{"role": "user", "content": f"msg-{i}"}],
                )
                for i in range(8)
            ]
            await asyncio.gather(*tasks)

        assert peak_concurrent == 2


class TestLitellmParamsForwarding:
    """Verify that litellm_params from the model_list reach the underlying API call."""

    @pytest.mark.asyncio
    async def test_custom_litellm_params_reach_api_call(self):
        """Custom params in litellm_params are forwarded to litellm.acompletion."""
        test_router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_key": "test-key",
                        "custom_param": "must_be_preserved",
                    },
                }
            ],
            num_retries=0,
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _mock_response("ok")
            await test_router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
            )

        call_kwargs = mock.call_args[1]
        assert call_kwargs.get("custom_param") == "must_be_preserved", (
            f"custom_param not found in litellm.acompletion kwargs: {call_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_conftest_litellm_params_reach_api_call(self):
        """litellm_params from conftest's EVA_MODEL_LIST reach litellm.acompletion via the global router."""
        router.init(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_key": "test-key",
                        "custom_param": "must_be_preserved",
                    },
                }
            ]
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _mock_response("ok")
            await router.get().acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
            )

        call_kwargs = mock.call_args[1]
        assert call_kwargs.get("custom_param") == "must_be_preserved", (
            f"custom_param not found in litellm.acompletion kwargs: {call_kwargs}"
        )

        router.reset()
