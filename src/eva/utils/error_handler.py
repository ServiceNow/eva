"""Centralized error handling utility using LiteLLM native exception handling.

This module provides unified error categorization, source identification, and retry logic
by leveraging LiteLLM's built-in exception types and attributes instead of string pattern matching.
"""

import asyncio
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime

from litellm.exceptions import (
    # Network/Connection Errors
    APIConnectionError,
    # Base exceptions
    APIError,
    # 4xx Client Errors
    AuthenticationError,
    BadRequestError,
    # Other
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    # 5xx Server Errors
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
)

from eva.models.results import ErrorDetails
from eva.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ErrorInfo:
    """Structured error information from categorization."""

    error_type: str  # Maps to ErrorDetails.error_type
    error_source: str  # Provider or component name
    is_retryable: bool
    status_code: int | None
    original_exception: Exception


def categorize_error(error: Exception) -> ErrorInfo:
    """Categorize any error using LiteLLM exceptions with fallbacks.

    Priority order:
    1. LiteLLM exceptions (use native attributes)
    2. Known non-LLM exceptions (asyncio.TimeoutError, etc.)
    3. String pattern fallbacks (for TTS/STT/tool errors)

    Args:
        error: The exception to categorize

    Returns:
        ErrorInfo with error_type, error_source, is_retryable, status_code
    """
    error_str = str(error).lower()
    error_source = get_error_source(error)
    status_code = getattr(error, "status_code", None)

    # LiteLLM timeout errors
    if isinstance(error, Timeout):
        return ErrorInfo(
            error_type="timeout_error",
            error_source=error_source,
            is_retryable=True,
            status_code=status_code,
            original_exception=error,
        )

    # Network/connection errors
    if isinstance(error, APIConnectionError):
        return ErrorInfo(
            error_type="network_error",
            error_source=error_source,
            is_retryable=True,
            status_code=status_code,
            original_exception=error,
        )

    # Rate limit errors (429)
    if isinstance(error, RateLimitError):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=True,
            status_code=status_code,
            original_exception=error,
        )

    # Server errors (5xx) - retryable
    if isinstance(error, (ServiceUnavailableError, InternalServerError)):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=True,
            status_code=status_code,
            original_exception=error,
        )

    # Authentication errors (401) - not retryable
    if isinstance(error, AuthenticationError):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=False,
            status_code=status_code,
            original_exception=error,
        )

    # Permission errors (403) - not retryable
    if isinstance(error, PermissionDeniedError):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=False,
            status_code=status_code,
            original_exception=error,
        )

    # Bad request errors (400) - not retryable
    if isinstance(
        error,
        (
            BadRequestError,
            ContextWindowExceededError,
            ContentPolicyViolationError,
            InvalidRequestError,
            UnprocessableEntityError,
        ),
    ):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=False,
            status_code=status_code,
            original_exception=error,
        )

    # Not found errors (404) - not retryable
    if isinstance(error, NotFoundError):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=False,
            status_code=status_code,
            original_exception=error,
        )

    # Budget exceeded errors - not retryable
    if isinstance(error, BudgetExceededError):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=False,
            status_code=status_code,
            original_exception=error,
        )

    # Generic LiteLLM API errors
    if isinstance(error, APIError):
        return ErrorInfo(
            error_type="llm_error",
            error_source=error_source,
            is_retryable=True,  # Generic API errors are retryable
            status_code=status_code,
            original_exception=error,
        )

    # Non-LLM errors - asyncio timeout
    if isinstance(error, asyncio.TimeoutError):
        return ErrorInfo(
            error_type="timeout_error",
            error_source="system",
            is_retryable=True,
            status_code=None,
            original_exception=error,
        )

    # Non-LLM errors - TTS providers (string pattern matching)
    if "cartesia" in error_str or "elevenlabs" in error_str:
        tts_source = "cartesia" if "cartesia" in error_str else "elevenlabs"
        return ErrorInfo(
            error_type="tts_error",
            error_source=tts_source,
            is_retryable=True,
            status_code=None,
            original_exception=error,
        )

    # Non-LLM errors - STT providers (string pattern matching)
    if "deepgram" in error_str or "assemblyai" in error_str:
        stt_source = "deepgram" if "deepgram" in error_str else "assemblyai"
        return ErrorInfo(
            error_type="stt_error",
            error_source=stt_source,
            is_retryable=True,
            status_code=None,
            original_exception=error,
        )

    # Non-LLM errors - Tool execution
    if "tool" in error_str or "function" in error_str:
        return ErrorInfo(
            error_type="tool_error",
            error_source="tool_executor",
            is_retryable=False,
            status_code=None,
            original_exception=error,
        )

    # Non-LLM errors - Port pool
    if "port" in error_str:
        return ErrorInfo(
            error_type="system_error",
            error_source="port_pool",
            is_retryable=False,
            status_code=None,
            original_exception=error,
        )

    # Default fallback
    return ErrorInfo(
        error_type="system_error",
        error_source="unknown",
        is_retryable=False,
        status_code=None,
        original_exception=error,
    )


def get_error_source(error: Exception) -> str:
    """Get provider or component name from error.

    Priority order:
    1. exception.llm_provider attribute (LiteLLM native)
    2. Known exception types (asyncio, etc.)
    3. String pattern matching (TTS/STT providers)

    Args:
        error: The exception to identify

    Returns:
        Provider/component name (e.g., "openai", "azure", "anthropic", "cartesia", etc.)
    """
    # Priority 1: Use LiteLLM's native llm_provider attribute
    if hasattr(error, "llm_provider") and error.llm_provider:
        return error.llm_provider

    # Priority 2: Known exception types
    if isinstance(error, asyncio.TimeoutError):
        return "system"

    # Priority 3: String pattern matching for non-LLM providers
    error_str = str(error).lower()

    # TTS providers
    if "cartesia" in error_str:
        return "cartesia"
    if "elevenlabs" in error_str:
        return "elevenlabs"

    # STT providers
    if "deepgram" in error_str:
        return "deepgram"
    if "assemblyai" in error_str:
        return "assemblyai"

    # System components
    if "port" in error_str:
        return "port_pool"
    if "tool" in error_str or "function" in error_str:
        return "tool_executor"

    # Fallback to checking for provider names in error string
    for provider in ["openai", "azure", "anthropic", "google", "bedrock", "vertex"]:
        if provider in error_str:
            return provider

    return "unknown"


def is_retryable_error(error: Exception) -> bool:
    """Determine if error should be retried.

    Uses LiteLLM exception types to determine retry eligibility.

    Args:
        error: The exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    # Retryable LiteLLM errors
    retryable_types = (
        Timeout,
        APIConnectionError,
        RateLimitError,
        ServiceUnavailableError,
        InternalServerError,
        APIError,  # Generic API errors are retryable
    )

    if isinstance(error, retryable_types):
        return True

    # Retryable non-LLM errors
    if isinstance(error, asyncio.TimeoutError):
        return True

    # TTS/STT errors are generally retryable
    error_str = str(error).lower()
    if any(provider in error_str for provider in ["cartesia", "elevenlabs", "deepgram", "assemblyai"]):
        return True

    # Non-retryable by default
    return False


def create_error_details(
    error: Exception,
    retry_count: int = 0,
    retry_succeeded: bool = False,
) -> ErrorDetails:
    """Create ErrorDetails object from exception using centralized categorization.

    Args:
        error: The exception to convert
        retry_count: Number of retries attempted
        retry_succeeded: Whether retry succeeded

    Returns:
        ErrorDetails object with categorized information
    """
    error_info = categorize_error(error)

    # Use format_exception on the exception object directly so the traceback is correct regardless of whether we are inside an active except block.
    stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))

    return ErrorDetails(
        error_type=error_info.error_type,
        error_source=error_info.error_source,
        is_retryable=error_info.is_retryable,
        retry_count=retry_count,
        retry_succeeded=retry_succeeded,
        timestamps=[datetime.now(UTC).isoformat()],
        stack_trace=stack_trace,
        original_error=str(error),
    )
