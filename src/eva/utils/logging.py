"""Logging utilities for the voice agent benchmark framework."""

import contextvars
import logging
import sys

# ContextVar that tracks which record_id the current asyncio task is processing.
# Each ConversationWorker sets this at the start of its run() method.
# Because contextvars are asyncio-task-scoped, concurrent workers each see their own value.
current_record_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_record_id", default=None)


class RecordFilter(logging.Filter):
    """Filter that only allows log records from the matching asyncio task.

    When attached to a per-record FileHandler, this ensures that only logs
    emitted while ``current_record_id`` equals the target ``record_id`` are
    written to that record's log file.
    """

    def __init__(self, record_id: str) -> None:
        super().__init__()
        self.record_id = record_id

    def filter(self, record: logging.LogRecord) -> bool:
        return current_record_id.get() == self.record_id


def _bridge_loguru_to_stdlib() -> None:
    """Route pipecat's loguru logs into the stdlib ``pipecat`` logger.

    Pipecat logs through loguru (a separate logging system), so its records never
    reach the stdlib handlers configured here — including the smart-turn analyzer's
    ``End of Turn complete due to stop_secs`` / ``End of Turn result`` debug lines.
    This installs a loguru sink that re-emits every loguru record as a stdlib log
    record under the ``pipecat`` logger, preserving the original level and source
    location. Called once from ``setup_logging`` so per-record file handlers (which
    attach to the ``pipecat`` logger) capture these lines.

    No-op if loguru is not installed.
    """
    try:
        from loguru import logger as _loguru_logger
    except ImportError:
        return

    class _InterceptHandler:
        def write(self, message) -> None:  # loguru sink protocol
            record = message.record
            # Map loguru level name to a stdlib level number (fallback to the numeric no).
            level = logging.getLevelName(record["level"].name)
            if not isinstance(level, int):
                level = record["level"].no
            std_logger = logging.getLogger("pipecat")
            log_record = std_logger.makeRecord(
                "pipecat",
                level,
                record["file"].path,
                record["line"],
                record["message"],
                (),
                None,
                func=record["function"],
            )
            std_logger.handle(log_record)

    # Replace loguru's default stderr sink with our stdlib bridge at DEBUG so the
    # smart-turn debug lines are captured; the stdlib handlers do the final filtering.
    _loguru_logger.remove()
    _loguru_logger.add(_InterceptHandler(), level="DEBUG", format="{message}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the ``eva`` hierarchy.

    Names that already start with `eva` are used as-is.
    All other names (e.g. `__main__`, test modules) are prefixed with
    `eva.` so they inherit the handlers configured by `setup_logging()`.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    if not name.startswith("eva"):
        name = f"eva.{name}"
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger for eva
    root_logger = logging.getLogger("eva")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Don't propagate to root logger
    root_logger.propagate = False

    # Bridge pipecat's loguru logs into the stdlib ``pipecat`` logger so framework
    # diagnostics (e.g. smart-turn end-of-turn reasons) land in the same sinks.
    _bridge_loguru_to_stdlib()
    pipecat_logger = logging.getLogger("pipecat")
    pipecat_logger.setLevel(logging.DEBUG)
    pipecat_logger.addHandler(console_handler)
    if log_file:
        pipecat_logger.addHandler(file_handler)
    pipecat_logger.propagate = False

    root_logger.debug(f"Logging configured: level={level}, file={log_file}")


def add_record_log_file(record_id: str, log_file_path: str) -> logging.FileHandler:
    """Add a file handler to capture all logs for a specific record.

    Args:
        record_id: Record ID (used as handler name)
        log_file_path: Path to the log file

    Returns:
        The created file handler (store this to remove it later)
    """
    # Create formatter
    format_string = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    formatter = logging.Formatter(format_string)

    # Create file handler with a filter so only logs from *this* record's
    # asyncio task are written (prevents cross-worker pollution).
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.set_name(f"record_{record_id}")  # Name it for easy removal
    file_handler.addFilter(RecordFilter(record_id))

    # Add to eva logger (captures all eva.* logs)
    eva_logger = logging.getLogger("eva")
    eva_logger.addHandler(file_handler)

    # Also add to pipecat logger to capture pipecat framework logs
    pipecat_logger = logging.getLogger("pipecat")
    pipecat_logger.addHandler(file_handler)
    pipecat_logger.setLevel(logging.DEBUG)

    return file_handler


def remove_record_log_file(file_handler: logging.FileHandler) -> None:
    """Remove a file handler that was added for a specific record.

    Args:
        file_handler: The file handler to remove
    """
    if file_handler is None:
        return

    # Remove from eva logger
    eva_logger = logging.getLogger("eva")
    eva_logger.removeHandler(file_handler)

    # Remove from pipecat logger
    pipecat_logger = logging.getLogger("pipecat")
    pipecat_logger.removeHandler(file_handler)

    # Close the file handler
    file_handler.close()
