"""Logging utilities for the voice agent benchmark framework."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
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

    root_logger.debug(f"Logging configured: level={level}, file={log_file}")
