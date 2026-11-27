"""The loop watchdog dumps thread stacks only when the event loop is blocked."""

import asyncio
import gc
import time
from unittest.mock import MagicMock

import pytest

from eva.utils import loop_watchdog
from eva.utils.loop_watchdog import ENV_VAR, LoopWatchdog


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("", None), ("0", None), ("2", 2.0), (" 1.5 ", 1.5)])
def test_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, value)
    watchdog = LoopWatchdog.from_env()
    assert (watchdog.threshold_s if watchdog else None) == expected


def test_from_env_rejects_non_numbers(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "yes")
    with pytest.raises(ValueError, match=ENV_VAR):
        LoopWatchdog.from_env()


def _block_the_loop(seconds: float) -> None:
    time.sleep(seconds)


@pytest.fixture
def mock_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(loop_watchdog, "logger", logger)
    return logger


async def test_blocked_loop_dumps_stacks_and_reports_on_recovery(tmp_path, mock_logger):
    out = tmp_path / "stacks.txt"
    with out.open("w") as f, LoopWatchdog(0.3, file=f, heartbeat_s=0.05):
        await asyncio.sleep(0.1)
        _block_the_loop(0.8)
        await asyncio.sleep(0.2)

    dump = out.read_text()
    assert "Timeout (0:00:00.300000)!" in dump
    assert "_block_the_loop" in dump
    mock_logger.warning.assert_called_once()
    message = mock_logger.warning.call_args.args[0]
    assert message.startswith("Event loop was blocked for at least 0.")
    assert "process CPU" in message and "GC" in message


async def test_running_loop_never_dumps(tmp_path, mock_logger):
    out = tmp_path / "stacks.txt"
    with out.open("w") as f, LoopWatchdog(0.3, file=f, heartbeat_s=0.05):
        for _ in range(10):
            await asyncio.sleep(0.06)

    assert "Timeout" not in out.read_text()
    mock_logger.warning.assert_not_called()


async def test_exit_disarms(tmp_path, mock_logger):
    out = tmp_path / "stacks.txt"
    with out.open("w") as f:
        with LoopWatchdog(0.3, file=f, heartbeat_s=0.05) as watchdog:
            await asyncio.sleep(0.1)
        _block_the_loop(0.6)

    assert "Timeout" not in out.read_text()
    assert watchdog._on_gc not in gc.callbacks


def test_threshold_must_exceed_heartbeat():
    with pytest.raises(ValueError):
        LoopWatchdog(0.25, heartbeat_s=0.25)
