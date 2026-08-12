#!/usr/bin/env python3
"""CLI entry point for eva.

Used by both the `eva` console script (installed via pip/uv) and `python main.py`.
"""

import argparse
import asyncio
import faulthandler
import sys

from pydantic import ValidationError

faulthandler.enable()  # Print Python stack trace on segfaults (exit 139)


class _NoUsageHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Suppress the long, auto-generated `usage:` block in `--help` output."""

    def _format_usage(self, *args, **kwargs):
        return ""


def main():
    """Entry point for the `eva` console script."""
    # Import config first (lightweight) for fast --help and validation errors.
    # Heavy deps (pipecat, litellm, etc.) are imported only in run_benchmark.
    from pydantic_settings import CliSettingsSource

    from eva.models.config import RunConfig

    cli_source = CliSettingsSource(RunConfig, cli_parse_args=True, formatter_class=_NoUsageHelpFormatter)

    try:
        config = RunConfig(_cli_settings_source=cli_source, _env_file=".env")
    except ValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    from eva.run_benchmark import run_benchmark

    sys.exit(asyncio.run(run_benchmark(config)))


if __name__ == "__main__":
    main()
