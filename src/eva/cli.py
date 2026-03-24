#!/usr/bin/env python3
"""CLI entry point for the `eva` command (installed via pip install)."""

import asyncio
import sys


def main():
    """Entry point for the `eva` console script."""
    # Import config first (lightweight) for fast --help and validation errors.
    # Heavy deps (pipecat, litellm, etc.) are imported only in run_benchmark.
    from eva.models.config import RunConfig

    config = RunConfig(_cli_parse_args=True, _env_file=".env")

    from scripts.run_benchmark import main as run_main

    sys.exit(asyncio.run(run_main(config)))


if __name__ == "__main__":
    main()
