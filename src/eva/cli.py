#!/usr/bin/env python3
"""CLI entry point for eva.

Used by both the `eva` console script (installed via pip/uv) and `python main.py`.
"""

import asyncio
import sys

from pydantic import ValidationError


def main():
    """Entry point for the `eva` console script."""
    # Import config first (lightweight) for fast --help and validation errors.
    # Heavy deps (pipecat, litellm, etc.) are imported only in run_benchmark.
    from eva.models.config import RunConfig

    try:
        config = RunConfig(_cli_parse_args=True, _env_file=".env")
    except ValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    from eva.run_benchmark import run_benchmark

    sys.exit(asyncio.run(run_benchmark(config)))


if __name__ == "__main__":
    main()
