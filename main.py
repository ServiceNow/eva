#!/usr/bin/env python3
"""CLI script for running voice agent benchmarks.

Supports 2 modes:
1. `python main.py --domain ...`             New simulations + validate + rerun
2. `python main.py --run-id <existing_id>`   Validate existing + rerun failures
"""

import asyncio
import sys

from eva.models.config import RunConfig
from scripts.run_benchmark import main

if __name__ == "__main__":
    config = RunConfig(_cli_parse_args=True, _env_file=".env")
    sys.exit(asyncio.run(main(config)))
