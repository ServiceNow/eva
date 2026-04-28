#!/usr/bin/env python3
"""CLI entry point for eva.

Used by both the `eva` console script (installed via pip/uv) and `python main.py`.
"""

import asyncio
import os
import sys

from pydantic import ValidationError


def _extract_domain_spec() -> tuple[str | None, bool]:
    """Return (raw_domain_spec, came_from_argv).

    Looks at --domain / --domain=... in sys.argv first, then EVA_DOMAIN env var.
    """
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--domain" and i + 1 < len(argv):
            return argv[i + 1], True
        if arg.startswith("--domain="):
            return arg.split("=", 1)[1], True
    env = os.environ.get("EVA_DOMAIN")
    if env is not None:
        return env, False
    return None, False


def _strip_domain_from_argv() -> None:
    """Remove --domain (and its value) from sys.argv in place."""
    argv = sys.argv
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--domain":
            i += 2
            continue
        if a.startswith("--domain="):
            i += 1
            continue
        out.append(a)
        i += 1
    sys.argv = out


def _has_explicit_run_id() -> bool:
    return any(a == "--run-id" or a.startswith("--run-id=") for a in sys.argv) or "EVA_RUN_ID" in os.environ


def main():
    """Entry point for the `eva` console script."""
    # Import config first (lightweight) for fast --help and validation errors.
    # Heavy deps (pipecat, litellm, etc.) are imported only in run_benchmark.
    from eva.models.config import RunConfig

    spec, _ = _extract_domain_spec()
    domains = [d.strip() for d in spec.split(",")] if spec and "," in spec else None

    if domains is None:
        # Single-domain path — unchanged behavior.
        try:
            config = RunConfig(_cli_parse_args=True, _env_file=".env")
        except ValidationError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

        from eva.run_benchmark import run_benchmark

        sys.exit(asyncio.run(run_benchmark(config)))

    # Multi-domain path: loop, one RunConfig + run_benchmark per domain.
    # Dedupe preserving order, drop empties.
    seen: set[str] = set()
    ordered: list[str] = []
    for d in domains:
        if d and d not in seen:
            seen.add(d)
            ordered.append(d)

    explicit_run_id = _has_explicit_run_id()
    original_env = os.environ.get("EVA_DOMAIN")
    _strip_domain_from_argv()  # prevent pydantic-settings from re-reading the comma list

    from eva.run_benchmark import run_benchmark

    worst_exit = 0
    try:
        for domain in ordered:
            os.environ["EVA_DOMAIN"] = domain
            try:
                config = RunConfig(_cli_parse_args=True, _env_file=".env")
            except ValidationError as e:
                print(f"[domain={domain}] {e}", file=sys.stderr)
                worst_exit = max(worst_exit, 1)
                continue

            if explicit_run_id:
                config.run_id = f"{config.run_id}_{domain}"

            print(f"\n=== Running domain: {domain} ===\n", file=sys.stderr)
            exit_code = asyncio.run(run_benchmark(config))
            worst_exit = max(worst_exit, exit_code)
    finally:
        if original_env is None:
            os.environ.pop("EVA_DOMAIN", None)
        else:
            os.environ["EVA_DOMAIN"] = original_env

    sys.exit(worst_exit)


if __name__ == "__main__":
    main()
