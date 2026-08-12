#!/bin/sh
# Installed as /usr/local/bin/eva, which comes first on PATH, shadowing the real
# entry point at /opt/venv/bin/eva. Runs before every `eva` invocation, regardless of
# how the container is launched — Docker ENTRYPOINT, `sh -c 'eva ... && eva ...'`, or
# a Toolkit job's custom `command` (which replaces the image's ENTRYPOINT outright, so
# hooking this at the container level instead of here would silently miss those).
#
# Krisp VIVA SDK is proprietary and can't be baked into the image at build time
# (see vendor/krisp/README.md), so it's installed here at container start instead,
# from whatever is bind-mounted at /app/vendor/krisp. It's optional: with no wheel
# present, eva still runs fine, just without the krisp_viva_turn strategy.
set -e

whl_count=$(ls /app/vendor/krisp/*.whl 2>/dev/null | wc -l)
if [ "$whl_count" -gt 1 ]; then
  echo "Expected at most one Krisp wheel in vendor/krisp/, found $whl_count" >&2
  exit 1
elif [ "$whl_count" -eq 1 ]; then
  echo "Installing Krisp VIVA SDK..."
  uv pip install --python /opt/venv/bin/python --no-cache /app/vendor/krisp/*.whl
else
  echo "No Krisp wheel in vendor/krisp/ — krisp_viva_turn will be unavailable"
fi

# Full path, not bare "eva": avoids re-resolving PATH back to this wrapper, and keeps
# argparse's default program name in --help/usage/error output as "eva".
exec /opt/venv/bin/eva "$@"
