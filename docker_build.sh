#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-latest}"
PUSH="${2:-}"
shift 2 2>/dev/null || true

GIT_COMMIT_SHA=$(git rev-parse HEAD)
GIT_BRANCH=$(git branch --show-current)
GIT_DIRTY=$([[ -n $(git status --porcelain) ]] && echo true || echo false)
GIT_DIFF_HASH=$(git diff | shasum -a 256 | cut -c1-12)

IMAGE="registry.console.elementai.com/snow.core_llm/eva:$TAG"

# Use BuildKit — the legacy builder ("Step X/Y" output) does not cache
# multi-stage builds well and re-runs apt-get / re-copies layers needlessly.
export DOCKER_BUILDKIT=1

docker build \
  --platform linux/amd64 \
  --build-arg GIT_COMMIT_SHA="$GIT_COMMIT_SHA" \
  --build-arg GIT_BRANCH="$GIT_BRANCH" \
  --build-arg GIT_DIRTY="$GIT_DIRTY" \
  --build-arg GIT_DIFF_HASH="$GIT_DIFF_HASH" \
  -t "$IMAGE" \
  "$@" .

if [[ "$PUSH" == "--push" ]]; then
  docker push "$IMAGE"
fi
