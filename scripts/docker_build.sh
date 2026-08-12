#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Missing image tag. Usage: $0 <image:tag> [--push]}"
PUSH="${2:-}"
shift 2>/dev/null || true

GIT_COMMIT_SHA=$(git rev-parse HEAD)
GIT_BRANCH=$(git branch --show-current)
GIT_DIRTY=$([[ -n $(git status --porcelain) ]] && echo true || echo false)
GIT_DIFF_HASH=$(git diff | shasum -a 256 | cut -c1-12)

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
