#!/bin/bash
# Build and push EVA-Bench Docker image to toolkit registry

set -e

export ACCOUNT_ID=$(eai account get --field id)

IMAGE_NAME="eva-bench"
VERSION="${1:-latest}"  # Default to 'latest', or use arg
export IMAGE="registry.console.elementai.com/$ACCOUNT_ID/$IMAGE_NAME:$VERSION"

echo "Building Docker image: $IMAGE"
echo "Account ID: $ACCOUNT_ID"
echo ""


docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f Dockerfile \
    -t $IMAGE \
    --push \
    .

echo ""
echo "✅ Image built and pushed: $IMAGE"
echo ""
echo "To use this image in toolkit jobs:"
echo "  eai job new --image $IMAGE --cpu 4 --mem 8 ..."
