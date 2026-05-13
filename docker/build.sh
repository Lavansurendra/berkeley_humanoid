#!/bin/bash

set -e  # Exit on error

# Detect script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile"

# Configuration
IMAGE_NAME="locomotion"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "========================================"
echo "Building Locomotion Docker Image"
echo "========================================"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "${DOCKERFILE_PATH}" ]; then
    echo "ERROR: Dockerfile not found at ${DOCKERFILE_PATH}"
    exit 1
fi

# Build the Docker image from project root with explicit Dockerfile path
echo "Building Docker image..."
echo "This may take 10-20 minutes on first build..."
echo ""

docker build \
    -f "${DOCKERFILE_PATH}" \
    --tag "${FULL_IMAGE_NAME}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

# Check build status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Build successful!"
    echo "========================================"
    echo "Image: ${FULL_IMAGE_NAME}"
    echo ""
    echo "Run the container: docker/run.sh"
    echo ""
else
    echo ""
    echo "========================================"
    echo "❌ Build failed!"
    echo "========================================"
    echo "Check the error messages above"
    exit 1
fi
