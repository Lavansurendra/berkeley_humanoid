#!/bin/bash
# Run script for AlohaMini Docker container

set -e  # Exit on error

# Detect script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
IMAGE_NAME="locomotion"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="locomotion-dev"

# Parse command line arguments
MODE="interactive"
if [ "$1" == "--jupyter" ]; then
    MODE="jupyter"
fi

echo "========================================"
echo "Running Locomotion Docker Container"
echo "========================================"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Mode: ${MODE}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check if image exists
if ! docker image inspect "${FULL_IMAGE_NAME}" > /dev/null 2>&1; then
    echo "ERROR: Docker image '${FULL_IMAGE_NAME}' not found"
    echo "Please build the image first: docker/build.sh"
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
fi

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

# Common Docker run arguments
DOCKER_ARGS=(
    --name "${CONTAINER_NAME}"
    --rm
    -it
    -v "~/workspace"
    -w /workspace
    --privileged
)

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    DOCKER_ARGS+=(
        --runtime=nvidia
        --gpus all
        -e NVIDIA_DRIVER_CAPABILITIES=all
    )
fi

if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    DOCKER_ARGS+=(
        --pid=host
    )
fi


docker run "${DOCKER_ARGS[@]}" \
    "${FULL_IMAGE_NAME}" \
    /bin/bash