#!/bin/bash
# Build script for InSilicoVA Docker image with platform detection
# Usage: ./build-docker.sh

set -e

echo "🐳 InSilicoVA Docker Build Script"
echo "================================"

# Detect platform
PLATFORM=$(uname -m)
echo "Detected platform: $PLATFORM"

# Set platform-specific variables
if [[ "$PLATFORM" == "arm64" ]]; then
    echo "🍎 Building for ARM64 (Apple Silicon)"
    DOCKER_IMAGE="insilicova-arm64:latest"
    DOCKER_PLATFORM="linux/arm64"
elif [[ "$PLATFORM" == "x86_64" ]]; then
    echo "💻 Building for AMD64 (Intel/AMD)"
    DOCKER_IMAGE="insilicova-amd64:latest"
    DOCKER_PLATFORM="linux/amd64"
else
    echo "❌ Unsupported platform: $PLATFORM"
    echo "Supported platforms: arm64 (Apple Silicon), x86_64 (Intel/AMD)"
    exit 1
fi

echo "Building image: $DOCKER_IMAGE"
echo "Platform: $DOCKER_PLATFORM"
echo ""

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    echo "❌ Dockerfile not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build \
    -t "$DOCKER_IMAGE" \
    --platform "$DOCKER_PLATFORM" \
    .

# Verify build
if docker images "$DOCKER_IMAGE" | grep -q "$DOCKER_IMAGE"; then
    echo ""
    echo "✅ Build successful!"
    echo "📋 Image details:"
    docker images "$DOCKER_IMAGE"
    echo ""
    echo "🚀 To run the image:"
    echo "   docker run -v \$(pwd):/workspace -w /workspace -it $DOCKER_IMAGE bash"
    echo ""
    echo "🧪 To test R packages:"
    echo "   docker run $DOCKER_IMAGE R -e \"library(openVA); library(InSilicoVA)\""
    echo ""
    echo "📝 Model configuration for this platform:"
    echo "   docker_image: \"$DOCKER_IMAGE\""
    echo "   docker_platform: \"$DOCKER_PLATFORM\""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi