#!/bin/bash
# Build script for InSilicoVA Docker image with platform detection
# Usage: ./build-docker.sh [--test-only]
#
# Options:
#   --test-only    Only test existing image without building

set -e

echo "🐳 InSilicoVA Docker Build Script"
echo "================================"

# Parse command line arguments
TEST_ONLY=false
for arg in "$@"; do
    case $arg in
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--test-only]"
            echo ""
            echo "Options:"
            echo "  --test-only    Only test existing image without building"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    echo ""
    echo "Installation instructions:"
    echo "  macOS: https://docs.docker.com/desktop/mac/install/"
    echo "  Linux: https://docs.docker.com/engine/install/"
    echo "  Windows: https://docs.docker.com/desktop/windows/install/"
    exit 1
fi

echo "✅ Docker is available"
docker --version

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

# Function to test the Docker image
test_docker_image() {
    local image_name="$1"
    echo ""
    echo "🧪 Testing Docker image: $image_name"
    echo "Testing R packages..."
    
    if docker run --rm "$image_name" R -e "library(openVA); library(InSilicoVA); cat('✅ R packages loaded successfully\\n')"; then
        echo ""
        echo "✅ Docker image test passed!"
        echo "🎉 Docker setup is ready for InSilicoVA evaluation!"
        echo ""
        echo "📝 Model configuration for this platform:"
        echo "   docker_image: \"$image_name\""
        echo "   docker_platform: \"$DOCKER_PLATFORM\""
        echo ""
        echo "🚀 Usage examples:"
        echo "   # Interactive R session:"
        echo "   docker run -v \$(pwd):/workspace -w /workspace -it $image_name R"
        echo ""
        echo "   # Run AP-only evaluation:"
        echo "   docker run -v \$(pwd):/workspace -w /workspace $image_name \\"
        echo "     python baseline/run_ap_only_insilico.py"
        echo ""
        echo "   # Run with poetry:"
        echo "   poetry run python baseline/run_ap_only_insilico.py"
        return 0
    else
        echo ""
        echo "❌ Docker image test failed!"
        echo "🔧 The image exists but R packages may not be working correctly."
        return 1
    fi
}

# If test-only flag, just test existing image
if [[ "$TEST_ONLY" == "true" ]]; then
    echo "Running in test-only mode..."
    
    # Check if image exists
    if docker images "$DOCKER_IMAGE" | grep -q "$DOCKER_IMAGE"; then
        test_docker_image "$DOCKER_IMAGE"
        exit $?
    else
        echo "❌ Docker image $DOCKER_IMAGE not found!"
        echo "Please build the image first with: $0"
        exit 1
    fi
fi

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    echo "❌ Dockerfile not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
echo "🏗️  This may take 10-15 minutes on first build..."
echo ""

if docker build -t "$DOCKER_IMAGE" --platform "$DOCKER_PLATFORM" .; then
    echo ""
    echo "✅ Build successful!"
    echo "📋 Image details:"
    docker images "$DOCKER_IMAGE"
    
    # Test the built image
    test_docker_image "$DOCKER_IMAGE"
    exit $?
else
    echo ""
    echo "❌ Build failed!"
    echo "🔧 Please check the error messages above and try again."
    exit 1
fi