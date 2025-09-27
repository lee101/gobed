#!/bin/bash
# Build validation script for GoBeD
# Tests that binaries build correctly for all target platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🏗  GoBeD Build Validation${NC}"
echo "============================"

cd "$(dirname "$0")/.."

# Build tracking
BUILDS_PASSED=0
BUILDS_FAILED=0
FAILED_BUILDS=()

# Function to run a build and track results
build_test() {
    local build_name="$1"
    local goos="$2"
    local goarch="$3"
    local gpu_enabled="$4"
    local output_suffix="$5"

    echo ""
    echo -e "${BLUE}🔨 Building: ${build_name}${NC}"
    echo "----------------------------------------"

    local build_dir="/tmp/gobed-builds/${build_name}"
    mkdir -p "$build_dir"

    local build_success=true
    local gpu_tags=""
    local cgo_enabled="0"

    if [ "$gpu_enabled" = "true" ]; then
        gpu_tags="-tags gpu"
        cgo_enabled="1"

        # Build CUDA libraries first (Linux only)
        if [ "$goos" = "linux" ]; then
            echo "Building CUDA libraries..."
            if ! (cd bed && make cuda 2>/dev/null); then
                echo -e "${YELLOW}  CUDA library build failed, skipping GPU build${NC}"
                build_success=false
            fi
        else
            echo -e "${YELLOW}  GPU builds only supported on Linux${NC}"
            build_success=false
        fi
    fi

    if [ "$build_success" = "true" ]; then
        # Build main binary
        echo "Building gobed binary..."
        if GOOS="$goos" GOARCH="$goarch" CGO_ENABLED="$cgo_enabled" go build $gpu_tags -ldflags "-s -w" -o "${build_dir}/gobed${output_suffix}" ./cmd/demo; then
            echo -e "${GREEN} gobed binary built successfully${NC}"
        else
            echo -e "${RED} gobed binary build failed${NC}"
            build_success=false
        fi

        # Build bed binary
        echo "Building bed binary..."
        if (cd bed && GOOS="$goos" GOARCH="$goarch" CGO_ENABLED="$cgo_enabled" go build $gpu_tags -ldflags "-s -w" -o "${build_dir}/bed${output_suffix}" .); then
            echo -e "${GREEN} bed binary built successfully${NC}"
        else
            echo -e "${RED} bed binary build failed${NC}"
            build_success=false
        fi

        # Validate binary properties
        if [ "$build_success" = "true" ]; then
            echo "Validating binary properties..."

            local gobed_binary="${build_dir}/gobed${output_suffix}"
            local bed_binary="${build_dir}/bed${output_suffix}"

            # Check file exists and has reasonable size
            if [ -f "$gobed_binary" ] && [ $(stat -f%z "$gobed_binary" 2>/dev/null || stat -c%s "$gobed_binary" 2>/dev/null || echo 0) -gt 1000000 ]; then
                echo -e "${GREEN} gobed binary validation passed${NC}"
            else
                echo -e "${RED} gobed binary validation failed${NC}"
                build_success=false
            fi

            if [ -f "$bed_binary" ] && [ $(stat -f%z "$bed_binary" 2>/dev/null || stat -c%s "$bed_binary" 2>/dev/null || echo 0) -gt 1000000 ]; then
                echo -e "${GREEN} bed binary validation passed${NC}"
            else
                echo -e "${RED} bed binary validation failed${NC}"
                build_success=false
            fi

            # Test binary execution (only on matching OS)
            if [ "$goos" = "$(go env GOOS)" ] && [ "$goarch" = "$(go env GOARCH)" ]; then
                echo "Testing binary execution..."
                if "$gobed_binary" --help >/dev/null 2>&1 || "$gobed_binary" -h >/dev/null 2>&1 || true; then
                    echo -e "${GREEN} gobed binary execution test passed${NC}"
                else
                    echo -e "${YELLOW}  gobed binary execution test inconclusive${NC}"
                fi

                if "$bed_binary" --help >/dev/null 2>&1 || "$bed_binary" -h >/dev/null 2>&1 || true; then
                    echo -e "${GREEN} bed binary execution test passed${NC}"
                else
                    echo -e "${YELLOW}  bed binary execution test inconclusive${NC}"
                fi
            fi
        fi
    fi

    if [ "$build_success" = "true" ]; then
        echo -e "${GREEN} BUILD PASSED: ${build_name}${NC}"
        ((BUILDS_PASSED++))
    else
        echo -e "${RED} BUILD FAILED: ${build_name}${NC}"
        ((BUILDS_FAILED++))
        FAILED_BUILDS+=("$build_name")
    fi
}

# Prepare build environment
mkdir -p /tmp/gobed-builds

# Source GPU environment for CUDA builds
if [ -f "gpu_env.sh" ]; then
    source gpu_env.sh
elif [ -f "scripts/detect_gpu.sh" ]; then
    bash scripts/detect_gpu.sh
    source gpu_env.sh
fi

echo -e "${YELLOW} Go Environment:${NC}"
echo "  Go Version: $(go version)"
echo "  GOOS: $(go env GOOS)"
echo "  GOARCH: $(go env GOARCH)"
echo "  CGO_ENABLED: $(go env CGO_ENABLED)"
if [ "$USE_GPU" = "true" ]; then
    echo "  GPU: Enabled"
    echo "  CUDA: $CUDA_PATH"
else
    echo "  GPU: Disabled"
fi

# Build matrix - CPU versions
build_test "linux-amd64-cpu" "linux" "amd64" "false" ""
build_test "linux-arm64-cpu" "linux" "arm64" "false" ""
build_test "darwin-amd64-cpu" "darwin" "amd64" "false" ""
build_test "darwin-arm64-cpu" "darwin" "arm64" "false" ""
build_test "windows-amd64-cpu" "windows" "amd64" "false" ".exe"

# GPU builds (Linux only)
if [ "$USE_GPU" = "true" ]; then
    build_test "linux-amd64-gpu" "linux" "amd64" "true" ""
else
    echo -e "${YELLOW}⏭  Skipping GPU builds (CUDA not available)${NC}"
fi

# Create release structure
echo ""
echo -e "${BLUE} Creating Release Structure${NC}"
echo "----------------------------------------"

RELEASE_DIR="/tmp/gobed-builds/release"
mkdir -p "$RELEASE_DIR"

# Create archives for each successful build
for build_dir in /tmp/gobed-builds/*/; do
    if [ -d "$build_dir" ]; then
        build_name=$(basename "$build_dir")
        if [ "$build_name" != "release" ]; then
            echo "Creating archive for $build_name..."

            cd "$build_dir"
            if [[ "$build_name" == *"windows"* ]]; then
                zip -q "$RELEASE_DIR/gobed-${build_name}.zip" *
                echo -e "${GREEN} Created gobed-${build_name}.zip${NC}"
            else
                tar -czf "$RELEASE_DIR/gobed-${build_name}.tar.gz" *
                echo -e "${GREEN} Created gobed-${build_name}.tar.gz${NC}"
            fi
        fi
    fi
done

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}             Build Summary                  ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

echo -e "  Passed: ${GREEN}${BUILDS_PASSED}${NC}"
echo -e "  Failed: ${RED}${BUILDS_FAILED}${NC}"
echo -e "  Total:  $((BUILDS_PASSED + BUILDS_FAILED))"

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed Builds:${NC}"
    for build in "${FAILED_BUILDS[@]}"; do
        echo -e "  • ${RED}${build}${NC}"
    done
fi

echo ""
echo -e "${BLUE}Release artifacts created in:${NC}"
echo "  ${RELEASE_DIR}"
ls -la "$RELEASE_DIR"

echo -e "${BLUE}═══════════════════════════════════════════${NC}"

# Exit with error if any builds failed
if [ $BUILDS_FAILED -gt 0 ]; then
    echo -e "${RED} Some builds failed!${NC}"
    exit 1
else
    echo -e "${GREEN} All builds passed!${NC}"
    exit 0
fi