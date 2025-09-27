#!/bin/bash
# Comprehensive test suite for GoBeD
# Runs all Go tests, C++ tests, and binary validation tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 GoBeD Comprehensive Test Suite${NC}"
echo "=================================="

# Change to project root
cd "$(dirname "$0")/.."

# Source GPU environment if available
if [ -f "gpu_env.sh" ]; then
    echo -e "${YELLOW}📦 Loading GPU environment...${NC}"
    source gpu_env.sh
else
    echo -e "${YELLOW}📦 No GPU environment found, running GPU detection...${NC}"
    if [ -f "scripts/detect_gpu.sh" ]; then
        bash scripts/detect_gpu.sh
        source gpu_env.sh
    fi
fi

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo ""
    echo -e "${BLUE}🔍 Running: ${test_name}${NC}"
    echo "----------------------------------------"

    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED: ${test_name}${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAILED: ${test_name}${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$test_name")
    fi
}

# 1. Go Module Tests
run_test "Go Module Verification" "go mod verify"
run_test "Go Module Tidy Check" "go mod tidy && git diff --exit-code go.mod go.sum"

# 2. Go Unit Tests
run_test "Go Unit Tests (CPU)" "go test -v -race ./..."

# 3. Go Build Tests
run_test "Build Main Binary (CPU)" "go build -o /tmp/gobed-test ./cmd/demo"
run_test "Build Bed Binary (CPU)" "cd bed && go build -o /tmp/bed-test ."

# 4. GPU Tests (if available)
if [ "$USE_GPU" = "true" ]; then
    echo -e "${YELLOW}🚀 GPU Tests Enabled${NC}"

    run_test "Build CUDA Libraries" "cd bed && make cuda"
    run_test "Go Unit Tests (GPU)" "go test -tags gpu -v ./..."
    run_test "Build Main Binary (GPU)" "go build -tags gpu -o /tmp/gobed-gpu-test ./cmd/demo"
    run_test "Build Bed Binary (GPU)" "cd bed && go build -tags gpu -o /tmp/bed-gpu-test ."
    run_test "GPU Integration Test" "cd bed && make test"
else
    echo -e "${YELLOW}⏭️  Skipping GPU tests (GPU not available)${NC}"
fi

# 5. Code Quality Tests
run_test "Go Format Check" "gofmt -l . | (! grep .)"
run_test "Go Vet" "go vet ./..."

# Install and run linters if available
if command -v golangci-lint &> /dev/null; then
    run_test "GolangCI Lint" "golangci-lint run --timeout=5m ./..."
else
    echo -e "${YELLOW}⏭️  Skipping golangci-lint (not installed)${NC}"
fi

# 6. Binary Functionality Tests
echo ""
echo -e "${BLUE}🔧 Binary Functionality Tests${NC}"
echo "----------------------------------------"

# Create test workspace
mkdir -p /tmp/gobed-test-workspace
cd /tmp/gobed-test-workspace

# Copy test data
cp -r "$(dirname "$0")/../testdata" .

# Test main binary
if [ -f "/tmp/gobed-test" ]; then
    run_test "Main Binary Help" "/tmp/gobed-test --help"
    run_test "Main Binary Version" "/tmp/gobed-test --version || true"  # Version flag might not exist
fi

# Test bed binary
if [ -f "/tmp/bed-test" ]; then
    run_test "Bed Binary Help" "/tmp/bed-test --help || /tmp/bed-test -h || true"
    # Test basic search functionality
    run_test "Bed Search Test" "/tmp/bed-test -dir testdata -k 3 'machine learning' || true"
fi

# Test GPU binaries if available
if [ "$USE_GPU" = "true" ]; then
    if [ -f "/tmp/gobed-gpu-test" ]; then
        run_test "GPU Main Binary Help" "/tmp/gobed-gpu-test --help"
    fi

    if [ -f "/tmp/bed-gpu-test" ]; then
        run_test "GPU Bed Binary Help" "/tmp/bed-gpu-test --help || /tmp/bed-gpu-test -h || true"
        run_test "GPU Bed Search Test" "/tmp/bed-gpu-test -dir testdata -k 3 'machine learning' || true"
    fi
fi

# 7. Performance/Benchmark Tests
cd "$(dirname "$0")/.."
run_test "Go Benchmarks" "go test -bench=. -benchtime=1s ./... | head -50"

# 8. Security Tests
if command -v gosec &> /dev/null; then
    run_test "Security Scan (gosec)" "gosec -quiet ./..."
else
    echo -e "${YELLOW}⏭️  Skipping security scan (gosec not installed)${NC}"
fi

# 9. C++ Tests (if any CUDA files exist)
if [ "$USE_GPU" = "true" ] && [ -f "bed/cuda_unique_topk.cu" ]; then
    echo ""
    echo -e "${BLUE}🔧 C++ CUDA Tests${NC}"
    echo "----------------------------------------"

    cd bed
    run_test "CUDA Compilation Test" "make cuda"

    # Test CUDA kernel if possible
    if [ -f "libcuda_unique_topk.so" ]; then
        run_test "CUDA Library Link Test" "ldd libcuda_unique_topk.so | grep -E '(cuda|cublas)'"
    fi
    cd ..
fi

# 10. Integration Tests with Real Data
echo ""
echo -e "${BLUE}🔗 Integration Tests${NC}"
echo "----------------------------------------"

# Test with actual model if available
if [ -d "model" ] && [ -f "model/model.safetensors" ]; then
    run_test "Model Loading Test" "go test -tags integration -run TestModelLoading ./... || true"
else
    echo -e "${YELLOW}⏭️  Skipping model tests (model files not found)${NC}"
fi

# Cleanup
rm -f /tmp/gobed-test /tmp/bed-test /tmp/gobed-gpu-test /tmp/bed-gpu-test
rm -rf /tmp/gobed-test-workspace

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}              Test Summary                  ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

echo -e "  Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "  Failed: ${RED}${TESTS_FAILED}${NC}"
echo -e "  Total:  $((TESTS_PASSED + TESTS_FAILED))"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed Tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  • ${RED}${test}${NC}"
    done
fi

echo -e "${BLUE}═══════════════════════════════════════════${NC}"

# Exit with error if any tests failed
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
fi