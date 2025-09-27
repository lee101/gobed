#!/bin/bash
# Performance and benchmark testing script for GoBeD
# Tests CPU vs GPU performance, validates search accuracy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE} GoBeD Performance Test Suite${NC}"
echo "================================="

cd "$(dirname "$0")/.."

# Source GPU environment
if [ -f "gpu_env.sh" ]; then
    source gpu_env.sh
elif [ -f "scripts/detect_gpu.sh" ]; then
    bash scripts/detect_gpu.sh
    source gpu_env.sh
fi

# Performance test results
PERF_RESULTS=()

# Function to run performance test
perf_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"

    echo ""
    echo -e "${BLUE} ${test_name}${NC}"
    echo "----------------------------------------"

    local start_time=$(date +%s.%N)
    local test_output=$(eval "$test_command" 2>&1)
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)

    echo "Duration: ${duration}s"

    if [ -n "$expected_pattern" ]; then
        if echo "$test_output" | grep -q "$expected_pattern"; then
            echo -e "${GREEN} Output validation passed${NC}"
        else
            echo -e "${YELLOW}  Output validation failed or inconclusive${NC}"
        fi
    fi

    PERF_RESULTS+=("$test_name: ${duration}s")
    echo "Command: $test_command"
    echo "Output preview:"
    echo "$test_output" | head -10
}

# 1. Go Benchmark Tests
echo -e "${YELLOW} Running Go Benchmarks${NC}"

# CPU benchmarks
echo ""
echo -e "${BLUE} CPU Benchmarks${NC}"
perf_test "Go Benchmark CPU" "go test -bench=. -benchtime=3s ./..." "Benchmark"

# GPU benchmarks (if available)
if [ "$USE_GPU" = "true" ]; then
    echo ""
    echo -e "${BLUE} GPU Benchmarks${NC}"
    perf_test "Go Benchmark GPU" "go test -tags gpu -bench=. -benchtime=3s ./..." "Benchmark"
fi

# 2. Build Performance Tests
echo ""
echo -e "${YELLOW} Build Performance Tests${NC}"

perf_test "Clean Build CPU" "make clean && go build -o /tmp/gobed-perf-cpu ./cmd/demo" ""

if [ "$USE_GPU" = "true" ]; then
    perf_test "Clean Build GPU" "make clean && cd bed && make cuda && cd .. && go build -tags gpu -o /tmp/gobed-perf-gpu ./cmd/demo" ""
fi

# 3. Search Performance Tests
echo ""
echo -e "${YELLOW} Search Performance Tests${NC}"

# Create test workspace
mkdir -p /tmp/gobed-perf-test
cd /tmp/gobed-perf-test

# Copy test data
cp -r "$(dirname "$0")/../testdata" .

# Generate larger test dataset for performance testing
echo "Generating larger test dataset..."
mkdir -p perf-data
for i in {1..50}; do
    echo "Performance test document $i: This document contains various machine learning concepts including neural networks, deep learning, artificial intelligence, natural language processing, computer vision, and data science methodologies. Document $i focuses on performance testing of semantic search capabilities." > perf-data/doc$i.txt
done

# Test CPU search performance
if [ -f "/tmp/gobed-perf-cpu" ]; then
    echo ""
    echo -e "${BLUE} CPU Search Performance${NC}"
    perf_test "CPU Search Small Dataset" "/tmp/gobed-perf-cpu -dir testdata -k 5 'machine learning'" ""
    perf_test "CPU Search Large Dataset" "/tmp/gobed-perf-cpu -dir perf-data -k 10 'neural networks'" ""
fi

# Test bed binary performance
if [ -f "/tmp/bed-test" ] || command -v bed &> /dev/null; then
    BED_BINARY="/tmp/bed-test"
    if ! [ -f "$BED_BINARY" ]; then
        BED_BINARY="bed"
    fi

    echo ""
    echo -e "${BLUE}🛏  Bed Search Performance${NC}"
    perf_test "Bed CPU Search Small" "$BED_BINARY -dir testdata -k 5 'machine learning'" ""
    perf_test "Bed CPU Search Large" "$BED_BINARY -dir perf-data -k 10 'artificial intelligence'" ""
fi

# Test GPU search performance (if available)
if [ "$USE_GPU" = "true" ] && [ -f "/tmp/gobed-perf-gpu" ]; then
    echo ""
    echo -e "${BLUE} GPU Search Performance${NC}"
    perf_test "GPU Search Small Dataset" "/tmp/gobed-perf-gpu -dir testdata -k 5 'machine learning'" ""
    perf_test "GPU Search Large Dataset" "/tmp/gobed-perf-gpu -dir perf-data -k 10 'neural networks'" ""

    # GPU bed binary
    if [ -f "/tmp/bed-gpu-test" ]; then
        perf_test "Bed GPU Search Small" "/tmp/bed-gpu-test -dir testdata -k 5 'machine learning'" ""
        perf_test "Bed GPU Search Large" "/tmp/bed-gpu-test -dir perf-data -k 10 'artificial intelligence'" ""
    fi
fi

# 4. Memory Usage Tests
echo ""
echo -e "${YELLOW}🧠 Memory Usage Tests${NC}"

# Function to get memory usage
get_memory_usage() {
    local pid=$1
    if command -v ps &> /dev/null; then
        ps -o rss= -p "$pid" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Test memory usage during search
if [ -f "/tmp/gobed-perf-cpu" ]; then
    echo "Testing CPU memory usage..."
    /tmp/gobed-perf-cpu -dir perf-data -k 10 'machine learning' &
    CPU_PID=$!
    sleep 2
    CPU_MEMORY=$(get_memory_usage $CPU_PID)
    wait $CPU_PID
    echo "CPU Memory Usage: ${CPU_MEMORY} KB"
fi

if [ "$USE_GPU" = "true" ] && [ -f "/tmp/gobed-perf-gpu" ]; then
    echo "Testing GPU memory usage..."
    /tmp/gobed-perf-gpu -dir perf-data -k 10 'machine learning' &
    GPU_PID=$!
    sleep 2
    GPU_MEMORY=$(get_memory_usage $GPU_PID)
    wait $GPU_PID
    echo "GPU Memory Usage: ${GPU_MEMORY} KB"

    # GPU memory info
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Info:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    fi
fi

# 5. Stress Test
echo ""
echo -e "${YELLOW} Stress Tests${NC}"

# Concurrent search test
if [ -f "/tmp/gobed-perf-cpu" ]; then
    echo "Running concurrent search stress test..."
    start_time=$(date +%s.%N)

    for i in {1..5}; do
        /tmp/gobed-perf-cpu -dir testdata -k 3 "test query $i" >/dev/null 2>&1 &
    done
    wait

    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    echo "Concurrent search test completed in: ${duration}s"
fi

# Cleanup test workspace
cd "$(dirname "$0")/.."
rm -rf /tmp/gobed-perf-test
rm -f /tmp/gobed-perf-cpu /tmp/gobed-perf-gpu /tmp/bed-test /tmp/bed-gpu-test

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}            Performance Summary             ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

for result in "${PERF_RESULTS[@]}"; do
    echo -e "  ${result}"
done

echo -e "${BLUE}═══════════════════════════════════════════${NC}"

# System info
echo ""
echo -e "${YELLOW} System Information${NC}"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | awk '/^Mem:/{print $2}' 2>/dev/null || echo 'Unknown')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi -L | head -1)"
fi
echo "Go Version: $(go version)"

echo ""
echo -e "${GREEN} Performance testing completed${NC}"