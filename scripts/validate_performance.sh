#!/bin/bash

# Performance validation script for gobed
# Checks if performance meets the specified targets

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}       GoBeD Performance Validation         ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

# Performance targets
TARGET_LATENCY_MS=1.0      # < 1ms search latency
TARGET_QPS=1000000          # > 1M queries per second
TARGET_MEMORY_GB=4          # < 4GB for 240k docs
TARGET_INDEX_RATE=150000    # > 150k embeddings/sec

# Find the latest benchmark file
BENCH_FILE=$(ls -t benchmark-results/bench_*.txt 2>/dev/null | head -1)

if [ -z "$BENCH_FILE" ]; then
    echo -e "${YELLOW}⚠️  No benchmark results found. Running benchmarks...${NC}"
    make bench
    BENCH_FILE=$(ls -t benchmark-results/bench_*.txt 2>/dev/null | head -1)
fi

if [ -z "$BENCH_FILE" ]; then
    echo -e "${RED}✗ Failed to run benchmarks${NC}"
    exit 1
fi

echo -e "${GREEN}📊 Analyzing: $BENCH_FILE${NC}"
echo ""

# Initialize counters
PASSED=0
FAILED=0
WARNINGS=0

# Function to check a metric
check_metric() {
    local name="$1"
    local value="$2"
    local target="$3"
    local comparison="$4"  # "lt" for less than, "gt" for greater than
    local unit="$5"

    if [ -z "$value" ]; then
        echo -e "${YELLOW}⚠️  $name: No data available${NC}"
        ((WARNINGS++))
        return
    fi

    # Remove any non-numeric characters except . and -
    value=$(echo "$value" | sed 's/[^0-9.-]//g')

    # Compare based on type
    local result
    if [ "$comparison" = "lt" ]; then
        result=$(echo "$value < $target" | bc -l)
    else
        result=$(echo "$value > $target" | bc -l)
    fi

    if [ "$result" = "1" ]; then
        echo -e "${GREEN}✅ $name: $value $unit (target: $comparison $target $unit)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $name: $value $unit (target: $comparison $target $unit)${NC}"
        ((FAILED++))
    fi
}

# Extract metrics from benchmark file

# 1. Search Latency
echo -e "${BLUE}Search Performance:${NC}"
if grep -q "BenchmarkSearch" "$BENCH_FILE"; then
    # Extract ns/op and convert to ms
    LATENCY_NS=$(grep "BenchmarkSearch" "$BENCH_FILE" | awk '{print $3}' | head -1)
    if [ -n "$LATENCY_NS" ]; then
        LATENCY_MS=$(echo "scale=3; $LATENCY_NS / 1000000" | bc)
        check_metric "Search Latency" "$LATENCY_MS" "$TARGET_LATENCY_MS" "lt" "ms"
    fi
fi

# 2. Throughput (QPS)
if grep -q "QPS" "$BENCH_FILE"; then
    QPS=$(grep "QPS" "$BENCH_FILE" | awk '{print $2}' | tail -1)
    check_metric "Throughput" "$QPS" "$TARGET_QPS" "gt" "QPS"
fi

# 3. Memory Usage
echo -e "\n${BLUE}Memory Performance:${NC}"
if grep -q "B/op" "$BENCH_FILE"; then
    BYTES_PER_OP=$(grep "BenchmarkSearch" "$BENCH_FILE" | awk '{print $5}' | head -1)
    if [ -n "$BYTES_PER_OP" ]; then
        MB_PER_OP=$(echo "scale=3; $BYTES_PER_OP / 1048576" | bc)
        echo -e "  Memory per operation: ${MB_PER_OP} MB"
    fi
fi

# 4. Indexing Performance
echo -e "\n${BLUE}Indexing Performance:${NC}"
if grep -q "BenchmarkIndex" "$BENCH_FILE"; then
    INDEX_NS=$(grep "BenchmarkIndex" "$BENCH_FILE" | awk '{print $3}' | head -1)
    if [ -n "$INDEX_NS" ]; then
        INDEX_RATE=$(echo "scale=0; 1000000000 / $INDEX_NS" | bc)
        check_metric "Index Rate" "$INDEX_RATE" "$TARGET_INDEX_RATE" "gt" "docs/sec"
    fi
fi

# Additional checks from test output
echo -e "\n${BLUE}Test Results:${NC}"

# Check if GPU is being used
if grep -q "GPU mode" "$BENCH_FILE"; then
    echo -e "${GREEN}✅ GPU acceleration enabled${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  GPU acceleration status unknown${NC}"
    ((WARNINGS++))
fi

# Check for test failures
if grep -q "FAIL" "$BENCH_FILE"; then
    FAILURES=$(grep -c "FAIL" "$BENCH_FILE")
    echo -e "${RED}✗ Found $FAILURES test failures${NC}"
    ((FAILED++))
else
    echo -e "${GREEN}✅ All tests passed${NC}"
    ((PASSED++))
fi

# Performance comparison
echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}              Summary Report                 ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

# Overall status
TOTAL=$((PASSED + FAILED + WARNINGS))
SUCCESS_RATE=$((PASSED * 100 / (PASSED + FAILED + 1)))

echo -e "Tests Passed:  ${GREEN}$PASSED${NC}"
echo -e "Tests Failed:  ${RED}$FAILED${NC}"
echo -e "Warnings:      ${YELLOW}$WARNINGS${NC}"
echo -e "Success Rate:  $SUCCESS_RATE%"

# GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read line; do
        echo -e "  $line"
    done
fi

# Performance grade
echo -e "\n${BLUE}Performance Grade:${NC}"
if [ "$FAILED" -eq 0 ] && [ "$SUCCESS_RATE" -ge 90 ]; then
    echo -e "${GREEN}⭐⭐⭐⭐⭐ EXCELLENT - All targets met!${NC}"
    EXIT_CODE=0
elif [ "$SUCCESS_RATE" -ge 75 ]; then
    echo -e "${GREEN}⭐⭐⭐⭐ GOOD - Most targets met${NC}"
    EXIT_CODE=0
elif [ "$SUCCESS_RATE" -ge 50 ]; then
    echo -e "${YELLOW}⭐⭐⭐ FAIR - Some optimization needed${NC}"
    EXIT_CODE=1
else
    echo -e "${RED}⭐⭐ NEEDS IMPROVEMENT - Multiple targets missed${NC}"
    EXIT_CODE=1
fi

# Recommendations
if [ "$FAILED" -gt 0 ]; then
    echo -e "\n${BLUE}Recommendations:${NC}"

    if grep -q "Search Latency.*✗" <<< "$FAILURES"; then
        echo -e "  • Enable GPU acceleration for faster search"
        echo -e "  • Consider using IVF clustering for large datasets"
        echo -e "  • Implement batch processing for queries"
    fi

    if grep -q "Throughput.*✗" <<< "$FAILURES"; then
        echo -e "  • Enable parallel query processing"
        echo -e "  • Use memory-mapped files for large indices"
        echo -e "  • Consider sharding the index"
    fi

    if grep -q "Memory.*✗" <<< "$FAILURES"; then
        echo -e "  • Enable INT8 quantization"
        echo -e "  • Implement memory pooling"
        echo -e "  • Use streaming for large batches"
    fi
fi

echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"

exit $EXIT_CODE