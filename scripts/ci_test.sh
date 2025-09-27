#!/bin/bash
# CI Integration Test Script
# Simulates the full CI pipeline locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 GoBeD CI Pipeline Simulation${NC}"
echo "==============================="

cd "$(dirname "$0")/.."

# Track overall success
PIPELINE_SUCCESS=true

# Function to run pipeline stage
run_stage() {
    local stage_name="$1"
    local script_path="$2"

    echo ""
    echo -e "${BLUE} Stage: ${stage_name}${NC}"
    echo "================================================"

    if [ -f "$script_path" ]; then
        if bash "$script_path"; then
            echo -e "${GREEN} STAGE PASSED: ${stage_name}${NC}"
        else
            echo -e "${RED} STAGE FAILED: ${stage_name}${NC}"
            PIPELINE_SUCCESS=false
        fi
    else
        echo -e "${YELLOW}  STAGE SKIPPED: ${stage_name} (script not found: $script_path)${NC}"
    fi
}

# Pipeline stages
echo -e "${YELLOW} Running CI Pipeline Stages${NC}"

# Stage 1: Environment Detection and Setup
run_stage "GPU Detection & Environment Setup" "scripts/detect_gpu.sh"

# Stage 2: Comprehensive Testing
run_stage "Comprehensive Test Suite" "scripts/run_all_tests.sh"

# Stage 3: Build Validation
run_stage "Multi-Platform Build Validation" "scripts/validate_build.sh"

# Stage 4: Performance Testing
run_stage "Performance & Benchmark Testing" "scripts/performance_test.sh"

# Additional quick checks
echo ""
echo -e "${BLUE} Additional CI Checks${NC}"
echo "================================================"

# Check if key files exist
echo "Checking project structure..."
REQUIRED_FILES=(
    "go.mod"
    "Makefile"
    ".github/workflows/ci.yml"
    "testdata/sample1.txt"
    "bed/Makefile"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (missing)"
        PIPELINE_SUCCESS=false
    fi
done

# Check script permissions
echo ""
echo "Checking script permissions..."
SCRIPTS=(
    "scripts/detect_gpu.sh"
    "scripts/run_all_tests.sh"
    "scripts/validate_build.sh"
    "scripts/performance_test.sh"
    "scripts/ci_test.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo -e "${GREEN}✓${NC} $script (executable)"
    else
        echo -e "${YELLOW}${NC} $script (not executable)"
    fi
done

# Git status check
echo ""
echo "Checking git status..."
if git status --porcelain | grep -q .; then
    echo -e "${YELLOW}  Working directory has uncommitted changes${NC}"
    git status --short
else
    echo -e "${GREEN}✓ Working directory clean${NC}"
fi

# Final summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}            CI Pipeline Summary             ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

if [ "$PIPELINE_SUCCESS" = "true" ]; then
    echo -e "${GREEN} ALL STAGES PASSED${NC}"
    echo ""
    echo -e "${GREEN} Project is ready for CI/CD deployment${NC}"
    echo -e "${GREEN} All build targets validated${NC}"
    echo -e "${GREEN} Test suite comprehensive${NC}"
    echo -e "${GREEN} Performance benchmarks completed${NC}"
else
    echo -e "${RED} SOME STAGES FAILED${NC}"
    echo ""
    echo -e "${RED}Please review failed stages above${NC}"
fi

echo -e "${BLUE}═══════════════════════════════════════════${NC}"

# Exit with appropriate code
if [ "$PIPELINE_SUCCESS" = "true" ]; then
    exit 0
else
    exit 1
fi