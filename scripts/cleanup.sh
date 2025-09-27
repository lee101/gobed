#!/bin/bash

# Project Cleanup and Organization Script

set -e

echo "🧹 Cleaning up gobed project..."

# Create organized directory structure
echo "📁 Creating organized directory structure..."
mkdir -p test/unit
mkdir -p test/integration
mkdir -p test/benchmark
mkdir -p test/testdata
mkdir -p build
mkdir -p scripts
mkdir -p docs
mkdir -p examples

# Clean up build artifacts
echo "🗑️  Removing build artifacts..."
rm -f *.o
rm -f *.bak
rm -f test_200k test_200k_real
rm -f test_ai_gpu test_ai_simple
rm -f test_cached_search test_cuda_perf test_cuda_simple
rm -f test_gpu_quality test_real_model test_simple
rm -f unified_benchmark
rm -f bed_tool_new

# Move test files to proper locations
echo "📦 Organizing test files..."

# Move benchmark files
for file in benchmark*.py *benchmark*.go *bench*.sh; do
    if [ -f "$file" ]; then
        mv "$file" test/benchmark/ 2>/dev/null || true
    fi
done

# Move test scripts
for file in test_*.sh demo_*.sh; do
    if [ -f "$file" ]; then
        mv "$file" test/integration/ 2>/dev/null || true
    fi
done

# Move test Go files (but keep main test files in place)
for file in *_test.go; do
    if [ -f "$file" ] && [[ ! "$file" =~ ^(gobed_test|main_test)\.go$ ]]; then
        mv "$file" test/unit/ 2>/dev/null || true
    fi
done

# Move Python test files
for file in test_*.py; do
    if [ -f "$file" ]; then
        mv "$file" test/integration/ 2>/dev/null || true
    fi
done

# Clean up bed directory
echo "📦 Organizing bed/ directory..."
cd bed/
rm -f *.o bed bed_search bed_tool
mkdir -p tests benchmarks

# Move bed test files
for file in test*.py benchmark*.py; do
    if [ -f "$file" ]; then
        if [[ "$file" == benchmark* ]]; then
            mv "$file" benchmarks/ 2>/dev/null || true
        else
            mv "$file" tests/ 2>/dev/null || true
        fi
    fi
done

cd ..

# Move CUDA files to gpu directory
echo "📦 Organizing CUDA files..."
for file in cuda_*.cu cuda_*.o; do
    if [ -f "$file" ]; then
        mv "$file" gpu/ 2>/dev/null || true
    fi
done

# Move documentation
echo "📚 Organizing documentation..."
for file in *_SUMMARY.md *_STATUS.md EXPECTED_RESULTS.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/ 2>/dev/null || true
    fi
done

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Binaries
*.o
*.a
*.so
*.exe
*.test
*.out

# Build directories
/build/
/dist/
/bin/

# Test binaries
test_*
!test_*.go
!test_*.py
!test_*.sh

# Backup files
*.bak
*.tmp
*.swp
*~

# IDE files
.vscode/
.idea/
*.sublime-*

# Go specific
vendor/
*.prof
coverage.txt
coverage.html

# Python
__pycache__/
*.pyc
.pytest_cache/

# CUDA/GPU
*.cubin
*.ptx
*.fatbin

# Model files (keep in repo)
# *.safetensors
# *.bin

# Logs
*.log

# OS files
.DS_Store
Thumbs.db

# Project specific
gpu_env.sh
/tmp/
EOF
fi

echo "✅ Cleanup complete!"
echo ""
echo "📊 Project structure:"
echo "  test/         - All test files"
echo "    unit/       - Unit tests"
echo "    integration/- Integration tests"
echo "    benchmark/  - Benchmarks"
echo "    testdata/   - Test data files"
echo "  bed/          - Bed tool sources"
echo "    tests/      - Bed-specific tests"
echo "    benchmarks/ - Bed benchmarks"
echo "  gpu/          - GPU/CUDA code"
echo "  scripts/      - Build and utility scripts"
echo "  docs/         - Documentation"
echo "  examples/     - Usage examples"
echo ""
echo "Run 'source ./gpu_env.sh' to set up GPU environment"