#!/bin/bash
set -e

echo "🚀 Building and running Gobed bulk embedding benchmark..."
echo ""

# Build the benchmark
echo "🔨 Building benchmark..."
go build -o bulk_benchmark ./cmd/bulk_benchmark
echo "✅ Built successfully"
echo ""

# Check system info
echo "💻 System Information:"
echo "  CPU: $(nproc) cores"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "  GPU: $GPU_INFO"
fi
echo "  Go version: $(go version | cut -d' ' -f3)"
echo ""

# Run the benchmark
echo "🏃 Running benchmark..."
echo ""

# Record start time
start_time=$(date +%s)

./bulk_benchmark

# Record end time and duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "⏱️  Total benchmark time: ${duration} seconds"
echo ""
echo "🎯 Results saved in benchmark_results_*.json"

# Cleanup
rm -f bulk_benchmark