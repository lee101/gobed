#!/bin/bash

# Comprehensive profiling script for Gobed search engine
set -e

echo "=== Gobed Search Engine Profiling Suite ==="
echo "This script will run comprehensive performance profiling"
echo ""

# Create profiles directory
mkdir -p profiles

# Build the profiling benchmark
echo "Building profiling benchmark..."
go build -o profile_bench profile_bench.go
echo "✓ Build completed"

# Run the profiling benchmark
echo ""
echo "Starting profiling benchmark..."
echo "This will take approximately 3-5 minutes"
echo ""

# Run with nice priority to minimize system interference
nice -n 10 ./profile_bench

echo ""
echo "=== Profiling Analysis Commands ==="
echo ""

# CPU Profile Analysis
if [ -f "profiles/cpu_profile.pprof" ]; then
    echo "CPU Profile Analysis:"
    echo "  Basic analysis: go tool pprof profiles/cpu_profile.pprof"
    echo "  Top functions: go tool pprof -top profiles/cpu_profile.pprof"
    echo "  Function list: go tool pprof -list=main profiles/cpu_profile.pprof"
    echo "  Web view: go tool pprof -http=:8080 profiles/cpu_profile.pprof"
    echo ""
    
    # Generate CPU profile summary
    echo "CPU Profile Top Functions:"
    go tool pprof -top -nodecount=10 profiles/cpu_profile.pprof
    echo ""
fi

# Memory Profile Analysis  
if [ -f "profiles/mem_profile.pprof" ]; then
    echo "Memory Profile Analysis:"
    echo "  Basic analysis: go tool pprof profiles/mem_profile.pprof"
    echo "  Top allocations: go tool pprof -top profiles/mem_profile.pprof"
    echo "  Web view: go tool pprof -http=:8081 profiles/mem_profile.pprof"
    echo ""
    
    # Generate memory profile summary
    echo "Memory Profile Top Allocations:"
    go tool pprof -top -nodecount=10 profiles/mem_profile.pprof
    echo ""
fi

# Additional analysis commands
echo "Advanced Analysis Commands:"
echo "  Compare profiles: go tool pprof -base=old.pprof new.pprof"
echo "  Generate flame graph: go tool pprof -http=:8080 profiles/cpu_profile.pprof"
echo "  Export to SVG: go tool pprof -svg profiles/cpu_profile.pprof > cpu_profile.svg"
echo ""

echo "Performance Optimization Recommendations:"
echo "1. Look for high CPU usage in:"
echo "   - SIMD dot product operations"
echo "   - Vector indexing and search"
echo "   - Memory allocations in hot paths"
echo ""
echo "2. Check memory efficiency:"
echo "   - Large allocations in batch processing"
echo "   - Vector storage overhead"
echo "   - Index memory usage"
echo ""
echo "3. Concurrency bottlenecks:"
echo "   - Lock contention in search operations"
echo "   - Channel blocking in async operations"
echo "   - Goroutine overhead"
echo ""

echo "✓ Profiling completed! Check the profiles/ directory for detailed analysis."