#!/bin/bash

# Test script for searching ai.txt with real embeddings
set -e

echo " Building bed_real with CUDA support..."
cd /home/lee/code/gobed/bed

# Build CUDA library
nvcc -c cuda_search.cu -o cuda_search.o
ar rcs libcuda_search.a cuda_search.o

# Build bed_real binary (only the specific file)
CGO_ENABLED=1 go build -o bed_real ./bed_real.go

echo " Build complete"
echo ""

# Test searching ai.txt specifically
AI_FILE="/home/lee/code/gobed/testdata/ai.txt"

if [ ! -f "$AI_FILE" ]; then
    echo " Error: ai.txt not found at $AI_FILE"
    exit 1
fi

echo " Testing searches on ai.txt:"
echo "================================"

# Create a temp directory with just ai.txt
TEMP_DIR=$(mktemp -d)
cp "$AI_FILE" "$TEMP_DIR/ai.txt"

echo "Test 1: Search for 'anime' (should find anime-related content)"
echo "----------------------------------------------------------------"
./bed_real -dir "$TEMP_DIR" -k 10 -debug anime

echo ""
echo "Test 2: Search for 'father' (should find father-related content)"
echo "-----------------------------------------------------------------"
./bed_real -dir "$TEMP_DIR" -k 10 father

echo ""
echo "Test 3: Search for 'friend' (should find friend-related content)"
echo "-----------------------------------------------------------------"
./bed_real -dir "$TEMP_DIR" -k 10 friend

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo " All tests complete!"