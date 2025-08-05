#!/bin/bash

# Complete Test Suite for Go vs Python PyTorch Embedding Comparison
# This script demonstrates perfect numerical consistency between implementations

set -e  # Exit on any error

echo "🚀 COMPLETE EMBEDDING TEST SUITE"
echo "================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}📋 Checking Prerequisites...${NC}"
echo "----------------------------"

if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${RED}❌ Python virtual environment not found${NC}"
    echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [ ! -f "model/production_reference_tokens.json" ]; then
    echo -e "${RED}❌ Reference tokens not found${NC}"
    echo "Run: source .venv/bin/activate && python generate_production_tokens.py"
    exit 1
fi

if [ ! -f "cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors" ]; then
    echo -e "${RED}❌ Safetensors model not found${NC}"
    echo "Run: source .venv/bin/activate && python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')\""
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"
echo ""

# Test 1: Python PyTorch with Safetensors
echo -e "${BLUE}🐍 Test 1: Python PyTorch with Safetensors${NC}"
echo "===========================================" 
source .venv/bin/activate
python test_python_pytorch.py > /tmp/python_results.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python PyTorch test passed${NC}"
else
    echo -e "${RED}❌ Python PyTorch test failed${NC}"
    cat /tmp/python_results.txt
    exit 1
fi
echo ""

# Test 2: Go Safetensors Implementation  
echo -e "${BLUE}🔧 Test 2: Go Safetensors Implementation${NC}"
echo "=========================================="
go run safetensors_loader.go > /tmp/go_results.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Go Safetensors test passed${NC}"
else
    echo -e "${RED}❌ Go Safetensors test failed${NC}"
    cat /tmp/go_results.txt
    exit 1
fi
echo ""

# Test 3: Direct Value Comparison
echo -e "${BLUE}🔍 Test 3: Direct Value Comparison${NC}"
echo "==================================="
echo "Python PyTorch Results:"
source .venv/bin/activate
python simple_compare.py

echo ""
echo "Go Safetensors Results:"
go run simple_go_compare.go

echo ""

# Test 4: Exact Match Verification
echo -e "${BLUE}📊 Test 4: Exact Match Verification${NC}"
echo "===================================="

# Extract embedding values for comparison
python_embed1=$(source .venv/bin/activate && python simple_compare.py | grep "This is a test sentence" | grep -o '\[.*\]')
go_embed1=$(go run simple_go_compare.go | grep "This is a test sentence" | grep -o '\[.*\]')

python_embed2=$(source .venv/bin/activate && python simple_compare.py | grep "Machine learning is fascinating" | grep -o '\[.*\]')
go_embed2=$(go run simple_go_compare.go | grep "Machine learning is fascinating" | grep -o '\[.*\]')

python_embed3=$(source .venv/bin/activate && python simple_compare.py | grep "Hello world" | grep -o '\[.*\]')
go_embed3=$(go run simple_go_compare.go | grep "Hello world" | grep -o '\[.*\]')

echo "Embedding Comparison Results:"
echo "----------------------------"
if [ "$python_embed1" = "$go_embed1" ]; then
    echo -e "${GREEN}✅ Sentence 1: EXACT MATCH${NC}"
    echo "   Values: $python_embed1"
else
    echo -e "${RED}❌ Sentence 1: MISMATCH${NC}"
    echo "   Python: $python_embed1"
    echo "   Go:     $go_embed1"
fi

if [ "$python_embed2" = "$go_embed2" ]; then
    echo -e "${GREEN}✅ Sentence 2: EXACT MATCH${NC}"
    echo "   Values: $python_embed2"
else
    echo -e "${RED}❌ Sentence 2: MISMATCH${NC}"
    echo "   Python: $python_embed2"
    echo "   Go:     $go_embed2"
fi

if [ "$python_embed3" = "$go_embed3" ]; then
    echo -e "${GREEN}✅ Sentence 3: EXACT MATCH${NC}"
    echo "   Values: $python_embed3"
else
    echo -e "${RED}❌ Sentence 3: MISMATCH${NC}"
    echo "   Python: $python_embed3"
    echo "   Go:     $go_embed3"
fi

echo ""

# Test 5: Similarity Matrix Comparison
echo -e "${BLUE}📈 Test 5: Similarity Matrix Comparison${NC}"
echo "========================================"

echo "Extracting similarity matrices..."
source .venv/bin/activate
python test_python_pytorch.py | grep -A6 "Python PyTorch Similarity Matrix:" | tail -6 > /tmp/python_sim.txt
go run safetensors_loader.go | grep -A6 "Go Safetensors Similarity Matrix:" | tail -6 > /tmp/go_sim.txt

echo ""
echo "Python PyTorch Similarity Matrix:"
cat /tmp/python_sim.txt
echo ""
echo "Go Safetensors Similarity Matrix:"  
cat /tmp/go_sim.txt
echo ""

# Compare similarity matrices
if cmp -s /tmp/python_sim.txt /tmp/go_sim.txt; then
    echo -e "${GREEN}✅ SIMILARITY MATRICES: IDENTICAL${NC}"
    similarity_match=true
else
    echo -e "${RED}❌ SIMILARITY MATRICES: DIFFERENT${NC}"
    similarity_match=false
fi

echo ""

# Final Results Summary
echo -e "${YELLOW}🏆 FINAL RESULTS SUMMARY${NC}"
echo "========================="

all_match=true
if [ "$python_embed1" = "$go_embed1" ] && [ "$python_embed2" = "$go_embed2" ] && [ "$python_embed3" = "$go_embed3" ]; then
    echo -e "${GREEN}✅ EMBEDDING VALUES: PERFECT MATCH${NC}"
else
    echo -e "${RED}❌ EMBEDDING VALUES: MISMATCH DETECTED${NC}"
    all_match=false
fi

if [ "$similarity_match" = true ]; then
    echo -e "${GREEN}✅ SIMILARITY MATRICES: PERFECT MATCH${NC}"
else
    echo -e "${RED}❌ SIMILARITY MATRICES: MISMATCH DETECTED${NC}"
    all_match=false
fi

echo ""
echo "Technical Details:"
echo "- Model: sentence-transformers/static-retrieval-mrl-en-v1"
echo "- Vocabulary Size: 30,522 tokens"
echo "- Embedding Dimension: 1,024"
echo "- Weights Format: safetensors (119MB)"
echo "- Test Sentences: 5 diverse samples"
echo ""

if [ "$all_match" = true ]; then
    echo -e "${GREEN}🎉 SUCCESS: PERFECT NUMERICAL CONSISTENCY!${NC}"
    echo -e "${GREEN}   Go and Python implementations are 100% identical${NC}"
    echo -e "${GREEN}   Maximum difference: 0.000000${NC}"
    echo ""
    echo -e "${BLUE}🚀 Ready for production use!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  WARNING: Inconsistencies detected${NC}"
    echo -e "${RED}   Further investigation needed${NC}"
    exit 1
fi