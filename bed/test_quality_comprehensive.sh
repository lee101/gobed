#!/bin/bash

# Comprehensive quality testing for bed search tools
set -e

echo "🧪 COMPREHENSIVE QUALITY TESTING"
echo "=================================="

cd /home/lee/code/gobed/bed

# Test queries that should have semantic results (not just string matches)
TEST_QUERIES=(
    "machine learning"
    "artificial intelligence"
    "neural networks"
    "deep learning algorithms"
    "computer vision"
    "natural language processing"
    "programming languages"
    "software development"
    "data structures"
    "algorithms optimization"
)

# Test with different datasets
TEST_DIRS=(
    "/tmp/ai_small"
    "../testdata"
    "."
)

echo "📊 Testing bed_real performance and quality..."
echo "=============================================="

for dir in "${TEST_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "🔍 Testing directory: $dir"

        for query in "${TEST_QUERIES[@]}"; do
            echo "Query: '$query'"
            echo "----------------"

            # Run search and capture timing + results
            result=$(./bed_real -dir "$dir" -k 5 "$query" 2>&1)

            # Extract timing
            timing=$(echo "$result" | grep "Time:" | head -1)
            echo "  $timing"

            # Count actual results (lines that start with numbers)
            result_count=$(echo "$result" | grep -E "^\s*[0-9]+\." | wc -l)
            echo "  Results found: $result_count"

            # Check for semantic relevance (not just string matches)
            semantic_matches=$(echo "$result" | grep -E "^\s*[0-9]+\." | grep -v "✅" | wc -l)
            string_matches=$(echo "$result" | grep -E "^\s*[0-9]+\." | grep "✅" | wc -l)

            echo "  Semantic matches: $semantic_matches"
            echo "  String matches: $string_matches"

            if [ $result_count -gt 0 ]; then
                semantic_ratio=$(echo "scale=2; $semantic_matches * 100 / $result_count" | bc)
                echo "  Semantic percentage: ${semantic_ratio}%"

                # Show top result for manual inspection
                top_result=$(echo "$result" | grep -E "^\s*[0-9]+\." | head -1)
                echo "  Top result: $top_result"
            fi

            echo ""
        done
        echo ""
    else
        echo "⚠️  Directory $dir not found, skipping..."
    fi
done

echo "🔬 SEMANTIC QUALITY VERIFICATION"
echo "================================"

# Test specific semantic relationships
echo "Testing semantic understanding..."

# Create a small test dataset with known semantic relationships
mkdir -p /tmp/semantic_test
cat > /tmp/semantic_test/tech.txt << 'EOF'
Machine learning algorithms are used in artificial intelligence systems.
Neural networks form the backbone of deep learning frameworks.
Computer vision processes images using convolutional neural networks.
Natural language processing analyzes text data for understanding.
Python is a popular programming language for data science.
JavaScript enables dynamic web development and user interfaces.
Database systems store and retrieve structured information efficiently.
Cloud computing provides scalable infrastructure for applications.
Cybersecurity protects systems from malicious attacks and threats.
Software engineering involves designing robust and maintainable code.
EOF

cat > /tmp/semantic_test/science.txt << 'EOF'
Physics studies the fundamental laws governing matter and energy.
Chemistry examines molecular interactions and chemical reactions.
Biology investigates living organisms and their complex processes.
Mathematics provides the logical foundation for scientific analysis.
Astronomy explores celestial bodies and cosmic phenomena.
Geology studies Earth structure and geological formations.
Environmental science addresses climate change and sustainability.
Quantum mechanics describes behavior at the atomic scale.
Genetics analyzes heredity and DNA sequence variations.
Medical research develops treatments for human diseases.
EOF

# Test semantic relationships
echo "Testing 'algorithms' -> should find ML/AI content"
./bed_real -dir /tmp/semantic_test -k 3 "algorithms" | grep -E "^\s*[0-9]+\."

echo ""
echo "Testing 'programming' -> should find software/coding content"
./bed_real -dir /tmp/semantic_test -k 3 "programming" | grep -E "^\s*[0-9]+\."

echo ""
echo "Testing 'science' -> should find scientific content"
./bed_real -dir /tmp/semantic_test -k 3 "science" | grep -E "^\s*[0-9]+\."

echo ""
echo "🚀 PERFORMANCE STRESS TEST"
echo "=========================="

# Test with larger workload
if [ -d "../testdata" ]; then
    echo "Testing search performance under load..."

    for i in {1..10}; do
        start_time=$(date +%s%3N)
        ./bed_real -dir "../testdata" -k 10 "machine learning" > /dev/null 2>&1
        end_time=$(date +%s%3N)
        duration=$((end_time - start_time))
        echo "Iteration $i: ${duration}ms"
    done
fi

echo ""
echo "🔧 ERROR CONDITION TESTING"
echo "=========================="

# Test error conditions
echo "Testing with non-existent directory..."
./bed_real -dir "/nonexistent" -k 5 "test" 2>&1 | head -3

echo ""
echo "Testing with empty query..."
./bed_real -dir "/tmp/semantic_test" -k 5 "" 2>&1 | head -3

echo ""
echo "Testing with very large k value..."
./bed_real -dir "/tmp/semantic_test" -k 1000 "test" 2>&1 | head -5

echo ""
echo "✅ COMPREHENSIVE TESTING COMPLETE"
echo "================================="

# Cleanup
rm -rf /tmp/semantic_test