#!/bin/bash
set -e

echo "=================================================================================="
echo "🚀 Gobed Setup: Download Model Weights"
echo "=================================================================================="
echo "This script will download the real static-retrieval-mrl-en-v1 model weights (119MB)"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is required but not installed"
    echo "   Please install Python 3.7+ first"
    exit 1
fi

# Check if Go is available
if ! command -v go &> /dev/null; then
    echo "❌ Error: go is required but not installed"
    echo "   Please install Go 1.21+ first"
    exit 1
fi

echo "✅ Python3 and Go found"
echo ""

# Create model directory
echo "📁 Creating model directory..."
mkdir -p model
echo "✅ Directory created"
echo ""

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
PY_DEPS=(huggingface-hub safetensors sentence-transformers transformers torch)
MISSING=()
for dep in "${PY_DEPS[@]}"; do
    if ! pip3 show "$dep" &> /dev/null; then
        MISSING+=("$dep")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing: ${MISSING[*]}"
    pip3 install "${MISSING[@]}"
else
    echo "✅ Python dependencies already installed"
fi
echo ""

# Check if model already exists
if [ -f "model/real_model.safetensors" ]; then
    echo "✅ Model weights already downloaded (model/real_model.safetensors)"
else
    echo "📥 Downloading real model weights..."
    python3 download_real_model.py
    echo "✅ Model weights downloaded"
fi
echo ""

# Check if tokenizer already exists
if [ -f "model/tokenizer.json" ]; then
    echo "✅ Tokenizer already downloaded (model/tokenizer.json)"
else
    echo "📥 Downloading tokenizer.json from static-retrieval-mrl-en-v1..."
    curl -L -o model/tokenizer.json "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding/tokenizer.json"
    echo "✅ Tokenizer downloaded"
fi
echo ""

# Generate reference tokens if needed
if [ -f "model/real_reference_tokens.json" ]; then
    echo "✅ Reference tokens already exist"
else
    echo "🔤 Generating reference tokens..."
    python3 generate_real_reference.py
    echo "✅ Reference tokens generated"
fi
echo ""

# Test Go build
echo "🔨 Testing Go build..."
go build -o /tmp/gobed_test cmd/demo/main.go
rm /tmp/gobed_test
echo "✅ Go build successful"
echo ""

echo "=================================================================================="
echo "✨ Setup Complete!"
echo "=================================================================================="
echo ""
echo "You can now run the demo:"
echo "  go run cmd/demo/main.go"
echo ""
echo "Or use gobed as a library in your project:"
echo "  go get github.com/lee101/gobed"
echo ""
echo "Optional: For GPU acceleration support, run:"
echo "  ./setup_libtorch.sh"
echo ""