#!/bin/bash
set -e

echo "=================================================================================="
echo "🚀 Gobed Setup: Download LibTorch and Model Weights"
echo "=================================================================================="
echo "This script will download and setup everything needed to run Gobed from scratch:"
echo "  • LibTorch (for future GPU acceleration)"
echo "  • Real static-retrieval-mrl-en-v1 model weights (119MB)"
echo "  • Python dependencies for model download"
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
    echo "   Please install Go 1.19+ first"
    exit 1
fi

echo "✅ Python3 and Go found"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p model
mkdir -p libtorch
echo "✅ Directories created"
echo ""

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if ! pip3 show sentence-transformers huggingface-hub safetensors numpy &> /dev/null; then
    echo "Installing: sentence-transformers huggingface-hub safetensors numpy"
    pip3 install sentence-transformers huggingface-hub safetensors numpy
else
    echo "✅ Python dependencies already installed"
fi
echo ""

# Download LibTorch (CPU version for broader compatibility)
echo "🔥 Downloading LibTorch..."
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
LIBTORCH_ZIP="libtorch-cpu.zip"

if [ ! -f "libtorch/libtorch/lib/libtorch.so" ] && [ ! -f "libtorch/libtorch/lib/libtorch.dylib" ]; then
    echo "Downloading LibTorch CPU (2.1.0) - ~200MB..."
    if command -v wget &> /dev/null; then
        wget -O "$LIBTORCH_ZIP" "$LIBTORCH_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$LIBTORCH_ZIP" "$LIBTORCH_URL"
    else
        echo "❌ Error: Need wget or curl to download LibTorch"
        exit 1
    fi
    
    echo "Extracting LibTorch..."
    if command -v unzip &> /dev/null; then
        unzip -q "$LIBTORCH_ZIP" -d libtorch/
    else
        echo "❌ Error: unzip is required to extract LibTorch"
        exit 1
    fi
    
    rm "$LIBTORCH_ZIP"
    echo "✅ LibTorch downloaded and extracted"
else
    echo "✅ LibTorch already exists"
fi
echo ""

# Set LibTorch environment variables
export LIBTORCH="$(pwd)/libtorch/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"

# Download the real model weights
echo "🤖 Downloading real static-retrieval-mrl-en-v1 model..."
if [ ! -f "model/real_model.safetensors" ]; then
    echo "Creating model download script..."
    cat > download_model.py << 'EOF'
#!/usr/bin/env python3
import os
import json
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import shutil

def main():
    print("🔄 Downloading static-retrieval-mrl-en-v1 model...")
    
    # Download model
    model_name = "sentence-transformers/static-retrieval-mrl-en-v1"
    cache_dir = "./real_model_cache"
    
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        print(f"✅ Model downloaded to: {model_path}")
        
        # Find and copy safetensors file
        safetensors_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.safetensors'):
                    full_path = os.path.join(root, file)
                    safetensors_files.append(full_path)
        
        if safetensors_files:
            main_safetensors = safetensors_files[0]  # Take first safetensors file
            dest_path = "./model/real_model.safetensors"
            shutil.copy2(main_safetensors, dest_path)
            print(f"✅ Copied model weights to: {dest_path}")
            
            # Get file size
            size_mb = os.path.getsize(dest_path) / (1024*1024)
            print(f"📊 Model size: {size_mb:.1f} MB")
            
        else:
            print("❌ No safetensors files found!")
            return False
            
        # Generate reference tokens
        print("📝 Generating reference tokens...")
        model = SentenceTransformer(model_path)
        
        demo_sentences = [
            "Machine learning is fascinating.",
            "Artificial intelligence will change the world.",
            "Deep learning models are powerful.",
            "Neural networks process information.",
            "Hello world",
            "Good morning everyone",
            "Hi there friend",
            "The weather is nice today.",
            "Birds are singing beautifully.",
            "Trees grow tall in the forest.",
            "Python is a programming language.",
            "JavaScript runs in browsers.",
            "Code should be readable.",
            "The cat sits on the mat",
            "Pizza tastes delicious.",
            "Mathematics requires practice.",
            "This is a test sentence.",
            "Technology is advancing rapidly",
            "Natural language processing"
        ]
        
        reference_tokens = {}
        for sentence in demo_sentences:
            inputs = model.tokenize([sentence])
            token_ids = inputs['input_ids'].tolist()
            reference_tokens[sentence] = {
                "token_ids": token_ids,
                "length": len(token_ids)
            }
        
        # Save reference tokens
        tokens_path = "./model/real_reference_tokens.json"
        with open(tokens_path, 'w') as f:
            json.dump(reference_tokens, f, indent=2)
        print(f"✅ Saved {len(reference_tokens)} reference tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF
    
    python3 download_model.py
    if [ $? -eq 0 ]; then
        rm download_model.py
        echo "✅ Model download completed"
    else
        echo "❌ Model download failed"
        exit 1
    fi
else
    echo "✅ Model weights already exist"
fi
echo ""

# Test the setup
echo "🧪 Testing setup..."
if [ ! -f "main.go" ]; then
    echo "❌ main.go not found. Make sure you're in the gobed directory"
    exit 1
fi

echo "Compiling Go code..."
if go build -o gobed main.go; then
    echo "✅ Go compilation successful"
    
    echo ""
    echo "🎯 Running quick test..."
    timeout 10s ./gobed | head -n 20
    
    rm gobed  # Clean up binary
    echo ""
    echo "✅ Test completed successfully"
else
    echo "❌ Go compilation failed"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "🎉 Setup Complete!"
echo "=================================================================================="
echo "✅ LibTorch installed: $(pwd)/libtorch/libtorch"
echo "✅ Model weights downloaded: $(pwd)/model/real_model.safetensors"
echo "✅ Reference tokens generated: $(pwd)/model/real_reference_tokens.json"
echo "✅ Go compilation tested"
echo ""
echo "🚀 Ready to run:"
echo "   go run main.go"
echo ""
echo "💡 For GPU support (future), set environment:"
echo "   export LIBTORCH=$(pwd)/libtorch/libtorch"
echo "   export LD_LIBRARY_PATH=\$LIBTORCH/lib:\$LD_LIBRARY_PATH"
echo ""
echo "📊 Total downloads: ~320MB (LibTorch + Model)"
echo "=================================================================================="