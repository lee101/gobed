# 🚀 Complete Setup Guide

This guide shows how to set up Gobed from scratch, including LibTorch and real model weights.

## ⚡ Quick Start (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo> gobed
cd gobed

# 2. Run the automated setup script
./setup.sh

# 3. Run the demo
go run main.go
```

## 📋 What the Setup Script Does

The `setup.sh` script automatically handles:

### 1. Dependency Check
- ✅ Verifies Go 1.19+ is installed
- ✅ Verifies Python 3.7+ is available

### 2. Python Dependencies
Installs required packages:
```bash
pip3 install sentence-transformers huggingface-hub safetensors numpy
```

### 3. LibTorch Download (~200MB)
Downloads and extracts LibTorch CPU version:
- **URL**: `https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip`
- **Location**: `./libtorch/libtorch/`
- **Purpose**: Future GPU acceleration support

### 4. Real Model Weights (119MB)
Downloads the actual `sentence-transformers/static-retrieval-mrl-en-v1` model:
- **Source**: HuggingFace Hub
- **Format**: Safetensors
- **Location**: `./model/real_model.safetensors`

### 5. Reference Token Generation
Creates tokenization data for 19 demo sentences covering:
- Technology terms (ML, AI, deep learning)
- Programming concepts (Python, JavaScript)  
- Greetings (Hello, Good morning)
- Nature descriptions (Weather, birds, trees)
- Random phrases for contrast

### 6. Validation
- Compiles the Go code
- Runs a quick test to verify everything works
- Shows sample output

## 🔧 Manual Setup Alternative

If you prefer manual setup:

```bash
# Install Python packages
pip3 install sentence-transformers huggingface-hub safetensors numpy

# Download LibTorch (optional)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip -d libtorch/

# Download model using Python
python3 -c "
from huggingface_hub import snapshot_download
import shutil, os, json
from sentence_transformers import SentenceTransformer

# Download model
model_path = snapshot_download('sentence-transformers/static-retrieval-mrl-en-v1')
print(f'Downloaded to: {model_path}')

# Find safetensors file
for root, dirs, files in os.walk(model_path):
    for file in files:
        if file.endswith('.safetensors'):
            src = os.path.join(root, file)
            os.makedirs('model', exist_ok=True)
            shutil.copy2(src, 'model/real_model.safetensors')
            print('Copied model weights')
            break

# Generate reference tokens (simplified)
model = SentenceTransformer(model_path)
sentences = ['Hello world', 'Machine learning is fascinating.', 'This is a test sentence.']
tokens = {}
for s in sentences:
    inputs = model.tokenize([s])
    tokens[s] = {'token_ids': inputs['input_ids'].tolist(), 'length': len(inputs['input_ids'])}

with open('model/real_reference_tokens.json', 'w') as f:
    json.dump(tokens, f, indent=2)
print('Generated reference tokens')
"
```

## 📊 Download Summary

After setup completes, you'll have:

| Component | Size | Purpose |
|-----------|------|---------|
| **LibTorch** | ~200MB | Future GPU acceleration |
| **Model Weights** | 119MB | Real safetensors embedding matrix |
| **Reference Tokens** | ~2KB | Tokenization for 19 demo sentences |
| **HF Cache** | ~50MB | HuggingFace model cache |
| **Total** | ~370MB | Complete setup |

## ✅ Verification

The setup script ends with a validation test. You should see:

```
🧪 Testing setup...
Compiling Go code...
✅ Go compilation successful

🎯 Running quick test...
================================================================================
🚀 Gobed: Real Embedding Model Demo
================================================================================
Model: sentence-transformers/static-retrieval-mrl-en-v1 (REAL WEIGHTS)

🔄 Loading real static-retrieval-mrl-en-v1 model...
✅ Model loaded in 576ms (vocab: 30522, dims: 1024)
...
✅ Test completed successfully

🎉 Setup Complete!
✅ LibTorch installed
✅ Model weights downloaded  
✅ Reference tokens generated
✅ Go compilation tested
```

## 🚀 Next Steps

After successful setup:

1. **Run the full demo**: `go run main.go`
2. **Integrate the API**: Use `LoadModel()`, `Encode()`, `Similarity()` functions
3. **Benchmark performance**: Compare with your Python implementations
4. **Deploy**: Single binary with embedded model weights

## 🐛 Troubleshooting

**"python3 not found"**
- Install Python 3.7+: https://python.org/downloads/

**"go not found"**  
- Install Go 1.19+: https://golang.org/dl/

**"unzip not found"**
- Ubuntu/Debian: `sudo apt install unzip`
- macOS: Included by default
- Windows: Use built-in or install 7-zip

**Download fails**
- Check internet connection
- Try manual download links provided above
- Some corporate networks block HuggingFace - use VPN if needed

**Permission denied on setup.sh**
```bash
chmod +x setup.sh
./setup.sh
```

## 💡 Tips

- **Disk Space**: Ensure 500MB+ free space before setup
- **Network**: Downloads ~320MB total, ensure stable connection  
- **Python Packages**: Virtual environment recommended but not required
- **Go Version**: Use `go version` to check you have 1.19+

The setup is designed to be robust and provide clear error messages if anything goes wrong!