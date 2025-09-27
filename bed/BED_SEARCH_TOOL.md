# bed - GPU-Accelerated Semantic Search Tool

Ultra-fast semantic search for code and text files using static embeddings with optional GPU acceleration.

## Overview

`bed` is a command-line tool for semantic search across filesystems. It uses pre-trained static embeddings to find semantically similar content, achieving sub-millisecond search latency on GPU-enabled systems.

### Key Features
- **Sub-millisecond search** on GPU (0.02ms on RTX 3090)
- **Auto-detects CUDA** and falls back to CPU gracefully
- **Int8 quantization** for 4x memory reduction
- **Smart file chunking** with overlap for better context
- **Language-aware filtering** (ignores build artifacts, dependencies)

## Installation

### Prerequisites

- Go 1.21+
- Model files (119MB, downloaded via setup.sh)
- Optional: CUDA 12.x for GPU acceleration

### Basic Setup (CPU)

```bash
# Clone repository
git clone https://github.com/lee101/gobed
cd gobed

# Download model files
./setup.sh

# Build bed tool
cd bed
go build -o bed bed_cuda.go
```

### GPU Setup

```bash
# Detect GPU and set environment
../scripts/detect_gpu.sh
source gpu_env.sh

# Build with GPU support
go build -tags="gpu cuda" -o bed bed_cuda.go

# Or use build script
./build_bed.sh
```

## Usage

### Basic Search

```bash
# Search current directory
./bed "neural networks"

# Search specific directory
./bed -dir /path/to/code "authentication flow"

# Get more results
./bed -k 20 "database migration"
```

### Command-line Options

```
bed [OPTIONS] [SEARCH_QUERY]

Options:
  -search string   Search query (or provide as first argument)
  -dir string     Directory to search (default: current directory)
  -k int          Number of results to show (default: 12)
  -chunk int      Lines per chunk for indexing (default: 20)
  -debug          Enable debug output with timing stats
  -gpu            Use GPU acceleration (default: auto-detect)

Examples:
  bed "search query"                    # Search current directory
  bed -dir ~/projects "async handler"   # Search specific directory
  bed -k 5 -debug "error handling"     # Top 5 results with debug info
  bed --gpu=false "test"               # Force CPU-only mode
```

## How It Works

### Indexing Process

1. **File Discovery**: Recursively walks directory, filtering by:
   - Text file extensions (.go, .py, .js, .md, etc.)
   - Ignores common artifacts (node_modules, .git, build/, etc.)
   - Skips binary files and files >10MB

2. **Chunking Strategy**:
   - Small files (<20 lines): One embedding per line for precision
   - Large files: 20-line chunks with 25% overlap for context
   - Each chunk maintains file path and line numbers

3. **Embedding Generation**:
   - Uses pre-quantized Int8 512-dimensional static embeddings
   - Model: `sentence-transformers/static-retrieval-mrl-en-v1`
   - Simple token→vector lookup (not BERT)

4. **GPU Acceleration** (when available):
   - Batch processing of embeddings
   - CUDA kernel fusion for similarity computation
   - Automatic IVF clustering at 50K+ documents

### Search Process

1. **Query Embedding**: Converts search query to 512-dim vector
2. **Similarity Computation**: Cosine similarity against all chunks
3. **Ranking**: Top-K results by similarity score
4. **Display**: Shows file path, line numbers, and preview

## Performance Benchmarks

### Search Latency

| Dataset Size | CPU Mode | GPU Mode | Speedup |
|-------------|----------|----------|---------|
| 1K lines    | 2ms      | 0.3ms    | 6.7x    |
| 10K lines   | 18ms     | 1.2ms    | 15x     |
| 100K lines  | 180ms    | 8ms      | 22.5x   |
| 243K lines  | 450ms    | 20ms     | 22.5x   |

### Indexing Speed

| Dataset Size | CPU Mode | GPU Mode |
|-------------|----------|----------|
| 10K lines   | 0.8s     | 0.2s     |
| 100K lines  | 7.5s     | 1.8s     |
| 1M lines    | 75s      | 18s      |

## CUDA Auto-detection

The tool automatically detects CUDA availability through:

1. **Build-time Detection** (`build_bed.sh`):
   - Checks for nvidia-smi and CUDA libraries
   - Sets `GPU_ENABLED` and `CUDA_ENABLED` env vars

2. **Runtime Detection** (in bed_cuda.go):
   - Checks environment variables first
   - Falls back to checking `/usr/local/cuda/lib64/libcudart.so`
   - Gracefully degrades to CPU if CUDA unavailable

3. **Manual Override**:
   ```bash
   # Force GPU mode
   ./bed --gpu=true "query"

   # Force CPU mode
   ./bed --gpu=false "query"
   ```

## Ignored Patterns

The tool intelligently skips common non-source files:

### Directories
- Version control: `.git`, `.svn`, `.hg`
- Dependencies: `node_modules`, `vendor`, `venv`, `target`
- Build outputs: `dist`, `build`, `bin`, `out`
- IDE/Editor: `.idea`, `.vscode`, `.vs`
- Caches: `__pycache__`, `.cache`, `.next`

### File Extensions
- Binary: `.exe`, `.dll`, `.so`, `.o`
- Archives: `.zip`, `.tar`, `.gz`
- Media: `.jpg`, `.png`, `.mp4`, `.pdf`
- Databases: `.db`, `.sqlite`
- Lock files: `.lock`, `package-lock.json`

## Advanced Features

### Debug Mode

```bash
./bed --debug "search term"
```

Shows:
- Model loading time and memory usage
- Number of files/chunks indexed
- Indexing duration
- Search latency
- GPU vs CPU mode status

### Custom Chunking

```bash
# Smaller chunks for fine-grained search
./bed -chunk 10 "specific function"

# Larger chunks for more context
./bed -chunk 50 "architectural pattern"
```

### Integration Examples

#### Git Integration
```bash
# Search changed files
git diff --name-only | xargs -I {} ./bed -dir {} "TODO"

# Search specific commit
git show --name-only COMMIT_HASH | xargs -I {} ./bed -dir {} "bug"
```

#### Find + bed
```bash
# Search only Python files
find . -name "*.py" -type f | xargs dirname | sort -u | xargs -I {} ./bed -dir {} "import numpy"
```

#### Watch Mode (with entr)
```bash
# Re-search on file changes
ls *.go | entr -c ./bed "function signature"
```

## Troubleshooting

### CUDA Not Detected
```bash
# Check CUDA installation
nvidia-smi
ls -la /usr/local/cuda/lib64/

# Run detection script
../scripts/detect_gpu.sh

# Set environment manually
export CUDA_ENABLED=true
export GPU_ENABLED=true
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Model Files Not Found
```bash
# Ensure model files exist
ls -la ../model/
# Should contain:
# - real_model.safetensors
# - tokenizer.json

# Re-download if missing
cd .. && ./setup.sh
```

### Build Errors
```bash
# Clean build
go clean -cache
go mod tidy

# Build with verbose output
go build -v -x -tags="gpu cuda" -o bed bed_cuda.go
```

### Performance Issues
- Ensure GPU mode is active: `./bed --debug "test"`
- Check GPU utilization: `nvidia-smi`
- Reduce chunk size for faster indexing: `-chunk 10`
- Limit search scope: `-dir specific/path`

## Implementation Details

### Model Architecture
- **Type**: Static embeddings with mean pooling
- **Dimensions**: 512 (quantized from 1024)
- **Quantization**: Int8 (4x memory reduction)
- **Vocabulary**: 30,522 tokens
- **Source**: Hugging Face static-retrieval-mrl-en-v1

### Memory Usage
- **CPU Mode**: ~200MB for model + indexed data
- **GPU Mode**: ~150MB (model in VRAM)
- **Scaling**: ~1KB per document chunk

### Accuracy vs Speed Trade-offs
- Int8 quantization: -2% accuracy, 4x memory reduction
- 512 dims vs 1024: -5% accuracy, 2x faster
- GPU batching: 10-20x faster, identical accuracy

## Future Enhancements

Planned improvements:
- Persistent index cache for large codebases
- Incremental indexing for file changes
- Multi-GPU support for massive datasets
- Faiss integration for billion-scale search
- Language-specific tokenization
- Semantic code refactoring suggestions