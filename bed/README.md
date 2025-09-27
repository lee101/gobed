# Bed - Blazing Fast Semantic Search Tool 🛏

A high-performance semantic search tool built on GPU-accelerated embeddings, designed as the fastest alternative to ripgrep for semantic code search.

## Features

-  **GPU-Accelerated**: Leverages CUDA for 15x+ speedup on RTX series GPUs
-  **Async Indexing**: 26.8x faster indexing with worker pools
- 🧠 **Semantic Search**: Find code by meaning, not just text patterns
-  **Line-Level Results**: Shows exact line matches with context
- 📁 **Smart Filtering**: Respects .bedignore and .gitignore patterns
-  **Persistent Indexes**: Save and load embeddings for instant startup
- 🎨 **Colored Output**: Beautiful ripgrep-style colored results
-  **Progress Indicators**: Real-time indexing progress with ETA

## Quick Start

```bash
# Index current directory (first time)
bed index

# Search semantically 
bed "function for making art"
bed "error handling code"
bed "database connection setup"

# Search with options
bed "neural network" --limit 10 --context 3
bed "authentication" --ignore-case --color always
```

## Installation

```bash
# Build from source
go build -o bed main.go

# With GPU support (requires CUDA 12.8+)
go build -tags="gpu cuda" -o bed main.go
```

## Usage

```
bed [flags] <query>

Flags:
  -l, --limit int       Maximum number of results (default 5)
  -c, --context int     Lines of context around matches (default 2)
  -i, --ignore-case     Case-insensitive search
      --color string    When to colorize output (auto|always|never) (default "auto")
      --no-index        Skip indexing, use existing index only
      --force-index     Force re-indexing even if index exists
      --gpu             Enable GPU acceleration (default auto-detect)
      --threshold float Minimum similarity threshold (0.0-1.0) (default 0.7)
```

## Examples

```bash
# Find functions related to file operations
bed "file reading and writing functions"

# Find error handling patterns
bed "try catch exception handling"  

# Find database-related code
bed "sql query execution"

# Search with high precision
bed "authentication middleware" --threshold 0.85

# Force rebuild index
bed "search query" --force-index
```

Built with ❤ and blazing fast Go + CUDA.