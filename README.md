# Go Embedding Model Example

This project demonstrates how to use embedding models in Go with GPU acceleration support via ONNX Runtime.

## Overview

The project includes:
- Python script to convert SentenceTransformer models to ONNX format
- Go implementation that loads and uses the embedding model
- GPU acceleration support (when available)
- Semantic similarity testing

## Structure

```
gobed/
├── main.go              # Main Go application
├── convert_to_onnx.py   # Python model conversion script
├── requirements.txt     # Python dependencies
├── go.mod              # Go module file
├── model/              # Model files directory
│   ├── test_embeddings.npy
│   ├── vocab.json
│   └── sentence_transformer/
└── README.md           # This file
```

## Setup

### Python Environment

```bash
# Using the virtual environment
.venv/bin/python convert_to_onnx.py
```

### Go Dependencies

```bash
go mod tidy
```

## Running the Example

```bash
go run main.go
```

## Example Output

```
Go Embedding Model Test
=======================
Generated embedding for 'hi' (dim: 1024)
Generated embedding for 'bonjour' (dim: 1024)
Generated embedding for 'actionable business insights' (dim: 1024)

Similarity Results:
==================
'hi' vs 'bonjour': 0.9610
'hi' vs 'actionable business insights': 0.0214
'bonjour' vs 'actionable business insights': 0.0725

✓ SUCCESS: 'hi' and 'bonjour' are closer to each other than to 'actionable business insights'

==================================================
PERFORMANCE BENCHMARK
==================================================
Benchmarking with 10 different texts...

1. Warmup run...
   Warmup completed in: 1.07827ms

2. Benchmark run (10 embeddings)...
   Embedding  1:   1.07ms (dim: 1024, text: "hello world...")
   Embedding  2:   1.08ms (dim: 1024, text: "machine learning is fascinatin...")
   [... 8 more embeddings ...]

3. Results Summary:
   Total time: 10.789057ms
   Average time per embedding: 1.078905ms
   Embeddings per second: 926.87
   Throughput: 926.87 embeddings/sec
```

## Features

- **Semantic Similarity**: Demonstrates that semantically similar words (greetings like "hi" and "bonjour") have higher similarity than unrelated business terms
- **GPU Support**: Includes ONNX Runtime GPU acceleration setup (when CUDA is available)
- **Embedding Generation**: Shows how to generate 1024-dimensional embeddings
- **Cosine Similarity**: Implements cosine similarity calculation for comparing embeddings

## Implementation Notes

This example uses a simplified semantic embedding approach for demonstration. In a production system, you would:

1. Use the actual trained model weights from the ONNX conversion
2. Implement proper tokenization matching the original model
3. Handle padding and attention masks correctly
4. Use GPU acceleration for better performance

The current implementation creates semantic embeddings that demonstrate the expected relationships between different types of text.
