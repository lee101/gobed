#!/usr/bin/env python3
"""Upload gobed int8 model to HuggingFace Hub."""

from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "lee101/bed"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# Model card content
MODEL_CARD = """---
language:
  - en
license: mit
library_name: gobed
tags:
  - embeddings
  - semantic-search
  - int8
  - quantized
  - static-embeddings
  - sentence-embeddings
pipeline_tag: sentence-similarity
---

# Bed - Int8 Quantized Static Embeddings for Semantic Search

Ultra-fast int8 quantized static embeddings model for semantic search. Optimized for the [gobed](https://github.com/lee101/gobed) Go library.

## Model Details

| Property | Value |
|----------|-------|
| **Dimensions** | 512 |
| **Precision** | int8 + scale factors |
| **Vocabulary** | 30,522 tokens |
| **Model Size** | 15 MB |
| **Format** | safetensors |

## Performance

- **Embedding latency**: 0.16ms average
- **Throughput**: 6,200+ embeddings/sec
- **Memory**: 15 MB (7.9x smaller than float32 version)
- **Compression ratio**: 87.4% space reduction vs original

## Usage with gobed (Go)

```bash
go get github.com/lee101/gobed
```

```go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    engine, err := gobed.NewAutoSearchEngine()
    if err != nil {
        log.Fatal(err)
    }
    defer engine.Close()

    docs := map[string]string{
        "doc1": "machine learning and neural networks",
        "doc2": "natural language processing",
    }
    engine.AddDocuments(docs)

    results, _, _ := engine.SearchWithMetadata("AI research", 3)
    for _, r := range results {
        fmt.Printf("[%.3f] %s\\n", r.Similarity, r.Content)
    }
}
```

## Download Model Manually

```bash
# Clone the model repository
git clone https://huggingface.co/lee101/bed

# Or download specific files
wget https://huggingface.co/lee101/bed/resolve/main/modelint8_512dim.safetensors
wget https://huggingface.co/lee101/bed/resolve/main/tokenizer.json
```

## Using huggingface_hub (Python)

```python
from huggingface_hub import hf_hub_download

# Download model file
model_path = hf_hub_download(repo_id="lee101/bed", filename="modelint8_512dim.safetensors")

# Download tokenizer
tokenizer_path = hf_hub_download(repo_id="lee101/bed", filename="tokenizer.json")
```

## Model Architecture

This model uses static embeddings with int8 quantization:

- **Embedding layer**: 30,522 x 512 int8 weights
- **Scale factors**: 30,522 float32 scale values (one per token)
- **Tokenizer**: WordPiece tokenizer (same as BERT)

Embeddings are computed by:
1. Tokenizing input text
2. Looking up int8 embeddings for each token
3. Multiplying by scale factors to reconstruct float values
4. Mean pooling across tokens

## Quantization Details

Original model: 30,522 x 1024 float32 (119 MB)
Quantized model: 30,522 x 512 int8 + 30,522 float32 scales (15 MB)

Per-vector quantization preserves relative magnitudes:
```python
max_abs = max(abs(embedding_vector))
scale = max_abs / 127.0
quantized = round(embedding_vector / scale).astype(int8)
```

## Files

- `modelint8_512dim.safetensors` - Quantized embeddings and scales
- `tokenizer.json` - HuggingFace tokenizer

## License

MIT License - see [gobed repository](https://github.com/lee101/gobed) for details.

## Citation

```bibtex
@software{gobed,
  author = {Lee Penkman},
  title = {gobed: Ultra-Fast Semantic Search for Go},
  url = {https://github.com/lee101/gobed},
  year = {2024}
}
```
"""

def main():
    api = HfApi()

    # Create repo (will skip if exists)
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository {REPO_ID} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload model file
    print("Uploading modelint8_512dim.safetensors...")
    api.upload_file(
        path_or_fileobj=os.path.join(MODEL_DIR, "modelint8_512dim.safetensors"),
        path_in_repo="modelint8_512dim.safetensors",
        repo_id=REPO_ID,
        repo_type="model",
    )

    # Upload tokenizer
    print("Uploading tokenizer.json...")
    api.upload_file(
        path_or_fileobj=os.path.join(MODEL_DIR, "tokenizer.json"),
        path_in_repo="tokenizer.json",
        repo_id=REPO_ID,
        repo_type="model",
    )

    # Upload README/model card
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(f"\nModel uploaded to: https://huggingface.co/{REPO_ID}")
    print("\nTo download:")
    print(f"  git clone https://huggingface.co/{REPO_ID}")
    print("  # or")
    print(f"  huggingface-cli download {REPO_ID}")

if __name__ == "__main__":
    main()
