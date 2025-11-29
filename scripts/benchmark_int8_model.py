#!/usr/bin/env python3
"""
Benchmark script for the int8 512-dim model
Tests loading, embedding, and performance
"""

import numpy as np
import json
import struct
import time
import os
from pathlib import Path

def read_safetensors(filepath):
    """Read safetensors file format"""
    with open(filepath, 'rb') as f:
        # Read header length (8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]

        # Read header JSON
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)

        # Read tensor data
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue

            dtype = info['dtype']
            shape = info['shape']
            offset_start, offset_end = info['data_offsets']

            # Seek to tensor data position (8 + header_len + offset)
            f.seek(8 + header_len + offset_start)
            num_bytes = offset_end - offset_start

            # Read raw bytes
            raw_data = f.read(num_bytes)

            # Parse based on dtype
            if dtype == 'F32':
                # Float32 data
                tensor_data = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)
            elif dtype == 'I8':
                # Int8 data
                tensor_data = np.frombuffer(raw_data, dtype=np.int8).reshape(shape)
            else:
                print(f"Unsupported dtype: {dtype} for tensor {name}")
                continue

            tensors[name] = tensor_data

    return tensors, header

def simple_tokenize(text, vocab):
    """Simple tokenization that mirrors the Go implementation"""
    text = text.lower().strip()
    words = text.split()

    tokens = []

    # Add [CLS] token
    cls_id = vocab.get('[CLS]', vocab.get('cls', 101))
    tokens.append(cls_id)

    for word in words:
        # Try exact match
        if word in vocab:
            tokens.append(vocab[word])
            continue

        # Try with ## prefix
        subword = f"##${word}"
        if subword in vocab:
            tokens.append(vocab[subword])
            continue

        # Try partial matches
        found = False
        for token, token_id in vocab.items():
            if len(token) > 3 and token.startswith('##') and token[2:] in word:
                tokens.append(token_id)
                found = True
                break

        # Fallback to [UNK]
        if not found:
            unk_id = vocab.get('[UNK]', vocab.get('unk', 100))
            tokens.append(unk_id)

    # Add [SEP] token
    sep_id = vocab.get('[SEP]', vocab.get('sep', 102))
    tokens.append(sep_id)

    return tokens

def embed_tokens(tokens, embeddings, scales):
    """Embed tokens using int8 embeddings"""
    if len(tokens) == 0:
        return np.zeros(512, dtype=np.float32)

    result = np.zeros(512, dtype=np.float32)
    valid_tokens = 0

    for token in tokens:
        if 0 <= token < len(embeddings):
            embedding = embeddings[token].astype(np.float32)
            scale = scales[token]
            result += embedding * scale
            valid_tokens += 1

    if valid_tokens > 0:
        result /= valid_tokens

    return result

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def benchmark_model():
    """Benchmark the int8 model"""
    print(" Benchmarking Int8 512-dim Model")
    print("=" * 50)

    # Paths
    model_path = Path("model/modelint8_512dim.safetensors")
    tokenizer_path = Path("model/tokenizer.json")

    if not model_path.exists():
        print(f" Model not found: {model_path}")
        return

    if not tokenizer_path.exists():
        print(f" Tokenizer not found: {tokenizer_path}")
        return

    # Load model
    print(f"📚 Loading model from {model_path}")
    start_time = time.time()
    tensors, header = read_safetensors(model_path)
    load_time = time.time() - start_time

    embeddings = tensors['embeddings.weight']
    scales = tensors['embeddings.scales']

    print(f" Model loaded in {load_time:.3f}s")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Scales shape: {scales.shape}")
    print(f"   Memory usage: {embeddings.nbytes + scales.nbytes / 1024 / 1024:.1f} MB")

    # Load tokenizer vocab
    print(f" Loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path) as f:
        tokenizer_data = json.load(f)

    vocab = tokenizer_data['model']['vocab']
    print(f" Loaded vocab with {len(vocab)} tokens")

    # Test texts
    test_texts = [
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing",
        "computer vision applications",
        "artificial intelligence systems",
        "data science and analytics",
        "reinforcement learning agents",
        "transformer model architectures",
        "convolutional neural networks",
        "recurrent neural networks"
    ]

    # Warm up
    print("\n Warming up...")
    for text in test_texts[:3]:
        tokens = simple_tokenize(text, vocab)
        _ = embed_tokens(tokens, embeddings, scales)

    # Benchmark embedding speed
    print("\n  Benchmarking embedding speed...")
    num_iterations = 1000

    start_time = time.time()
    for i in range(num_iterations):
        text = test_texts[i % len(test_texts)]
        tokens = simple_tokenize(text, vocab)
        embedding = embed_tokens(tokens, embeddings, scales)

    total_time = time.time() - start_time
    avg_latency = (total_time / num_iterations) * 1000  # ms
    throughput = num_iterations / total_time

    print(f" Performance Results:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average latency: {avg_latency:.3f}ms")
    print(f"   Throughput: {throughput:.0f} embeddings/sec")

    # Test similarity computation
    print("\n Testing similarity computation...")

    test_pairs = [
        ("machine learning", "machine learning"),
        ("deep learning", "neural networks"),
        ("computer vision", "image processing"),
        ("hello world", "machine learning"),
    ]

    for text1, text2 in test_pairs:
        tokens1 = simple_tokenize(text1, vocab)
        tokens2 = simple_tokenize(text2, vocab)

        emb1 = embed_tokens(tokens1, embeddings, scales)
        emb2 = embed_tokens(tokens2, embeddings, scales)

        similarity = cosine_similarity(emb1, emb2)
        print(f"   Similarity('{text1}', '{text2}') = {similarity:.4f}")

    # Memory analysis
    print("\n Memory Analysis:")

    # Original model size (float32, 1024 dims)
    original_size = 30522 * 1024 * 4  # bytes
    int8_size = embeddings.nbytes + scales.nbytes

    compression_ratio = original_size / int8_size
    print(f"   Original model (float32, 1024d): {original_size / 1024 / 1024:.1f} MB")
    print(f"   Int8 model (512d): {int8_size / 1024 / 1024:.1f} MB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Space saved: {(1 - int8_size / original_size) * 100:.1f}%")

    # Quality analysis
    print("\n Quality Analysis:")

    # Check embedding statistics
    embedding_stats = {
        'min': embeddings.min(),
        'max': embeddings.max(),
        'mean': embeddings.mean(),
        'std': embeddings.std()
    }

    scale_stats = {
        'min': scales.min(),
        'max': scales.max(),
        'mean': scales.mean(),
        'std': scales.std()
    }

    print(f"   Embedding int8 stats: min={embedding_stats['min']}, max={embedding_stats['max']}, "
          f"mean={embedding_stats['mean']:.3f}, std={embedding_stats['std']:.3f}")
    print(f"   Scale stats: min={scale_stats['min']:.6f}, max={scale_stats['max']:.6f}, "
          f"mean={scale_stats['mean']:.6f}, std={scale_stats['std']:.6f}")

    print("\n Benchmark complete!")

if __name__ == "__main__":
    benchmark_model()