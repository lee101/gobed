#!/usr/bin/env python3
"""
INT8 vs Python Float32 Comparison Test

This script compares Go INT8 quantization with Python's float32 implementation
to verify accuracy and validate the quantization approach.
"""

import numpy as np
import json
import struct
import os
from typing import List, Tuple
import time

def load_safetensors(filepath: str) -> np.ndarray:
    """Load weights from safetensors format"""
    with open(filepath, 'rb') as f:
        # Read header length
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        
        # Read and parse header
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        
        # Read data
        data = f.read()
        
        # Get embedding weights
        embed_info = header.get('embedding.weight')
        if not embed_info:
            raise ValueError("embedding.weight not found in safetensors")
        
        start, end = embed_info['data_offsets']
        shape = embed_info['shape']
        
        # Convert bytes to float32 array
        tensor_bytes = data[start:end]
        weights = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
        
    return weights

def quantize_to_int8(weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Quantize float32 weights to INT8"""
    # Find min and max
    min_val = weights.min()
    max_val = weights.max()
    
    # Calculate scale and zero point for symmetric quantization
    scale = (max_val - min_val) / 255.0
    zero_point = int(-128 - min_val / scale)
    
    # Quantize
    weights_int8 = np.round(weights / scale + zero_point).astype(np.int8)
    
    return weights_int8, scale, zero_point

def dequantize_int8(weights_int8: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize INT8 weights back to float32"""
    return (weights_int8.astype(np.float32) - zero_point) * scale

def compute_embedding_int8(weights_int8: np.ndarray, token_ids: List[int], 
                          scale: float, zero_point: int) -> np.ndarray:
    """Compute embedding using INT8 weights"""
    # Sum embeddings for valid tokens
    embedding = np.zeros(weights_int8.shape[1], dtype=np.int32)
    valid_tokens = 0
    
    for token_id in token_ids:
        if 0 <= token_id < weights_int8.shape[0]:
            embedding += weights_int8[token_id].astype(np.int32)
            valid_tokens += 1
    
    # Mean pooling
    if valid_tokens > 0:
        embedding = embedding / valid_tokens
    
    # Dequantize to float
    embedding_float = embedding.astype(np.float32) / scale
    
    # Convert to 0-255 range (assuming original is roughly [-1, 1])
    embedding_uint8 = ((embedding_float + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    return embedding_uint8

def cosine_similarity_int8(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between INT8 vectors"""
    # Center around 0
    a_centered = a.astype(np.int16) - 128
    b_centered = b.astype(np.int16) - 128
    
    # Compute similarity
    dot_product = np.dot(a_centered, b_centered)
    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def main():
    print("=" * 80)
    print("🐍 PYTHON INT8 QUANTIZATION TEST")
    print("=" * 80)
    
    # Find model path
    model_path = "../../model/real_model.safetensors"
    if not os.path.exists(model_path):
        model_path = "model/real_model.safetensors"
    
    print(f"\n📁 Loading model from: {model_path}")
    
    # Load weights
    print("🔄 Loading Float32 weights...")
    weights_f32 = load_safetensors(model_path)
    vocab_size, embed_dim = weights_f32.shape
    print(f" Loaded weights: shape={weights_f32.shape}, dtype={weights_f32.dtype}")
    
    # Quantize to INT8
    print("\n Quantizing to INT8...")
    weights_int8, scale, zero_point = quantize_to_int8(weights_f32)
    print(f"  Scale: {scale:.6f}")
    print(f"  Zero point: {zero_point}")
    print(f"  INT8 range: [{weights_int8.min()}, {weights_int8.max()}]")
    
    # Test with sample token IDs
    test_cases = [
        ("Hello world", [7592, 2088]),  # Example token IDs
        ("Machine learning", [3698, 2143]),
        ("Python", [7145]),
    ]
    
    print("\n Computing embeddings:")
    embeddings_f32 = []
    embeddings_int8 = []
    
    for text, token_ids in test_cases:
        # Float32 embedding
        emb_f32 = np.zeros(embed_dim)
        for tid in token_ids:
            if 0 <= tid < vocab_size:
                emb_f32 += weights_f32[tid]
        emb_f32 /= len(token_ids)  # Mean pooling
        embeddings_f32.append(emb_f32)
        
        # INT8 embedding
        emb_int8 = compute_embedding_int8(weights_int8, token_ids, scale, zero_point)
        embeddings_int8.append(emb_int8)
        
        print(f"  {text}: F32 mean={emb_f32.mean():.3f}, INT8 mean={emb_int8.mean():.1f}")
    
    # Compare similarities
    print("\n📐 Similarity comparison:")
    for i in range(len(test_cases)):
        for j in range(i + 1, len(test_cases)):
            # Float32 similarity
            sim_f32 = np.dot(embeddings_f32[i], embeddings_f32[j]) / (
                np.linalg.norm(embeddings_f32[i]) * np.linalg.norm(embeddings_f32[j])
            )
            
            # INT8 similarity
            sim_int8 = cosine_similarity_int8(embeddings_int8[i], embeddings_int8[j])
            
            print(f"  '{test_cases[i][0]}' vs '{test_cases[j][0]}':")
            print(f"    Float32: {sim_f32:.4f}")
            print(f"    INT8:    {sim_int8:.4f}")
            print(f"    Diff:    {abs(sim_f32 - sim_int8):.4f}")
    
    # Memory comparison
    print("\n Memory usage:")
    f32_size = vocab_size * embed_dim * 4 / (1024 * 1024)
    int8_size = vocab_size * embed_dim * 1 / (1024 * 1024)
    print(f"  Float32: {f32_size:.2f} MB")
    print(f"  INT8:    {int8_size:.2f} MB")
    print(f"  Reduction: {(1 - int8_size/f32_size)*100:.1f}%")
    
    # Quantization error analysis
    print("\n Quantization error analysis:")
    
    # Sample some weights for comparison
    sample_idx = np.random.choice(vocab_size, 100)
    for idx in sample_idx[:5]:  # Show first 5
        original = weights_f32[idx, :10]  # First 10 dims
        quantized = weights_int8[idx, :10]
        dequantized = dequantize_int8(quantized, scale, zero_point)
        error = np.abs(original - dequantized).mean()
        
        if idx == sample_idx[0]:
            print(f"  Token {idx} (first 3 values):")
            for k in range(3):
                print(f"    Original: {original[k]:8.4f}")
                print(f"    INT8:     {quantized[k]:4d}")
                print(f"    Dequant:  {dequantized[k]:8.4f}")
                print(f"    Error:    {abs(original[k] - dequantized[k]):8.4f}")
    
    # Overall statistics
    total_weights = vocab_size * embed_dim
    print(f"\n Statistics:")
    print(f"  Total weights: {total_weights:,}")
    print(f"  Compression ratio: 4x")
    print(f"  Theoretical speedup: 2-4x (with SIMD)")
    
    print("\n Python INT8 test completed!")
    print(" Results can be compared with Go implementation")

if __name__ == "__main__":
    main()