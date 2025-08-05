#!/usr/bin/env python3
"""
Comprehensive comparison of all three embedding approaches:
1. Python PyTorch (with safetensors)
2. Python ONNX 
3. Go Safetensors (results will be compared externally)
"""

import json
import numpy as np
import torch
import onnxruntime as ort
from safetensors import safe_open

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def load_safetensors_weights(safetensors_path):
    """Load embedding weights from safetensors file."""
    tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def pytorch_forward(weights, token_ids):
    """PyTorch forward pass."""
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    
    embeddings = weights['embedding.weight'][token_ids]
    mask = token_ids != 0
    if mask.sum() == 0:
        return torch.zeros(embeddings.shape[-1])
    
    masked_embeddings = embeddings[mask]
    return masked_embeddings.mean(dim=0)

def onnx_forward(session, token_ids):
    """ONNX forward pass."""
    # Pad token_ids to a fixed length if needed
    max_len = 512
    if len(token_ids) < max_len:
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    elif len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    
    token_ids_array = np.array([token_ids], dtype=np.int64)
    outputs = session.run(["embeddings"], {"input_ids": token_ids_array})
    return outputs[0][0]

def main():
    print("Comprehensive Embedding Comparison")
    print("==================================")
    
    # Load reference tokens
    with open("model/production_reference_tokens.json", "r") as f:
        reference_tokens = json.load(f)
    
    # Load models
    print("Loading models...")
    
    # 1. PyTorch safetensors
    safetensors_path = "cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors"
    pytorch_weights = load_safetensors_weights(safetensors_path)
    print("✓ PyTorch safetensors loaded")
    
    # 2. ONNX model
    onnx_session = ort.InferenceSession("model/production_embedding_model.onnx")
    print("✓ ONNX model loaded")
    
    # Test sentences
    sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today.",
        "Python is a programming language.",
        "Hello world"
    ]
    
    pytorch_embeddings = []
    onnx_embeddings = []
    
    print("\nGenerating embeddings...")
    print("=" * 80)
    
    for sentence in sentences:
        if sentence not in reference_tokens:
            print(f"Warning: No tokens for '{sentence}'")
            continue
            
        token_ids = reference_tokens[sentence]["token_ids"]
        
        # PyTorch
        pytorch_emb = pytorch_forward(pytorch_weights, token_ids).numpy()
        pytorch_embeddings.append(pytorch_emb)
        
        # ONNX
        onnx_emb = onnx_forward(onnx_session, token_ids)
        onnx_embeddings.append(onnx_emb)
        
        # Print first 5 values for comparison
        print(f"'{sentence}':")
        print(f"  PyTorch: [{pytorch_emb[0]:.3f}, {pytorch_emb[1]:.3f}, {pytorch_emb[2]:.3f}, {pytorch_emb[3]:.3f}, {pytorch_emb[4]:.3f}]")
        print(f"  ONNX:    [{onnx_emb[0]:.3f}, {onnx_emb[1]:.3f}, {onnx_emb[2]:.3f}, {onnx_emb[3]:.3f}, {onnx_emb[4]:.3f}]")
        
        # Calculate difference
        diff = np.abs(pytorch_emb - onnx_emb).max()
        print(f"  Max diff: {diff:.6f}")
        print()
    
    # Similarity matrices
    print("\nSimilarity Matrices")
    print("=" * 50)
    
    print("\nPyTorch Similarity Matrix:")
    print("      S1    S2    S3    S4    S5  ")
    for i, emb1 in enumerate(pytorch_embeddings):
        row = f"S{i+1}  "
        for emb2 in pytorch_embeddings:
            sim = cosine_similarity(emb1, emb2)
            row += f"{sim:5.3f} "
        print(row)
    
    print("\nONNX Similarity Matrix:")
    print("      S1    S2    S3    S4    S5  ")
    for i, emb1 in enumerate(onnx_embeddings):
        row = f"S{i+1}  "
        for emb2 in onnx_embeddings:
            sim = cosine_similarity(emb1, emb2)
            row += f"{sim:5.3f} "
        print(row)
    
    # Overall consistency check
    print("\nConsistency Analysis")
    print("=" * 30)
    
    total_diff = 0.0
    max_diff = 0.0
    
    for i, (pytorch_emb, onnx_emb) in enumerate(zip(pytorch_embeddings, onnx_embeddings)):
        diff = np.abs(pytorch_emb - onnx_emb).mean()
        max_diff_sentence = np.abs(pytorch_emb - onnx_emb).max()
        total_diff += diff
        max_diff = max(max_diff, max_diff_sentence)
        print(f"Sentence {i+1}: Mean diff = {diff:.6f}, Max diff = {max_diff_sentence:.6f}")
    
    avg_diff = total_diff / len(pytorch_embeddings)
    print(f"\nOverall: Average diff = {avg_diff:.6f}, Max diff = {max_diff:.6f}")
    
    if avg_diff < 0.001:
        print("✅ EXCELLENT: PyTorch and ONNX are highly consistent!")
    elif avg_diff < 0.01:
        print("✅ GOOD: PyTorch and ONNX are reasonably consistent.")
    else:
        print("⚠️  WARNING: Significant differences detected.")
    
    print("\nNow run the Go safetensors implementation:")
    print("go run safetensors_loader.go")
    print("\nThe Go implementation should produce identical results to PyTorch!")

if __name__ == "__main__":
    main()