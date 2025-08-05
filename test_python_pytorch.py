#!/usr/bin/env python3
"""
Test PyTorch embedding with the actual model weights from safetensors.
"""

import json
import numpy as np
import torch
from safetensors import safe_open

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def load_model_weights(safetensors_path):
    """Load embedding weights from safetensors file."""
    tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            print(f"Loaded tensor '{key}': {tensors[key].shape} {tensors[key].dtype}")
    return tensors

def forward_embedding(weights, token_ids):
    """Perform forward pass through embedding layer with mean pooling."""
    # Convert to tensor if needed
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    
    # Get embeddings for all tokens  
    embeddings = weights['embedding.weight'][token_ids]  # [seq_len, embed_dim]
    
    # Mean pooling (excluding padding tokens if token_id == 0)
    mask = token_ids != 0
    if mask.sum() == 0:  # All padding tokens
        return torch.zeros(embeddings.shape[-1])
    
    masked_embeddings = embeddings[mask]
    mean_embedding = masked_embeddings.mean(dim=0)
    
    return mean_embedding

def main():
    print("Python PyTorch Embedding Test")
    print("=============================")
    
    # Load the safetensors model
    safetensors_path = "cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors"
    weights = load_model_weights(safetensors_path)
    
    # Load reference tokens
    with open("model/production_reference_tokens.json", "r") as f:
        reference_tokens = json.load(f)
    
    # Test sentences
    sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today.",
        "Python is a programming language.",
        "Hello world"
    ]
    
    embeddings = []
    
    print("\nGenerating embeddings...")
    for sentence in sentences:
        if sentence in reference_tokens:
            token_ids = reference_tokens[sentence]["token_ids"]
            embedding = forward_embedding(weights, token_ids)
            embeddings.append(embedding.numpy())
            
            print(f"'{sentence}' -> [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, {embedding[3]:.3f}, {embedding[4]:.3f}]")
        else:
            print(f"Warning: No tokens found for '{sentence}'")
            embeddings.append(None)
    
    # Calculate similarity matrix
    print("\nPython PyTorch Similarity Matrix:")
    print("      S1    S2    S3    S4    S5  ")
    for i, emb1 in enumerate(embeddings):
        if emb1 is None:
            continue
        row = f"S{i+1}  "
        for emb2 in embeddings:
            if emb2 is None:
                row += "  --- "
                continue
            sim = cosine_similarity(emb1, emb2)
            row += f"{sim:5.3f} "
        print(row)
    
    print("\nPython PyTorch embedding test completed!")
    print("This uses the actual model weights from safetensors.")

if __name__ == "__main__":
    main()