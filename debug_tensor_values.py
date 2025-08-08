#!/usr/bin/env python3
"""
Debug by checking what the actual tensor values look like vs expected.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def load_safetensors_weights():
    """Load the same weights our Go code is using."""
    from safetensors import safe_open
    
    with safe_open("./model/real_model.safetensors", framework="pt", device="cpu") as f:
        embedding_weight = f.get_tensor("embedding.weight")
        return embedding_weight

def main():
    print("🔍 Debugging tensor values...")
    
    # Load the model
    model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
    model = SentenceTransformer(model_path)
    
    # Load our safetensors weights directly
    weights = load_safetensors_weights()
    print(f"Safetensors weights shape: {weights.shape}")
    print(f"Safetensors dtype: {weights.dtype}")
    
    # Test sentence
    sentence = "This is a test sentence."
    inputs = model.tokenize([sentence])
    token_ids = inputs['input_ids'].tolist()
    print(f"Token IDs: {token_ids}")
    
    # Manual computation using the exact same weights
    print(f"\n🧪 Manual computation:")
    
    # Get embeddings for each token
    token_embeddings = []
    for token_id in token_ids:
        emb = weights[token_id]
        token_embeddings.append(emb)
        print(f"  Token {token_id}: first 5 values = [{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, {emb[3]:.3f}, {emb[4]:.3f}]")
    
    # Stack and mean pool
    stacked = torch.stack(token_embeddings)
    mean_pooled = torch.mean(stacked, dim=0)
    print(f"  Mean pooled: [{mean_pooled[0]:.3f}, {mean_pooled[1]:.3f}, {mean_pooled[2]:.3f}, {mean_pooled[3]:.3f}, {mean_pooled[4]:.3f}]")
    
    # L2 normalize
    normalized = torch.nn.functional.normalize(mean_pooled.unsqueeze(0), p=2, dim=1)
    print(f"  Normalized: [{normalized[0][0]:.3f}, {normalized[0][1]:.3f}, {normalized[0][2]:.3f}, {normalized[0][3]:.3f}, {normalized[0][4]:.3f}]")
    
    # Compare with model output
    full_output = model.encode([sentence])
    print(f"  Model output: [{full_output[0][0]:.3f}, {full_output[0][1]:.3f}, {full_output[0][2]:.3f}, {full_output[0][3]:.3f}, {full_output[0][4]:.3f}]")
    
    # Check if they match
    diff = np.abs(normalized[0][:5].detach().numpy() - full_output[0][:5])
    max_diff = np.max(diff)
    print(f"  Max difference: {max_diff:.6f}")
    
    if max_diff < 0.01:
        print("  ✅ Manual computation matches!")
    else:
        print("  ❌ Manual computation differs")
    
    print(f"\nGo implementation should:")
    print(f"  1. Load weights as [vocab_size, embed_dim] = {weights.shape}")
    print(f"  2. For tokens {token_ids}")
    print(f"  3. Mean pool -> [{mean_pooled[0]:.6f}, {mean_pooled[1]:.6f}, {mean_pooled[2]:.6f}, {mean_pooled[3]:.6f}, {mean_pooled[4]:.6f}]")
    print(f"  4. L2 normalize -> [{normalized[0][0]:.6f}, {normalized[0][1]:.6f}, {normalized[0][2]:.6f}, {normalized[0][3]:.6f}, {normalized[0][4]:.6f}]")

if __name__ == "__main__":
    main()