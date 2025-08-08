#!/usr/bin/env python3
"""
Debug the real model to understand the exact computation pipeline.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("🔍 Debugging real model computation pipeline...")
    
    # Load the model
    model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
    model = SentenceTransformer(model_path)
    
    sentence = "This is a test sentence."
    print(f"Testing: '{sentence}'")
    
    # Step 1: Tokenize
    inputs = model.tokenize([sentence])
    token_ids = inputs['input_ids'].tolist()
    print(f"Token IDs: {token_ids}")
    
    # Step 2: Get the embedding layer
    print(f"\nModel components:")
    for i, module in enumerate(model.modules()):
        module_name = module.__class__.__name__
        if hasattr(module, 'word_embeddings'):
            print(f"  {i}: {module_name} with word_embeddings")
            embeddings_layer = module.word_embeddings
            vocab_size, embed_dim = embeddings_layer.weight.shape
            print(f"     Embedding shape: [{vocab_size}, {embed_dim}]")
            
            # Get raw embeddings for our tokens
            token_tensor = torch.tensor([token_ids])
            raw_embeddings = embeddings_layer(token_tensor)
            print(f"     Raw embeddings shape: {raw_embeddings.shape}")
            
            # Mean pooling
            mean_pooled = torch.mean(raw_embeddings, dim=1)
            print(f"     Mean pooled shape: {mean_pooled.shape}")
            print(f"     Mean pooled sample: [{mean_pooled[0][0]:.3f}, {mean_pooled[0][1]:.3f}, {mean_pooled[0][2]:.3f}, {mean_pooled[0][3]:.3f}, {mean_pooled[0][4]:.3f}]")
            
            # L2 normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            print(f"     Normalized sample: [{normalized[0][0]:.3f}, {normalized[0][1]:.3f}, {normalized[0][2]:.3f}, {normalized[0][3]:.3f}, {normalized[0][4]:.3f}]")
            
            break
    
    # Step 3: Full model output for comparison
    full_output = model.encode([sentence])
    print(f"\nFull model output sample: [{full_output[0][0]:.3f}, {full_output[0][1]:.3f}, {full_output[0][2]:.3f}, {full_output[0][3]:.3f}, {full_output[0][4]:.3f}]")
    
    # Check if they match
    if 'normalized' in locals():
        diff = np.abs(normalized[0][:5].detach().numpy() - full_output[0][:5])
        max_diff = np.max(diff)
        print(f"Max difference: {max_diff:.6f}")
        
        if max_diff < 0.001:
            print("✅ Manual computation matches sentence-transformers!")
        else:
            print("❌ Manual computation differs from sentence-transformers")
            print("   This suggests there are additional processing steps")

if __name__ == "__main__":
    main()