#!/usr/bin/env python3
"""
Inspect the actual model architecture to understand the computation.
"""

import torch
from sentence_transformers import SentenceTransformer

def main():
    print("🔍 Inspecting model architecture...")
    
    # Load the model
    model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
    model = SentenceTransformer(model_path)
    
    print(f"Model: {type(model)}")
    print(f"Model modules: {len(model)}")
    
    for i, module in enumerate(model):
        print(f"\nModule {i}: {type(module)}")
        print(f"  Class: {module.__class__.__name__}")
        
        # Check if it's a Transformer module
        if hasattr(module, 'auto_model'):
            print(f"  Has auto_model: {type(module.auto_model)}")
            transformer = module.auto_model
            
            # Look for embeddings
            if hasattr(transformer, 'embeddings'):
                embeddings = transformer.embeddings
                print(f"  Embeddings: {type(embeddings)}")
                
                if hasattr(embeddings, 'word_embeddings'):
                    word_embeddings = embeddings.word_embeddings
                    print(f"  Word embeddings shape: {word_embeddings.weight.shape}")
                    print(f"  Word embeddings type: {type(word_embeddings)}")
                    
                    # Test manual computation
                    sentence = "This is a test sentence."
                    inputs = model.tokenize([sentence])
                    token_ids = inputs['input_ids'].tolist()
                    
                    print(f"\n🧪 Manual computation test:")
                    print(f"  Token IDs: {token_ids}")
                    
                    # Get embeddings for tokens
                    token_tensor = torch.tensor([token_ids])
                    token_embeddings = word_embeddings(token_tensor)
                    print(f"  Raw token embeddings shape: {token_embeddings.shape}")
                    
                    # Mean pool (skip special tokens if needed)
                    mean_pooled = torch.mean(token_embeddings, dim=1)
                    print(f"  Mean pooled shape: {mean_pooled.shape}")
                    print(f"  Mean pooled sample: [{mean_pooled[0][0]:.3f}, {mean_pooled[0][1]:.3f}, {mean_pooled[0][2]:.3f}, {mean_pooled[0][3]:.3f}, {mean_pooled[0][4]:.3f}]")
                    
                    # L2 normalize
                    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                    print(f"  Normalized sample: [{normalized[0][0]:.3f}, {normalized[0][1]:.3f}, {normalized[0][2]:.3f}, {normalized[0][3]:.3f}, {normalized[0][4]:.3f}]")
                    
                    # Compare with full model
                    full_output = model.encode([sentence])
                    print(f"  Full model sample: [{full_output[0][0]:.3f}, {full_output[0][1]:.3f}, {full_output[0][2]:.3f}, {full_output[0][3]:.3f}, {full_output[0][4]:.3f}]")
                    
                    # Check match
                    import numpy as np
                    diff = np.abs(normalized[0][:5].detach().numpy() - full_output[0][:5])
                    max_diff = np.max(diff)
                    print(f"  Max difference: {max_diff:.6f}")
                    
                    if max_diff < 0.01:
                        print("  ✅ Manual computation matches!")
                    else:
                        print("  ❌ Manual computation differs - additional processing needed")
        
        # Check if it's a Pooling module
        if hasattr(module, 'pooling_mode_mean_tokens'):
            print(f"  Pooling module found")
            print(f"  Mean pooling: {module.pooling_mode_mean_tokens}")
            print(f"  Max pooling: {getattr(module, 'pooling_mode_max_tokens', False)}")
            print(f"  CLS pooling: {getattr(module, 'pooling_mode_cls_token', False)}")

if __name__ == "__main__":
    main()