#!/usr/bin/env python3
"""
Debug the StaticEmbedding model to understand how it works.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def debug_static_embedding():
    print("🔍 Debugging StaticEmbedding model")
    print("=" * 50)
    
    # Load the model
    model_path = "model/sentence_transformer"
    print(f"📂 Loading model from {model_path}")
    model = SentenceTransformer(model_path, device='cpu')
    
    # Get the StaticEmbedding module
    static_embedding = model[0]
    print(f"📝 Model type: {type(static_embedding)}")
    print(f"📝 Config: {static_embedding.__dict__}")
    
    # Test sentence
    test_sentence = "hello world"
    print(f"\n🧪 Testing with sentence: '{test_sentence}'")
    
    # Get SentenceTransformer output
    st_output = model.encode([test_sentence])
    print(f"📊 SentenceTransformer output: {st_output.shape}")
    print(f"📊 Sample: {st_output[0][:5]}")
    
    # Tokenize manually
    tokenizer = model.tokenizer
    encoded = tokenizer.encode(test_sentence, add_special_tokens=True)
    print(f"\n🔤 Tokenization:")
    print(f"  Token IDs: {encoded.ids}")
    print(f"  Tokens: {[tokenizer.decode([tid]) for tid in encoded.ids[:10]]}")
    
    # Prepare inputs for StaticEmbedding
    max_len = 512
    if len(encoded.ids) > max_len:
        token_ids = encoded.ids[:max_len]
    else:
        token_ids = encoded.ids + [0] * (max_len - len(encoded.ids))
    
    # Create features dict as StaticEmbedding expects
    features = {
        'input_ids': torch.tensor([token_ids], dtype=torch.long),
        'offsets': torch.tensor([0], dtype=torch.long)  # Start offset for EmbeddingBag
    }
    
    print(f"\n📏 Input features:")
    print(f"  input_ids shape: {features['input_ids'].shape}")
    print(f"  offsets shape: {features['offsets'].shape}")
    print(f"  Token sample: {features['input_ids'][0][:10]}")
    
    # Run through StaticEmbedding
    try:
        with torch.no_grad():
            static_output = static_embedding(features)
        
        print(f"\n📊 StaticEmbedding output: {static_output['sentence_embedding'].shape}")
        print(f"📊 Sample: {static_output['sentence_embedding'][0][:5]}")
        
        # Compare
        similarity = np.dot(static_output['sentence_embedding'][0].numpy(), st_output[0]) / (
            np.linalg.norm(static_output['sentence_embedding'][0].numpy()) * np.linalg.norm(st_output[0])
        )
        print(f"📊 StaticEmbedding vs SentenceTransformer similarity: {similarity:.6f}")
        
        if similarity > 0.95:
            print("✅ Direct StaticEmbedding matches SentenceTransformer!")
        else:
            print("❌ Direct StaticEmbedding doesn't match SentenceTransformer")
            
    except Exception as e:
        print(f"❌ StaticEmbedding failed: {e}")
        
    # Let's also try without offsets to see if that's the issue
    print(f"\n🔧 Trying without offsets...")
    try:
        features_no_offsets = {
            'input_ids': features['input_ids']
        }
        
        with torch.no_grad():
            static_output_no_offsets = static_embedding(features_no_offsets)
            
        print(f"📊 No offsets output: {static_output_no_offsets['sentence_embedding'].shape}")
        print(f"📊 Sample: {static_output_no_offsets['sentence_embedding'][0][:5]}")
        
    except Exception as e:
        print(f"❌ No offsets failed: {e}")
        
    # Check the embedding layer directly
    print(f"\n🔍 Embedding layer info:")
    embedding_layer = static_embedding.embedding
    print(f"  Type: {type(embedding_layer)}")
    print(f"  Num embeddings: {embedding_layer.num_embeddings}")
    print(f"  Embedding dim: {embedding_layer.embedding_dim}")
    print(f"  Mode: {embedding_layer.mode}")
    
    # Test direct embedding lookup
    test_tokens = torch.tensor([encoded.ids[:4]], dtype=torch.long)  # Just first few real tokens
    test_offsets = torch.tensor([0], dtype=torch.long)
    
    with torch.no_grad():
        direct_embed = embedding_layer(test_tokens.flatten(), test_offsets)
        print(f"  Direct embedding: {direct_embed.shape}")
        print(f"  Direct sample: {direct_embed[0][:5]}")

if __name__ == "__main__":
    debug_static_embedding()
