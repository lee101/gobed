#!/usr/bin/env python3
"""
Inspect the StaticEmbedding module to understand its structure better.
"""

from sentence_transformers import SentenceTransformer
import torch

def inspect_static_embedding():
    print("Loading sentence-transformers/static-retrieval-mrl-en-v1...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    # Get the StaticEmbedding module
    static_embedding = model._modules['0']
    print(f"StaticEmbedding module: {static_embedding}")
    print(f"StaticEmbedding attributes: {dir(static_embedding)}")
    
    # Check the embedding layer
    embedding_layer = static_embedding.embedding
    print(f"Embedding layer: {embedding_layer}")
    print(f"Embedding layer attributes: {dir(embedding_layer)}")
    print(f"Embedding num_embeddings: {embedding_layer.num_embeddings}")
    print(f"Embedding embedding_dim: {embedding_layer.embedding_dim}")
    print(f"Embedding mode: {embedding_layer.mode}")
    
    # Check tokenizer
    tokenizer = model.tokenizer
    print(f"Tokenizer: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Tokenizer dir: {dir(tokenizer)}")
    
    # Test forward pass
    test_sentence = "This is a test sentence."
    print(f"\nTesting with: '{test_sentence}'")
    
    # Tokenize
    tokens = tokenizer(test_sentence, return_tensors='pt')
    print(f"Input IDs: {tokens['input_ids']}")
    print(f"Attention mask: {tokens['attention_mask']}")
    
    # Run through StaticEmbedding
    features = {'input_ids': tokens['input_ids']}
    output_features = static_embedding(features)
    print(f"StaticEmbedding output keys: {list(output_features.keys())}")
    for key, value in output_features.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Compare with full model
    full_embedding = model.encode([test_sentence], convert_to_tensor=True)
    print(f"Full model embedding shape: {full_embedding.shape}")
    print(f"Full model embedding (first 5): {full_embedding[0][:5]}")

if __name__ == "__main__":
    inspect_static_embedding()
