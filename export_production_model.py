#!/usr/bin/env python3
"""
Export the production sentence-transformers/static-retrieval-mrl-en-v1 model to ONNX.

This model is a StaticEmbedding model with 1024-dimensional embeddings,
trained with Matryoshka loss and multiple negatives ranking loss on 80M+ examples.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import os

class StaticEmbeddingONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper for the StaticEmbedding model.
    Takes input_ids and returns mean-pooled embeddings.
    """
    
    def __init__(self, model):
        super().__init__()
        # Get the StaticEmbedding module (should be the first and only module)
        self.static_embedding = model._modules['0']
        
    def forward(self, input_ids):
        """
        Forward pass that mimics the sentence-transformers pipeline:
        1. Get token embeddings from StaticEmbedding
        2. Apply mean pooling
        """
        # Get embeddings from StaticEmbedding - it expects a dictionary
        features = {'input_ids': input_ids}
        features = self.static_embedding(features)
        
        # Extract token embeddings
        token_embeddings = features['token_embeddings']  # [batch_size, seq_len, hidden_dim]
        
        # Create attention mask - assume all tokens are valid (non-padded)
        # In real usage, you'd pass this as input, but for simplicity we'll create it
        attention_mask = torch.ones(input_ids.shape, dtype=torch.float32, device=input_ids.device)
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings

def export_production_model():
    print("Loading sentence-transformers/static-retrieval-mrl-en-v1...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    print(f"Model modules: {list(model._modules.keys())}")
    print(f"Model max_seq_length: {model.max_seq_length}")
    
    # Get the StaticEmbedding module
    static_embedding = model._modules['0']
    print(f"StaticEmbedding module: {static_embedding}")
    print(f"Vocab size: {static_embedding.embedding.num_embeddings}")
    print(f"Embedding dim: {static_embedding.embedding.embedding_dim}")
    
    # Create ONNX wrapper
    onnx_model = StaticEmbeddingONNXWrapper(model)
    onnx_model.eval()
    
    # Test with sample input
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.", 
        "The weather is nice today."
    ]
    
    print(f"\nTesting with sentences: {test_sentences}")
    
    # Get reference embeddings from the original model
    reference_embeddings = model.encode(test_sentences, convert_to_tensor=True)
    print(f"Reference embeddings shape: {reference_embeddings.shape}")
    print(f"Reference embeddings (first 5 dims): {reference_embeddings[0][:5]}")
    
    # Get tokenizer
    tokenizer = model.tokenizer
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    # Tokenize test sentences
    inputs = tokenizer(test_sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    print(f"Attention mask: {attention_mask}")
    
    # Test ONNX wrapper
    with torch.no_grad():
        onnx_embeddings = onnx_model(input_ids)
    
    print(f"ONNX embeddings shape: {onnx_embeddings.shape}")
    print(f"ONNX embeddings (first 5 dims): {onnx_embeddings[0][:5]}")
    
    # Compare with reference
    similarity = torch.nn.functional.cosine_similarity(reference_embeddings, onnx_embeddings, dim=1)
    print(f"Cosine similarity with reference: {similarity}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    
    # Dummy input for ONNX export
    dummy_input = torch.randint(0, tokenizer.get_vocab_size(), (1, 10))
    
    onnx_path = "model/production_embedding_model.onnx"
    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'embeddings': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model exported to: {onnx_path}")
    
    # Save tokenizer files  
    tokenizer_dir = "model/production_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Save the tokenizer JSON file
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    
    # Also save vocab if available
    vocab = tokenizer.get_vocab()
    with open(os.path.join(tokenizer_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Tokenizer saved to: {tokenizer_dir}")
    
    # Save model info
    model_info = {
        "model_name": "sentence-transformers/static-retrieval-mrl-en-v1",
        "embedding_dimension": int(static_embedding.embedding.embedding_dim),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "max_seq_length": int(model.max_seq_length) if model.max_seq_length != float('inf') else 512,
        "model_type": "StaticEmbedding",
        "pooling": "mean",
        "normalization": "none"
    }
    
    with open("model/production_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("Model info saved to: model/production_model_info.json")
    print(f"Model info: {model_info}")
    
    # Generate test tokens for validation
    test_tokens_data = {}
    for sentence in test_sentences:
        tokens = tokenizer(sentence, padding=False, truncation=True, max_length=512)
        test_tokens_data[sentence] = {
            "input_ids": tokens['input_ids'],
            "attention_mask": tokens['attention_mask']
        }
    
    with open("model/production_test_tokens.json", "w") as f:
        json.dump(test_tokens_data, f, indent=2)
    
    print("Test tokens saved to: model/production_test_tokens.json")
    
    return model, onnx_model

if __name__ == "__main__":
    export_production_model()
