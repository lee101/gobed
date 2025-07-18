#!/usr/bin/env python3
"""
Simple export of the production static-retrieval-mrl-en-v1 model to ONNX.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import os

class ProductionStaticEmbedding(nn.Module):
    """
    ONNX-compatible wrapper for the production StaticEmbedding model.
    """
    
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer
        
    def forward(self, input_ids):
        """
        Forward pass: input_ids -> embeddings (mean pooled)
        
        The original StaticEmbedding uses EmbeddingBag with mode='mean'
        which automatically applies mean pooling.
        """
        # EmbeddingBag expects a 1D tensor of indices
        # For batch processing, we need to flatten and provide offsets
        batch_size, seq_len = input_ids.shape
        
        # Flatten input_ids
        flat_input_ids = input_ids.view(-1)
        
        # Create offsets for each sequence in the batch
        offsets = torch.arange(0, batch_size * seq_len, seq_len, dtype=torch.long)
        
        # Apply embedding bag (automatically does mean pooling)
        embeddings = self.embedding(flat_input_ids, offsets)
        
        return embeddings

def export_production_model_simple():
    print("Loading production model...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    model.cpu()  # Move to CPU for consistent processing
    
    # Get the embedding layer
    static_embedding = model._modules['0']
    embedding_layer = static_embedding.embedding
    
    print(f"Vocab size: {embedding_layer.num_embeddings}")
    print(f"Embedding dim: {embedding_layer.embedding_dim}")
    
    # Create ONNX wrapper
    onnx_model = ProductionStaticEmbedding(embedding_layer)
    onnx_model.eval()
    
    # Test with a simple example
    test_input = torch.tensor([[101, 2023, 2003, 1037, 3231, 6251, 102]])  # Example BERT tokens
    print(f"Test input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = onnx_model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (first 5 dims): {output[0][:5]}")
    
    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = "model/production_embedding_model.onnx"
    
    torch.onnx.export(
        onnx_model,
        test_input,
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
    
    # Save model info
    model_info = {
        "model_name": "sentence-transformers/static-retrieval-mrl-en-v1",
        "embedding_dimension": embedding_layer.embedding_dim,
        "vocab_size": embedding_layer.num_embeddings,
        "max_seq_length": 512,
        "model_type": "StaticEmbedding",
        "pooling": "mean",
        "normalization": "none"
    }
    
    with open("model/production_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("Model info saved to: model/production_model_info.json")
    print(f"Model info: {model_info}")
    
    # Copy tokenizer files from cached model
    tokenizer_src = "/home/lee/code/gobed/cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding"
    tokenizer_dst = "model/production_tokenizer"
    
    os.makedirs(tokenizer_dst, exist_ok=True)
    
    # Copy tokenizer.json
    import shutil
    shutil.copy(f"{tokenizer_src}/tokenizer.json", f"{tokenizer_dst}/tokenizer.json")
    print(f"Tokenizer copied to: {tokenizer_dst}")
    
    return model, onnx_model

if __name__ == "__main__":
    export_production_model_simple()
