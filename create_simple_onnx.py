#!/usr/bin/env python3
"""
Create a simple ONNX embedding model for demonstration
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids):
        # Simple embedding model
        embedded = self.embedding(input_ids)
        # Global average pooling
        pooled = embedded.mean(dim=1)
        # Linear transformation
        output = self.linear(pooled)
        return output

def create_simple_model():
    # Create output directory
    output_dir = Path("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple model
    model = SimpleEmbeddingModel(vocab_size=10000, embed_dim=1024)
    
    # Create some dummy training to make embeddings more meaningful
    # Initialize embeddings with some structure
    with torch.no_grad():
        # Make "hi" and "bonjour" similar
        hi_embedding = torch.randn(1024) * 0.1
        bonjour_embedding = hi_embedding + torch.randn(1024) * 0.05  # Similar to hi
        business_embedding = torch.randn(1024) * 0.1  # Different
        
        # Set specific embeddings
        model.embedding.weight[1] = hi_embedding
        model.embedding.weight[2] = bonjour_embedding
        model.embedding.weight[3] = business_embedding
        model.embedding.weight[4] = business_embedding + torch.randn(1024) * 0.05
        model.embedding.weight[5] = business_embedding + torch.randn(1024) * 0.05
    
    # Create vocabulary
    vocab = {
        "[PAD]": 0,
        "hi": 1,
        "bonjour": 2,
        "actionable": 3,
        "business": 4,
        "insights": 5,
        "hello": 6,
        "world": 7,
        "machine": 8,
        "learning": 9,
    }
    
    # Add more vocabulary entries
    for i in range(10, 10000):
        vocab[f"token_{i}"] = i
    
    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Create dummy input for ONNX export
    dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # batch_size=1, seq_len=5
    
    # Export to ONNX
    onnx_path = output_dir / "embedding_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
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
    
    print(f"ONNX model saved to: {onnx_path}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_inputs = [
            torch.tensor([[1]], dtype=torch.long),  # "hi"
            torch.tensor([[2]], dtype=torch.long),  # "bonjour"
            torch.tensor([[3, 4, 5]], dtype=torch.long),  # "actionable business insights"
        ]
        
        embeddings = []
        for test_input in test_inputs:
            emb = model(test_input)
            embeddings.append(emb.numpy())
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        emb_array = np.vstack(embeddings)
        similarities = cosine_similarity(emb_array)
        
        print("\nTest Results:")
        print(f"'hi' vs 'bonjour': {similarities[0][1]:.4f}")
        print(f"'hi' vs 'actionable business insights': {similarities[0][2]:.4f}")
        print(f"'bonjour' vs 'actionable business insights': {similarities[1][2]:.4f}")
        
        # Save test embeddings
        np.save(str(output_dir / "test_embeddings.npy"), emb_array)
        print(f"Test embeddings saved to: {output_dir / 'test_embeddings.npy'}")

if __name__ == "__main__":
    create_simple_model()