#!/usr/bin/env python3
"""
Test native PyTorch model directly (no ONNX conversion).
"""

from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def test_native_pytorch():
    """Test the native PyTorch SentenceTransformer model."""
    print("=== NATIVE PYTORCH MODEL TEST ===")
    print("Loading SentenceTransformer model...")
    
    # Load the model
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.", 
        "The weather is nice today.",
        "Python is a programming language.",
        "Hello world"
    ]
    
    print("\nGenerating embeddings with native PyTorch...")
    embeddings = model.encode(sentences)
    
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings[0].shape}")
    
    # Show first 5 values of each embedding
    print("\nNative PyTorch Results:")
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        print(f"  {i+1}. '{sentence}' -> [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, {embedding[3]:.3f}, {embedding[4]:.3f}]")
    
    # Calculate similarity matrix
    print("\nNative PyTorch Similarity Matrix:")
    print("      S1    S2    S3    S4    S5  ")
    for i, emb1 in enumerate(embeddings):
        row = f"S{i+1}  "
        for j, emb2 in enumerate(embeddings):
            sim = cosine_similarity(emb1, emb2)
            row += f"{sim:5.3f} "
        print(row)
    
    return embeddings

if __name__ == "__main__":
    native_embeddings = test_native_pytorch()
    print("\nNative PyTorch test completed!")
