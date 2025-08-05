#!/usr/bin/env python3
"""
Comprehensive validation comparing Python and Go embeddings.
"""

import json
import numpy as np
import onnxruntime as ort

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def load_reference_tokens():
    """Load reference tokens from JSON file."""
    with open("model/production_reference_tokens.json", "r") as f:
        return json.load(f)

def test_python_onnx():
    """Test Python ONNX inference."""
    print("Testing Python ONNX inference...")
    
    # Load model
    session = ort.InferenceSession("model/production_embedding_model.onnx")
    
    # Load reference tokens
    reference_tokens = load_reference_tokens()
    
    sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.", 
        "The weather is nice today.",
        "Python is a programming language.",
        "Hello world"
    ]
    
    embeddings = []
    for sentence in sentences:
        if sentence in reference_tokens:
            token_ids = reference_tokens[sentence]["token_ids"]
            token_ids_array = np.array([token_ids], dtype=np.int64)
            
            # Run inference
            outputs = session.run(["embeddings"], {"input_ids": token_ids_array})
            embedding = outputs[0][0]  # Extract the embedding vector
            embeddings.append(embedding)
            
            print(f"'{sentence}' -> [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, {embedding[3]:.3f}, {embedding[4]:.3f}]")
    
    # Calculate similarity matrix
    print("\nPython ONNX Similarity Matrix:")
    print("      S1    S2    S3    S4    S5  ")
    for i, emb1 in enumerate(embeddings):
        row = f"S{i+1}  "
        for j, emb2 in enumerate(embeddings):
            sim = cosine_similarity(emb1, emb2)
            row += f"{sim:5.3f} "
        print(row)
    
    return embeddings

if __name__ == "__main__":
    python_embeddings = test_python_onnx()
    print("\nPython ONNX test completed!")
