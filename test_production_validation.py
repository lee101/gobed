#!/usr/bin/env python3
"""
Simple validation of the production ONNX model.
"""

import torch
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import numpy as np

def simple_validation():
    print("Loading models...")
    
    # Load original model
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    model.cpu()
    
    # Load ONNX session
    session = ort.InferenceSession("model/production_embedding_model.onnx")
    
    # Test sentence
    test_sentence = "This is a test sentence."
    print(f"Test sentence: '{test_sentence}'")
    
    # Get reference embedding
    ref_embedding = model.encode([test_sentence], convert_to_tensor=True)
    print(f"Reference embedding shape: {ref_embedding.shape}")
    print(f"Reference embedding (first 5): {ref_embedding[0][:5]}")
    
    # Tokenize for ONNX
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(test_sentence)
    print(f"Tokens: {tokens}")
    
    # Prepare ONNX input
    input_ids = np.array([tokens], dtype=np.int64)
    print(f"ONNX input shape: {input_ids.shape}")
    
    # Run ONNX inference
    onnx_output = session.run(['embeddings'], {'input_ids': input_ids})
    onnx_embedding = torch.from_numpy(onnx_output[0])
    print(f"ONNX embedding shape: {onnx_embedding.shape}")
    print(f"ONNX embedding (first 5): {onnx_embedding[0][:5]}")
    
    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_embedding, onnx_embedding, dim=1
    )
    print(f"Cosine similarity: {cos_sim.item():.6f}")
    
    if cos_sim.item() > 0.99:
        print("✓ SUCCESS: Models match well!")
    else:
        print("✗ ISSUE: Models don't match well")

if __name__ == "__main__":
    simple_validation()
