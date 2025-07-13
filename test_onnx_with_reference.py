#!/usr/bin/env python3
"""
Test the current ONNX model with our exact test sentences to validate it's working.
"""

import json
import numpy as np
import onnxruntime as ort

def test_onnx_model():
    print("🧪 Testing ONNX Model with Reference Tokens")
    print("=" * 50)
    
    # Load reference tokens
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    # Load ONNX model
    session = ort.InferenceSession("model/embedding_model.onnx")
    
    test_sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful"
    ]
    
    print("📊 ONNX Model Results:")
    embeddings = []
    
    for sentence in test_sentences:
        if sentence in ref_tokens:
            token_ids = ref_tokens[sentence]["token_ids"]
            # Ensure we pad to 512 tokens
            if len(token_ids) < 512:
                token_ids = token_ids + [0] * (512 - len(token_ids))
            elif len(token_ids) > 512:
                token_ids = token_ids[:512]
            
            # Run ONNX inference
            input_tensor = np.array([token_ids], dtype=np.int64)
            onnx_outputs = session.run(None, {'input_ids': input_tensor})
            embedding = onnx_outputs[0][0]
            embeddings.append(embedding)
            
            print(f"'{sentence}':")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Sample values: {embedding[:5]}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
        else:
            print(f"❌ No reference tokens for '{sentence}'")
    
    # Calculate distances and similarities
    print("\n📏 Distance and Similarity Analysis:")
    
    # Squared Euclidean distances
    dist1 = np.sum((embeddings[0] - embeddings[1]) ** 2)
    dist2 = np.sum((embeddings[0] - embeddings[2]) ** 2) 
    dist3 = np.sum((embeddings[1] - embeddings[2]) ** 2)
    
    print(f"Squared Euclidean Distances:")
    print(f"  '{test_sentences[0]}' vs '{test_sentences[1]}': {dist1:.6f}")
    print(f"  '{test_sentences[0]}' vs '{test_sentences[2]}': {dist2:.6f}")
    print(f"  '{test_sentences[1]}' vs '{test_sentences[2]}': {dist3:.6f}")
    
    # Cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim1 = cosine_similarity(embeddings[0], embeddings[1])
    sim2 = cosine_similarity(embeddings[0], embeddings[2])
    sim3 = cosine_similarity(embeddings[1], embeddings[2])
    
    print(f"\nCosine Similarities:")
    print(f"  '{test_sentences[0]}' vs '{test_sentences[1]}': {sim1:.8f}")
    print(f"  '{test_sentences[0]}' vs '{test_sentences[2]}': {sim2:.8f}")
    print(f"  '{test_sentences[1]}' vs '{test_sentences[2]}': {sim3:.8f}")
    
    # Check if results are realistic (embeddings should be different)
    if dist1 != dist2 and dist1 != dist3 and dist2 != dist3:
        print("\n✅ SUCCESS: Embeddings are differentiated (not identical)")
    else:
        print("\n❌ PROBLEM: Some embeddings are identical")
    
    # Check if distances have reasonable range
    min_dist = min(dist1, dist2, dist3)
    max_dist = max(dist1, dist2, dist3)
    
    if max_dist / min_dist > 1.1:  # At least 10% difference
        print("✅ SUCCESS: Distance values have reasonable range")
    else:
        print("❌ PROBLEM: Distance values are too similar")
    
    return embeddings

if __name__ == "__main__":
    test_onnx_model()
