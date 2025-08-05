#!/usr/bin/env python3
"""
Check what our ONNX model actually outputs vs SentenceTransformer
"""

import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def test_onnx_vs_sentence_transformer():
    print("🔍 Comparing ONNX model vs SentenceTransformer...")
    
    # Load both models
    print("📥 Loading models...")
    sentence_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    onnx_session = ort.InferenceSession("model/embedding_model.onnx")
    
    # Check ONNX model info
    print(f"\n🔧 ONNX Model Info:")
    for inp in onnx_session.get_inputs():
        print(f"  Input: {inp.name}, type: {inp.type}, shape: {inp.shape}")
    for out in onnx_session.get_outputs():
        print(f"  Output: {out.name}, type: {out.type}, shape: {out.shape}")
    
    # Test with simple text
    test_text = "hello world"
    print(f"\n📝 Testing with: '{test_text}'")
    
    # Get SentenceTransformer result
    sentence_embedding = sentence_model.encode([test_text])[0]
    print(f"\n✅ SentenceTransformer result:")
    print(f"  Shape: {sentence_embedding.shape}")
    print(f"  First 10: {sentence_embedding[:10]}")
    print(f"  Range: [{sentence_embedding.min():.3f}, {sentence_embedding.max():.3f}]")
    print(f"  Mean: {sentence_embedding.mean():.6f}, Std: {sentence_embedding.std():.6f}")
    
    # Test ONNX with simple input (what our Go code does)
    # CLS + hello + world + SEP + padding
    simple_input = np.array([[101, 7592, 2088, 102] + [0] * 508], dtype=np.int64)
    
    onnx_result = onnx_session.run(None, {"input_ids": simple_input})
    onnx_embedding = onnx_result[0][0]  # First output, first sequence
    
    print(f"\n🤖 ONNX result (simple tokenization):")
    print(f"  Input shape: {simple_input.shape}")
    print(f"  Output shape: {onnx_embedding.shape}")
    print(f"  First 10: {onnx_embedding[:10]}")
    print(f"  Range: [{onnx_embedding.min():.3f}, {onnx_embedding.max():.3f}]")
    print(f"  Mean: {onnx_embedding.mean():.6f}, Std: {onnx_embedding.std():.6f}")
    
    # Compare similarity to itself (should be 1.0)
    sentence_self_sim = cosine_similarity([sentence_embedding], [sentence_embedding])[0][0]
    onnx_self_sim = cosine_similarity([onnx_embedding], [onnx_embedding])[0][0]
    
    print(f"\n🎯 Self-similarity check:")
    print(f"  SentenceTransformer: {sentence_self_sim:.8f}")
    print(f"  ONNX: {onnx_self_sim:.8f}")
    
    # Cross-similarity (should be much lower if they're different)
    cross_sim = cosine_similarity([sentence_embedding], [onnx_embedding])[0][0]
    print(f"  Cross-similarity: {cross_sim:.8f}")
    
    if cross_sim < 0.5:
        print(f"\n❌ WARNING: Cross-similarity is very low ({cross_sim:.3f})")
        print(f"   This suggests our ONNX processing is completely different!")
    
    # Test multiple different texts with ONNX
    print(f"\n🧪 Testing ONNX with multiple texts...")
    test_inputs = [
        # Different simple token combinations
        np.array([[101, 7592, 2088, 102] + [0] * 508], dtype=np.int64),  # hello world
        np.array([[101, 4633, 102] + [0] * 509], dtype=np.int64),        # weather
        np.array([[101, 3698, 4083, 102] + [0] * 508], dtype=np.int64),  # machine learning
    ]
    
    onnx_embeddings = []
    for i, input_ids in enumerate(test_inputs):
        result = onnx_session.run(None, {"input_ids": input_ids})
        embedding = result[0][0]
        onnx_embeddings.append(embedding)
        print(f"  Text {i+1}: mean={embedding.mean():.6f}, std={embedding.std():.6f}")
    
    # Check ONNX similarities
    print(f"\n📊 ONNX Similarities:")
    for i in range(len(onnx_embeddings)):
        for j in range(i + 1, len(onnx_embeddings)):
            sim = cosine_similarity([onnx_embeddings[i]], [onnx_embeddings[j]])[0][0]
            print(f"  Text {i+1} vs Text {j+1}: {sim:.8f}")

if __name__ == "__main__":
    test_onnx_vs_sentence_transformer()
