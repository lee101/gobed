#!/usr/bin/env python3
"""
Test the same model in Python to compare results with our Go implementation
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort

def test_sentence_transformer():
    print("=" * 60)
    print("PYTHON SENTENCE TRANSFORMER TEST")
    print("=" * 60)
    
    # Load the same model we converted
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    
    # Test with the same texts
    test_texts = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful"
    ]
    
    # Generate embeddings
    embeddings = model.encode(test_texts)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    
    for i, text in enumerate(test_texts):
        print(f"\nText: '{text}'")
        print(f"  First 5 values: {embeddings[i][:5]}")
        print(f"  Stats: mean={embeddings[i].mean():.6f}, std={embeddings[i].std():.6f}")
        print(f"  Range: [{embeddings[i].min():.6f}, {embeddings[i].max():.6f}]")
    
    # Calculate similarities
    print(f"\nSimilarity Results (Python SentenceTransformer):")
    print("=" * 50)
    
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"'{test_texts[i]}' vs '{test_texts[j]}': {similarity:.8f}")

def test_onnx_model():
    print("\n" + "=" * 60)
    print("PYTHON ONNX MODEL TEST")
    print("=" * 60)
    
    # Test the ONNX model directly
    session = ort.InferenceSession("model/embedding_model.onnx")
    
    # Print input/output info
    print(f"ONNX Model inputs: {[input.name for input in session.get_inputs()]}")
    print(f"ONNX Model outputs: {[output.name for output in session.get_outputs()]}")
    
    for inp in session.get_inputs():
        print(f"  Input '{inp.name}': {inp.type}, shape: {inp.shape}")
    for out in session.get_outputs():
        print(f"  Output '{out.name}': {out.type}, shape: {out.shape}")
    
    # Test with simple tokenized input
    test_inputs = [
        # CLS + "hello" + "world" + SEP + padding
        np.array([[101, 7592, 2088, 102] + [0] * 508], dtype=np.int64),
        # CLS + "weather" + "nice" + SEP + padding  
        np.array([[101, 4633, 3835, 102] + [0] * 508], dtype=np.int64),
        # CLS + "machine" + "learning" + SEP + padding
        np.array([[101, 3698, 4083, 102] + [0] * 508], dtype=np.int64),
    ]
    
    embeddings = []
    for i, input_ids in enumerate(test_inputs):
        result = session.run(None, {"input_ids": input_ids})
        embedding = result[0][0]  # Get first batch, first sequence
        embeddings.append(embedding)
        
        print(f"\nONNX Input {i+1}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {embedding.shape}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Stats: mean={embedding.mean():.6f}, std={embedding.std():.6f}")
        print(f"  Range: [{embedding.min():.6f}, {embedding.max():.6f}]")
    
    # Calculate similarities
    print(f"\nSimilarity Results (ONNX Direct):")
    print("=" * 40)
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"Input {i+1} vs Input {j+1}: {similarity:.8f}")

if __name__ == "__main__":
    test_sentence_transformer()
    test_onnx_model()
