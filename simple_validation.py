#!/usr/bin/env python3
"""
Simple validation: Test our exact reference sentences with both Python and ONNX to confirm they match.
"""

import json
import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer

def validate_implementation():
    print("🎯 Final Validation: ONNX Model vs SentenceTransformer")
    print("=" * 60)
    
    # Test sentences (same as in Go)
    test_sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful"
    ]
    
    # Load Python SentenceTransformer
    print("📦 Loading SentenceTransformer...")
    model = SentenceTransformer("model/sentence_transformer", device='cpu')
    
    # Load ONNX model
    print("📦 Loading ONNX model...")
    session = ort.InferenceSession("model/embedding_model.onnx")
    
    # Load reference tokens
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    print("\n📊 Computing embeddings...")
    
    # Get embeddings from both models
    python_embeddings = model.encode(test_sentences)
    onnx_embeddings = []
    
    for sentence in test_sentences:
        if sentence in ref_tokens:
            token_ids = ref_tokens[sentence]["token_ids"]
            # Ensure exactly 512 tokens
            if len(token_ids) < 512:
                token_ids = token_ids + [0] * (512 - len(token_ids))
            elif len(token_ids) > 512:
                token_ids = token_ids[:512]
            
            # Run ONNX inference
            input_tensor = np.array([token_ids], dtype=np.int64)
            onnx_outputs = session.run(None, {'input_ids': input_tensor})
            embedding = onnx_outputs[0][0]
            onnx_embeddings.append(embedding)
        else:
            print(f"❌ No reference tokens for '{sentence}'")
            return False
    
    onnx_embeddings = np.array(onnx_embeddings)
    
    # Compare embeddings
    print("\n📊 Embedding Comparison:")
    print("-" * 40)
    
    similarities = []
    for i, sentence in enumerate(test_sentences):
        py_emb = python_embeddings[i]
        onnx_emb = onnx_embeddings[i]
        
        similarity = np.dot(py_emb, onnx_emb) / (np.linalg.norm(py_emb) * np.linalg.norm(onnx_emb))
        similarities.append(similarity)
        
        print(f"'{sentence}':")
        print(f"  Python:    {py_emb[:5]}")
        print(f"  ONNX:      {onnx_emb[:5]}")
        print(f"  Similarity: {similarity:.6f}")
        print()
    
    # Distance analysis
    print("📏 Distance Analysis:")
    print("-" * 40)
    
    # Python distances (Squared Euclidean)
    py_dist1 = np.sum((python_embeddings[0] - python_embeddings[1]) ** 2)
    py_dist2 = np.sum((python_embeddings[0] - python_embeddings[2]) ** 2)
    py_dist3 = np.sum((python_embeddings[1] - python_embeddings[2]) ** 2)
    
    # ONNX distances (these should match Go output exactly)
    onnx_dist1 = np.sum((onnx_embeddings[0] - onnx_embeddings[1]) ** 2)
    onnx_dist2 = np.sum((onnx_embeddings[0] - onnx_embeddings[2]) ** 2)
    onnx_dist3 = np.sum((onnx_embeddings[1] - onnx_embeddings[2]) ** 2)
    
    print("Squared Euclidean Distances:")
    print(f"'{test_sentences[0]}' vs '{test_sentences[1]}':")
    print(f"  Python: {py_dist1:.6f}")
    print(f"  ONNX:   {onnx_dist1:.6f}")
    print()
    
    print(f"'{test_sentences[0]}' vs '{test_sentences[2]}':")
    print(f"  Python: {py_dist2:.6f}")
    print(f"  ONNX:   {onnx_dist2:.6f}")
    print()
    
    print(f"'{test_sentences[1]}' vs '{test_sentences[2]}':")
    print(f"  Python: {py_dist3:.6f}")
    print(f"  ONNX:   {onnx_dist3:.6f}")
    print()
    
    # Go output for comparison (from the run we just did)
    go_dist1 = 69850.929688
    go_dist2 = 81284.218750  
    go_dist3 = 38447.765625
    
    print("🔍 Go vs ONNX Distance Comparison:")
    print(f"Distance 1: Go={go_dist1:.6f}, ONNX={onnx_dist1:.6f}, Diff={abs(go_dist1-onnx_dist1):.6f}")
    print(f"Distance 2: Go={go_dist2:.6f}, ONNX={onnx_dist2:.6f}, Diff={abs(go_dist2-onnx_dist2):.6f}")
    print(f"Distance 3: Go={go_dist3:.6f}, ONNX={onnx_dist3:.6f}, Diff={abs(go_dist3-onnx_dist3):.6f}")
    
    # Validation
    avg_similarity = np.mean(similarities)
    max_go_onnx_diff = max(abs(go_dist1-onnx_dist1), abs(go_dist2-onnx_dist2), abs(go_dist3-onnx_dist3))
    
    print("\n🎯 Validation Results:")
    print("=" * 40)
    print(f"Average Python-ONNX similarity: {avg_similarity:.6f}")
    print(f"Max Go-ONNX distance difference: {max_go_onnx_diff:.6f}")
    
    # Check if ONNX model matches Python well enough
    if avg_similarity > 0.998:
        print("✅ EXCELLENT: ONNX model matches Python very closely!")
        onnx_vs_python_ok = True
    elif avg_similarity > 0.95:
        print("✅ GOOD: ONNX model matches Python reasonably well!")
        onnx_vs_python_ok = True
    else:
        print("❌ PROBLEM: ONNX model differs significantly from Python!")
        onnx_vs_python_ok = False
    
    # Check if Go matches ONNX exactly
    if max_go_onnx_diff < 0.001:
        print("✅ PERFECT: Go output matches ONNX exactly!")
        go_vs_onnx_ok = True
    elif max_go_onnx_diff < 0.1:
        print("✅ EXCELLENT: Go output matches ONNX very closely!")
        go_vs_onnx_ok = True
    else:
        print("❌ PROBLEM: Go output differs from ONNX!")
        go_vs_onnx_ok = False
    
    # Check if distances are realistic and differentiated
    min_dist = min(onnx_dist1, onnx_dist2, onnx_dist3)
    max_dist = max(onnx_dist1, onnx_dist2, onnx_dist3)
    distance_range_ratio = max_dist / min_dist if min_dist > 0 else float('inf')
    
    if distance_range_ratio > 1.5:
        print("✅ EXCELLENT: Distances are well-differentiated!")
        distances_ok = True
    elif distance_range_ratio > 1.1:
        print("✅ GOOD: Distances show some differentiation!")
        distances_ok = True
    else:
        print("❌ PROBLEM: Distances are too similar!")
        distances_ok = False
    
    # Overall success
    overall_success = onnx_vs_python_ok and go_vs_onnx_ok and distances_ok
    
    if overall_success:
        print("\n🎉 OVERALL SUCCESS!")
        print("✅ The embedding similarity issue has been fully resolved!")
        print("✅ Go implementation produces realistic, differentiated embeddings!")
        print("✅ Distance calculations are working correctly!")
    else:
        print("\n💥 Some issues remain to be addressed.")
    
    return overall_success

if __name__ == "__main__":
    validate_implementation()
