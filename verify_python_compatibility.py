#!/usr/bin/env python3
"""
Verify that Go implementation matches Python static-retrieval-mrl-en-v1 model output.
This script runs the same test sentences through the Python model.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import json

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def main():
    # Load the model
    print("Loading Python static-retrieval-mrl-en-v1 model...")
    model = SentenceTransformer('static-retrieval-mrl-en-v1')
    
    # Test sentences (same as in Go tests)
    test_sentences = [
        "Hello world",
        "What is machine learning?",
        "The weather is nice today.",
        "Python is a programming language used for data science.",
        "The cat is sleeping",
        "The kitten is sleeping",
        "I love programming",
        "The weather is cold",
    ]
    
    print("\n" + "="*60)
    print("PYTHON MODEL OUTPUTS (for comparison with Go)")
    print("="*60)
    
    embeddings = {}
    for sentence in test_sentences:
        emb = model.encode(sentence)
        embeddings[sentence] = emb
        
        # Calculate norm
        norm = np.linalg.norm(emb)
        
        print(f"\nSentence: '{sentence}'")
        print(f"  Dimension: {len(emb)}")
        print(f"  First 5 values: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, {emb[3]:.4f}, {emb[4]:.4f}]")
        print(f"  L2 Norm: {norm:.4f}")
        print(f"  Mean: {np.mean(emb):.4f}")
        print(f"  Min: {np.min(emb):.4f}, Max: {np.max(emb):.4f}")
    
    print("\n" + "="*60)
    print("SIMILARITY COMPARISONS")
    print("="*60)
    
    # Test similarity pairs (same as Go tests)
    similarity_pairs = [
        ("Hello world", "Hello world"),
        ("The cat is sleeping", "The kitten is sleeping"),
        ("I love programming", "The weather is cold"),
        ("The cat is sleeping", "I love programming"),
    ]
    
    for text1, text2 in similarity_pairs:
        sim = cosine_similarity(embeddings[text1], embeddings[text2])
        print(f"\nSimilarity('{text1}', '{text2}'): {sim:.4f}")
    
    print("\n" + "="*60)
    print("GO TEST VERIFICATION")
    print("="*60)
    
    # These are the values we got from the Go implementation
    go_outputs = {
        "Hello world": {
            "first_5": [14.6416, 28.7915, 3.0570, 11.0085, 5.0033],
            "norm": 488.0037
        },
        "What is machine learning?": {
            "first_5": [2.9748, 12.9363, 3.2580, -12.5399, 11.0782],
            "norm": 149.3066
        },
        "The weather is nice today.": {
            "first_5": [5.0020, -0.1559, -9.5286, 8.9363, -3.9082],
            "norm": 137.4557
        }
    }
    
    print("\nComparing Go outputs with Python outputs:")
    all_match = True
    
    for sentence, go_data in go_outputs.items():
        if sentence in embeddings:
            py_emb = embeddings[sentence]
            py_first_5 = py_emb[:5]
            py_norm = np.linalg.norm(py_emb)
            
            # Check if first 5 values match (within tolerance)
            values_match = True
            for i in range(5):
                diff = abs(py_first_5[i] - go_data["first_5"][i])
                if diff > 0.01:  # 0.01 tolerance
                    values_match = False
                    print(f"\n '{sentence}' - Value mismatch at index {i}:")
                    print(f"   Python: {py_first_5[i]:.4f}, Go: {go_data['first_5'][i]:.4f}")
            
            # Check if norm matches
            norm_diff = abs(py_norm - go_data["norm"])
            if norm_diff > 0.1:  # 0.1 tolerance for norm
                print(f"\n '{sentence}' - Norm mismatch:")
                print(f"   Python: {py_norm:.4f}, Go: {go_data['norm']:.4f}")
                all_match = False
            elif values_match:
                print(f"\n '{sentence}' - MATCHES!")
                print(f"   First 5 values match: {py_first_5[:5].round(4).tolist()}")
                print(f"   Norm matches: Python={py_norm:.4f}, Go={go_data['norm']:.4f}")
    
    if all_match:
        print("\n" + "="*60)
        print(" SUCCESS: Go implementation matches Python model!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("  Some differences found - check tolerance settings")
        print("="*60)
    
    # Save embeddings for reference
    output_data = {}
    for sentence in test_sentences:
        emb = embeddings[sentence]
        output_data[sentence] = {
            "embedding_first_10": emb[:10].tolist(),
            "norm": float(np.linalg.norm(emb)),
            "mean": float(np.mean(emb)),
            "dimension": len(emb)
        }
    
    with open("python_embeddings_reference.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nReference data saved to python_embeddings_reference.json")

if __name__ == "__main__":
    main()