#!/usr/bin/env python3
"""
Simple test to see what actual similarities should be from SentenceTransformer
"""

import time
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("🚀 Testing SentenceTransformer similarities...")
    
    # Load the same model
    print("📥 Loading model...")
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    
    # Test with the same texts as our Go app
    test_texts = [
        "hello world",
        "the weather is nice today",
        "machine learning algorithms are powerful"
    ]
    
    print(f"📝 Testing {len(test_texts)} texts:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. '{text}'")
    
    # Generate embeddings
    print("\n🧠 Generating embeddings...")
    start = time.time()
    embeddings = model.encode(test_texts)
    elapsed = time.time() - start
    
    print(f"✅ Generated embeddings in {elapsed:.3f}s")
    print(f"📊 Embedding shape: {embeddings.shape}")
    print(f"📊 Embedding dtype: {embeddings.dtype}")
    
    # Show embedding stats
    print(f"\n📈 Embedding Statistics:")
    for i, text in enumerate(test_texts):
        emb = embeddings[i]
        print(f"  '{text}':")
        print(f"    First 5: [{emb[0]:.6f}, {emb[1]:.6f}, {emb[2]:.6f}, {emb[3]:.6f}, {emb[4]:.6f}]")
        print(f"    Mean: {emb.mean():.6f}, Std: {emb.std():.6f}")
        print(f"    Range: [{emb.min():.6f}, {emb.max():.6f}]")
    
    # Calculate similarities
    print(f"\n🎯 Similarity Results:")
    print("=" * 50)
    
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"'{test_texts[i]}' vs '{test_texts[j]}': {sim:.8f}")
    
    # Also test some very different texts to see range
    print(f"\n🔬 Testing with very different texts:")
    diverse_texts = [
        "cat",
        "dog", 
        "mathematical equation",
        "financial markets crashed",
        "I love pizza"
    ]
    
    diverse_embeddings = model.encode(diverse_texts)
    
    for i in range(len(diverse_texts)):
        for j in range(i + 1, len(diverse_texts)):
            sim = cosine_similarity([diverse_embeddings[i]], [diverse_embeddings[j]])[0][0]
            print(f"'{diverse_texts[i]}' vs '{diverse_texts[j]}': {sim:.8f}")

if __name__ == "__main__":
    main()
