#!/usr/bin/env python3
"""
Python baseline implementation using sentence-transformers for comparison with Go implementation.
This script uses the real multilingual-e5-base model for accurate embeddings.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    """Load the multilingual E5 model."""
    print("Loading multilingual-e5-base model...")
    start = time.time()
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.3f}s")
    return model

def encode_texts(model, texts):
    """Encode texts and measure performance."""
    print(f"\nEncoding {len(texts)} texts...")
    
    # Add E5 prefix as required
    prefixed_texts = [f"query: {text}" for text in texts]
    
    # Warmup
    print("Warmup run...")
    start = time.time()
    _ = model.encode([prefixed_texts[0]])
    warmup_time = time.time() - start
    print(f"Warmup completed in {warmup_time:.3f}s")
    
    # Actual encoding
    print("Encoding all texts...")
    start = time.time()
    embeddings = model.encode(prefixed_texts)
    total_time = time.time() - start
    
    avg_time = total_time / len(texts)
    throughput = len(texts) / total_time
    
    print(f"Total encoding time: {total_time:.3f}s")
    print(f"Average time per text: {avg_time:.3f}s")
    print(f"Throughput: {throughput:.2f} embeddings/sec")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings

def calculate_similarities(embeddings, texts):
    """Calculate cosine similarities between embeddings."""
    print("\nCalculating similarities...")
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)
    
    # Print specific comparisons
    sim_hi_bonjour = similarities[0, 1]
    sim_hi_business = similarities[0, 2]  
    sim_bonjour_business = similarities[1, 2]
    
    print(f"'hi' vs 'bonjour': {sim_hi_bonjour:.4f}")
    print(f"'hi' vs 'actionable business insights': {sim_hi_business:.4f}")
    print(f"'bonjour' vs 'actionable business insights': {sim_bonjour_business:.4f}")
    
    # Check if greetings are more similar to each other
    if sim_hi_bonjour > sim_hi_business and sim_hi_bonjour > sim_bonjour_business:
        print("✓ SUCCESS: 'hi' and 'bonjour' are closer to each other than to 'actionable business insights'")
    else:
        print("✗ Unexpected similarity pattern")
    
    return similarities

def benchmark_performance(model):
    """Run comprehensive performance benchmark."""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    benchmark_texts = [
        "hello world",
        "machine learning is fascinating", 
        "artificial intelligence and deep learning",
        "natural language processing",
        "computer vision and image recognition",
        "data science and analytics",
        "software engineering best practices",
        "distributed systems architecture",
        "cloud computing and microservices",
        "performance optimization techniques"
    ]
    
    prefixed_texts = [f"query: {text}" for text in benchmark_texts]
    
    print(f"Benchmarking with {len(benchmark_texts)} different texts...")
    
    # Individual timing
    times = []
    embeddings = []
    
    for i, text in enumerate(prefixed_texts):
        start = time.time()
        embedding = model.encode([text])
        elapsed = time.time() - start
        times.append(elapsed)
        embeddings.append(embedding[0])
        
        print(f"   Embedding {i+1:2d}: {elapsed*1000:6.2f}ms (dim: {len(embedding[0])}, text: \"{benchmark_texts[i][:30]}...\")")
    
    total_time = sum(times)
    avg_time = np.mean(times)
    
    print(f"\nResults Summary:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average time per embedding: {avg_time:.3f}s")
    print(f"   Embeddings per second: {len(benchmark_texts)/total_time:.2f}")
    print(f"   Throughput: {1.0/avg_time:.2f} embeddings/sec")
    
    return np.array(embeddings)

def save_embeddings_for_comparison(embeddings, texts, filename="python_embeddings.npy"):
    """Save embeddings for comparison with Go implementation."""
    print(f"\nSaving embeddings to {filename}")
    np.save(filename, embeddings)
    
    # Also save text labels
    with open(filename.replace('.npy', '_texts.txt'), 'w') as f:
        for text in texts:
            f.write(f"{text}\n")

def main():
    print("Python Multilingual E5 Embedding Comparison")
    print("=" * 50)
    
    # Load model
    model = load_model()
    
    # Test texts (same as Go implementation)
    test_texts = [
        "hi",
        "bonjour", 
        "actionable business insights"
    ]
    
    # Generate embeddings
    embeddings = encode_texts(model, test_texts)
    
    # Calculate similarities  
    similarities = calculate_similarities(embeddings, test_texts)
    
    # Save for comparison
    save_embeddings_for_comparison(embeddings, test_texts)
    
    # Run performance benchmark
    benchmark_embeddings = benchmark_performance(model)
    
    print(f"\nPython implementation completed successfully!")
    print(f"Embeddings saved for comparison with Go implementation.")

if __name__ == "__main__":
    main()