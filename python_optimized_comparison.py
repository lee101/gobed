#!/usr/bin/env python3
"""
Optimized Python comparison focusing only on pure inference time.
This creates a fair comparison by separating model loading from inference timing.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class OptimizedE5Model:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        """Load model once during initialization."""
        print(f"Loading {model_name}...")
        load_start = time.time()
        
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available for fair comparison
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
        
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.3f}s")
        
        # Warmup the model
        print("Warming up model...")
        self._warmup()
        print("Warmup completed")
    
    def _warmup(self):
        """Warmup the model with a few inference calls."""
        warmup_texts = ["query: hello", "query: world", "query: test"]
        for _ in range(3):
            _ = self.model.encode(warmup_texts, show_progress_bar=False)
    
    def encode_single(self, text):
        """Encode a single text - this is what we benchmark."""
        # Add E5 prefix
        prefixed_text = f"query: {text}"
        
        # Single inference call
        embedding = self.model.encode([prefixed_text], show_progress_bar=False)
        return embedding[0]
    
    def encode_batch(self, texts):
        """Encode multiple texts efficiently."""
        prefixed_texts = [f"query: {text}" for text in texts]
        embeddings = self.model.encode(prefixed_texts, show_progress_bar=False)
        return embeddings

def benchmark_inference(model):
    """Benchmark pure inference time, excluding model loading."""
    print("\n" + "="*60)
    print("OPTIMIZED PYTHON INFERENCE BENCHMARK")
    print("="*60)
    
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
    
    print(f"Benchmarking {len(benchmark_texts)} texts with optimized inference...")
    
    # Benchmark individual inference times
    print("\nPure inference benchmarks:")
    times = []
    embeddings = []
    
    for i, text in enumerate(benchmark_texts):
        # Time only the inference call
        start = time.time()
        embedding = model.encode_single(text)
        elapsed = time.time() - start
        
        times.append(elapsed)
        embeddings.append(embedding)
        
        print(f"   Text {i+1:2d}: {elapsed*1000:8.2f}ms - \"{text[:40]}\"")
    
    # Calculate statistics
    total_time = sum(times)
    avg_time = np.mean(times)
    throughput = len(benchmark_texts) / total_time
    
    print(f"\nPerformance Summary:")
    print(f"   Total inference time: {total_time:.3f}s")
    print(f"   Average per inference: {avg_time:.3f}s")
    print(f"   Throughput: {throughput:.2f} inferences/sec")
    print(f"   Latency: {avg_time*1000:.2f}ms per inference")
    print(f"   Range: {min(times)*1000:.2f}ms - {max(times)*1000:.2f}ms")
    
    return times, embeddings

def benchmark_batch_inference(model):
    """Compare batch vs individual inference."""
    print("\n" + "="*60) 
    print("BATCH VS INDIVIDUAL INFERENCE")
    print("="*60)
    
    test_texts = [
        "hello world",
        "machine learning is fascinating", 
        "artificial intelligence and deep learning",
        "natural language processing",
        "computer vision and image recognition"
    ]
    
    # Individual inference
    print("Individual inference timing:")
    individual_start = time.time()
    individual_embeddings = []
    for text in test_texts:
        embedding = model.encode_single(text)
        individual_embeddings.append(embedding)
    individual_time = time.time() - individual_start
    
    # Batch inference
    print("Batch inference timing:")
    batch_start = time.time()
    batch_embeddings = model.encode_batch(test_texts)
    batch_time = time.time() - batch_start
    
    print(f"\nResults:")
    print(f"   Individual: {individual_time:.3f}s ({individual_time/len(test_texts)*1000:.2f}ms per text)")
    print(f"   Batch: {batch_time:.3f}s ({batch_time/len(test_texts)*1000:.2f}ms per text)")
    print(f"   Speedup: {individual_time/batch_time:.2f}x")
    
    return individual_time, batch_time

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_semantic_similarity(model):
    """Test semantic similarity with optimized inference."""
    print("\n" + "="*40)
    print("SEMANTIC SIMILARITY TEST")
    print("="*40)
    
    test_texts = [
        "hi",
        "bonjour", 
        "actionable business insights"
    ]
    
    print("Generating embeddings...")
    embeddings = []
    times = []
    
    for text in test_texts:
        start = time.time()
        embedding = model.encode_single(text)
        elapsed = time.time() - start
        
        embeddings.append(embedding)
        times.append(elapsed)
        print(f"'{text}': {elapsed*1000:.2f}ms (dim: {len(embedding)})")
    
    # Calculate similarities
    print("\nSimilarity Results:")
    sim1 = cosine_similarity(embeddings[0], embeddings[1])
    sim2 = cosine_similarity(embeddings[0], embeddings[2])
    sim3 = cosine_similarity(embeddings[1], embeddings[2])
    
    print(f"'{test_texts[0]}' vs '{test_texts[1]}': {sim1:.4f}")
    print(f"'{test_texts[0]}' vs '{test_texts[2]}': {sim2:.4f}")
    print(f"'{test_texts[1]}' vs '{test_texts[2]}': {sim3:.4f}")
    
    if sim1 > sim2 and sim1 > sim3:
        print("✓ SUCCESS: Greetings are more similar to each other")
    else:
        print("⚠ Greetings similarity pattern not as expected")
    
    return embeddings, times

def main():
    print("Optimized Python E5 Embedding Comparison")
    print("=" * 50)
    
    # Load model (one-time cost, like Go implementation)
    model = OptimizedE5Model()
    
    # Test semantic similarity
    embeddings, similarity_times = test_semantic_similarity(model)
    
    # Benchmark pure inference 
    inference_times, benchmark_embeddings = benchmark_inference(model)
    
    # Compare batch vs individual
    individual_time, batch_time = benchmark_batch_inference(model)
    
    print(f"\n🎉 Python optimized inference completed!")
    print(f"Average inference time: {np.mean(inference_times)*1000:.2f}ms")

if __name__ == "__main__":
    main()