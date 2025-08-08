#!/usr/bin/env python3
"""
Production Python comparison using the exact same model for verification.
This ensures we get identical results between Go and Python implementations.
"""

import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json
import os

class ProductionPythonModel:
    def __init__(self, model_name="sentence-transformers/static-retrieval-mrl-en-v1"):
        """Load model once during initialization (separated from inference)."""
        print(f"🔄 Loading Python model: {model_name}")
        load_start = time.time()
        
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("🚀 Model loaded on GPU")
        else:
            print("💻 Model loaded on CPU")
        
        # Load reference tokens for exact comparison
        self.reference_tokens = self.load_reference_tokens("model/production_reference_tokens.json")
        
        load_time = time.time() - load_start
        print(f"✅ Model loaded successfully in {load_time:.3f}s")
        print(f"📦 Reference tokens: {len(self.reference_tokens)} sentences")
        
        # Warmup
        print("🔥 Warming up model...")
        self.warmup()
        print("✅ Warmup completed")
    
    def load_reference_tokens(self, tokens_path):
        """Load reference tokens for exact comparison with Go."""
        if not os.path.exists(tokens_path):
            print(f"⚠️  Reference tokens not found: {tokens_path}")
            return {}
            
        with open(tokens_path, 'r') as f:
            reference_tokens = json.load(f)
        
        print(f"✅ Loaded reference tokens for {len(reference_tokens)} sentences")
        return reference_tokens
    
    def warmup(self):
        """Warmup the model with a few inference calls."""
        if self.reference_tokens:
            sample_text = list(self.reference_tokens.keys())[0]
            for _ in range(5):
                _ = self.encode_text(sample_text)
    
    def encode_text(self, text):
        """Encode a single text - this is what we benchmark."""
        # Use sentence-transformers directly for the exact model behavior
        embedding = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return embedding[0]
    
    def batch_encode_texts(self, texts):
        """Encode multiple texts efficiently."""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings

def benchmark_pure_inference(model):
    """Benchmark pure inference time, excluding model loading."""
    print("\n" + "="*70)
    print("🚀 PRODUCTION PYTHON INFERENCE BENCHMARK")
    print("="*70)
    
    # Get test sentences from reference tokens
    if not model.reference_tokens:
        print("❌ No reference tokens available for benchmarking")
        return
    
    sentences = list(model.reference_tokens.keys())[:10]  # First 10 sentences
    
    print(f"Benchmarking {len(sentences)} sentences with production model...")
    
    # Individual inference benchmarks
    print("\n⏱️  Pure inference benchmarks:")
    times = []
    embeddings = []
    
    for i, sentence in enumerate(sentences):
        # Time ONLY the inference call
        start = time.time()
        embedding = model.encode_text(sentence)
        elapsed = time.time() - start
        
        times.append(elapsed)
        embeddings.append(embedding)
        
        # Display with truncated sentence
        display_sentence = sentence
        if len(display_sentence) > 35:
            display_sentence = display_sentence[:32] + "..."
        
        print(f"   S{i+1:2d}: {elapsed*1000000:8.2f}μs - \"{display_sentence}\"")
    
    # Calculate performance statistics
    total_time = sum(times)
    avg_time = np.mean(times)
    throughput = len(sentences) / total_time
    
    print(f"\n📊 Performance Summary:")
    print(f"   Total inference time: {total_time:.6f}s")
    print(f"   Average per inference: {avg_time:.6f}s")
    print(f"   Throughput: {throughput:.0f} inferences/sec")
    print(f"   Latency: {avg_time*1000000:.2f}μs per inference")
    print(f"   Range: {min(times)*1000000:.2f}μs - {max(times)*1000000:.2f}μs")
    
    # Test batch processing
    print("\n📦 Testing batch processing...")
    batch_start = time.time()
    batch_embeddings = model.batch_encode_texts(sentences)
    batch_time = time.time() - batch_start
    
    batch_throughput = len(sentences) / batch_time
    print(f"   Batch time: {batch_time:.6f}s ({batch_throughput:.0f} texts/sec)")
    print(f"   Batch avg: {batch_time*1000000/len(sentences):.2f}μs per text")
    
    # Accuracy verification
    print("\n🎯 Accuracy verification:")
    expected = [3.483, -2.513, 3.576, -0.724, 1.369]
    
    # Find "This is a test sentence." if available
    test_sentence_idx = -1
    for i, sentence in enumerate(sentences):
        if sentence == "This is a test sentence.":
            test_sentence_idx = i
            break
    
    if test_sentence_idx >= 0:
        embedding = embeddings[test_sentence_idx]
        max_diff = max(abs(embedding[i] - expected[i]) for i in range(5))
        
        print(f"   Expected: [{expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f}, {expected[3]:.3f}, {expected[4]:.3f}]")
        print(f"   Actual:   [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, {embedding[3]:.3f}, {embedding[4]:.3f}]")
        print(f"   Max diff: {max_diff:.6f}")
        
        if max_diff < 0.001:
            print("   ✅ PERFECT MATCH!")
        elif max_diff < 0.01:
            print("   ✅ EXCELLENT MATCH!")
        else:
            print("   ⚠️  Moderate match")
    
    # Similarity test
    if len(embeddings) >= 2:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]
        print(f"   Sample similarity (S1 vs S2): {sim:.4f}")
    
    return embeddings, times

def save_embeddings_for_comparison(embeddings, sentences):
    """Save embeddings for comparison with Go implementation."""
    print(f"\n💾 Saving embeddings for Go comparison...")
    
    # Save embeddings
    np.save("python_production_embeddings.npy", np.array(embeddings))
    
    # Save sentences
    with open("python_production_sentences.txt", "w") as f:
        for sentence in sentences:
            f.write(f"{sentence}\n")
    
    print("✅ Python embeddings saved for comparison")

def main():
    print("================================================================================")
    print("🚀 PRODUCTION PYTHON EMBEDDING - INFERENCE BENCHMARK")
    print("================================================================================")
    print("Model: sentence-transformers/static-retrieval-mrl-en-v1")
    print("Purpose: Exact comparison with Go implementation")
    print("")
    
    # Load model (one-time cost)
    model = ProductionPythonModel()
    
    # Benchmark pure inference
    embeddings, times = benchmark_pure_inference(model)
    
    # Save for comparison with Go
    if model.reference_tokens:
        sentences = list(model.reference_tokens.keys())[:10]
        save_embeddings_for_comparison(embeddings, sentences)
    
    print("\n" + "="*80)
    print("✅ Python production benchmark completed!")
    print("🎯 Ready for Go vs Python comparison")
    print("="*80)

if __name__ == "__main__":
    main()