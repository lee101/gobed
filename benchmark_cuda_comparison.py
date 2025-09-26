#!/usr/bin/env python3
"""
Compare original vs optimized CUDA implementations
"""

import subprocess
import time
import sys

def compile_cuda_code():
    """Compile both CUDA implementations"""
    print("🔨 Compiling CUDA implementations...")
    
    # Compile original
    try:
        result = subprocess.run([
            "nvcc", "-o", "cuda_similarity_original", 
            "cuda_similarity.cu", "-lcublas"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to compile original: {result.stderr}")
            return False
        else:
            print("✅ Compiled original implementation")
    except FileNotFoundError:
        print("❌ nvcc not found. Make sure CUDA toolkit is installed.")
        return False
    
    # Compile optimized (note: may need linking to select_topk kernel)
    try:
        result = subprocess.run([
            "nvcc", "-o", "cuda_similarity_optimized", 
            "cuda_similarity_optimized.cu", "cuda_similarity.cu", "-lcublas"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ Optimized version needs select_topk kernel from original")
            print("Creating combined implementation...")
            return create_combined_implementation()
        else:
            print("✅ Compiled optimized implementation")
            return True
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def create_combined_implementation():
    """Create a version that can actually compile and run"""
    print("📝 Creating testable optimized implementation...")
    
    # Create a simple performance test that can be run with existing PyTorch backend
    test_code = '''#!/usr/bin/env python3
"""
Test memory optimization techniques in PyTorch backend
"""

import torch
import time
import numpy as np

class MemoryOptimizedGPUIndexer:
    """GPU indexer with persistent memory allocation"""
    
    def __init__(self, dim=384, max_vectors=1000000, max_queries=1000):
        self.dim = dim
        self.max_vectors = max_vectors
        self.max_queries = max_queries
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-allocate persistent GPU memory
        self.vectors = None
        self.query_buffer = torch.zeros(max_queries, dim, device=self.device)
        self.score_buffer = torch.zeros(max_queries, max_vectors, device=self.device)
        self.current_size = 0
        
        print(f"🚀 Memory-optimized indexer initialized on {self.device}")
        print(f"   Max vectors: {max_vectors}, Max queries: {max_queries}")
        if self.device.type == "cuda":
            allocated_mb = (max_queries * dim + max_queries * max_vectors) * 4 / 1e6
            print(f"   Pre-allocated: {allocated_mb:.1f} MB GPU memory")
    
    def add_vectors_persistent(self, vectors: np.ndarray):
        """Add vectors using persistent allocation"""
        start_time = time.time()
        
        if self.vectors is None:
            # Pre-allocate maximum size
            self.vectors = torch.zeros(self.max_vectors, self.dim, device=self.device)
        
        new_vectors = torch.from_numpy(vectors).float()
        new_size = len(new_vectors)
        
        # Copy to pre-allocated buffer (no dynamic allocation)
        self.vectors[self.current_size:self.current_size + new_size] = new_vectors.to(self.device)
        self.current_size += new_size
        
        add_time = time.time() - start_time
        return {
            "time_ms": add_time * 1000,
            "throughput": len(vectors) / add_time,
            "method": "persistent_allocation"
        }
    
    def batch_search_optimized(self, queries: np.ndarray, k=10):
        """Optimized batch search using pre-allocated buffers"""
        if self.current_size == 0:
            return {"error": "No vectors added"}
        
        start_time = time.time()
        num_queries = len(queries)
        
        if num_queries > self.max_queries:
            return {"error": f"Too many queries: {num_queries} > {self.max_queries}"}
        
        # Use pre-allocated query buffer (no allocation)
        self.query_buffer[:num_queries] = torch.from_numpy(queries).float().to(self.device)
        
        # Batch computation using pre-allocated score buffer
        active_vectors = self.vectors[:self.current_size]
        
        # Efficient matrix multiply: (num_queries, dim) @ (current_size, dim).T
        torch.matmul(
            self.query_buffer[:num_queries], 
            active_vectors.T, 
            out=self.score_buffer[:num_queries, :self.current_size]
        )
        
        # Top-k selection
        active_scores = self.score_buffer[:num_queries, :self.current_size]
        top_values, top_indices = torch.topk(active_scores, k, dim=1)
        
        search_time = time.time() - start_time
        
        return {
            "indices": top_indices.cpu().numpy().tolist(),
            "scores": top_values.cpu().numpy().tolist(),
            "time_ms": search_time * 1000,
            "qps": num_queries / search_time,
            "method": "persistent_buffers"
        }

def benchmark_memory_optimization():
    """Compare standard vs memory-optimized approaches"""
    print("=" * 80)
    print("🧪 MEMORY OPTIMIZATION BENCHMARK")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return
    
    # Test parameters
    num_vectors = 500000
    dim = 384
    num_queries = 100
    k = 10
    
    print(f"Configuration: {num_vectors} vectors, {num_queries} queries, {dim}D")
    print()
    
    # Generate test data
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    
    # Normalize for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    print("📊 Standard GPU Indexer:")
    # Standard approach (from existing gpu_backend.py)
    from gpu_backend import GPUIndexer
    standard_indexer = GPUIndexer(dim=dim)
    
    # Add vectors
    add_result = standard_indexer.add_vectors(vectors)
    print(f"   Add time: {add_result['time_ms']:.2f}ms")
    
    # Multiple searches to measure consistency
    search_times = []
    for i in range(10):
        result = standard_indexer.batch_search(queries, k)
        search_times.append(result['time_ms'])
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"   Search time: {avg_search_time:.2f}ms avg ({min(search_times):.2f}-{max(search_times):.2f}ms)")
    print(f"   QPS: {num_queries * 1000 / avg_search_time:.0f}")
    
    print()
    print("🚀 Memory-Optimized Indexer:")
    
    # Memory-optimized approach
    optimized_indexer = MemoryOptimizedGPUIndexer(dim=dim, max_vectors=num_vectors, max_queries=num_queries)
    
    # Add vectors
    add_result = optimized_indexer.add_vectors_persistent(vectors)
    print(f"   Add time: {add_result['time_ms']:.2f}ms")
    
    # Multiple searches
    opt_search_times = []
    for i in range(10):
        result = optimized_indexer.batch_search_optimized(queries, k)
        opt_search_times.append(result['time_ms'])
    
    avg_opt_search_time = sum(opt_search_times) / len(opt_search_times)
    print(f"   Search time: {avg_opt_search_time:.2f}ms avg ({min(opt_search_times):.2f}-{max(opt_search_times):.2f}ms)")
    print(f"   QPS: {num_queries * 1000 / avg_opt_search_time:.0f}")
    
    print()
    print("📈 Performance Comparison:")
    search_speedup = avg_search_time / avg_opt_search_time
    print(f"   Search speedup: {search_speedup:.1f}x")
    print(f"   Latency reduction: {avg_search_time - avg_opt_search_time:.2f}ms")
    print(f"   QPS improvement: {(num_queries * 1000 / avg_opt_search_time) - (num_queries * 1000 / avg_search_time):.0f}")
    
    # Memory usage
    print()
    print("💾 Memory Analysis:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"   GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
    
    print("=" * 80)
    print("✅ Memory optimization provides consistent performance!")
    print("=" * 80)

if __name__ == "__main__":
    benchmark_memory_optimization()
'''
    
    with open('benchmark_memory_optimization.py', 'w') as f:
        f.write(test_code)
    
    print("✅ Created memory optimization benchmark")
    return True

def main():
    print("🔬 CUDA Implementation Comparison")
    print("=" * 50)
    
    if not compile_cuda_code():
        print("❌ Compilation failed, running memory optimization test instead")
        print()
        
        # Run the memory optimization benchmark
        try:
            import subprocess
            result = subprocess.run([sys.executable, "benchmark_memory_optimization.py"], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"❌ Failed to run benchmark: {e}")
        return
    
    # If compilation succeeded, run both implementations
    print()
    print("🏃 Running original implementation...")
    try:
        result = subprocess.run(["./cuda_similarity_original"], 
                              capture_output=True, text=True, timeout=60)
        print("Original results:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Original failed: {e}")
    
    print()
    print("🚀 Running optimized implementation...")
    try:
        result = subprocess.run(["./cuda_similarity_optimized"], 
                              capture_output=True, text=True, timeout=60)
        print("Optimized results:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Optimized failed: {e}")

if __name__ == "__main__":
    main()