#!/usr/bin/env python3
"""
Simple test for GPU search without needing CUDA compilation.
Uses PyTorch's built-in CUDA operations.
"""

import torch
import torch.nn as nn
import time
import numpy as np

def test_gpu_search():
    """Test GPU search using PyTorch built-in ops."""
    
    print(" GPU Search Test (PyTorch)")
    print("="*50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    device = torch.device("cuda")
    print(f" Using GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test configurations
    configs = [
        (10_000, "Small"),
        (50_000, "Medium"),
        (100_000, "Large"),
        (500_000, "Extra Large"),
    ]
    
    for n_vectors, label in configs:
        print(f"\n Testing {label} ({n_vectors:,} vectors):")
        
        # Create random INT8 database
        db = torch.randint(-128, 127, (n_vectors, 512), dtype=torch.int8, device=device)
        query = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(query.float(), db.float().T)
            torch.cuda.synchronize()
        
        # Benchmark INT8 -> float32 matmul
        iterations = 100 if n_vectors <= 100_000 else 20
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(iterations):
            # Convert to float and compute
            scores = torch.matmul(query.float(), db.float().T)
            top_k = torch.topk(scores, k=10)
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        avg_latency = (elapsed / iterations) * 1000  # ms
        
        print(f"   Average latency: {avg_latency:.2f} ms")
        print(f"   Throughput: {1000/avg_latency:.0f} QPS")
        print(f"   Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        # Batch test
        batch_size = 32
        queries = torch.randint(-128, 127, (batch_size, 512), dtype=torch.int8, device=device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(10):
            batch_scores = torch.matmul(queries.float(), db.float().T)
            batch_top_k = torch.topk(batch_scores, k=10, dim=1)
            torch.cuda.synchronize()
        
        batch_elapsed = time.perf_counter() - start
        batch_latency = (batch_elapsed / 10) * 1000
        batch_throughput = (batch_size * 10) / batch_elapsed
        
        print(f"   Batch({batch_size}) latency: {batch_latency:.2f} ms")
        print(f"   Batch throughput: {batch_throughput:.0f} QPS")
    
    # Test custom INT8 computation
    print("\n Testing optimized INT8 computation:")
    n = 100_000
    db = torch.randint(-128, 127, (n, 512), dtype=torch.int8, device=device)
    query = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
    
    # Method 1: Direct float conversion
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        scores1 = torch.matmul(query.float(), db.float().T)
        torch.cuda.synchronize()
    t1 = (time.perf_counter() - start) / 100 * 1000
    
    # Method 2: Using int32 accumulation (simulating __dp4a)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        # Cast to int32 for accumulation
        scores2 = torch.matmul(query.to(torch.int32), db.to(torch.int32).T)
        torch.cuda.synchronize()
    t2 = (time.perf_counter() - start) / 100 * 1000
    
    print(f"   Float32 matmul: {t1:.2f} ms")
    print(f"   Int32 matmul: {t2:.2f} ms")
    print(f"   Speedup: {t1/t2:.2f}x")
    
    print("\n GPU search test complete!")
    
    # Estimate for 1M vectors
    print("\n Estimates for 1M vectors:")
    est_latency = avg_latency * (1_000_000 / n_vectors)
    print(f"   Expected latency: {est_latency:.1f} ms")
    print(f"   Expected QPS: {1000/est_latency:.0f}")

if __name__ == "__main__":
    test_gpu_search()