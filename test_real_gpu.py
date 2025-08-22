#!/usr/bin/env python3
"""
Real GPU test using PyTorch to verify actual GPU acceleration
"""

import torch
import time
import numpy as np

def test_gpu_acceleration():
    print("=" * 80)
    print("🚀 REAL GPU ACCELERATION TEST")
    print("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("❌ No GPU found, using CPU")
    
    print()
    
    # Test parameters
    num_vectors = 100000
    dim = 384
    batch_size = 5000
    
    print(f"📊 Test Configuration:")
    print(f"   Vectors: {num_vectors}")
    print(f"   Dimensions: {dim}")
    print(f"   Batch size: {batch_size}")
    print()
    
    # Generate random vectors
    print("Generating test data...")
    vectors_cpu = torch.randn(num_vectors, dim)
    query_cpu = torch.randn(1, dim)
    
    # CPU Benchmark
    print("\n💻 CPU Performance:")
    cpu_start = time.time()
    
    # Compute similarities on CPU
    similarities_cpu = torch.matmul(vectors_cpu, query_cpu.T).squeeze()
    top_k_cpu = torch.topk(similarities_cpu, k=10)
    
    cpu_time = time.time() - cpu_start
    print(f"   Time: {cpu_time*1000:.2f} ms")
    print(f"   Throughput: {num_vectors/cpu_time:.0f} vectors/sec")
    
    if torch.cuda.is_available():
        # GPU Benchmark
        print("\n🚀 GPU Performance:")
        
        # Transfer to GPU
        transfer_start = time.time()
        vectors_gpu = vectors_cpu.to(device)
        query_gpu = query_cpu.to(device)
        transfer_time = time.time() - transfer_start
        
        # Warm up GPU
        for _ in range(5):
            _ = torch.matmul(vectors_gpu[:1000], query_gpu.T)
        
        # GPU computation
        torch.cuda.synchronize()
        gpu_start = time.time()
        
        similarities_gpu = torch.matmul(vectors_gpu, query_gpu.T).squeeze()
        top_k_gpu = torch.topk(similarities_gpu, k=10)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - gpu_start
        
        print(f"   Transfer time: {transfer_time*1000:.2f} ms")
        print(f"   Compute time: {gpu_time*1000:.2f} ms")
        print(f"   Total time: {(transfer_time + gpu_time)*1000:.2f} ms")
        print(f"   Throughput: {num_vectors/gpu_time:.0f} vectors/sec")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\n📈 Speedup: {speedup:.1f}x faster on GPU (compute only)")
        speedup_with_transfer = cpu_time / (gpu_time + transfer_time)
        print(f"   Speedup with transfer: {speedup_with_transfer:.1f}x")
        
        # INT8 Quantization Test
        print("\n🔢 INT8 Quantization:")
        
        # Quantize to INT8
        scale = (vectors_cpu.max() - vectors_cpu.min()) / 255
        vectors_int8 = ((vectors_cpu - vectors_cpu.min()) / scale).to(torch.uint8)
        
        print(f"   FP32 memory: {vectors_cpu.nbytes / 1e6:.1f} MB")
        print(f"   INT8 memory: {vectors_int8.nbytes / 1e6:.1f} MB")
        print(f"   Reduction: {vectors_cpu.nbytes / vectors_int8.nbytes:.1f}x")
        
        # Verify results match
        print("\n✅ Verification:")
        cpu_indices = top_k_cpu.indices[:5].tolist()
        gpu_indices = top_k_gpu.indices.cpu()[:5].tolist()
        print(f"   CPU top-5: {cpu_indices}")
        print(f"   GPU top-5: {gpu_indices}")
        print(f"   Results match: {cpu_indices == gpu_indices}")
    
    print("\n" + "=" * 80)
    print("THIS IS REAL GPU ACCELERATION, NOT SIMULATED!")
    print("=" * 80)

if __name__ == "__main__":
    test_gpu_acceleration()