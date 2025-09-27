#!/usr/bin/env python3
"""Test the compiled CUDA ops."""

import torch
import sys
import os

# Add build directory to path
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
os.environ['LD_LIBRARY_PATH'] = '/home/lee/code/gobed/gpu_search/cuda_ops/build:' + os.environ.get('LD_LIBRARY_PATH', '')

print(" Testing Custom CUDA Ops")
print("=" * 50)

# Load the compiled library
try:
    torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')
    print(" Loaded libgobed_ann_ops.so")
except Exception as e:
    print(f" Failed to load library: {e}")
    sys.exit(1)

# Check available ops
print("\n Available custom ops:")
if hasattr(torch.ops, 'gobed_ann'):
    ops = dir(torch.ops.gobed_ann)
    for op in ops:
        if not op.startswith('_'):
            print(f"  - gobed_ann::{op}")
else:
    print("   No gobed_ann namespace found")

# Test INT8 dot product
print("\n Testing INT8 dot product:")
device = torch.device("cuda")

# Create test data
n = 10000
db = torch.randint(-128, 127, (n, 512), dtype=torch.int8, device=device)
query = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)

print(f"  Database: {db.shape} ({db.dtype})")
print(f"  Query: {query.shape} ({query.dtype})")

try:
    # Call custom op
    scores = torch.ops.gobed_ann.i8dot512_scores(query, db)
    print(f"   Scores computed: {scores.shape} ({scores.dtype})")
    print(f"  Score range: [{scores.min().item()}, {scores.max().item()}]")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        scores = torch.ops.gobed_ann.i8dot512_scores(query, db)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latency = (elapsed / 100) * 1000
    print(f"  Average latency: {latency:.2f} ms")
    print(f"  Throughput: {1000/latency:.0f} QPS")
    
except Exception as e:
    print(f"   Failed: {e}")

# Test batch INT8 dot product
print("\n Testing batch INT8 dot product:")
batch = 32
queries = torch.randint(-128, 127, (batch, 512), dtype=torch.int8, device=device)
print(f"  Queries: {queries.shape} ({queries.dtype})")

try:
    scores_batch = torch.ops.gobed_ann.i8dot512_batch(queries, db)
    print(f"   Batch scores computed: {scores_batch.shape} ({scores_batch.dtype})")
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        scores_batch = torch.ops.gobed_ann.i8dot512_batch(queries, db)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    batch_latency = (elapsed / 10) * 1000
    batch_throughput = (batch * 10) / elapsed
    print(f"  Batch latency: {batch_latency:.2f} ms")
    print(f"  Batch throughput: {batch_throughput:.0f} QPS")
    
except Exception as e:
    print(f"   Failed: {e}")

print("\n CUDA ops test complete!")