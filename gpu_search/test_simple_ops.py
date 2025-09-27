#!/usr/bin/env python3
"""
Simple test to isolate segfault issue
"""

import torch
import sys
import traceback

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_properties(0).name)
    print("CUDA version:", torch.version.cuda)

# Try to load the library
try:
    print("\nLoading CUDA ops library...")
    torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')
    print(" Library loaded successfully")
    
    # List available ops
    print("\nAvailable operations:")
    if hasattr(torch.ops, 'gobed_ann'):
        ops = dir(torch.ops.gobed_ann)
        for op in ops:
            if not op.startswith('_'):
                print(f"  - gobed_ann.{op}")
    
    # Test basic operation
    print("\nTesting basic i8dot512_scores operation...")
    device = torch.device("cuda")
    
    # Create small test tensors
    q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
    db = torch.randint(-128, 127, (10, 512), dtype=torch.int8, device=device)
    
    print(f"Query shape: {q.shape}, dtype: {q.dtype}")
    print(f"Database shape: {db.shape}, dtype: {db.dtype}")
    
    # Run operation
    result = torch.ops.gobed_ann.i8dot512_scores(q, db)
    print(f" Result shape: {result.shape}, dtype: {result.dtype}")
    
    # Test error handling
    print("\nTesting error handling...")
    try:
        # Wrong dtype
        q_float = torch.randn(512, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q_float, db)
        print(" Should have raised error for wrong dtype")
    except Exception as e:
        print(f" Correctly rejected wrong dtype: {type(e).__name__}")
    
    print("\n All basic tests passed!")
    
except Exception as e:
    print(f"\n Error: {e}")
    traceback.print_exc()