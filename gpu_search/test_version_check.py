#!/usr/bin/env python3
"""Test version check function separately"""

import torch
import sys

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')

def test_version_check():
    print("Testing version check function...")
    
    try:
        # Test if function exists
        if hasattr(torch.ops, 'gobed_ann'):
            ops = [op for op in dir(torch.ops.gobed_ann) if not op.startswith('_')]
            print(f"Available ops: {ops}")
            
            if 'check_cuda_capabilities' in ops:
                print("Calling check_cuda_capabilities...")
                result = torch.ops.gobed_ann.check_cuda_capabilities()
                print(f"Result: {result}")
            else:
                print("check_cuda_capabilities not found")
                
        else:
            print("gobed_ann namespace not found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_version_check()