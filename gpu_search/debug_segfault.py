#!/usr/bin/env python3
"""Debug segmentation fault step by step"""

import torch
import sys

def test_basic_load():
    print("1. Testing basic library load...")
    try:
        sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
        torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')
        print("   ✅ Library loaded")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_device_check():
    print("2. Testing CUDA device...")
    try:
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name()
            print(f"   ✅ CUDA available: {device}")
            return True
        else:
            print("   ❌ CUDA not available")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_simple_tensors():
    print("3. Testing simple tensor creation...")
    try:
        device = torch.device("cuda")
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (10, 512), dtype=torch.int8, device=device)
        print(f"   ✅ Tensors created: q={q.shape}, db={db.shape}")
        return True, q, db
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False, None, None

def test_kernel_call(q, db):
    print("4. Testing kernel call...")
    try:
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        print(f"   ✅ Kernel executed: result={result.shape}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Debugging segmentation fault...")
    
    if not test_basic_load():
        return
    
    if not test_device_check():
        return
    
    success, q, db = test_simple_tensors()
    if not success:
        return
    
    test_kernel_call(q, db)
    
    print("🎯 Debug complete")

if __name__ == "__main__":
    main()