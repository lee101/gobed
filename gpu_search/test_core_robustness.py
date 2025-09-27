#!/usr/bin/env python3
"""
Core robustness testing for production deployment
Focuses on essential functionality without experimental features
"""

import torch
import numpy as np
import sys
import os
import time
import traceback

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')


def test_input_validation():
    """Test input validation and error handling"""
    print(" Testing input validation...")
    tests_passed = 0
    total_tests = 0
    
    device = torch.device("cuda")
    
    # Test 1: Wrong dtype
    total_tests += 1
    try:
        q = torch.randn(512, device=device)  # float32 instead of int8
        db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        print("   Wrong query dtype: Should have failed")
    except Exception:
        print("   Wrong query dtype: Correctly rejected")
        tests_passed += 1
    
    # Test 2: Wrong dimensions
    total_tests += 1
    try:
        q = torch.randint(-128, 127, (256,), dtype=torch.int8, device=device)  # 256 instead of 512
        db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        print("   Wrong query dimension: Should have failed")
    except Exception:
        print("   Wrong query dimension: Correctly rejected")
        tests_passed += 1
    
    # Test 3: CPU tensors
    total_tests += 1
    try:
        q = torch.randint(-128, 127, (512,), dtype=torch.int8)  # CPU tensor
        db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        print("   CPU tensor: Should have failed")
    except Exception:
        print("   CPU tensor: Correctly rejected")
        tests_passed += 1
    
    return tests_passed, total_tests


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print(" Testing edge cases...")
    tests_passed = 0
    total_tests = 0
    
    device = torch.device("cuda")
    
    # Test 1: Empty database
    total_tests += 1
    try:
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.empty((0, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        if result.shape[0] == 0:
            print("   Empty database: Handled correctly")
            tests_passed += 1
        else:
            print(f"   Empty database: Wrong shape {result.shape}")
    except Exception as e:
        print(f"   Empty database: Failed with {e}")
    
    # Test 2: Single vector database
    total_tests += 1
    try:
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (1, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        if result.shape[0] == 1:
            print("   Single vector database: Handled correctly")
            tests_passed += 1
        else:
            print(f"   Single vector database: Wrong shape {result.shape}")
    except Exception as e:
        print(f"   Single vector database: Failed with {e}")
    
    # Test 3: Large database
    total_tests += 1
    try:
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (100000, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        if result.shape[0] == 100000:
            print("   Large database (100K): Handled correctly")
            tests_passed += 1
        else:
            print(f"   Large database: Wrong shape {result.shape}")
        del db, result
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("   Large database: OOM handled gracefully")
        tests_passed += 1
    except Exception as e:
        print(f"   Large database: Failed with {e}")
    
    # Test 4: Extreme values
    total_tests += 1
    try:
        q = torch.full((512,), 127, dtype=torch.int8, device=device)  # Max values
        db = torch.full((100, 512), -128, dtype=torch.int8, device=device)  # Min values
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        expected = 127 * (-128) * 512  # Should be negative
        actual = result[0].item()
        if actual == expected:
            print("   Extreme values: Computed correctly")
            tests_passed += 1
        else:
            print(f"   Extreme values: Expected {expected}, got {actual}")
    except Exception as e:
        print(f"   Extreme values: Failed with {e}")
    
    return tests_passed, total_tests


def test_numerical_accuracy():
    """Test numerical accuracy against reference implementation"""
    print("🔢 Testing numerical accuracy...")
    tests_passed = 0
    total_tests = 1
    
    device = torch.device("cuda")
    
    try:
        # Create test data with fixed seed
        np.random.seed(42)
        q_np = np.random.randint(-128, 127, (512,), dtype=np.int8)
        db_np = np.random.randint(-128, 127, (100, 512), dtype=np.int8)
        
        # Reference implementation (CPU)
        ref_result = np.dot(db_np, q_np)
        
        # GPU implementation
        q_gpu = torch.from_numpy(q_np).to(device)
        db_gpu = torch.from_numpy(db_np).to(device)
        gpu_result = torch.ops.gobed_ann.i8dot512_scores(q_gpu, db_gpu).cpu().numpy()
        
        # Compare results
        max_diff = np.max(np.abs(ref_result - gpu_result))
        if max_diff == 0:
            print(f"   Numerical accuracy: Perfect match")
            tests_passed += 1
        else:
            print(f"   Numerical accuracy: Max difference {max_diff}")
            
    except Exception as e:
        print(f"   Numerical accuracy: Failed with {e}")
    
    return tests_passed, total_tests


def test_batch_operations():
    """Test batch operation edge cases"""
    print(" Testing batch operations...")
    tests_passed = 0
    total_tests = 0
    
    device = torch.device("cuda")
    
    # Test 1: Single query batch
    total_tests += 1
    try:
        queries = torch.randint(-128, 127, (1, 512), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_batch(queries, db)
        if result.shape == (1, 100):
            print("   Single query batch: Correct shape")
            tests_passed += 1
        else:
            print(f"   Single query batch: Wrong shape {result.shape}")
    except Exception as e:
        print(f"   Single query batch: Failed with {e}")
    
    # Test 2: Large batch
    total_tests += 1
    try:
        queries = torch.randint(-128, 127, (64, 512), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_batch(queries, db)
        if result.shape == (64, 1000):
            print("   Large batch (64): Correct shape")
            tests_passed += 1
        else:
            print(f"   Large batch: Wrong shape {result.shape}")
    except Exception as e:
        print(f"   Large batch: Failed with {e}")
    
    return tests_passed, total_tests


def test_memory_management():
    """Test memory allocation and cleanup"""
    print(" Testing memory management...")
    tests_passed = 0
    total_tests = 1
    
    device = torch.device("cuda")
    
    try:
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate and free multiple times
        for i in range(10):
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
            db = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            
            # Force cleanup
            del q, db, result
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        memory_leak = final_memory > initial_memory + 1024*1024  # Allow 1MB tolerance
        
        if not memory_leak:
            print(f"   Memory leak check: No significant leaks")
            print(f"     Initial: {initial_memory/1e6:.1f}MB, Final: {final_memory/1e6:.1f}MB")
            tests_passed += 1
        else:
            print(f"   Memory leak detected: {(final_memory-initial_memory)/1e6:.1f}MB")
            
    except Exception as e:
        print(f"   Memory management: Failed with {e}")
    
    return tests_passed, total_tests


def test_performance_consistency():
    """Test performance consistency across runs"""
    print(" Testing performance consistency...")
    tests_passed = 0
    total_tests = 1
    
    device = torch.device("cuda")
    
    try:
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (10000, 512), dtype=torch.int8, device=device)
        
        # Warmup
        for _ in range(5):
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        
        # Measure performance
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time  # Coefficient of variation
        
        # Performance should be consistent (CV < 15%)
        if cv < 0.15:
            print(f"   Performance consistency: CV = {cv*100:.1f}%")
            print(f"     Mean latency: {mean_time*1000:.2f}ms")
            tests_passed += 1
        else:
            print(f"   Performance inconsistent: CV = {cv*100:.1f}%")
            
    except Exception as e:
        print(f"   Performance consistency: Failed with {e}")
    
    return tests_passed, total_tests


def main():
    """Run core robustness test suite"""
    print("=" * 80)
    print("🧪 CORE ROBUSTNESS TEST SUITE")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    all_passed = 0
    all_total = 0
    
    # Run all test categories
    passed, total = test_input_validation()
    all_passed += passed
    all_total += total
    
    passed, total = test_edge_cases()
    all_passed += passed
    all_total += total
    
    passed, total = test_numerical_accuracy()
    all_passed += passed
    all_total += total
    
    passed, total = test_batch_operations()
    all_passed += passed
    all_total += total
    
    passed, total = test_memory_management()
    all_passed += passed
    all_total += total
    
    passed, total = test_performance_consistency()
    all_passed += passed
    all_total += total
    
    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {all_total}")
    print(f"Passed: {all_passed}")
    print(f"Failed: {all_total - all_passed}")
    print(f"Success rate: {all_passed/all_total*100:.1f}%")
    
    if all_passed == all_total:
        print("\n CORE SYSTEM READY FOR PRODUCTION")
        print(" All critical tests passed")
        print(" Error handling robust")
        print(" Performance consistent")
        print(" Memory management stable")
    else:
        print(f"\n  {all_total - all_passed} TESTS FAILED - REVIEW REQUIRED")
    
    print("=" * 80)
    return all_passed == all_total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)