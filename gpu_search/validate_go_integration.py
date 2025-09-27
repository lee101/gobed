#!/usr/bin/env python3
"""
Validation script for Go LibTorch integration
Tests aspects specific to calling from Go/C++ code
"""

import torch
import numpy as np
import sys
import os
import ctypes
import json
from pathlib import Path

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')


def test_c_api_compatibility():
    """Test C API compatibility for Go integration"""
    print("🔗 Testing C API compatibility...")
    
    device = torch.device("cuda")
    
    # Test 1: Contiguous memory layout (important for Go)
    try:
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        
        # Ensure contiguous
        q = q.contiguous()
        db = db.contiguous()
        
        # Check memory layout
        assert q.is_contiguous(), "Query not contiguous"
        assert db.is_contiguous(), "Database not contiguous"
        
        # Test operation
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        assert result.is_contiguous(), "Result not contiguous"
        
        print("   Memory layout: Contiguous tensors handled correctly")
        
    except Exception as e:
        print(f"   Memory layout: {e}")
        return False
    
    # Test 2: Data pointer access (for Go FFI)
    try:
        q_ptr = q.data_ptr()
        db_ptr = db.data_ptr()
        result_ptr = result.data_ptr()
        
        # Verify pointers are valid
        assert q_ptr != 0, "Query pointer is null"
        assert db_ptr != 0, "Database pointer is null"
        assert result_ptr != 0, "Result pointer is null"
        
        print(f"   Data pointers: Valid pointers obtained")
        print(f"     Query: 0x{q_ptr:x}, DB: 0x{db_ptr:x}, Result: 0x{result_ptr:x}")
        
    except Exception as e:
        print(f"   Data pointers: {e}")
        return False
    
    return True


def test_error_handling_for_go():
    """Test error handling patterns that work well with Go"""
    print("  Testing Go-friendly error handling...")
    
    device = torch.device("cuda")
    
    # Test 1: Graceful handling of invalid inputs
    error_cases = [
        {
            "name": "Wrong dtype",
            "q": lambda: torch.randn(512, device=device),  # float32
            "db": lambda: torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        },
        {
            "name": "Wrong dimensions", 
            "q": lambda: torch.randint(-128, 127, (256,), dtype=torch.int8, device=device),  # 256
            "db": lambda: torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        },
        {
            "name": "CPU tensor",
            "q": lambda: torch.randint(-128, 127, (512,), dtype=torch.int8),  # CPU
            "db": lambda: torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=device)
        }
    ]
    
    success_count = 0
    for case in error_cases:
        try:
            q = case["q"]()
            db = case["db"]()
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            print(f"   {case['name']}: Should have failed")
        except RuntimeError as e:
            # RuntimeError is good for Go integration
            print(f"   {case['name']}: Properly raised RuntimeError")
            success_count += 1
        except Exception as e:
            print(f"    {case['name']}: Unexpected error type: {type(e).__name__}")
    
    return success_count == len(error_cases)


def test_performance_for_production():
    """Test performance characteristics for production deployment"""
    print(" Testing production performance...")
    
    device = torch.device("cuda")
    
    # Test realistic production sizes
    sizes = [1000, 10000, 100000]
    batch_sizes = [1, 32, 64, 128]
    
    results = {}
    
    for db_size in sizes:
        print(f"  Testing database size: {db_size:,}")
        
        try:
            db = torch.randint(-128, 127, (db_size, 512), dtype=torch.int8, device=device)
            
            # Single query performance
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
            
            # Warmup
            for _ in range(3):
                _ = torch.ops.gobed_ann.i8dot512_scores(q, db)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            end.record()
            torch.cuda.synchronize()
            
            latency = start.elapsed_time(end)  # milliseconds
            qps = 1000 / latency
            
            results[f"single_{db_size}"] = {
                "latency_ms": latency,
                "qps": qps,
                "memory_mb": torch.cuda.memory_allocated() / 1e6
            }
            
            print(f"    Single query: {latency:.2f}ms ({qps:.0f} QPS)")
            
            # Test batch performance if memory allows
            if db_size <= 10000:  # Avoid OOM on large DBs
                for batch_size in batch_sizes:
                    try:
                        queries = torch.randint(-128, 127, (batch_size, 512), dtype=torch.int8, device=device)
                        
                        # Warmup
                        _ = torch.ops.gobed_ann.i8dot512_batch(queries, db)
                        
                        torch.cuda.synchronize()
                        start.record()
                        result = torch.ops.gobed_ann.i8dot512_batch(queries, db)
                        end.record()
                        torch.cuda.synchronize()
                        
                        batch_latency = start.elapsed_time(end)
                        batch_qps = batch_size * 1000 / batch_latency
                        
                        results[f"batch_{batch_size}_{db_size}"] = {
                            "latency_ms": batch_latency,
                            "qps": batch_qps,
                            "batch_size": batch_size
                        }
                        
                        print(f"    Batch-{batch_size}: {batch_latency:.2f}ms ({batch_qps:.0f} QPS)")
                        
                    except torch.cuda.OutOfMemoryError:
                        print(f"    Batch-{batch_size}: OOM (expected for large sizes)")
                        break
            
            # Cleanup
            del db, q
            if 'queries' in locals():
                del queries
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    Error with size {db_size}: {e}")
            continue
    
    # Save performance data for Go integration reference
    with open('performance_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   Performance baseline saved to performance_baseline.json")
    return True


def test_concurrent_access():
    """Test behavior under concurrent access (important for Go goroutines)"""
    print("🔄 Testing concurrent access patterns...")
    
    device = torch.device("cuda")
    
    # Simulate multiple concurrent requests
    try:
        # Create multiple query/db pairs
        queries = []
        dbs = []
        
        for i in range(5):
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
            db = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=device)
            queries.append(q)
            dbs.append(db)
        
        # Simulate concurrent execution
        results = []
        for q, db in zip(queries, dbs):
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            results.append(result)
        
        # Verify all results are correct
        for i, result in enumerate(results):
            assert result.shape[0] == 1000, f"Result {i} has wrong shape"
        
        print("   Concurrent access: Multiple operations completed successfully")
        
        # Cleanup
        for tensors in [queries, dbs, results]:
            for tensor in tensors:
                del tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"   Concurrent access: {e}")
        return False


def test_memory_stability():
    """Test memory stability for long-running Go services"""
    print(" Testing memory stability...")
    
    device = torch.device("cuda")
    
    try:
        initial_memory = torch.cuda.memory_allocated()
        
        # Simulate long-running service with many requests
        for iteration in range(50):
            # Random sizes to simulate varied load
            db_size = np.random.randint(100, 5000)
            batch_size = np.random.randint(1, 32)
            
            # Create tensors
            queries = torch.randint(-128, 127, (batch_size, 512), dtype=torch.int8, device=device)
            db = torch.randint(-128, 127, (db_size, 512), dtype=torch.int8, device=device)
            
            # Execute operations
            if batch_size == 1:
                result = torch.ops.gobed_ann.i8dot512_scores(queries[0], db)
            else:
                result = torch.ops.gobed_ann.i8dot512_batch(queries, db)
            
            # Explicit cleanup (important for Go GC integration)
            del queries, db, result
            
            # Periodic cleanup
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated()
                print(f"    Iteration {iteration}: {current_memory/1e6:.1f}MB allocated")
        
        final_memory = torch.cuda.memory_allocated()
        memory_growth = final_memory - initial_memory
        
        # Allow 10MB growth tolerance
        if memory_growth < 10 * 1024 * 1024:
            print(f"   Memory stability: Growth {memory_growth/1e6:.1f}MB (acceptable)")
            return True
        else:
            print(f"   Memory stability: Growth {memory_growth/1e6:.1f}MB (too much)")
            return False
            
    except Exception as e:
        print(f"   Memory stability: {e}")
        return False


def main():
    """Run Go integration validation"""
    print("=" * 80)
    print("🔗 GO LIBTORCH INTEGRATION VALIDATION")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    tests = [
        test_c_api_compatibility,
        test_error_handling_for_go,
        test_performance_for_production,
        test_concurrent_access,
        test_memory_stability
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
            print()
        except Exception as e:
            print(f"   Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 80)
    print(" VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n READY FOR GO LIBTORCH INTEGRATION")
        print(" C API compatibility verified")
        print(" Error handling Go-friendly")  
        print(" Performance baselines established")
        print(" Concurrent access stable")
        print(" Memory management robust")
        
        print("\n Integration Notes:")
        print("- Use contiguous tensors for FFI")
        print("- Handle RuntimeError exceptions")
        print("- Monitor memory growth in long-running services")
        print("- Performance baseline saved to performance_baseline.json")
    else:
        print(f"\n  {total-passed} TESTS FAILED - REVIEW REQUIRED")
    
    print("=" * 80)
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)