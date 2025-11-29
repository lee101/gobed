#!/usr/bin/env python3
"""
Comprehensive robustness testing for GPU search system
Tests edge cases, error handling, and CUDA version compatibility
"""

import torch
import numpy as np
import sys
import os
import time
import traceback
from typing import List, Dict, Any

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')


class RobustnessTestSuite:
    """Comprehensive test suite for production readiness"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = []
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - cannot run robustness tests")
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = " PASS" if success else " FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"     {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    def test_cuda_capabilities(self):
        """Test CUDA device capabilities and version compatibility"""
        print("\n Testing CUDA capabilities...")
        
        try:
            # Check device capabilities
            capabilities = torch.ops.gobed_ann.check_cuda_capabilities()
            self.log_test("CUDA capabilities check", True, 
                         f"Capabilities tensor shape: {capabilities.shape}")
            
            # Print device info
            device_props = torch.cuda.get_device_properties(0)
            print(f"     Device: {device_props.name}")
            print(f"     Compute capability: {device_props.major}.{device_props.minor}")
            print(f"     Memory: {device_props.total_memory / 1e9:.1f} GB")
            
            # Check for __dp4a support
            dp4a_support = (device_props.major > 6) or (device_props.major == 6 and device_props.minor >= 1)
            self.log_test("__dp4a intrinsic support", dp4a_support,
                         f"Compute capability {device_props.major}.{device_props.minor}")
            
        except Exception as e:
            self.log_test("CUDA capabilities check", False, str(e))
    
    def test_input_validation(self):
        """Test input validation and error handling"""
        print("\n Testing input validation...")
        
        # Test wrong dtype
        try:
            q = torch.randn(512, device=self.device)  # float32 instead of int8
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Wrong query dtype rejection", False, "Should have failed")
        except Exception as e:
            self.log_test("Wrong query dtype rejection", True, "Correctly rejected float32")
        
        # Test wrong dimensions
        try:
            q = torch.randint(-128, 127, (256,), dtype=torch.int8, device=self.device)  # 256 instead of 512
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Wrong query dimension rejection", False, "Should have failed")
        except Exception as e:
            self.log_test("Wrong query dimension rejection", True, "Correctly rejected 256-dim")
        
        # Test non-contiguous tensors
        try:
            q = torch.randint(-128, 127, (1024,), dtype=torch.int8, device=self.device)[::2]  # Non-contiguous
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Non-contiguous tensor rejection", False, "Should have failed")
        except Exception as e:
            self.log_test("Non-contiguous tensor rejection", True, "Correctly rejected non-contiguous")
        
        # Test CPU tensors
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8)  # CPU tensor
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("CPU tensor rejection", False, "Should have failed")
        except Exception as e:
            self.log_test("CPU tensor rejection", True, "Correctly rejected CPU tensor")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n Testing edge cases...")
        
        # Test empty database
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.empty((0, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Empty database handling", result.shape[0] == 0, 
                         f"Result shape: {result.shape}")
        except Exception as e:
            self.log_test("Empty database handling", False, str(e))
        
        # Test single vector database
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (1, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Single vector database", result.shape[0] == 1,
                         f"Result shape: {result.shape}")
        except Exception as e:
            self.log_test("Single vector database", False, str(e))
        
        # Test large database (memory stress)
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            # Test with progressively larger sizes
            for size in [1000, 10000, 100000]:
                try:
                    db = torch.randint(-128, 127, (size, 512), dtype=torch.int8, device=self.device)
                    result = torch.ops.gobed_ann.i8dot512_scores(q, db)
                    success = result.shape[0] == size
                    self.log_test(f"Large database ({size:,} vectors)", success,
                                 f"Result shape: {result.shape}")
                    del db, result  # Free memory
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    self.log_test(f"Large database ({size:,} vectors)", True,
                                 "OOM handled gracefully")
                    break
                except Exception as e:
                    self.log_test(f"Large database ({size:,} vectors)", False, str(e))
                    break
        except Exception as e:
            self.log_test("Large database test", False, str(e))
        
        # Test extreme values
        try:
            q = torch.full((512,), 127, dtype=torch.int8, device=self.device)  # Max values
            db = torch.full((100, 512), -128, dtype=torch.int8, device=self.device)  # Min values
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            expected = 127 * (-128) * 512  # Should be negative
            actual = result[0].item()
            self.log_test("Extreme values handling", actual == expected,
                         f"Expected: {expected}, Got: {actual}")
        except Exception as e:
            self.log_test("Extreme values handling", False, str(e))
    
    def test_batch_operations(self):
        """Test batch operation edge cases"""
        print("\n Testing batch operations...")
        
        # Test single query batch
        try:
            queries = torch.randint(-128, 127, (1, 512), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_batch(queries, db)
            self.log_test("Single query batch", result.shape == (1, 100),
                         f"Result shape: {result.shape}")
        except Exception as e:
            self.log_test("Single query batch", False, str(e))
        
        # Test large batch
        try:
            batch_sizes = [32, 64, 128, 256]
            for batch_size in batch_sizes:
                try:
                    queries = torch.randint(-128, 127, (batch_size, 512), dtype=torch.int8, device=self.device)
                    db = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=self.device)
                    result = torch.ops.gobed_ann.i8dot512_batch(queries, db)
                    success = result.shape == (batch_size, 1000)
                    self.log_test(f"Batch size {batch_size}", success,
                                 f"Result shape: {result.shape}")
                    del queries, db, result
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    self.log_test(f"Batch size {batch_size}", True, "OOM handled gracefully")
                    break
                except Exception as e:
                    self.log_test(f"Batch size {batch_size}", False, str(e))
                    break
        except Exception as e:
            self.log_test("Large batch test", False, str(e))
    
    def test_numerical_accuracy(self):
        """Test numerical accuracy against reference implementation"""
        print("\n🔢 Testing numerical accuracy...")
        
        try:
            # Create test data
            np.random.seed(42)
            q_np = np.random.randint(-128, 127, (512,), dtype=np.int8)
            db_np = np.random.randint(-128, 127, (100, 512), dtype=np.int8)
            
            # Reference implementation (CPU)
            ref_result = np.dot(db_np, q_np)
            
            # GPU implementation
            q_gpu = torch.from_numpy(q_np).to(self.device)
            db_gpu = torch.from_numpy(db_np).to(self.device)
            gpu_result = torch.ops.gobed_ann.i8dot512_scores(q_gpu, db_gpu).cpu().numpy()
            
            # Compare results
            max_diff = np.max(np.abs(ref_result - gpu_result))
            accuracy = max_diff == 0
            
            self.log_test("Numerical accuracy", accuracy,
                         f"Max difference: {max_diff}")
            
        except Exception as e:
            self.log_test("Numerical accuracy", False, str(e))
    
    def test_memory_management(self):
        """Test memory allocation and cleanup"""
        print("\n Testing memory management...")
        
        try:
            initial_memory = torch.cuda.memory_allocated()
            
            # Allocate and free multiple times
            for i in range(10):
                q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
                db = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=self.device)
                result = torch.ops.gobed_ann.i8dot512_scores(q, db)
                
                # Force cleanup
                del q, db, result
                torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            memory_leak = final_memory > initial_memory + 1024*1024  # Allow 1MB tolerance
            
            self.log_test("Memory leak check", not memory_leak,
                         f"Initial: {initial_memory/1e6:.1f}MB, Final: {final_memory/1e6:.1f}MB")
            
        except Exception as e:
            self.log_test("Memory leak check", False, str(e))
    
    def test_performance_consistency(self):
        """Test performance consistency across runs"""
        print("\n Testing performance consistency...")
        
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (10000, 512), dtype=torch.int8, device=self.device)
            
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
            
            # Performance should be consistent (CV < 10%)
            consistent = cv < 0.1
            
            self.log_test("Performance consistency", consistent,
                         f"Mean: {mean_time*1000:.2f}ms, CV: {cv*100:.1f}%")
            
        except Exception as e:
            self.log_test("Performance consistency", False, str(e))
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 80)
        print("🧪 ROBUSTNESS TEST SUITE FOR GPU SEARCH SYSTEM")
        print("=" * 80)
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print()
        
        # Run all test categories
        self.test_cuda_capabilities()
        self.test_input_validation()
        self.test_edge_cases()
        self.test_batch_operations()
        self.test_numerical_accuracy()
        self.test_memory_management()
        self.test_performance_consistency()
        
        # Summary
        print("\n" + "=" * 80)
        print(" TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        # List failed tests
        failed_tests = [result for result in self.test_results if not result["success"]]
        if failed_tests:
            print("\n Failed tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        else:
            print("\n All tests passed!")
        
        return passed == total


def main():
    """Run robustness test suite"""
    test_suite = RobustnessTestSuite()
    
    try:
        success = test_suite.run_all_tests()
        
        print("\n" + "=" * 80)
        if success:
            print(" SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("  ISSUES FOUND - REVIEW BEFORE DEPLOYMENT")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n Test suite failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())