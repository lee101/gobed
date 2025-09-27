#!/usr/bin/env python3
"""
Final robustness test suite - using correct library path
"""

import torch
import numpy as np
import time
import gc
import subprocess
import os
import json

# Load the correct library
torch.ops.load_library('/home/lee/code/gobed/gpu_search/libgobed_ann_ops.so')

class RobustnessChecker:
    def __init__(self):
        self.device = torch.device("cuda")
        self.results = {
            "tests": [],
            "issues": [],
            "recommendations": []
        }
        
    def log_test(self, name, passed, details=""):
        """Log test result"""
        self.results["tests"].append({
            "name": name,
            "passed": passed,
            "details": details
        })
        status = "" if passed else ""
        print(f"{status} {name}: {details}")
        
    def test_error_handling(self):
        """Test error handling robustness"""
        print("\n Testing Error Handling")
        print("-" * 50)
        
        # Test 1: Wrong dtype
        try:
            q = torch.randn(512, device=self.device)
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Wrong dtype rejection", False, "Should have failed")
        except:
            self.log_test("Wrong dtype rejection", True, "Correctly rejected float32")
            
        # Test 2: Wrong dimensions
        try:
            q = torch.randint(-128, 127, (256,), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Wrong dimension rejection", False, "Should have failed")
        except:
            self.log_test("Wrong dimension rejection", True, "Correctly rejected 256-dim")
            
        # Test 3: Empty database
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.empty((0, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("Empty database", result.shape[0] == 0, f"Shape: {result.shape}")
        except Exception as e:
            self.log_test("Empty database", False, str(e))
            self.results["issues"].append("Empty database handling failed")
            
        # Test 4: CPU tensor rejection
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8)  # CPU
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            torch.ops.gobed_ann.i8dot512_scores(q, db)
            self.log_test("CPU tensor rejection", False, "Should have failed")
        except:
            self.log_test("CPU tensor rejection", True, "Correctly rejected CPU tensor")
            
    def test_memory_stability(self):
        """Test for memory leaks"""
        print("\n Testing Memory Stability")
        print("-" * 50)
        
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated()
        
        # Run 100 iterations
        for i in range(100):
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (5000, 512), dtype=torch.int8, device=self.device)
            scores = torch.ops.gobed_ann.i8dot512_scores(q, db)
            del q, db, scores
            
        torch.cuda.empty_cache()
        final_mem = torch.cuda.memory_allocated()
        
        leak = final_mem - initial_mem
        self.log_test("Memory leak check", leak < 10_000_000, f"Leak: {leak/1e6:.2f} MB")
        
        if leak > 10_000_000:
            self.results["issues"].append(f"Potential memory leak: {leak/1e6:.2f} MB")
            
    def test_large_scale(self):
        """Test with varying database sizes"""
        print("\n📏 Testing Scale")
        print("-" * 50)
        
        sizes = [100, 1000, 10000, 100000, 1000000]
        
        for size in sizes:
            try:
                q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
                db = torch.randint(-128, 127, (size, 512), dtype=torch.int8, device=self.device)
                
                start = time.perf_counter()
                scores = torch.ops.gobed_ann.i8dot512_scores(q, db)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                throughput = size / elapsed / 1e6
                self.log_test(f"Scale {size:>7}", True, f"{elapsed*1000:.2f}ms, {throughput:.2f}M vec/s")
                
                del q, db, scores
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                self.log_test(f"Scale {size:>7}", False, "OOM")
                self.results["recommendations"].append(f"Implement batching for sizes > {sizes[sizes.index(size)-1]}")
                break
            except Exception as e:
                self.log_test(f"Scale {size:>7}", False, str(e))
                
    def test_edge_values(self):
        """Test extreme values"""
        print("\n Testing Edge Values")
        print("-" * 50)
        
        # Maximum values
        q_max = torch.full((512,), 127, dtype=torch.int8, device=self.device)
        db_max = torch.full((100, 512), 127, dtype=torch.int8, device=self.device)
        scores_max = torch.ops.gobed_ann.i8dot512_scores(q_max, db_max)
        expected_max = 127 * 127 * 512
        
        self.log_test("Maximum values", 
                     torch.all(scores_max == expected_max).item(),
                     f"Expected: {expected_max}, Got: {scores_max[0].item()}")
        
        # Minimum values
        q_min = torch.full((512,), -128, dtype=torch.int8, device=self.device)
        db_min = torch.full((100, 512), -128, dtype=torch.int8, device=self.device)
        scores_min = torch.ops.gobed_ann.i8dot512_scores(q_min, db_min)
        expected_min = 128 * 128 * 512
        
        self.log_test("Minimum values",
                     torch.all(scores_min == expected_min).item(),
                     f"Expected: {expected_min}, Got: {scores_min[0].item()}")
        
        # Mixed values
        q_zero = torch.zeros((512,), dtype=torch.int8, device=self.device)
        db_rand = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
        scores_zero = torch.ops.gobed_ann.i8dot512_scores(q_zero, db_rand)
        
        self.log_test("Zero query",
                     torch.all(scores_zero == 0).item(),
                     f"All scores should be 0, got range [{scores_zero.min()}, {scores_zero.max()}]")
                     
    def test_batch_operations(self):
        """Test batch query operations"""
        print("\n Testing Batch Operations")
        print("-" * 50)
        
        try:
            queries = torch.randint(-128, 127, (32, 512), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (10000, 512), dtype=torch.int8, device=self.device)
            
            start = time.perf_counter()
            batch_scores = torch.ops.gobed_ann.i8dot512_batch(queries, db)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            self.log_test("Batch processing",
                         batch_scores.shape == (32, 10000),
                         f"Shape: {batch_scores.shape}, Time: {elapsed*1000:.2f}ms")
                         
        except Exception as e:
            self.log_test("Batch processing", False, str(e))
            self.results["issues"].append("Batch processing failed")
            
    def check_tools(self):
        """Check available profiling tools"""
        print("\n Checking Profiling Tools")
        print("-" * 50)
        
        tools = {
            'nvidia-smi': 'GPU monitoring',
            'nsys': 'System profiler',
            'ncu': 'Kernel profiler',
            'compute-sanitizer': 'Memory checker',
            'valgrind': 'CPU memory checker'
        }
        
        available = []
        for tool, desc in tools.items():
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode == 0:
                available.append(tool)
                print(f" {tool:20} - {desc}")
            else:
                print(f" {tool:20} - {desc}")
                
        if 'compute-sanitizer' not in available:
            self.results["recommendations"].append("Install CUDA toolkit for compute-sanitizer")
        if 'nsys' not in available:
            self.results["recommendations"].append("Install Nsight Systems for profiling")
            
        return available
        
    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 60)
        print(" ROBUSTNESS REPORT")
        print("=" * 60)
        
        # Count results
        passed = sum(1 for t in self.results["tests"] if t["passed"])
        total = len(self.results["tests"])
        
        print(f"\nTest Results: {passed}/{total} passed")
        
        if self.results["issues"]:
            print("\n Issues Found:")
            for issue in self.results["issues"]:
                print(f"  • {issue}")
        else:
            print("\n No critical issues found!")
            
        if self.results["recommendations"]:
            print("\n Recommendations:")
            for rec in self.results["recommendations"]:
                print(f"  • {rec}")
                
        # Add general recommendations
        print("\n Best Practices:")
        print("  • Run compute-sanitizer --tool memcheck regularly")
        print("  • Use nsys profile for performance analysis")
        print("  • Monitor GPU memory usage in production")
        print("  • Implement graceful OOM handling")
        print("  • Add telemetry for operation latencies")
        
        # Save report
        with open('robustness_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n📁 Report saved to robustness_report.json")
        
        return passed == total
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n GPU SEARCH ROBUSTNESS ANALYSIS")
        print("=" * 60)
        
        # Check environment
        print("\n Environment:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Run tests
        self.test_error_handling()
        self.test_memory_stability()
        self.test_large_scale()
        self.test_edge_values()
        self.test_batch_operations()
        
        # Check tools
        tools = self.check_tools()
        
        # Generate report
        all_passed = self.generate_report()
        
        if all_passed:
            print("\n System is robust and production-ready!")
        else:
            print("\n Some tests failed - review issues above")
            
        return 0 if all_passed else 1


def main():
    checker = RobustnessChecker()
    return checker.run_all_tests()


if __name__ == "__main__":
    exit(main())