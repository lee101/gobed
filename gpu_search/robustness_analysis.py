#!/usr/bin/env python3
"""
Comprehensive robustness analysis and improvement script for GPU search
"""

import torch
import numpy as np
import sys
import os
import time
import subprocess
import psutil
import traceback
from typing import List, Dict, Any, Optional
import json

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')


class RobustnessAnalyzer:
    """Analyze and improve robustness of GPU search system"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.issues = []
        self.recommendations = []
        
    def check_cuda_environment(self):
        """Check CUDA environment and compatibility"""
        print("\n CUDA Environment Analysis")
        print("=" * 60)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.issues.append("CUDA not available - GPU operations will fail")
            return False
            
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        cuda_version = torch.version.cuda
        
        print(f" GPU: {device_props.name}")
        print(f" CUDA Version: {cuda_version}")
        print(f" Compute Capability: {device_props.major}.{device_props.minor}")
        print(f" Memory: {device_props.total_memory / 1e9:.1f} GB")
        
        # Check for __dp4a support (required for INT8 operations)
        dp4a_support = (device_props.major > 6) or (device_props.major == 6 and device_props.minor >= 1)
        if not dp4a_support:
            self.issues.append(f"GPU lacks __dp4a support (CC {device_props.major}.{device_props.minor} < 6.1)")
            self.recommendations.append("Use GPU with compute capability >= 6.1 for INT8 operations")
        else:
            print(f" __dp4a INT8 support: Available")
            
        # Check CUDA/PyTorch compatibility
        if cuda_version:
            major, minor = map(int, cuda_version.split('.')[:2])
            if major < 11:
                self.recommendations.append("Consider upgrading to CUDA 11+ for better performance")
                
        return True
        
    def test_error_handling(self):
        """Test error handling for various edge cases"""
        print("\n Error Handling Tests")
        print("=" * 60)
        
        test_cases = []
        
        # Test 1: Wrong dtype
        print("Testing wrong dtype handling...")
        try:
            q = torch.randn(512, device=self.device)  # float32 instead of int8
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            test_cases.append(("Wrong dtype", False, "Should have raised error"))
        except Exception as e:
            test_cases.append(("Wrong dtype", True, "Correctly rejected"))
            
        # Test 2: Wrong dimensions
        print("Testing dimension validation...")
        try:
            q = torch.randint(-128, 127, (256,), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            test_cases.append(("Wrong dimensions", False, "Should have raised error"))
        except Exception as e:
            test_cases.append(("Wrong dimensions", True, "Correctly rejected"))
            
        # Test 3: Empty database
        print("Testing empty database handling...")
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.empty((0, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            if result.shape[0] == 0:
                test_cases.append(("Empty database", True, "Handled correctly"))
            else:
                test_cases.append(("Empty database", False, "Unexpected result shape"))
        except Exception as e:
            test_cases.append(("Empty database", False, f"Failed: {str(e)}"))
            
        # Test 4: Very large database
        print("Testing large database handling...")
        try:
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            # Test with 10M vectors (should work on 16GB GPU)
            db = torch.randint(-128, 127, (10_000_000, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            test_cases.append(("Large database (10M)", True, "Handled successfully"))
        except torch.cuda.OutOfMemoryError:
            test_cases.append(("Large database (10M)", False, "Out of memory"))
            self.recommendations.append("Implement batched processing for large databases")
        except Exception as e:
            test_cases.append(("Large database (10M)", False, str(e)))
            
        # Test 5: Non-contiguous tensors
        print("Testing non-contiguous tensor handling...")
        try:
            q = torch.randint(-128, 127, (1024,), dtype=torch.int8, device=self.device)[::2]
            db = torch.randint(-128, 127, (100, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            test_cases.append(("Non-contiguous", False, "Should have raised error"))
        except Exception as e:
            test_cases.append(("Non-contiguous", True, "Correctly rejected"))
            
        # Print results
        print("\n Error Handling Results:")
        for test_name, passed, details in test_cases:
            status = "" if passed else ""
            print(f"  {status} {test_name}: {details}")
            
        failed_tests = [t for t, p, _ in test_cases if not p]
        if failed_tests:
            self.issues.append(f"Failed error handling tests: {', '.join(failed_tests)}")
            
    def test_memory_patterns(self):
        """Test for memory leaks and inefficient patterns"""
        print("\n Memory Pattern Analysis")
        print("=" * 60)
        
        # Get initial memory state
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        print(f"Initial GPU memory: {initial_memory / 1e6:.1f} MB")
        
        # Run operations in a loop to detect leaks
        print("Running memory leak test (100 iterations)...")
        memory_samples = []
        
        for i in range(100):
            q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
            db = torch.randint(-128, 127, (10000, 512), dtype=torch.int8, device=self.device)
            result = torch.ops.gobed_ann.i8dot512_scores(q, db)
            
            if i % 10 == 0:
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated()
                memory_samples.append(current_memory)
                
            # Cleanup
            del q, db, result
            
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Analyze memory pattern
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples)
        
        print(f"Final GPU memory: {final_memory / 1e6:.1f} MB")
        print(f"Memory growth: {memory_growth / 1e6:.1f} MB")
        print(f"Peak memory: {max_memory / 1e6:.1f} MB")
        
        if memory_growth > 10 * 1e6:  # More than 10MB growth
            self.issues.append(f"Potential memory leak detected: {memory_growth / 1e6:.1f} MB growth")
            self.recommendations.append("Review CUDA kernel memory management")
        else:
            print(" No significant memory leaks detected")
            
    def test_concurrent_operations(self):
        """Test thread safety and concurrent operations"""
        print("\n🔄 Concurrent Operations Test")
        print("=" * 60)
        
        import threading
        import queue
        
        errors = queue.Queue()
        results = queue.Queue()
        
        def worker(thread_id):
            try:
                for _ in range(10):
                    q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
                    db = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=self.device)
                    result = torch.ops.gobed_ann.i8dot512_scores(q, db)
                    results.put((thread_id, result.shape))
            except Exception as e:
                errors.put((thread_id, str(e)))
                
        # Launch concurrent threads
        threads = []
        num_threads = 4
        print(f"Launching {num_threads} concurrent threads...")
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
            
        # Check results
        if errors.empty():
            print(f" All {num_threads} threads completed successfully")
            print(f" Total operations: {results.qsize()}")
        else:
            while not errors.empty():
                thread_id, error = errors.get()
                self.issues.append(f"Thread {thread_id} failed: {error}")
                
    def profile_performance(self):
        """Profile GPU performance characteristics"""
        print("\n Performance Profiling")
        print("=" * 60)
        
        sizes = [100, 1000, 10000, 100000, 1000000]
        results = []
        
        for size in sizes:
            try:
                q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=self.device)
                db = torch.randint(-128, 127, (size, 512), dtype=torch.int8, device=self.device)
                
                # Warmup
                for _ in range(3):
                    _ = torch.ops.gobed_ann.i8dot512_scores(q, db)
                    
                torch.cuda.synchronize()
                
                # Benchmark
                start = time.perf_counter()
                for _ in range(10):
                    _ = torch.ops.gobed_ann.i8dot512_scores(q, db)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / 10
                
                throughput = size / elapsed
                results.append((size, elapsed * 1000, throughput))
                print(f"  Size {size:8d}: {elapsed*1000:6.2f} ms, {throughput/1e6:.2f} M vec/s")
                
            except Exception as e:
                print(f"  Size {size:8d}: Failed - {str(e)}")
                
        # Check for performance issues
        if results:
            # Check if throughput scales linearly
            if len(results) > 1:
                throughputs = [t for _, _, t in results]
                if max(throughputs) / min(throughputs) > 10:
                    self.recommendations.append("Performance scaling issues detected - review kernel efficiency")
                    
    def check_nvidia_tools(self):
        """Check available NVIDIA profiling tools"""
        print("\n NVIDIA Tool Availability")
        print("=" * 60)
        
        tools = {
            'nvidia-smi': 'GPU monitoring',
            'nvprof': 'Legacy profiler (deprecated)',
            'ncu': 'Nsight Compute (kernel profiler)',
            'nsys': 'Nsight Systems (system profiler)',
            'cuda-memcheck': 'Memory checker',
            'compute-sanitizer': 'Modern memory/race checker'
        }
        
        available_tools = []
        
        for tool, description in tools.items():
            try:
                result = subprocess.run(['which', tool], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f" {tool:20s} - {description}")
                    available_tools.append(tool)
                else:
                    print(f" {tool:20s} - {description}")
            except:
                print(f" {tool:20s} - {description}")
                
        if 'nsys' in available_tools:
            self.recommendations.append("Use 'nsys profile python your_script.py' for system-wide profiling")
        if 'ncu' in available_tools:
            self.recommendations.append("Use 'ncu --target-processes all python your_script.py' for kernel profiling")
        if 'compute-sanitizer' in available_tools:
            self.recommendations.append("Use 'compute-sanitizer --tool memcheck python your_script.py' for memory checking")
        else:
            self.recommendations.append("Install CUDA toolkit for access to profiling tools")
            
    def generate_report(self):
        """Generate comprehensive robustness report"""
        print("\n" + "=" * 60)
        print(" ROBUSTNESS ANALYSIS REPORT")
        print("=" * 60)
        
        if self.issues:
            print("\n Issues Found:")
            for issue in self.issues:
                print(f"  • {issue}")
        else:
            print("\n No critical issues found")
            
        if self.recommendations:
            print("\n Recommendations:")
            for rec in self.recommendations:
                print(f"  • {rec}")
                
        # Save report
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'issues': self.issues,
            'recommendations': self.recommendations,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None
        }
        
        with open('robustness_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\n📁 Report saved to robustness_report.json")
        
    def run_full_analysis(self):
        """Run complete robustness analysis"""
        print("\n Starting Comprehensive Robustness Analysis")
        print("=" * 60)
        
        if not self.check_cuda_environment():
            print(" CUDA environment check failed - aborting")
            return
            
        self.test_error_handling()
        self.test_memory_patterns()
        self.test_concurrent_operations()
        self.profile_performance()
        self.check_nvidia_tools()
        self.generate_report()


def main():
    analyzer = RobustnessAnalyzer()
    analyzer.run_full_analysis()
    
    # Additional valgrind command suggestion
    print("\n For CPU memory leak detection with valgrind:")
    print("   valgrind --leak-check=full --show-leak-kinds=all \\")
    print("            --track-origins=yes --verbose \\")
    print("            python your_script.py")
    
    print("\n For GPU memory checking:")
    print("   compute-sanitizer --tool memcheck python your_script.py")
    print("   compute-sanitizer --tool racecheck python your_script.py")
    print("   compute-sanitizer --tool initcheck python your_script.py")


if __name__ == "__main__":
    main()