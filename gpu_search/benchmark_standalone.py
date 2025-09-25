#!/usr/bin/env python3
"""Benchmark the LibTorch-free CUDA library against the original LibTorch version."""

import ctypes
import numpy as np
import time
import os
import sys
from typing import Tuple

# Load the LibTorch-free library
lib_path = os.path.join(os.path.dirname(__file__), 'cuda_ops/build/libgobed_ann_ops.so')
if not os.path.exists(lib_path):
    print(f"❌ Library not found: {lib_path}")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)

# Define function signatures
lib.cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cuda_malloc.restype = ctypes.c_int

lib.cuda_free.argtypes = [ctypes.c_void_p]
lib.cuda_free.restype = ctypes.c_int

lib.cuda_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuda_memcpy_h2d.restype = ctypes.c_int

lib.cuda_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuda_memcpy_d2h.restype = ctypes.c_int

lib.cuda_synchronize.argtypes = []
lib.cuda_synchronize.restype = ctypes.c_int

lib.i8dot512_scores.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.i8dot512_scores.restype = ctypes.c_int

lib.i8dot512_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
lib.i8dot512_batch.restype = ctypes.c_int

def cuda_malloc(size: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    result = lib.cuda_malloc(ctypes.byref(ptr), size)
    if result != 0:
        raise RuntimeError(f"cuda_malloc failed with error {result}")
    return ptr

def cuda_free(ptr: ctypes.c_void_p):
    result = lib.cuda_free(ptr)
    if result != 0:
        raise RuntimeError(f"cuda_free failed with error {result}")

def cuda_memcpy_h2d(dst: ctypes.c_void_p, src_array: np.ndarray):
    size = src_array.nbytes
    result = lib.cuda_memcpy_h2d(dst, src_array.ctypes.data_as(ctypes.c_void_p), size)
    if result != 0:
        raise RuntimeError(f"cuda_memcpy_h2d failed with error {result}")

def cuda_memcpy_d2h(dst_array: np.ndarray, src: ctypes.c_void_p):
    size = dst_array.nbytes
    result = lib.cuda_memcpy_d2h(dst_array.ctypes.data_as(ctypes.c_void_p), src, size)
    if result != 0:
        raise RuntimeError(f"cuda_memcpy_d2h failed with error {result}")

def benchmark_standalone_ops(N: int, B: int = 1) -> Tuple[float, np.ndarray]:
    """Benchmark the LibTorch-free operations."""

    # Generate test data
    query = np.random.randint(-128, 128, size=(512,), dtype=np.int8)
    db = np.random.randint(-128, 128, size=(N, 512), dtype=np.int8)

    if B > 1:
        queries = np.random.randint(-128, 128, size=(B, 512), dtype=np.int8)
        result_shape = (B, N)
    else:
        result_shape = (N,)

    # Allocate device memory
    query_gpu = cuda_malloc(query.nbytes)
    db_gpu = cuda_malloc(db.nbytes)
    result_gpu = cuda_malloc(np.prod(result_shape) * 4)  # int32

    if B > 1:
        queries_gpu = cuda_malloc(queries.nbytes)
        cuda_memcpy_h2d(queries_gpu, queries)

    try:
        # Copy data to device
        cuda_memcpy_h2d(query_gpu, query)
        cuda_memcpy_h2d(db_gpu, db)

        # Warm up
        for _ in range(3):
            if B > 1:
                lib.i8dot512_batch(queries_gpu, db_gpu, result_gpu, B, N)
            else:
                lib.i8dot512_scores(query_gpu, db_gpu, result_gpu, N)
            lib.cuda_synchronize()

        # Benchmark
        start_time = time.perf_counter()

        for _ in range(10):
            if B > 1:
                result = lib.i8dot512_batch(queries_gpu, db_gpu, result_gpu, B, N)
            else:
                result = lib.i8dot512_scores(query_gpu, db_gpu, result_gpu, N)
            if result != 0:
                raise RuntimeError(f"CUDA operation failed with error {result}")

        lib.cuda_synchronize()
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10 * 1000  # ms

        # Copy results back
        results = np.zeros(result_shape, dtype=np.int32)
        cuda_memcpy_d2h(results, result_gpu)

        return avg_time, results

    finally:
        # Cleanup
        cuda_free(query_gpu)
        cuda_free(db_gpu)
        cuda_free(result_gpu)
        if B > 1:
            cuda_free(queries_gpu)

def benchmark_cpu_reference(queries: np.ndarray, db: np.ndarray) -> Tuple[float, np.ndarray]:
    """CPU reference implementation."""
    start_time = time.perf_counter()

    if queries.ndim == 1:
        # Single query
        results = np.dot(db.astype(np.int32), queries.astype(np.int32))
    else:
        # Batch queries
        results = np.dot(db.astype(np.int32), queries.astype(np.int32).T).T

    end_time = time.perf_counter()
    cpu_time = (end_time - start_time) * 1000  # ms

    return cpu_time, results

def main():
    print("=== LibTorch-Free CUDA Library Benchmark ===\n")

    # Test configurations
    configs = [
        (1000, 1, "1K vectors, single query"),
        (10000, 1, "10K vectors, single query"),
        (100000, 1, "100K vectors, single query"),
        (1000, 8, "1K vectors, batch of 8"),
        (10000, 8, "10K vectors, batch of 8"),
        (100000, 8, "100K vectors, batch of 8"),
    ]

    for N, B, description in configs:
        print(f"📊 {description}")
        print("-" * 50)

        try:
            # Generate test data for CPU comparison
            if B == 1:
                query = np.random.randint(-128, 128, size=(512,), dtype=np.int8)
                test_queries = query
            else:
                query = np.random.randint(-128, 128, size=(B, 512), dtype=np.int8)
                test_queries = query

            db = np.random.randint(-128, 128, size=(N, 512), dtype=np.int8)

            # Benchmark GPU (LibTorch-free)
            gpu_time, gpu_results = benchmark_standalone_ops(N, B)

            # Benchmark CPU reference
            cpu_time, cpu_results = benchmark_cpu_reference(test_queries, db)

            # Verify correctness
            if B == 1:
                matches = np.allclose(gpu_results, cpu_results, atol=1)
                diff = np.abs(gpu_results - cpu_results).max()
            else:
                matches = np.allclose(gpu_results, cpu_results, atol=1)
                diff = np.abs(gpu_results - cpu_results).max()

            # Calculate throughput
            total_ops = N * B
            gpu_throughput = total_ops / gpu_time  # ops/ms
            cpu_throughput = total_ops / cpu_time  # ops/ms
            speedup = cpu_time / gpu_time

            print(f"  GPU Time:      {gpu_time:.2f} ms")
            print(f"  CPU Time:      {cpu_time:.2f} ms")
            print(f"  Speedup:       {speedup:.1f}x")
            print(f"  GPU Throughput: {gpu_throughput:.0f} ops/ms")
            print(f"  CPU Throughput: {cpu_throughput:.0f} ops/ms")
            print(f"  Correctness:   {'✅ PASS' if matches else '❌ FAIL'}")
            print(f"  Max Diff:      {diff}")
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

if __name__ == "__main__":
    main()