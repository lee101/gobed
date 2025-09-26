#!/usr/bin/env python3
"""Benchmark end-to-end search performance: indexing + search throughput."""

import ctypes
import numpy as np
import time
import os
import sys
from typing import Tuple, List

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

class LibTorchFreeSearchEngine:
    def __init__(self):
        self.db_gpu = None
        self.db_size = 0

    def index(self, vectors: np.ndarray) -> float:
        """Index vectors and return indexing time in ms."""
        if vectors.dtype != np.int8:
            raise ValueError("Vectors must be int8")
        if vectors.shape[1] != 512:
            raise ValueError("Vectors must be 512-dimensional")

        start_time = time.perf_counter()

        # Free existing index
        if self.db_gpu:
            lib.cuda_free(self.db_gpu)

        self.db_size = vectors.shape[0]

        # Allocate GPU memory
        size_bytes = vectors.nbytes
        self.db_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(self.db_gpu), size_bytes)
        if result != 0:
            raise RuntimeError(f"Failed to allocate GPU memory: {result}")

        # Copy to GPU
        result = lib.cuda_memcpy_h2d(self.db_gpu, vectors.ctypes.data_as(ctypes.c_void_p), size_bytes)
        if result != 0:
            raise RuntimeError(f"Failed to copy to GPU: {result}")

        lib.cuda_synchronize()
        end_time = time.perf_counter()

        return (end_time - start_time) * 1000  # ms

    def search(self, query: np.ndarray, k: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
        """Search for k nearest neighbors. Returns (time_ms, indices, scores)."""
        if self.db_gpu is None:
            raise RuntimeError("No vectors indexed")

        if query.dtype != np.int8:
            raise ValueError("Query must be int8")
        if query.shape[0] != 512:
            raise ValueError("Query must be 512-dimensional")

        start_time = time.perf_counter()

        # Allocate GPU memory for query and results
        query_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(query_gpu), query.nbytes)
        if result != 0:
            raise RuntimeError(f"Failed to allocate query GPU memory: {result}")

        scores_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(scores_gpu), self.db_size * 4)  # int32
        if result != 0:
            raise RuntimeError(f"Failed to allocate scores GPU memory: {result}")

        try:
            # Copy query to GPU
            result = lib.cuda_memcpy_h2d(query_gpu, query.ctypes.data_as(ctypes.c_void_p), query.nbytes)
            if result != 0:
                raise RuntimeError(f"Failed to copy query to GPU: {result}")

            # Execute search
            result = lib.i8dot512_scores(query_gpu, self.db_gpu, scores_gpu, self.db_size)
            if result != 0:
                raise RuntimeError(f"Search failed: {result}")

            # Copy results back
            scores = np.zeros(self.db_size, dtype=np.int32)
            result = lib.cuda_memcpy_d2h(scores.ctypes.data_as(ctypes.c_void_p), scores_gpu, scores.nbytes)
            if result != 0:
                raise RuntimeError(f"Failed to copy scores back: {result}")

            lib.cuda_synchronize()
            end_time = time.perf_counter()

            # Get top-k indices (highest scores for dot product)
            top_k_indices = np.argpartition(scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
            top_k_scores = scores[top_k_indices]

            search_time = (end_time - start_time) * 1000  # ms

            return search_time, top_k_indices, top_k_scores

        finally:
            lib.cuda_free(query_gpu)
            lib.cuda_free(scores_gpu)

    def batch_search(self, queries: np.ndarray, k: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
        """Batch search for multiple queries."""
        if self.db_gpu is None:
            raise RuntimeError("No vectors indexed")

        if queries.dtype != np.int8:
            raise ValueError("Queries must be int8")
        if queries.shape[1] != 512:
            raise ValueError("Queries must be 512-dimensional")

        B = queries.shape[0]
        start_time = time.perf_counter()

        # Allocate GPU memory
        queries_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(queries_gpu), queries.nbytes)
        if result != 0:
            raise RuntimeError(f"Failed to allocate queries GPU memory: {result}")

        scores_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(scores_gpu), B * self.db_size * 4)  # int32
        if result != 0:
            raise RuntimeError(f"Failed to allocate batch scores GPU memory: {result}")

        try:
            # Copy queries to GPU
            result = lib.cuda_memcpy_h2d(queries_gpu, queries.ctypes.data_as(ctypes.c_void_p), queries.nbytes)
            if result != 0:
                raise RuntimeError(f"Failed to copy queries to GPU: {result}")

            # Execute batch search
            result = lib.i8dot512_batch(queries_gpu, self.db_gpu, scores_gpu, B, self.db_size)
            if result != 0:
                raise RuntimeError(f"Batch search failed: {result}")

            # Copy results back
            scores = np.zeros((B, self.db_size), dtype=np.int32)
            result = lib.cuda_memcpy_d2h(scores.ctypes.data_as(ctypes.c_void_p), scores_gpu, scores.nbytes)
            if result != 0:
                raise RuntimeError(f"Failed to copy batch scores back: {result}")

            lib.cuda_synchronize()
            end_time = time.perf_counter()

            # Get top-k for each query
            batch_indices = np.zeros((B, k), dtype=np.int64)
            batch_scores = np.zeros((B, k), dtype=np.int32)

            for i in range(B):
                top_k_idx = np.argpartition(scores[i], -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[i][top_k_idx])[::-1]]
                batch_indices[i] = top_k_idx
                batch_scores[i] = scores[i][top_k_idx]

            search_time = (end_time - start_time) * 1000  # ms

            return search_time, batch_indices, batch_scores

        finally:
            lib.cuda_free(queries_gpu)
            lib.cuda_free(scores_gpu)

    def __del__(self):
        if hasattr(self, 'db_gpu') and self.db_gpu:
            lib.cuda_free(self.db_gpu)

def benchmark_indexing_performance():
    """Benchmark indexing performance for different dataset sizes."""
    print("=== Indexing Performance Benchmark ===\n")

    sizes = [1000, 10000, 100000, 500000, 1000000]

    engine = LibTorchFreeSearchEngine()

    for N in sizes:
        print(f"📊 Indexing {N:,} vectors (512D, INT8)")

        # Generate random data
        vectors = np.random.randint(-128, 128, size=(N, 512), dtype=np.int8)
        data_size_mb = vectors.nbytes / (1024 * 1024)

        try:
            # Benchmark indexing
            index_time = engine.index(vectors)

            throughput = N / index_time  # vectors/ms
            bandwidth = data_size_mb / (index_time / 1000)  # MB/s

            print(f"  Time:       {index_time:.2f} ms")
            print(f"  Throughput: {throughput:.0f} vectors/ms ({throughput * 1000:.0f} vectors/s)")
            print(f"  Bandwidth:  {bandwidth:.1f} MB/s")
            print(f"  Data size:  {data_size_mb:.1f} MB")
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

def benchmark_search_performance():
    """Benchmark search performance for different scenarios."""
    print("=== Search Performance Benchmark ===\n")

    # Index a large dataset
    N = 100000
    print(f"Indexing {N:,} vectors for search benchmarks...")
    vectors = np.random.randint(-128, 128, size=(N, 512), dtype=np.int8)

    engine = LibTorchFreeSearchEngine()
    index_time = engine.index(vectors)
    print(f"Indexed in {index_time:.2f} ms\n")

    # Single query benchmarks
    k_values = [10, 50, 100, 500, 1000]

    for k in k_values:
        print(f"📊 Single query search (k={k})")

        query = np.random.randint(-128, 128, size=(512,), dtype=np.int8)

        # Warm up
        for _ in range(3):
            engine.search(query, k)

        # Benchmark
        times = []
        for _ in range(10):
            search_time, indices, scores = engine.search(query, k)
            times.append(search_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        qps = 1000 / avg_time  # queries per second

        print(f"  Time:      {avg_time:.3f} ± {std_time:.3f} ms")
        print(f"  QPS:       {qps:.0f} queries/second")
        print(f"  Scanned:   {N:,} vectors")
        print(f"  Retrieved: {len(indices)} results")
        print()

def benchmark_batch_search():
    """Benchmark batch search performance."""
    print("=== Batch Search Performance Benchmark ===\n")

    # Index dataset
    N = 50000
    vectors = np.random.randint(-128, 128, size=(N, 512), dtype=np.int8)

    engine = LibTorchFreeSearchEngine()
    index_time = engine.index(vectors)
    print(f"Indexed {N:,} vectors in {index_time:.2f} ms\n")

    batch_sizes = [1, 4, 8, 16, 32, 64]
    k = 100

    for B in batch_sizes:
        print(f"📊 Batch search: {B} queries, k={k}")

        queries = np.random.randint(-128, 128, size=(B, 512), dtype=np.int8)

        # Warm up
        for _ in range(3):
            engine.batch_search(queries, k)

        # Benchmark
        times = []
        for _ in range(5):
            search_time, indices, scores = engine.batch_search(queries, k)
            times.append(search_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        qps = B * 1000 / avg_time  # queries per second
        ops_per_sec = B * N * 1000 / avg_time  # vector comparisons per second

        print(f"  Time:         {avg_time:.3f} ± {std_time:.3f} ms")
        print(f"  QPS:          {qps:.0f} queries/second")
        print(f"  Comparisons:  {ops_per_sec:.0f} ops/second")
        print(f"  Per query:    {avg_time/B:.3f} ms/query")
        print()

def benchmark_throughput_scaling():
    """Benchmark throughput scaling with dataset size."""
    print("=== Throughput Scaling Benchmark ===\n")

    dataset_sizes = [10000, 50000, 100000, 500000, 1000000]
    batch_size = 8
    k = 100

    for N in dataset_sizes:
        print(f"📊 Dataset: {N:,} vectors, batch: {batch_size}, k={k}")

        vectors = np.random.randint(-128, 128, size=(N, 512), dtype=np.int8)
        queries = np.random.randint(-128, 128, size=(batch_size, 512), dtype=np.int8)

        engine = LibTorchFreeSearchEngine()

        try:
            # Index
            index_time = engine.index(vectors)

            # Search (with warmup)
            for _ in range(2):
                engine.batch_search(queries, k)

            search_time, _, _ = engine.batch_search(queries, k)

            # Calculate metrics
            qps = batch_size * 1000 / search_time
            ops_per_sec = batch_size * N * 1000 / search_time
            throughput_per_query = N * 1000 / search_time  # vectors/second per query

            print(f"  Index time:   {index_time:.2f} ms")
            print(f"  Search time:  {search_time:.2f} ms")
            print(f"  QPS:          {qps:.0f}")
            print(f"  Ops/sec:      {ops_per_sec:.0f}")
            print(f"  Throughput:   {throughput_per_query:.0f} vectors/s per query")
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

def main():
    print("🚀 LibTorch-Free GPU Search Engine Benchmark")
    print("=" * 60)
    print()

    try:
        # Run all benchmarks
        benchmark_indexing_performance()
        benchmark_search_performance()
        benchmark_batch_search()
        benchmark_throughput_scaling()

        print("✅ All benchmarks completed!")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()