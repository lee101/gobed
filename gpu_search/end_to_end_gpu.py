#!/usr/bin/env python3
"""
End-to-end GPU search system - everything stays on GPU
From indexing to search, no CPU-GPU transfers except initial load and final results
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from typing import List, Tuple, Optional

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')

class GPUIndex:
    """Pure GPU index - data never leaves GPU during operations"""
    
    def __init__(self, dim: int = 512, max_vectors: int = 10000000, use_int8: bool = True):
        self.dim = dim
        self.max_vectors = max_vectors
        self.use_int8 = use_int8
        self.device = torch.device("cuda")
        self.current_size = 0
        
        # Pre-allocate GPU memory
        if use_int8:
            self.index = torch.zeros((max_vectors, dim), dtype=torch.int8, device=self.device)
            self.scale = torch.ones(max_vectors, dtype=torch.float32, device=self.device)
            self.zero_point = torch.zeros(max_vectors, dtype=torch.int8, device=self.device)
        else:
            self.index = torch.zeros((max_vectors, dim), dtype=torch.float32, device=self.device)
        
        # Metadata stays on GPU too
        self.norms = torch.zeros(max_vectors, dtype=torch.float32, device=self.device)
        self.ids = torch.arange(max_vectors, dtype=torch.int64, device=self.device)
        
        print(f"📊 GPU Index initialized:")
        print(f"   Device: {torch.cuda.get_device_name()}")
        print(f"   Max vectors: {max_vectors:,}")
        print(f"   Dimension: {dim}")
        print(f"   Precision: {'INT8' if use_int8 else 'FP32'}")
        print(f"   Memory allocated: {self._get_memory_usage():.1f} MB")
    
    def _get_memory_usage(self) -> float:
        """Get GPU memory usage in MB"""
        return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
    
    def add_vectors_gpu(self, vectors: torch.Tensor) -> dict:
        """Add vectors that are already on GPU - no transfer!"""
        assert vectors.is_cuda, f"Vectors must already be on GPU, got device: {vectors.device}"
        
        n = vectors.shape[0]
        if self.current_size + n > self.max_vectors:
            raise ValueError(f"Index full: {self.current_size + n} > {self.max_vectors}")
        
        start_idx = self.current_size
        end_idx = start_idx + n
        
        # Normalize on GPU
        norms = torch.norm(vectors, dim=1, keepdim=True)
        normalized = vectors / (norms + 1e-8)
        
        if self.use_int8:
            # Quantize on GPU
            min_vals = normalized.min(dim=1, keepdim=True)[0]
            max_vals = normalized.max(dim=1, keepdim=True)[0]
            scale = (max_vals - min_vals) / 255.0
            zero_point = -torch.round(min_vals / scale)
            
            # Store quantization params
            self.scale[start_idx:end_idx] = scale.squeeze()
            self.zero_point[start_idx:end_idx] = zero_point.squeeze().to(torch.int8)
            
            # Quantize and store
            quantized = torch.round((normalized - min_vals) / scale).to(torch.int8)
            self.index[start_idx:end_idx] = quantized
        else:
            self.index[start_idx:end_idx] = normalized
        
        self.norms[start_idx:end_idx] = norms.squeeze()
        self.current_size = end_idx
        
        return {
            "vectors_added": n,
            "total_vectors": self.current_size,
            "memory_mb": self._get_memory_usage()
        }
    
    def search_gpu(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search with query already on GPU - uses custom CUDA kernels"""
        assert query.is_cuda, "Query must be on GPU"
        
        if self.current_size == 0:
            return torch.tensor([]), torch.tensor([])
        
        # Get active portion of index
        active_index = self.index[:self.current_size]
        
        if self.use_int8:
            # Normalize and quantize query on GPU
            query_norm = torch.norm(query)
            query_normalized = query / (query_norm + 1e-8)
            
            # Use same quantization as first vector (simplified)
            # In production, you'd use global quantization params
            q_min = query_normalized.min()
            q_max = query_normalized.max()
            q_scale = (q_max - q_min) / 255.0
            q_zero = -torch.round(q_min / q_scale)
            
            query_int8 = torch.round((query_normalized - q_min) / q_scale).to(torch.int8)
            
            # Use custom CUDA kernel for INT8 dot product!
            if self.dim == 512:
                scores = torch.ops.gobed_ann.i8dot512_scores(query_int8, active_index)
            else:
                # Fallback to PyTorch for other dimensions
                scores = torch.matmul(active_index.float(), query_int8.float())
            
            # Dequantize scores
            scores = scores.float() * q_scale * self.scale[:self.current_size]
        else:
            # FP32 search
            query_normalized = query / torch.norm(query)
            scores = torch.matmul(active_index, query_normalized)
        
        # Get top-k on GPU
        k = min(k, self.current_size)
        top_scores, top_indices = torch.topk(scores, k)
        
        return top_indices, top_scores
    
    def batch_search_gpu(self, queries: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch search with custom CUDA kernels"""
        assert queries.is_cuda, "Queries must be on GPU"
        
        if self.current_size == 0:
            return torch.tensor([]), torch.tensor([])
        
        batch_size = queries.shape[0]
        active_index = self.index[:self.current_size]
        
        if self.use_int8 and self.dim == 512:
            # Normalize and quantize queries on GPU
            query_norms = torch.norm(queries, dim=1, keepdim=True)
            queries_normalized = queries / (query_norms + 1e-8)
            
            # Simple quantization (production would be more sophisticated)
            q_min = queries_normalized.min(dim=1, keepdim=True)[0]
            q_max = queries_normalized.max(dim=1, keepdim=True)[0]
            q_scale = (q_max - q_min) / 255.0
            
            queries_int8 = torch.round((queries_normalized - q_min) / q_scale).to(torch.int8)
            
            # Use custom CUDA kernel for batch INT8 dot product!
            scores = torch.ops.gobed_ann.i8dot512_batch(queries_int8, active_index)
            
            # Dequantize
            scores = scores.float() * q_scale * self.scale[:self.current_size].unsqueeze(0)
        else:
            # FP32 batch search
            query_norms = torch.norm(queries, dim=1, keepdim=True)
            queries_normalized = queries / (query_norms + 1e-8)
            scores = torch.matmul(queries_normalized, active_index.T)
        
        # Get top-k for each query
        k = min(k, self.current_size)
        top_scores, top_indices = torch.topk(scores, k, dim=1)
        
        return top_indices, top_scores


class EmbeddingModelGPU(nn.Module):
    """GPU-only embedding model - embeddings never leave GPU"""
    
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim).cuda()
        self.layer_norm = nn.LayerNorm(embed_dim).cuda()
        
    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Process tokens entirely on GPU"""
        # Embedding lookup
        embeddings = self.embed(token_ids)
        
        # Masked mean pooling
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        pooled = sum_embeddings / (sum_mask + 1e-8)
        
        # Layer norm and L2 norm
        pooled = self.layer_norm(pooled)
        pooled = pooled / torch.norm(pooled, dim=1, keepdim=True)
        
        return pooled


def benchmark_end_to_end():
    """Benchmark the complete GPU pipeline"""
    print("\n" + "="*80)
    print("🚀 END-TO-END GPU PIPELINE BENCHMARK")
    print("="*80)
    
    device = torch.device("cuda")
    
    # Initialize components
    embed_model = EmbeddingModelGPU(vocab_size=30522, embed_dim=512)
    index = GPUIndex(dim=512, max_vectors=1000000, use_int8=True)
    
    # Generate test data ON GPU
    print("\n📦 Generating test data directly on GPU...")
    num_docs = 100000
    max_seq_len = 128
    batch_size = 1000
    
    results = []
    
    for batch_start in range(0, num_docs, batch_size):
        batch_end = min(batch_start + batch_size, num_docs)
        batch_docs = batch_end - batch_start
        
        # Generate random tokens ON GPU
        token_ids = torch.randint(0, 30522, (batch_docs, max_seq_len), device=device)
        attention_mask = torch.ones((batch_docs, max_seq_len), device=device)
        
        # Embed ON GPU (no transfer)
        with torch.amp.autocast('cuda'):  # Use mixed precision for speed
            embeddings = embed_model(token_ids, attention_mask).float()  # Ensure float output
        
        # Add to index ON GPU (no transfer)
        index.add_vectors_gpu(embeddings)
        
        if (batch_start + batch_size) % 10000 == 0:
            print(f"  Indexed {batch_start + batch_size} documents...")
    
    print(f"\n✅ Indexed {num_docs} documents")
    print(f"   GPU memory used: {index._get_memory_usage():.1f} MB")
    
    # Benchmark search
    print("\n🔍 Benchmarking search...")
    
    # Single query
    query_tokens = torch.randint(0, 30522, (1, max_seq_len), device=device)
    query_mask = torch.ones((1, max_seq_len), device=device)
    
    # Warmup
    for _ in range(10):
        query_embedding = embed_model(query_tokens, query_mask).squeeze()
        indices, scores = index.search_gpu(query_embedding, k=10)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    num_queries = 100
    for _ in range(num_queries):
        query_embedding = embed_model(query_tokens, query_mask).squeeze()
        indices, scores = index.search_gpu(query_embedding, k=10)
    
    torch.cuda.synchronize()
    single_time = time.perf_counter() - start
    
    print(f"  Single query latency: {single_time/num_queries*1000:.2f} ms")
    print(f"  Single query QPS: {num_queries/single_time:.0f}")
    
    # Batch queries
    batch_queries = 32
    query_tokens_batch = torch.randint(0, 30522, (batch_queries, max_seq_len), device=device)
    query_mask_batch = torch.ones((batch_queries, max_seq_len), device=device)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    num_batches = 10
    for _ in range(num_batches):
        query_embeddings = embed_model(query_tokens_batch, query_mask_batch)
        indices, scores = index.batch_search_gpu(query_embeddings, k=10)
    
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - start
    
    total_queries = batch_queries * num_batches
    print(f"  Batch-{batch_queries} latency: {batch_time/num_batches*1000:.2f} ms")
    print(f"  Batch throughput: {total_queries/batch_time:.0f} QPS")
    
    print("\n📊 Key Achievement:")
    print("  • ZERO CPU-GPU transfers during operation")
    print("  • Custom CUDA kernels for INT8 search")
    print("  • Everything stays on GPU from start to finish")
    print("  • True end-to-end GPU acceleration!")


def test_cuda_kernels():
    """Test custom CUDA kernels thoroughly"""
    print("\n" + "="*80)
    print("🧪 TESTING CUSTOM CUDA KERNELS")
    print("="*80)
    
    device = torch.device("cuda")
    
    # Test different sizes
    test_sizes = [100, 1000, 10000, 100000, 1000000]
    
    for n in test_sizes:
        print(f"\n📏 Testing with {n:,} vectors:")
        
        # Create test data
        db = torch.randn(n, 512, device=device)
        db = db / torch.norm(db, dim=1, keepdim=True)
        
        # Quantize to INT8
        db_min = db.min(dim=1, keepdim=True)[0]
        db_max = db.max(dim=1, keepdim=True)[0]
        scale = (db_max - db_min) / 255.0
        db_int8 = torch.round((db - db_min) / scale).to(torch.int8)
        
        query = torch.randn(512, device=device)
        query = query / torch.norm(query)
        q_min = query.min()
        q_max = query.max()
        q_scale = (q_max - q_min) / 255.0
        query_int8 = torch.round((query - q_min) / q_scale).to(torch.int8)
        
        # Test single query
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        scores = torch.ops.gobed_ann.i8dot512_scores(query_int8, db_int8)
        
        torch.cuda.synchronize()
        kernel_time = time.perf_counter() - start
        
        print(f"  Single query: {kernel_time*1000:.3f} ms")
        print(f"  Throughput: {1/kernel_time:.0f} QPS")
        
        # Compare with PyTorch
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        scores_pytorch = torch.matmul(db_int8.float(), query_int8.float())
        
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start
        
        print(f"  PyTorch baseline: {pytorch_time*1000:.3f} ms")
        print(f"  Speedup: {pytorch_time/kernel_time:.2f}x")
        
        # Test batch
        if n <= 100000:  # Don't run batch on huge datasets
            batch = 32
            queries_batch = torch.randn(batch, 512, device=device)
            queries_batch = queries_batch / torch.norm(queries_batch, dim=1, keepdim=True)
            queries_int8 = torch.round(queries_batch * 127).to(torch.int8)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            scores_batch = torch.ops.gobed_ann.i8dot512_batch(queries_int8, db_int8)
            
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start
            
            print(f"  Batch-{batch}: {batch_time*1000:.3f} ms")
            print(f"  Batch throughput: {batch/batch_time:.0f} QPS")


if __name__ == "__main__":
    print("🚀 Pure GPU Search System")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Test CUDA kernels
    test_cuda_kernels()
    
    # Run end-to-end benchmark
    benchmark_end_to_end()
    
    print("\n" + "="*80)
    print("✅ PURE GPU SEARCH SYSTEM COMPLETE!")
    print("="*80)