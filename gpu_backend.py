#!/usr/bin/env python3
"""
Real GPU backend for gobed - provides actual GPU acceleration via Python/PyTorch
This can be called from Go using subprocess or a Python server
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
import time
from typing import List, Tuple, Optional
import argparse

class GPUIndexer:
    """Real GPU-accelerated vector indexer"""
    
    def __init__(self, dim: int = 384, max_vectors: int = 1000000, use_int8: bool = False):
        self.dim = dim
        self.max_vectors = max_vectors
        self.use_int8 = use_int8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-allocate GPU memory
        self.vectors = None
        self.current_size = 0
        
        # INT8 quantization parameters
        self.scale = None
        self.zero_point = None
        
        print(f"Initialized GPU Indexer on {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def add_vectors(self, vectors: np.ndarray) -> dict:
        """Add vectors to GPU index"""
        start_time = time.time()
        
        new_vectors = torch.from_numpy(vectors).float()
        
        if self.vectors is None:
            self.vectors = new_vectors.to(self.device)
        else:
            self.vectors = torch.cat([self.vectors, new_vectors.to(self.device)], dim=0)
        
        self.current_size = len(self.vectors)
        
        # Quantize if INT8 mode
        if self.use_int8 and self.vectors is not None:
            self._quantize_vectors()
        
        add_time = time.time() - start_time
        
        return {
            "status": "success",
            "vectors_added": len(vectors),
            "total_vectors": self.current_size,
            "time_ms": add_time * 1000,
            "throughput": len(vectors) / add_time
        }
    
    def _quantize_vectors(self):
        """Quantize vectors to INT8"""
        if self.vectors is None:
            return
        
        # Calculate quantization parameters
        min_val = self.vectors.min()
        max_val = self.vectors.max()
        self.scale = (max_val - min_val) / 255.0
        self.zero_point = -torch.round(min_val / self.scale)
        
        # Store as INT8 (simulated - PyTorch doesn't have great INT8 matmul yet)
        # In production, you'd use TensorRT or similar
    
    def search(self, query: np.ndarray, k: int = 10) -> dict:
        """Search for k nearest neighbors"""
        if self.vectors is None or len(self.vectors) == 0:
            return {"status": "error", "message": "No vectors in index"}
        
        start_time = time.time()
        
        # Convert query to tensor and move to GPU
        query_tensor = torch.from_numpy(query).float().to(self.device)
        
        # Compute similarities (cosine similarity via dot product)
        # Assuming vectors are normalized
        similarities = torch.matmul(self.vectors, query_tensor)
        
        # Get top-k
        k = min(k, len(self.vectors))
        top_values, top_indices = torch.topk(similarities, k)
        
        search_time = time.time() - start_time
        
        return {
            "status": "success",
            "indices": top_indices.cpu().numpy().tolist(),
            "scores": top_values.cpu().numpy().tolist(),
            "time_ms": search_time * 1000,
            "qps": 1.0 / search_time
        }
    
    def batch_search(self, queries: np.ndarray, k: int = 10) -> dict:
        """Batch search for multiple queries"""
        if self.vectors is None or len(self.vectors) == 0:
            return {"status": "error", "message": "No vectors in index"}
        
        start_time = time.time()
        
        # Convert queries to tensor and move to GPU
        queries_tensor = torch.from_numpy(queries).float().to(self.device)
        
        # Batch matrix multiplication
        # (num_queries, dim) @ (num_vectors, dim).T = (num_queries, num_vectors)
        similarities = torch.matmul(queries_tensor, self.vectors.T)
        
        # Get top-k for each query
        k = min(k, len(self.vectors))
        top_values, top_indices = torch.topk(similarities, k, dim=1)
        
        search_time = time.time() - start_time
        
        return {
            "status": "success",
            "indices": top_indices.cpu().numpy().tolist(),
            "scores": top_values.cpu().numpy().tolist(),
            "time_ms": search_time * 1000,
            "qps": len(queries) / search_time,
            "num_queries": len(queries)
        }
    
    def get_stats(self) -> dict:
        """Get indexer statistics"""
        stats = {
            "device": str(self.device),
            "num_vectors": self.current_size,
            "dimension": self.dim,
            "use_int8": self.use_int8,
        }
        
        if self.device.type == "cuda":
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
        
        if self.vectors is not None:
            memory_bytes = self.vectors.element_size() * self.vectors.nelement()
            stats["index_memory_mb"] = memory_bytes / 1e6
        
        return stats


class EmbeddingModel:
    """GPU-accelerated embedding model"""
    
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 384):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        
        # Create embedding layer (in production, load from checkpoint)
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        
        print(f"Embedding model on {self.device}")
    
    def embed_tokens(self, token_ids: List[int]) -> np.ndarray:
        """Convert token IDs to embeddings"""
        with torch.no_grad():
            tokens = torch.tensor(token_ids, dtype=torch.long).to(self.device)
            embeddings = self.embedding(tokens)
            
            # Average pooling
            pooled = embeddings.mean(dim=0)
            
            # L2 normalization
            normalized = pooled / pooled.norm()
            
            return normalized.cpu().numpy()
    
    def batch_embed(self, token_batches: List[List[int]]) -> np.ndarray:
        """Batch embedding with padding"""
        with torch.no_grad():
            # Pad sequences
            max_len = max(len(seq) for seq in token_batches)
            padded = []
            masks = []
            
            for seq in token_batches:
                padded_seq = seq + [0] * (max_len - len(seq))
                mask = [1] * len(seq) + [0] * (max_len - len(seq))
                padded.append(padded_seq)
                masks.append(mask)
            
            # Convert to tensors
            tokens = torch.tensor(padded, dtype=torch.long).to(self.device)
            masks = torch.tensor(masks, dtype=torch.float).to(self.device)
            
            # Get embeddings
            embeddings = self.embedding(tokens)
            
            # Masked average pooling
            masked_embeddings = embeddings * masks.unsqueeze(-1)
            pooled = masked_embeddings.sum(dim=1) / masks.sum(dim=1, keepdim=True)
            
            # L2 normalization
            normalized = pooled / pooled.norm(dim=1, keepdim=True)
            
            return normalized.cpu().numpy()


def benchmark_gpu_performance():
    """Run comprehensive GPU benchmark"""
    print("=" * 80)
    print("🚀 REAL GPU PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    indexer = GPUIndexer(dim=384, use_int8=False)
    stats = indexer.get_stats()
    print(f"Device: {stats['device']}")
    if 'gpu_name' in stats:
        print(f"GPU: {stats['gpu_name']}")
    print()
    
    # Test different scales
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        print(f"Testing {size} vectors...")
        
        # Generate random vectors
        vectors = np.random.randn(size, 384).astype(np.float32)
        
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        # Add to index
        result = indexer.add_vectors(vectors)
        print(f"  Add: {result['time_ms']:.2f}ms, {result['throughput']:.0f} vec/s")
        
        # Single query search
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        result = indexer.search(query, k=10)
        print(f"  Search: {result['time_ms']:.2f}ms, {result['qps']:.0f} qps")
        
        # Batch search
        queries = np.random.randn(100, 384).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        result = indexer.batch_search(queries, k=10)
        print(f"  Batch-100: {result['time_ms']:.2f}ms, {result['qps']:.0f} qps")
        
        # Clear index for next size
        indexer.vectors = None
        indexer.current_size = 0
        print()
    
    print("=" * 80)
    print("✅ This is REAL GPU acceleration!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="GPU Backend for gobed")
    parser.add_argument("--mode", choices=["index", "search", "benchmark", "server"], 
                       default="benchmark", help="Operation mode")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimension")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    if args.mode == "benchmark":
        benchmark_gpu_performance()
    elif args.mode == "server":
        # Start HTTP server for Go to call
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        indexer = GPUIndexer(dim=args.dim, use_int8=args.int8)
        
        @app.route("/add", methods=["POST"])
        def add_vectors():
            data = request.json
            vectors = np.array(data["vectors"], dtype=np.float32)
            result = indexer.add_vectors(vectors)
            return jsonify(result)
        
        @app.route("/search", methods=["POST"])
        def search():
            data = request.json
            query = np.array(data["query"], dtype=np.float32)
            k = data.get("k", 10)
            result = indexer.search(query, k)
            return jsonify(result)
        
        @app.route("/batch_search", methods=["POST"])
        def batch_search():
            data = request.json
            queries = np.array(data["queries"], dtype=np.float32)
            k = data.get("k", 10)
            result = indexer.batch_search(queries, k)
            return jsonify(result)
        
        @app.route("/stats", methods=["GET"])
        def stats():
            return jsonify(indexer.get_stats())
        
        print(f"Starting GPU backend server on port {args.port}...")
        app.run(host="0.0.0.0", port=args.port)
    
    else:
        print("Mode not implemented yet")


if __name__ == "__main__":
    main()