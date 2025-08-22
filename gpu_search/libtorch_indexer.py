#!/usr/bin/env python3
"""
LibTorch-based indexing system with end-to-end functionality.
Implements high-performance GPU indexing and search using torch.h and custom CUDA ops.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Load our custom CUDA operations
torch.ops.load_library('./cuda_ops/build/libgobed_ann_ops.so')

@dataclass
class IndexConfig:
    """Configuration for the indexing system"""
    device: str = "cuda:0"
    batch_size: int = 1024
    num_subquantizers: int = 64  # M in PQ
    codebook_size: int = 256     # K in PQ (8-bit codes)
    vector_dim: int = 512        # D 
    ivf_clusters: int = 4096     # IVF clusters
    probe_lists: int = 64        # Lists to probe during search
    rerank_k: int = 1000         # Candidates to rerank

class LibTorchIndexer(nn.Module):
    """High-performance GPU indexer using LibTorch and custom CUDA ops"""
    
    def __init__(self, config: IndexConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Index components
        self.database = None           # [N, D] original vectors
        self.ivf_centroids = None      # [num_clusters, D] IVF centroids
        self.pq_codebooks = None       # [M, K, D/M] PQ codebooks
        self.quantized_codes = None    # [N, M] quantized codes
        self.ivf_lists = None          # List assignments for each vector
        
        # Statistics
        self.num_vectors = 0
        self.is_trained = False
        self.index_built = False
        
        print(f"🚀 Initialized LibTorchIndexer on {self.device}")
        print(f"   Config: {config.num_subquantizers}x{config.codebook_size} PQ, {config.ivf_clusters} IVF")
    
    def train_index(self, training_vectors: torch.Tensor) -> None:
        """Train the IVF and PQ components using training data"""
        print(f"🔧 Training index with {training_vectors.shape[0]} vectors...")
        
        training_vectors = training_vectors.to(self.device, dtype=torch.int8)
        n_train, dim = training_vectors.shape
        
        assert dim == self.config.vector_dim, f"Expected {self.config.vector_dim}D vectors, got {dim}D"
        
        # 1. Train IVF centroids using k-means
        print("   Training IVF centroids...")
        self.ivf_centroids = self._train_ivf_centroids(training_vectors)
        
        # 2. Train PQ codebooks
        print("   Training PQ codebooks...")
        self.pq_codebooks = self._train_pq_codebooks(training_vectors)
        
        self.is_trained = True
        print("✅ Index training completed")
    
    def _train_ivf_centroids(self, vectors: torch.Tensor) -> torch.Tensor:
        """Train IVF centroids using k-means clustering"""
        # Simple k-means implementation for IVF training
        n, d = vectors.shape
        k = self.config.ivf_clusters
        
        # Initialize centroids randomly
        indices = torch.randperm(n, device=self.device)[:k]
        centroids = vectors[indices].float()
        
        # K-means iterations
        for iteration in range(20):  # Fixed iterations for efficiency
            # Assign vectors to closest centroids
            with torch.no_grad():
                # Compute distances using our custom CUDA op
                distances = torch.zeros(n, k, device=self.device)
                for i in range(k):
                    centroid_int8 = centroids[i].round().clamp(-128, 127).to(torch.int8)
                    scores = torch.ops.gobed_ann.i8dot512_scores(centroid_int8, vectors)
                    distances[:, i] = -scores  # Convert similarity to distance
                
                assignments = distances.argmin(dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = vectors[mask].float().mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]  # Keep old centroid if no assignments
            
            centroids = new_centroids
        
        return centroids.round().clamp(-128, 127).to(torch.int8)
    
    def _train_pq_codebooks(self, vectors: torch.Tensor) -> torch.Tensor:
        """Train Product Quantization codebooks"""
        n, d = vectors.shape
        m = self.config.num_subquantizers
        k = self.config.codebook_size
        subvec_dim = d // m
        
        # Initialize codebooks
        codebooks = torch.zeros(m, k, subvec_dim, device=self.device, dtype=torch.int8)
        
        # Train each subquantizer independently
        for i in range(m):
            start_idx = i * subvec_dim
            end_idx = (i + 1) * subvec_dim
            subvectors = vectors[:, start_idx:end_idx].float()
            
            # K-means for this subquantizer
            indices = torch.randperm(n, device=self.device)[:k]
            centroids = subvectors[indices]
            
            for _ in range(10):  # Fewer iterations for subquantizers
                # Assign to closest centroids
                distances = torch.cdist(subvectors, centroids)
                assignments = distances.argmin(dim=1)
                
                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                for j in range(k):
                    mask = assignments == j
                    if mask.sum() > 0:
                        new_centroids[j] = subvectors[mask].mean(dim=0)
                    else:
                        new_centroids[j] = centroids[j]
                
                centroids = new_centroids
            
            codebooks[i] = centroids.round().clamp(-128, 127).to(torch.int8)
        
        return codebooks
    
    def add_vectors(self, vectors: torch.Tensor) -> None:
        """Add vectors to the index"""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        
        print(f"📚 Adding {vectors.shape[0]} vectors to index...")
        
        vectors = vectors.to(self.device, dtype=torch.int8)
        n, d = vectors.shape
        
        # Store original vectors for reranking
        if self.database is None:
            self.database = vectors
        else:
            self.database = torch.cat([self.database, vectors], dim=0)
        
        # Assign to IVF lists
        ivf_assignments = self._assign_to_ivf_lists(vectors)
        if self.ivf_lists is None:
            self.ivf_lists = ivf_assignments
        else:
            # Adjust indices for concatenation
            offset = self.num_vectors
            adjusted_assignments = ivf_assignments + offset
            self.ivf_lists = torch.cat([self.ivf_lists, adjusted_assignments])
        
        # Quantize using PQ
        new_codes = self._quantize_vectors(vectors)
        if self.quantized_codes is None:
            self.quantized_codes = new_codes
        else:
            self.quantized_codes = torch.cat([self.quantized_codes, new_codes], dim=0)
        
        self.num_vectors += n
        self.index_built = True
        
        print(f"✅ Added vectors. Total: {self.num_vectors}")
    
    def _assign_to_ivf_lists(self, vectors: torch.Tensor) -> torch.Tensor:
        """Assign vectors to IVF lists"""
        n = vectors.shape[0]
        assignments = torch.zeros(n, device=self.device, dtype=torch.long)
        
        # Find closest centroid for each vector
        for i, centroid in enumerate(self.ivf_centroids):
            scores = torch.ops.gobed_ann.i8dot512_scores(centroid, vectors)
            if i == 0:
                best_scores = scores
                assignments.fill_(0)
            else:
                mask = scores > best_scores
                assignments[mask] = i
                best_scores = torch.maximum(best_scores, scores)
        
        return assignments
    
    def _quantize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Quantize vectors using trained PQ codebooks"""
        n, d = vectors.shape
        m = self.config.num_subquantizers
        k = self.config.codebook_size
        subvec_dim = d // m
        
        codes = torch.zeros(n, m, device=self.device, dtype=torch.uint8)
        
        for i in range(m):
            start_idx = i * subvec_dim
            end_idx = (i + 1) * subvec_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            # Find closest codeword for each subvector
            codebook = self.pq_codebooks[i]  # [K, subvec_dim]
            
            best_codes = torch.zeros(n, device=self.device, dtype=torch.uint8)
            best_scores = torch.full((n,), float('-inf'), device=self.device)
            
            for j in range(k):
                codeword = codebook[j]
                scores = torch.sum(subvectors.int() * codeword.int(), dim=1)
                mask = scores > best_scores
                best_codes[mask] = j
                best_scores = torch.maximum(best_scores, scores.float())
            
            codes[:, i] = best_codes
        
        return codes
    
    def search(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search for k nearest neighbors"""
        if not self.index_built:
            raise RuntimeError("Index must be built before searching")
        
        query = query.to(self.device, dtype=torch.int8)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Stage 1: IVF probe to get candidate lists
        candidate_ids = self._probe_ivf_lists(query[0])
        
        if len(candidate_ids) == 0:
            # Fallback: return empty results
            return torch.empty(0, dtype=torch.long, device=self.device), \
                   torch.empty(0, dtype=torch.float32, device=self.device)
        
        # Stage 2: PQ-based scoring of candidates
        pq_scores = self._score_with_pq(query[0], candidate_ids)
        
        # Stage 3: Select top candidates for reranking
        rerank_k = min(self.config.rerank_k, len(candidate_ids))
        top_indices = torch.topk(pq_scores, rerank_k, largest=True).indices
        rerank_candidates = candidate_ids[top_indices]
        
        # Stage 4: Exact reranking using original vectors
        final_scores = self._exact_rerank(query[0], rerank_candidates)
        
        # Return top-k results
        k = min(k, len(rerank_candidates))
        top_k_indices = torch.topk(final_scores, k, largest=True).indices
        result_ids = rerank_candidates[top_k_indices]
        result_scores = final_scores[top_k_indices]
        
        return result_ids, result_scores
    
    def _probe_ivf_lists(self, query: torch.Tensor) -> torch.Tensor:
        """Probe IVF lists to get candidate vectors"""
        # Find closest centroids
        scores = torch.zeros(self.config.ivf_clusters, device=self.device)
        for i, centroid in enumerate(self.ivf_centroids):
            score = torch.ops.gobed_ann.i8dot512_scores(centroid.unsqueeze(0), query.unsqueeze(0))
            scores[i] = score[0]
        
        # Select top probe_lists centroids
        probe_lists = min(self.config.probe_lists, self.config.ivf_clusters)
        top_lists = torch.topk(scores, probe_lists, largest=True).indices
        
        # Collect candidate vector IDs from these lists
        candidates = []
        for list_id in top_lists:
            mask = self.ivf_lists == list_id
            candidates.append(torch.where(mask)[0])
        
        if candidates:
            return torch.cat(candidates)
        else:
            return torch.empty(0, dtype=torch.long, device=self.device)
    
    def _score_with_pq(self, query: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Score candidates using PQ and ADC"""
        # Build PQ lookup table using our custom CUDA op
        lut = torch.ops.gobed_ann.build_pq_lut(query, self.pq_codebooks)
        
        # Get codes for candidates
        candidate_codes = self.quantized_codes[candidate_ids]
        
        # Compute ADC scores using our custom CUDA op
        scores = torch.ops.gobed_ann.adc_scan(lut, candidate_codes)
        
        return scores
    
    def _exact_rerank(self, query: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Exact reranking using original vectors"""
        candidates = self.database[candidate_ids]
        scores = torch.ops.gobed_ann.i8dot512_scores(query, candidates)
        return scores
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        
        return {
            "num_vectors": self.num_vectors,
            "vector_dim": self.config.vector_dim,
            "ivf_clusters": self.config.ivf_clusters,
            "pq_subquantizers": self.config.num_subquantizers,
            "device": str(self.device),
            "gpu_memory_mb": gpu_memory,
            "is_trained": self.is_trained,
            "index_built": self.index_built
        }
    
    def save_index(self, path: str) -> None:
        """Save the trained index"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained index")
        
        torch.save({
            'config': self.config,
            'ivf_centroids': self.ivf_centroids,
            'pq_codebooks': self.pq_codebooks,
            'database': self.database,
            'quantized_codes': self.quantized_codes,
            'ivf_lists': self.ivf_lists,
            'num_vectors': self.num_vectors,
            'is_trained': self.is_trained,
            'index_built': self.index_built
        }, path)
        
        print(f"💾 Saved index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load a trained index"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.ivf_centroids = checkpoint['ivf_centroids']
        self.pq_codebooks = checkpoint['pq_codebooks']
        self.database = checkpoint['database']
        self.quantized_codes = checkpoint['quantized_codes']
        self.ivf_lists = checkpoint['ivf_lists']
        self.num_vectors = checkpoint['num_vectors']
        self.is_trained = checkpoint['is_trained']
        self.index_built = checkpoint['index_built']
        
        print(f"📖 Loaded index from {path}")

def create_test_vectors(n: int, d: int, device: str = "cuda:0") -> torch.Tensor:
    """Create test vectors for benchmarking"""
    # Create structured test data that's more realistic than random
    vectors = torch.randn(n, d, device=device) * 50
    vectors = vectors.round().clamp(-128, 127).to(torch.int8)
    return vectors

def benchmark_indexer():
    """Comprehensive benchmark of the LibTorch indexer"""
    print("🔥 LibTorch Indexer Benchmark")
    print("=" * 40)
    
    # Configuration
    config = IndexConfig(
        device="cuda:0",
        batch_size=1024,
        num_subquantizers=64,
        codebook_size=256,
        vector_dim=512,
        ivf_clusters=1024,
        probe_lists=16,
        rerank_k=200
    )
    
    # Create test data
    print("🎯 Creating test data...")
    train_vectors = create_test_vectors(10000, 512)  # Training set
    index_vectors = create_test_vectors(100000, 512)  # Vectors to index
    query_vectors = create_test_vectors(100, 512)     # Query vectors
    
    # Initialize indexer
    indexer = LibTorchIndexer(config)
    
    # Benchmark training
    print("\n⏱️  Training benchmark...")
    start_time = time.time()
    indexer.train_index(train_vectors)
    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.2f}s")
    
    # Benchmark indexing
    print("\n⏱️  Indexing benchmark...")
    start_time = time.time()
    indexer.add_vectors(index_vectors)
    index_time = time.time() - start_time
    index_rate = len(index_vectors) / index_time
    print(f"   Indexing time: {index_time:.2f}s")
    print(f"   Indexing rate: {index_rate:.0f} vectors/sec")
    
    # Benchmark search
    print("\n⏱️  Search benchmark...")
    k = 10
    search_times = []
    
    for i in range(min(10, len(query_vectors))):
        start_time = time.time()
        ids, scores = indexer.search(query_vectors[i], k=k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        if i == 0:  # Print first result for verification
            print(f"   First query results: {len(ids)} results")
            if len(ids) > 0:
                print(f"   Top score: {scores[0].item():.2f}")
    
    avg_search_time = np.mean(search_times) * 1000  # Convert to ms
    qps = 1.0 / np.mean(search_times)
    
    print(f"   Average search time: {avg_search_time:.2f}ms")
    print(f"   Queries per second: {qps:.0f}")
    
    # Print statistics
    print("\n📊 Index Statistics:")
    stats = indexer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Memory usage
    total_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"   Total GPU memory: {total_memory:.1f} MB")
    
    return indexer

if __name__ == "__main__":
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available")
        exit(1)
    
    # Run benchmark
    indexer = benchmark_indexer()
    
    print("\n✅ LibTorch indexer benchmark completed!")