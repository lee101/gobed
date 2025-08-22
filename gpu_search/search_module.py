#!/usr/bin/env python3
"""
TorchScript wrapper module for GPU search operations
This module wraps our custom CUDA ops in a TorchScript-compatible interface
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

class GPUSearchModule(nn.Module):
    """
    TorchScript-compatible module for GPU-accelerated similarity search
    Implements IVF + OPQ + PQ + ADC + tiny re-rank architecture
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 pq_m: int = 64,           # Number of subquantizers
                 pq_k: int = 256,          # Codebook size (8-bit)
                 ivf_centroids: int = 4096, # IVF coarse centroids
                 device: str = "cuda"):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.pq_m = pq_m
        self.pq_k = pq_k
        self.ivf_centroids = ivf_centroids
        self.device = device
        
        # Initialize empty database
        self.register_buffer('database', torch.empty((0, embedding_dim), dtype=torch.int8))
        self.register_buffer('pq_codes', torch.empty((0, pq_m), dtype=torch.uint8))
        
        # Initialize PQ codebooks with random int8 values
        pq_codebooks = torch.randint(-128, 127, (pq_m, pq_k, embedding_dim // pq_m), dtype=torch.int8)
        self.register_buffer('pq_codebooks', pq_codebooks)
        
        # Initialize IVF centroids with random int8 values
        ivf_centroids = torch.randint(-128, 127, (ivf_centroids, embedding_dim), dtype=torch.int8)
        self.register_buffer('ivf_centroids_buf', ivf_centroids)
        
        self.register_buffer('ivf_assignments', torch.empty((0,), dtype=torch.long))
        
        # Load custom CUDA ops
        self._load_custom_ops()
    
    def _load_custom_ops(self):
        """Load custom CUDA operations"""
        try:
            # This will be available after building the CUDA extension
            torch.ops.load_library("libgobed_ann_ops.so")
            self.has_custom_ops = True
            print("✅ Loaded custom CUDA ops")
        except Exception as e:
            print(f"⚠️  Custom CUDA ops not available: {e}")
            self.has_custom_ops = False
    
    def load_database(self, embeddings: torch.Tensor) -> None:
        """
        Load embeddings database to GPU
        Args:
            embeddings: [N, D] int8 embeddings tensor
        """
        self.database = embeddings.to(self.device, dtype=torch.int8)
        
        # Move all buffers to the same device
        self.pq_codebooks = self.pq_codebooks.to(self.device)
        self.ivf_centroids_buf = self.ivf_centroids_buf.to(self.device)
        
        # Generate PQ codes for the database
        self._generate_pq_codes()
        
        # Assign to IVF centroids
        self._assign_ivf_centroids()
        
        print(f"📚 Loaded database: {self.database.shape[0]} vectors, {self.database.shape[1]} dims")
    
    def _generate_pq_codes(self):
        """Generate Product Quantization codes for database vectors"""
        N = self.database.shape[0]
        codes = torch.zeros((N, self.pq_m), dtype=torch.uint8, device=self.device)
        
        # Simple quantization - find nearest codeword for each subvector
        subvec_dim = self.embedding_dim // self.pq_m
        
        for m in range(self.pq_m):
            start_dim = m * subvec_dim
            end_dim = (m + 1) * subvec_dim
            
            # Extract subvectors
            db_subvecs = self.database[:, start_dim:end_dim]  # [N, subvec_dim]
            codebook = self.pq_codebooks[m]  # [K, subvec_dim]
            
            # Compute distances to all codewords
            distances = torch.cdist(db_subvecs.float(), codebook.float())  # [N, K]
            codes[:, m] = torch.argmin(distances, dim=1).to(torch.uint8)
        
        self.pq_codes = codes
    
    def _assign_ivf_centroids(self):
        """Assign database vectors to IVF centroids"""
        distances = torch.cdist(self.database.float(), self.ivf_centroids_buf.float())
        self.ivf_assignments = torch.argmin(distances, dim=1)
    
    @torch.jit.export
    def search(self, query: torch.Tensor, k: int = 10, nprobe: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform similarity search
        Args:
            query: [D] int8 query vector
            k: number of results to return
            nprobe: number of IVF centroids to probe
        Returns:
            scores: [k] similarity scores
            indices: [k] database indices
        """
        if self.database.shape[0] == 0:
            return torch.empty((0,), device=self.device), torch.empty((0,), dtype=torch.long, device=self.device)
        
        # Step 1: IVF coarse search - find nearest centroids
        centroid_scores = self._compute_scores_native(query, self.ivf_centroids_buf)
        _, top_centroids = torch.topk(centroid_scores, min(nprobe, centroid_scores.shape[0]))
        
        # Step 2: Find candidate vectors from selected centroids
        candidate_mask = torch.zeros(self.database.shape[0], dtype=torch.bool, device=self.device)
        for centroid_id in top_centroids:
            candidate_mask |= (self.ivf_assignments == centroid_id)
        
        candidate_indices = torch.where(candidate_mask)[0]
        
        if candidate_indices.shape[0] == 0:
            return torch.empty((0,), device=self.device), torch.empty((0,), dtype=torch.long, device=self.device)
        
        # Step 3: Product Quantization search using ADC
        if self.has_custom_ops:
            # Use custom CUDA ops for faster computation
            lut = torch.ops.gpu_search_ops.build_pq_lut(query, self.pq_codebooks)
            candidate_codes = self.pq_codes[candidate_indices]
            candidate_scores = torch.ops.gpu_search_ops.adc_scan(lut, candidate_codes)
        else:
            # Fallback to native PyTorch
            candidate_scores = self._adc_search_native(query, candidate_indices)
        
        # Step 4: Select top-k candidates
        top_k = min(k * 4, candidate_scores.shape[0])  # Over-select for reranking
        _, top_candidate_indices = torch.topk(candidate_scores, top_k)
        rerank_indices = candidate_indices[top_candidate_indices]
        
        # Step 5: Tiny re-rank with exact computation
        rerank_vectors = self.database[rerank_indices]
        if self.has_custom_ops:
            rerank_scores = torch.ops.gpu_search_ops.i8dot512_scores(query, rerank_vectors)
        else:
            rerank_scores = self._compute_scores_native(query, rerank_vectors)
        
        # Final top-k selection
        final_k = min(k, rerank_scores.shape[0])
        final_scores, final_indices = torch.topk(rerank_scores, final_k)
        final_db_indices = rerank_indices[final_indices]
        
        return final_scores, final_db_indices
    
    def _compute_scores_native(self, query: torch.Tensor, database: torch.Tensor) -> torch.Tensor:
        """Native PyTorch implementation of int8 dot product"""
        return torch.matmul(query.float(), database.float().T)
    
    def _adc_search_native(self, query: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        """Native PyTorch implementation of ADC search"""
        scores = torch.zeros(candidate_indices.shape[0], device=self.device)
        
        subvec_dim = self.embedding_dim // self.pq_m
        
        for m in range(self.pq_m):
            start_dim = m * subvec_dim
            end_dim = (m + 1) * subvec_dim
            
            query_sub = query[start_dim:end_dim]
            codebook = self.pq_codebooks[m]
            
            # Compute LUT for this subquantizer
            lut = torch.matmul(query_sub.float(), codebook.float().T)  # [K]
            
            # Lookup scores for candidate vectors
            candidate_codes = self.pq_codes[candidate_indices, m]
            scores += lut[candidate_codes]
        
        return scores
    
    @torch.jit.export
    def batch_search(self, queries: torch.Tensor, k: int = 10, nprobe: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch similarity search
        Args:
            queries: [B, D] int8 query vectors
            k: number of results per query
            nprobe: number of IVF centroids to probe
        Returns:
            scores: [B, k] similarity scores
            indices: [B, k] database indices
        """
        batch_size = queries.shape[0]
        all_scores = torch.zeros((batch_size, k), device=self.device)
        all_indices = torch.zeros((batch_size, k), dtype=torch.long, device=self.device)
        
        for i in range(batch_size):
            scores, indices = self.search(queries[i], k, nprobe)
            if scores.shape[0] > 0:
                actual_k = min(k, scores.shape[0])
                all_scores[i, :actual_k] = scores[:actual_k]
                all_indices[i, :actual_k] = indices[:actual_k]
        
        return all_scores, all_indices
    
    @torch.jit.export
    def get_stats(self) -> Tuple[int, float]:
        """
        Get database statistics
        Returns:
            num_vectors: number of vectors in database
            memory_mb: memory usage in MB
        """
        num_vectors = self.database.shape[0]
        memory_bytes = (
            self.database.numel() * self.database.element_size() +
            self.pq_codes.numel() * self.pq_codes.element_size() +
            self.pq_codebooks.numel() * self.pq_codebooks.element_size() +
            self.ivf_centroids_buf.numel() * self.ivf_centroids_buf.element_size() +
            self.ivf_assignments.numel() * self.ivf_assignments.element_size()
        )
        memory_mb = memory_bytes / (1024 * 1024)
        
        return num_vectors, memory_mb


def export_search_module() -> str:
    """
    Export the search module to TorchScript format
    Returns:
        path to exported model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = GPUSearchModule(device=device)
    module = module.to(device)
    
    # Create example data for tracing
    example_query = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
    example_database = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=device)
    
    # Load example database
    module.load_database(example_database)
    
    # Trace the module
    traced_module = torch.jit.script(module)
    
    # Save the traced module
    output_path = "/home/lee/code/gobed/model/gpu_search_module.pt"
    traced_module.save(output_path)
    
    print(f"✅ Exported GPU search module to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("🚀 Exporting GPU Search Module")
    export_search_module()