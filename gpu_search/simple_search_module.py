#!/usr/bin/env python3
"""
Simplified TorchScript search module without custom ops dependency
This provides the core functionality that can be exported to TorchScript
"""

import torch
import torch.nn as nn
from typing import Tuple

class SimpleGPUSearchModule(nn.Module):
    """
    Simplified TorchScript-compatible module for GPU-accelerated similarity search
    Uses native PyTorch operations only for TorchScript compatibility
    """
    
    def __init__(self, embedding_dim: int = 512, device: str = "cuda"):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize empty database
        self.register_buffer('database', torch.empty((0, embedding_dim), dtype=torch.int8))
        self.register_buffer('num_vectors', torch.tensor(0, dtype=torch.long))
    
    @torch.jit.export
    def load_database(self, embeddings: torch.Tensor) -> None:
        """
        Load embeddings database to GPU
        Args:
            embeddings: [N, D] int8 embeddings tensor
        """
        self.database = embeddings.to(self.device, dtype=torch.int8)
        self.num_vectors = torch.tensor(embeddings.shape[0], dtype=torch.long)
    
    @torch.jit.export
    def search(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform similarity search using int8 dot product
        Args:
            query: [D] int8 query vector
            k: number of results to return
        Returns:
            scores: [k] similarity scores
            indices: [k] database indices
        """
        if self.database.shape[0] == 0:
            return torch.empty((0,), device=self.device), torch.empty((0,), dtype=torch.long, device=self.device)
        
        # Compute similarity scores using int8 dot product
        # Convert to float32 for computation
        query_f32 = query.float()
        database_f32 = self.database.float()
        
        scores = torch.matmul(database_f32, query_f32)
        
        # Get top-k results
        actual_k = min(k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, actual_k)
        
        return top_scores, top_indices
    
    @torch.jit.export
    def batch_search(self, queries: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch similarity search
        Args:
            queries: [B, D] int8 query vectors
            k: number of results per query
        Returns:
            scores: [B, k] similarity scores
            indices: [B, k] database indices
        """
        if self.database.shape[0] == 0:
            batch_size = queries.shape[0]
            return (torch.empty((batch_size, 0), device=self.device), 
                   torch.empty((batch_size, 0), dtype=torch.long, device=self.device))
        
        batch_size = queries.shape[0]
        actual_k = min(k, self.database.shape[0])
        
        # Batch matrix multiplication
        queries_f32 = queries.float()  # [B, D]
        database_f32 = self.database.float()  # [N, D]
        
        # Compute all similarities at once
        all_scores = torch.matmul(queries_f32, database_f32.T)  # [B, N]
        
        # Get top-k for each query
        top_scores, top_indices = torch.topk(all_scores, actual_k, dim=1)  # [B, k]
        
        return top_scores, top_indices
    
    @torch.jit.export
    def get_stats(self) -> Tuple[int, float]:
        """
        Get database statistics
        Returns:
            num_vectors: number of vectors in database
            memory_mb: memory usage in MB
        """
        num_vectors = int(self.num_vectors.item())
        memory_bytes = self.database.numel() * self.database.element_size()
        memory_mb = float(memory_bytes) / (1024.0 * 1024.0)
        
        return num_vectors, memory_mb


def export_simple_search_module() -> str:
    """
    Export the simplified search module to TorchScript format
    Returns:
        path to exported model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = SimpleGPUSearchModule(device=device)
    module = module.to(device)
    
    # Create example data for tracing
    example_database = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=device)
    
    # Load example database
    module.load_database(example_database)
    
    # Script the module (no custom ops dependency)
    scripted_module = torch.jit.script(module)
    
    # Test the scripted module
    test_query = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
    test_scores, test_indices = scripted_module.search(test_query, 5)
    print(f" Test search returned {test_scores.shape[0]} results")
    
    # Test batch search
    test_queries = torch.randint(-128, 127, (8, 512), dtype=torch.int8, device=device)
    batch_scores, batch_indices = scripted_module.batch_search(test_queries, 5)
    print(f" Test batch search returned {batch_scores.shape} results")
    
    # Save the scripted module
    output_path = "/home/lee/code/gobed/model/simple_gpu_search_module.pt"
    scripted_module.save(output_path)
    
    print(f" Exported simple GPU search module to: {output_path}")
    return output_path


if __name__ == "__main__":
    print(" Exporting Simple GPU Search Module")
    export_simple_search_module()