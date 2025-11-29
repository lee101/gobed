#!/usr/bin/env python3
"""
TorchScript search module for GPU-accelerated ANN search.
Supports both Flat-INT8 and IVF-PQ search modes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class GPUSearchModule(nn.Module):
    """GPU-accelerated search module with Flat-INT8 and IVF-PQ support."""
    
    def __init__(self, 
                 db_i8: torch.Tensor,           # [N, 512] int8
                 db_scale: torch.Tensor,         # [N] float32 - per-vector scales
                 centroids: Optional[torch.Tensor] = None,    # [nlists, 512] float32
                 opq_R: Optional[torch.Tensor] = None,        # [512, 512] float32
                 pq_codebooks: Optional[torch.Tensor] = None, # [m, 256, dsub] float32
                 pq_codes: Optional[torch.Tensor] = None,     # [N, m] uint8
                 list_offsets: Optional[torch.Tensor] = None, # [nlists+1] int32
                 ids: Optional[torch.Tensor] = None):         # [N] int64
        super().__init__()
        
        # Register database
        self.register_buffer("db_i8", db_i8)
        self.register_buffer("db_scale", db_scale)
        self.register_buffer("ids", ids if ids is not None else torch.arange(db_i8.size(0)))
        
        # IVF-PQ components (optional)
        self.use_ivf_pq = centroids is not None
        if self.use_ivf_pq:
            self.register_buffer("centroids", centroids)
            self.register_buffer("opq_R", opq_R if opq_R is not None else torch.eye(512))
            self.register_buffer("pq_codebooks", pq_codebooks)
            self.register_buffer("pq_codes", pq_codes)
            self.register_buffer("list_offsets", list_offsets)
            
            self.nlists = centroids.size(0)
            self.m = pq_codebooks.size(0)
            self.dsub = 512 // self.m
    
    @torch.jit.export
    def search_flat(self, q_i8: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flat brute-force INT8 search."""
        # Compute INT8 dot products using custom CUDA op
        scores = torch.ops.gobed_ann.i8dot512_scores(q_i8, self.db_i8)
        
        # Get top-k
        vals, idx = torch.topk(scores, k=min(k, scores.size(0)), largest=True)
        
        # Return IDs and scores
        return self.ids[idx], vals
    
    @torch.jit.export
    def search_ivf_pq(self, q_i8: torch.Tensor, 
                      nprobe: int = 32, 
                      rerank: int = 256,
                      k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """IVF-PQ search with ADC and re-ranking."""
        
        if not self.use_ivf_pq:
            return self.search_flat(q_i8, k)
        
        # 1. Convert query to float for IVF routing
        q_f32 = q_i8.to(torch.float32)
        
        # 2. Find nearest IVF centroids
        dists = torch.cdist(q_f32.unsqueeze(0), self.centroids).squeeze(0)
        _, list_ids = torch.topk(dists, k=min(nprobe, self.nlists), largest=False)
        
        # 3. Gather codes from selected lists
        codes_sel, ids_sel = torch.ops.gobed_ann.gather_ivf_codes(
            self.pq_codes, self.ids, list_ids, self.list_offsets)
        
        if codes_sel.size(0) == 0:
            # No codes in selected lists
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
        
        # 4. Compute residual and rotate with OPQ
        # For simplicity, using centroid 0 as approximation
        residual = q_f32 - self.centroids[list_ids[0]]
        q_rot = torch.matmul(self.opq_R.t(), residual)
        q_rot = q_rot.view(self.m, self.dsub)
        
        # 5. Build PQ lookup table
        lut = torch.ops.gobed_ann.build_pq_lut(q_rot, self.pq_codebooks)
        
        # 6. ADC scan
        adc_scores = torch.ops.gobed_ann.adc_scan(codes_sel, lut)
        
        # 7. Get top candidates for re-ranking
        n_rerank = min(rerank, adc_scores.size(0))
        _, cand_idx = torch.topk(-adc_scores, k=n_rerank, largest=True)
        cand_ids = ids_sel[cand_idx]
        
        # 8. Re-rank with exact INT8 dot products
        cand_vecs = self.db_i8.index_select(0, cand_ids)
        exact_scores = torch.ops.gobed_ann.i8dot512_scores(q_i8, cand_vecs)
        
        # 9. Final top-k
        final_k = min(k, exact_scores.size(0))
        vals, idx = torch.topk(exact_scores, k=final_k, largest=True)
        
        return cand_ids[idx], vals
    
    @torch.jit.export  
    def batch_search_flat(self, queries_i8: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch flat search for multiple queries."""
        B = queries_i8.size(0)
        
        # Batch INT8 dot products
        all_scores = torch.ops.gobed_ann.i8dot512_batch(queries_i8, self.db_i8)
        
        # Top-k per query
        vals, idx = torch.topk(all_scores, k=min(k, all_scores.size(1)), dim=1, largest=True)
        
        # Map to IDs
        result_ids = self.ids[idx.view(-1)].view(B, -1)
        
        return result_ids, vals
    
    def forward(self, q_i8: torch.Tensor, 
                mode: str = "flat",
                nprobe: int = 32,
                rerank: int = 256,
                k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main search interface."""
        
        if mode == "flat":
            return self.search_flat(q_i8, k)
        elif mode == "ivf_pq":
            return self.search_ivf_pq(q_i8, nprobe, rerank, k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")


def create_demo_module(N: int = 100000, use_ivf_pq: bool = False):
    """Create a demo search module for testing."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate random INT8 database
    db_i8 = torch.randint(-128, 127, (N, 512), dtype=torch.int8, device=device)
    db_scale = torch.ones(N, dtype=torch.float32, device=device)
    ids = torch.arange(N, dtype=torch.long, device=device)
    
    if use_ivf_pq:
        # IVF-PQ components
        nlists = 1024
        m = 64
        dsub = 512 // m
        
        centroids = torch.randn(nlists, 512, dtype=torch.float32, device=device)
        opq_R = torch.eye(512, dtype=torch.float32, device=device)
        pq_codebooks = torch.randn(m, 256, dsub, dtype=torch.float32, device=device)
        pq_codes = torch.randint(0, 256, (N, m), dtype=torch.uint8, device=device)
        
        # Create dummy list offsets (uniform distribution)
        vecs_per_list = N // nlists
        list_offsets = torch.arange(0, N+1, vecs_per_list, dtype=torch.int32, device=device)
        if list_offsets.size(0) < nlists + 1:
            list_offsets = torch.cat([list_offsets, torch.tensor([N], device=device)])
        list_offsets = list_offsets[:nlists+1]
        
        module = GPUSearchModule(db_i8, db_scale, centroids, opq_R, 
                                 pq_codebooks, pq_codes, list_offsets, ids)
    else:
        module = GPUSearchModule(db_i8, db_scale, ids=ids)
    
    return module


def export_module(module: GPUSearchModule, path: str = "gpu_search_module.pt"):
    """Export module as TorchScript."""
    
    # Move to eval mode
    module.eval()
    
    # Script the module
    scripted = torch.jit.script(module)
    
    # Save
    scripted.save(path)
    print(f" Exported TorchScript module to {path}")
    
    return scripted


if __name__ == "__main__":
    print("Creating demo GPU search modules...")
    
    # Create flat search module
    flat_module = create_demo_module(N=100000, use_ivf_pq=False)
    export_module(flat_module, "gpu_search_flat.pt")
    
    # Create IVF-PQ module
    ivf_module = create_demo_module(N=1000000, use_ivf_pq=True)
    export_module(ivf_module, "gpu_search_ivf_pq.pt")
    
    # Test search
    print("\nTesting search...")
    q = torch.randint(-128, 127, (512,), dtype=torch.int8, device="cuda")
    
    # Flat search
    ids, scores = flat_module.search_flat(q, k=10)
    print(f"Flat search top-10 IDs: {ids.cpu().numpy()}")
    
    # Batch search
    queries = torch.randint(-128, 127, (4, 512), dtype=torch.int8, device="cuda")
    batch_ids, batch_scores = flat_module.batch_search_flat(queries, k=5)
    print(f"Batch search shape: {batch_ids.shape}")
    
    print(" Module creation and export complete!")