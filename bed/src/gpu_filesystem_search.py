#!/usr/bin/env python3
"""
GPU-accelerated filesystem search using gobed embeddings and IVF indexing
Efficient token-based search with hierarchical file/chunk indexing
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import hashlib
import mmap
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a text chunk"""
    file_path: str
    start_offset: int
    end_offset: int
    line_start: int
    line_end: int
    chunk_text: str
    chunk_hash: str


@dataclass
class SearchConfig:
    """Configuration for the search system"""
    device: str = "cuda:0"
    embedding_dim: int = 512
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64  # token overlap
    max_chunk_size: int = 1024  # max tokens per chunk
    ivf_clusters: int = 1024  # number of IVF clusters
    probe_lists: int = 32  # lists to probe during search
    batch_size: int = 256
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    index_path: str = ".bed_index"
    use_int8: bool = True
    num_workers: int = 4


class GPUFilesystemSearch:
    """GPU-accelerated filesystem search with IVF indexing"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Index components
        self.chunk_database = []  # List of ChunkInfo objects
        self.embeddings = None  # Tensor of embeddings [N, D]
        self.ivf_centroids = None  # IVF centroids [K, D]
        self.ivf_lists = {}  # Mapping from cluster_id to chunk indices
        self.file_index = {}  # Mapping from file_path to chunk indices

        # Stats
        self.num_chunks = 0
        self.num_files = 0
        self.index_size_mb = 0.0

        logger.info(f"Initialized GPU filesystem search on {self.device}")

    def chunk_file(self, file_path: str) -> List[ChunkInfo]:
        """Split a file into overlapping chunks for embedding"""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(self.config.max_file_size)

            lines = content.split('\n')
            current_chunk = []
            current_size = 0
            line_start = 0

            for line_num, line in enumerate(lines):
                # Simple token estimation (words)
                tokens = len(line.split())

                if current_size + tokens > self.config.chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = '\n'.join(current_chunk)
                    chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]

                    chunk_info = ChunkInfo(
                        file_path=file_path,
                        start_offset=sum(len(l) + 1 for l in lines[:line_start]),
                        end_offset=sum(len(l) + 1 for l in lines[:line_num]),
                        line_start=line_start,
                        line_end=line_num - 1,
                        chunk_text=chunk_text[:self.config.max_chunk_size * 10],  # Rough char limit
                        chunk_hash=chunk_hash
                    )
                    chunks.append(chunk_info)

                    # Start new chunk with overlap
                    overlap_lines = max(0, len(current_chunk) - 5)  # Keep last 5 lines
                    current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                    current_size = sum(len(l.split()) for l in current_chunk)
                    line_start = line_num - overlap_lines

                current_chunk.append(line)
                current_size += tokens

            # Add final chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]

                chunk_info = ChunkInfo(
                    file_path=file_path,
                    start_offset=sum(len(l) + 1 for l in lines[:line_start]),
                    end_offset=len(content),
                    line_start=line_start,
                    line_end=len(lines) - 1,
                    chunk_text=chunk_text[:self.config.max_chunk_size * 10],
                    chunk_hash=chunk_hash
                )
                chunks.append(chunk_info)

        except Exception as e:
            logger.warning(f"Failed to chunk file {file_path}: {e}")

        return chunks

    def embed_chunks_batch(self, chunks: List[ChunkInfo]) -> torch.Tensor:
        """Embed a batch of chunks using gobed (placeholder for actual embedding)"""
        # TODO: Integrate with actual gobed embedding model
        # For now, create random embeddings for testing
        embeddings = []

        for chunk in chunks:
            # Placeholder: create deterministic pseudo-random embedding based on text
            np.random.seed(hash(chunk.chunk_text) % (2**32))
            embedding = np.random.randn(self.config.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            if self.config.use_int8:
                # Quantize to int8
                embedding = (embedding * 127).astype(np.int8)

            embeddings.append(embedding)

        return torch.tensor(np.array(embeddings), device=self.device)

    def train_ivf_centroids(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Train IVF centroids using k-means clustering"""
        n, d = embeddings.shape
        k = min(self.config.ivf_clusters, n // 10)  # At least 10 points per cluster

        logger.info(f"Training {k} IVF centroids on {n} embeddings...")

        # Convert to float for k-means
        if embeddings.dtype == torch.int8:
            embeddings_float = embeddings.float() / 127.0
        else:
            embeddings_float = embeddings.float()

        # Initialize centroids with k-means++
        centroids = torch.zeros(k, d, device=self.device)
        indices = torch.randperm(n, device=self.device)[:k]
        centroids = embeddings_float[indices].clone()

        # K-means iterations
        for iteration in range(20):
            # Assign points to closest centroids
            distances = torch.cdist(embeddings_float, centroids)
            assignments = distances.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = embeddings_float[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]

            # Check convergence
            change = (new_centroids - centroids).abs().max()
            centroids = new_centroids

            if change < 1e-4:
                logger.info(f"K-means converged at iteration {iteration + 1}")
                break

        if self.config.use_int8:
            centroids = (centroids * 127).round().clamp(-128, 127).to(torch.int8)

        return centroids

    def build_ivf_lists(self, embeddings: torch.Tensor, centroids: torch.Tensor) -> Dict[int, List[int]]:
        """Assign embeddings to IVF lists"""
        n, d = embeddings.shape
        k = centroids.shape[0]

        # Convert to float for distance computation
        if embeddings.dtype == torch.int8:
            embeddings_float = embeddings.float() / 127.0
            centroids_float = centroids.float() / 127.0
        else:
            embeddings_float = embeddings.float()
            centroids_float = centroids.float()

        # Compute assignments
        ivf_lists = {i: [] for i in range(k)}

        # Process in batches for memory efficiency
        batch_size = min(10000, n)
        for i in range(0, n, batch_size):
            batch = embeddings_float[i:i+batch_size]
            distances = torch.cdist(batch, centroids_float)
            assignments = distances.argmin(dim=1)

            for j, cluster_id in enumerate(assignments.cpu().numpy()):
                ivf_lists[cluster_id].append(i + j)

        logger.info(f"Built {len(ivf_lists)} IVF lists")
        return ivf_lists

    def index_directory(self, directory: str, extensions: List[str] = None):
        """Index all files in a directory"""
        logger.info(f"Indexing directory: {directory}")

        if extensions is None:
            extensions = ['.py', '.go', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.md', '.txt']

        # Find all files
        all_files = []
        for ext in extensions:
            all_files.extend(Path(directory).rglob(f'*{ext}'))

        logger.info(f"Found {len(all_files)} files to index")

        # Chunk files in parallel
        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            chunk_lists = executor.map(self.chunk_file, [str(f) for f in all_files])
            for chunks in chunk_lists:
                all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")

        # Embed chunks in batches
        embeddings_list = []
        for i in range(0, len(all_chunks), self.config.batch_size):
            batch = all_chunks[i:i+self.config.batch_size]
            batch_embeddings = self.embed_chunks_batch(batch)
            embeddings_list.append(batch_embeddings)

        # Store chunks and embeddings
        self.chunk_database = all_chunks
        self.embeddings = torch.cat(embeddings_list, dim=0) if embeddings_list else torch.empty(0, self.config.embedding_dim)
        self.num_chunks = len(all_chunks)
        self.num_files = len(all_files)

        # Build file index
        self.file_index = {}
        for idx, chunk in enumerate(all_chunks):
            if chunk.file_path not in self.file_index:
                self.file_index[chunk.file_path] = []
            self.file_index[chunk.file_path].append(idx)

        # Train IVF if we have enough data
        if self.num_chunks > 100:
            self.ivf_centroids = self.train_ivf_centroids(self.embeddings)
            self.ivf_lists = self.build_ivf_lists(self.embeddings, self.ivf_centroids)
        else:
            logger.info("Not enough chunks for IVF, using brute-force search")

        # Calculate index size
        self.index_size_mb = (
            self.embeddings.numel() * self.embeddings.element_size() +
            (self.ivf_centroids.numel() * self.ivf_centroids.element_size() if self.ivf_centroids is not None else 0)
        ) / (1024 * 1024)

        logger.info(f"Index built: {self.num_chunks} chunks, {self.num_files} files, {self.index_size_mb:.2f} MB")

    def search(self, query: str, k: int = 10, use_ivf: bool = True) -> List[Tuple[ChunkInfo, float]]:
        """Search for similar chunks"""
        # Embed query
        query_chunk = ChunkInfo(
            file_path="query",
            start_offset=0,
            end_offset=len(query),
            line_start=0,
            line_end=0,
            chunk_text=query,
            chunk_hash=""
        )
        query_embedding = self.embed_chunks_batch([query_chunk])[0]

        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []

        # Search
        if use_ivf and self.ivf_centroids is not None:
            # IVF search
            results = self._search_ivf(query_embedding, k)
        else:
            # Brute-force search
            results = self._search_brute_force(query_embedding, k)

        return results

    def _search_ivf(self, query: torch.Tensor, k: int) -> List[Tuple[ChunkInfo, float]]:
        """Search using IVF index"""
        # Find closest centroids
        if query.dtype == torch.int8:
            query_float = query.float() / 127.0
            centroids_float = self.ivf_centroids.float() / 127.0
        else:
            query_float = query.float()
            centroids_float = self.ivf_centroids.float()

        centroid_distances = torch.cdist(query_float.unsqueeze(0), centroids_float).squeeze()
        probe_lists = min(self.config.probe_lists, len(self.ivf_lists))
        closest_centroids = centroid_distances.topk(probe_lists, largest=False).indices

        # Collect candidates from probe lists
        candidates = []
        for centroid_id in closest_centroids.cpu().numpy():
            candidates.extend(self.ivf_lists[centroid_id])

        if not candidates:
            return []

        # Score candidates
        candidate_embeddings = self.embeddings[candidates]

        if query.dtype == torch.int8:
            # Int8 dot product
            scores = torch.matmul(candidate_embeddings.float(), query.float())
        else:
            scores = torch.matmul(candidate_embeddings, query)

        # Get top-k
        actual_k = min(k, len(scores))
        top_scores, top_indices = scores.topk(actual_k)

        results = []
        for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
            chunk_idx = candidates[idx]
            results.append((self.chunk_database[chunk_idx], float(score)))

        return results

    def _search_brute_force(self, query: torch.Tensor, k: int) -> List[Tuple[ChunkInfo, float]]:
        """Brute-force search through all embeddings"""
        if query.dtype == torch.int8:
            scores = torch.matmul(self.embeddings.float(), query.float())
        else:
            scores = torch.matmul(self.embeddings, query)

        actual_k = min(k, len(scores))
        top_scores, top_indices = scores.topk(actual_k)

        results = []
        for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
            results.append((self.chunk_database[idx], float(score)))

        return results

    def save_index(self, path: str = None):
        """Save index to disk"""
        if path is None:
            path = self.config.index_path

        os.makedirs(path, exist_ok=True)

        # Save configuration
        with open(f"{path}/config.json", 'w') as f:
            json.dump(self.config.__dict__, f)

        # Save chunk database
        with open(f"{path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunk_database, f)

        # Save embeddings
        torch.save(self.embeddings, f"{path}/embeddings.pt")

        # Save IVF components
        if self.ivf_centroids is not None:
            torch.save(self.ivf_centroids, f"{path}/ivf_centroids.pt")
            with open(f"{path}/ivf_lists.pkl", 'wb') as f:
                pickle.dump(self.ivf_lists, f)

        # Save file index
        with open(f"{path}/file_index.pkl", 'wb') as f:
            pickle.dump(self.file_index, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: str = None):
        """Load index from disk"""
        if path is None:
            path = self.config.index_path

        # Load chunk database
        with open(f"{path}/chunks.pkl", 'rb') as f:
            self.chunk_database = pickle.load(f)

        # Load embeddings
        self.embeddings = torch.load(f"{path}/embeddings.pt", map_location=self.device)

        # Load IVF components if they exist
        if os.path.exists(f"{path}/ivf_centroids.pt"):
            self.ivf_centroids = torch.load(f"{path}/ivf_centroids.pt", map_location=self.device)
            with open(f"{path}/ivf_lists.pkl", 'rb') as f:
                self.ivf_lists = pickle.load(f)

        # Load file index
        with open(f"{path}/file_index.pkl", 'rb') as f:
            self.file_index = pickle.load(f)

        self.num_chunks = len(self.chunk_database)
        self.num_files = len(self.file_index)

        logger.info(f"Index loaded from {path}: {self.num_chunks} chunks, {self.num_files} files")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            'num_chunks': self.num_chunks,
            'num_files': self.num_files,
            'index_size_mb': self.index_size_mb,
            'embedding_dim': self.config.embedding_dim,
            'device': str(self.device),
            'use_ivf': self.ivf_centroids is not None,
        }

        if self.ivf_centroids is not None:
            stats['ivf_clusters'] = self.ivf_centroids.shape[0]
            stats['avg_points_per_cluster'] = self.num_chunks / self.ivf_centroids.shape[0]

        return stats


def main():
    """Example usage"""
    config = SearchConfig(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        chunk_size=256,
        ivf_clusters=256,
        probe_lists=16,
        use_int8=True
    )

    searcher = GPUFilesystemSearch(config)

    # Index current directory
    searcher.index_directory(".")

    # Save index
    searcher.save_index()

    # Example search
    results = searcher.search("embedding model GPU search", k=5)

    print("\nSearch Results:")
    for chunk, score in results:
        print(f"\nScore: {score:.4f}")
        print(f"File: {chunk.file_path}")
        print(f"Lines: {chunk.line_start}-{chunk.line_end}")
        print(f"Preview: {chunk.chunk_text[:200]}...")

    # Print stats
    print("\nIndex Stats:")
    for key, value in searcher.get_stats().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()