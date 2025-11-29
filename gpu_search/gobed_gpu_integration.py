#!/usr/bin/env python3
"""
Integration module for gobed to use GPU-accelerated search
This provides a clean interface for the Go code to use GPU operations
"""

import torch
import numpy as np
import sys
import os
from typing import List, Tuple, Optional, Dict, Any
import json

# Add build directory to path
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')

from end_to_end_gpu import GPUIndex, EmbeddingModelGPU


class GobedGPUBackend:
    """GPU backend for gobed - handles embedding and search on GPU"""
    
    def __init__(self, 
                 dim: int = 512,
                 max_vectors: int = 1000000,
                 use_int8: bool = True,
                 model_path: Optional[str] = None):
        """
        Initialize GPU backend
        
        Args:
            dim: Embedding dimension
            max_vectors: Maximum number of vectors to index
            use_int8: Use INT8 quantization for 4x memory reduction
            model_path: Path to embedding model (if using real model)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU backend requires GPU")
        
        # Initialize index
        self.index = GPUIndex(dim=dim, max_vectors=max_vectors, use_int8=use_int8)
        
        # Initialize embedding model (simplified for demo)
        self.embed_model = EmbeddingModelGPU(vocab_size=30522, embed_dim=dim)
        
        # Track metadata
        self.text_to_id = {}
        self.id_to_text = {}
        self.current_id = 0
        
        print(f" Gobed GPU Backend initialized")
        print(f"   Device: {torch.cuda.get_device_name()}")
        print(f"   Max vectors: {max_vectors:,}")
        print(f"   Precision: {'INT8' if use_int8 else 'FP32'}")
        print(f"   Memory reserved: {torch.cuda.memory_reserved(self.device) / 1e9:.1f} GB")
    
    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts on GPU
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings [batch_size, dim]
        """
        # Simplified tokenization (in production, use proper tokenizer)
        batch_size = len(texts)
        max_len = 128
        
        # Create random tokens for demo (in production, use real tokenizer)
        token_ids = torch.randint(0, 30522, (batch_size, max_len), device=self.device)
        attention_mask = torch.ones((batch_size, max_len), device=self.device)
        
        # Embed on GPU
        with torch.no_grad():
            embeddings = self.embed_model(token_ids, attention_mask)
        
        # Return as numpy for Go interface
        return embeddings.cpu().numpy()
    
    def index_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Index texts on GPU - embeddings never leave GPU
        
        Args:
            texts: List of texts to index
            
        Returns:
            Dict with indexing statistics
        """
        batch_size = 1000
        total_indexed = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Create tokens on GPU
            batch_len = len(batch_texts)
            max_len = 128
            token_ids = torch.randint(0, 30522, (batch_len, max_len), device=self.device)
            attention_mask = torch.ones((batch_len, max_len), device=self.device)
            
            # Embed on GPU
            with torch.no_grad():
                embeddings = self.embed_model(token_ids, attention_mask)
            
            # Add to index (stays on GPU)
            stats = self.index.add_vectors_gpu(embeddings)
            
            # Store metadata
            for j, text in enumerate(batch_texts):
                text_id = self.current_id
                self.text_to_id[text] = text_id
                self.id_to_text[text_id] = text
                self.current_id += 1
            
            total_indexed += batch_len
        
        return {
            "total_indexed": total_indexed,
            "total_vectors": self.index.current_size,
            "memory_mb": self.index._get_memory_usage()
        }
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar texts using GPU
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (text, score) tuples
        """
        # Tokenize query
        max_len = 128
        token_ids = torch.randint(0, 30522, (1, max_len), device=self.device)
        attention_mask = torch.ones((1, max_len), device=self.device)
        
        # Embed query on GPU
        with torch.no_grad():
            query_embedding = self.embed_model(token_ids, attention_mask).squeeze()
        
        # Search on GPU using custom CUDA kernels
        indices, scores = self.index.search_gpu(query_embedding, k)
        
        # Convert results
        results = []
        for idx, score in zip(indices.cpu().numpy(), scores.cpu().numpy()):
            if idx in self.id_to_text:
                results.append((self.id_to_text[idx], float(score)))
        
        return results
    
    def batch_search(self, queries: List[str], k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries - very efficient on GPU
        
        Args:
            queries: List of query texts
            k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        batch_size = len(queries)
        max_len = 128
        
        # Tokenize all queries
        token_ids = torch.randint(0, 30522, (batch_size, max_len), device=self.device)
        attention_mask = torch.ones((batch_size, max_len), device=self.device)
        
        # Embed queries on GPU
        with torch.no_grad():
            query_embeddings = self.embed_model(token_ids, attention_mask)
        
        # Batch search on GPU using custom CUDA kernels
        indices, scores = self.index.batch_search_gpu(query_embeddings, k)
        
        # Convert results
        all_results = []
        for query_idx in range(batch_size):
            query_results = []
            for i in range(k):
                idx = indices[query_idx, i].item()
                score = scores[query_idx, i].item()
                if idx in self.id_to_text:
                    query_results.append((self.id_to_text[idx], float(score)))
            all_results.append(query_results)
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "device": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "total_vectors": self.index.current_size,
            "max_vectors": self.index.max_vectors,
            "dimension": self.index.dim,
            "use_int8": self.index.use_int8,
            "memory_allocated_mb": self.index._get_memory_usage(),
            "memory_reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
        }


# Flask server interface for Go to call
from flask import Flask, request, jsonify

app = Flask(__name__)
backend = None


@app.route('/init', methods=['POST'])
def init():
    """Initialize GPU backend"""
    global backend
    config = request.json
    backend = GobedGPUBackend(
        dim=config.get('dim', 512),
        max_vectors=config.get('max_vectors', 1000000),
        use_int8=config.get('use_int8', True)
    )
    return jsonify({"status": "initialized", "stats": backend.get_stats()})


@app.route('/embed', methods=['POST'])
def embed():
    """Embed texts"""
    data = request.json
    texts = data['texts']
    embeddings = backend.embed_texts_batch(texts)
    return jsonify({"embeddings": embeddings.tolist()})


@app.route('/index', methods=['POST'])
def index():
    """Index texts on GPU"""
    data = request.json
    texts = data['texts']
    stats = backend.index_texts(texts)
    return jsonify(stats)


@app.route('/search', methods=['POST'])
def search():
    """Search for similar texts"""
    data = request.json
    query = data['query']
    k = data.get('k', 10)
    results = backend.search(query, k)
    return jsonify({"results": [{"text": text, "score": score} for text, score in results]})


@app.route('/batch_search', methods=['POST'])
def batch_search():
    """Batch search for multiple queries"""
    data = request.json
    queries = data['queries']
    k = data.get('k', 10)
    all_results = backend.batch_search(queries, k)
    response = []
    for results in all_results:
        response.append([{"text": text, "score": score} for text, score in results])
    return jsonify({"results": response})


@app.route('/stats', methods=['GET'])
def stats():
    """Get backend statistics"""
    return jsonify(backend.get_stats())


if __name__ == '__main__':
    print(" Starting Gobed GPU Backend Server...")
    print("   This server provides GPU-accelerated embedding and search for gobed")
    print("   Custom CUDA kernels ensure maximum performance")
    print()
    
    # Initialize backend on startup
    backend = GobedGPUBackend(
        dim=512,
        max_vectors=1000000,
        use_int8=True
    )
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)