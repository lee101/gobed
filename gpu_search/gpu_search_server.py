#!/usr/bin/env python3
"""
GPU Search Server for Gobed
Provides HTTP API for GPU-accelerated similarity search
"""

import torch
import numpy as np
from flask import Flask, request, jsonify
import time
import logging
from transformers import AutoTokenizer, AutoModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global database and model
db = None
device = None
tokenizer = None
model = None

def init_gpu():
    """Initialize GPU device and load embedding model."""
    global device, tokenizer, model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f" Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("  CUDA not available, using CPU")
    
    # Load embedding model on GPU
    try:
        model_name = "jinaai/jina-embeddings-v2-base-en"  # Alternative if gobed model not available
        logger.info(f"🔄 Loading embedding model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()  # Set to eval mode for inference
        logger.info(" Embedding model loaded on GPU")
    except Exception as e:
        logger.warning(f"  Could not load embedding model: {e}")
        tokenizer = None
        model = None
    
    return device

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    gpu_memory_mb = 0
    if db is not None:
        gpu_memory_mb = db.element_size() * db.nelement() / 1e6
    
    gpu_memory_total = 0
    gpu_memory_free = 0
    if torch.cuda.is_available():
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e6
        gpu_memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e6
    
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "database_loaded": db is not None,
        "database_size": len(db) if db is not None else 0,
        "gpu_memory_used_mb": gpu_memory_mb,
        "gpu_memory_total_mb": gpu_memory_total,
        "gpu_memory_free_mb": gpu_memory_free
    })

@app.route('/load', methods=['POST'])
def load_database():
    """Load embeddings into GPU memory."""
    global db
    
    try:
        data = request.json
        embeddings = np.array(data['embeddings'], dtype=np.int8)
        
        # Convert to torch tensor and move to GPU
        db = torch.from_numpy(embeddings).to(device)
        
        logger.info(f"Loaded {len(db)} embeddings to {device}")
        
        return jsonify({
            "status": "loaded",
            "count": len(db),
            "shape": list(db.shape),
            "device": str(device),
            "memory_mb": db.element_size() * db.nelement() / 1e6
        })
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Perform similarity search."""
    if db is None:
        return jsonify({"error": "Database not loaded"}), 400
    
    try:
        data = request.json
        query = torch.tensor(data['query'], dtype=torch.int8).to(device)
        k = data.get('k', 10)
        
        # Ensure k doesn't exceed database size
        k = min(k, len(db))
        
        # Perform search
        start = time.perf_counter()
        
        # Convert to float for computation
        scores = torch.matmul(query.float(), db.float().T)
        
        # Get top-k results
        values, indices = torch.topk(scores, k)
        
        # Synchronize for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        search_time = (time.perf_counter() - start) * 1000  # ms
        
        # Convert to Python lists
        ids = indices.cpu().tolist()
        scores_list = values.cpu().tolist()
        
        logger.info(f"Search completed in {search_time:.2f}ms, k={k}")
        
        return jsonify({
            "ids": ids,
            "scores": scores_list,
            "search_time_ms": search_time,
            "k": k
        })
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_search', methods=['POST'])
def batch_search():
    """Perform batch similarity search."""
    if db is None:
        return jsonify({"error": "Database not loaded"}), 400
    
    try:
        data = request.json
        queries = torch.tensor(data['queries'], dtype=torch.int8).to(device)
        k = data.get('k', 10)
        
        # Ensure k doesn't exceed database size
        k = min(k, len(db))
        
        # Perform batch search
        start = time.perf_counter()
        
        # Batch matrix multiplication
        scores = torch.matmul(queries.float(), db.float().T)
        
        # Get top-k for each query
        values, indices = torch.topk(scores, k, dim=1)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        search_time = (time.perf_counter() - start) * 1000  # ms
        
        # Convert to Python lists
        batch_ids = indices.cpu().tolist()
        batch_scores = values.cpu().tolist()
        
        logger.info(f"Batch search ({len(queries)} queries) completed in {search_time:.2f}ms")
        
        return jsonify({
            "batch_ids": batch_ids,
            "batch_scores": batch_scores,
            "batch_size": len(queries),
            "search_time_ms": search_time,
            "k": k,
            "qps": len(queries) * 1000 / search_time
        })
    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed_texts():
    """Generate embeddings for texts using GPU model."""
    if model is None or tokenizer is None:
        return jsonify({"error": "Embedding model not loaded"}), 400
    
    try:
        data = request.json
        texts = data['texts']
        
        start = time.perf_counter()
        
        # Tokenize texts
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to int8 for memory efficiency
        embeddings_np = embeddings.cpu().numpy()
        
        # Quantize to int8 (simple scaling)
        embeddings_int8 = []
        for emb in embeddings_np:
            # Scale to int8 range
            max_val = np.abs(emb).max()
            if max_val > 0:
                scaled = (emb / max_val * 127).astype(np.int8)
            else:
                scaled = np.zeros_like(emb, dtype=np.int8)
            embeddings_int8.append(scaled.tolist())
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        embed_time = (time.perf_counter() - start) * 1000  # ms
        
        logger.info(f"GPU embedded {len(texts)} texts in {embed_time:.2f}ms ({len(texts)*1000/embed_time:.0f} texts/sec)")
        
        return jsonify({
            "embeddings": embeddings_int8,
            "count": len(texts),
            "embed_time_ms": embed_time,
            "texts_per_sec": len(texts) * 1000 / embed_time,
            "dimensions": len(embeddings_int8[0]) if embeddings_int8 else 0
        })
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_database():
    """Clear the database from memory."""
    global db
    
    memory_freed = 0
    if db is not None:
        memory_freed = db.element_size() * db.nelement() / 1e6
    
    db = None
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info(f"🗑  Database cleared, freed {memory_freed:.1f} MB GPU memory")
    
    return jsonify({
        "status": "cleared",
        "memory_freed_mb": memory_freed
    })

@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Run a quick benchmark."""
    if db is None:
        return jsonify({"error": "Database not loaded"}), 400
    
    try:
        results = {}
        
        # Single query benchmark - use same dimensions as database
        query_dims = db.shape[1] if db is not None else 1024  # Default to model dimension
        query = torch.randint(-128, 127, (query_dims,), dtype=torch.int8, device=device)
        iterations = 100
        
        # Warmup
        for _ in range(10):
            torch.matmul(query.float(), db.float().T)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            scores = torch.matmul(query.float(), db.float().T)
            values, indices = torch.topk(scores, 10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        avg_latency = (elapsed / iterations) * 1000
        
        results['single_query'] = {
            'avg_latency_ms': avg_latency,
            'qps': 1000 / avg_latency,
            'iterations': iterations
        }
        
        # Batch benchmark
        batch_size = 128  # Increased for better GPU utilization
        queries = torch.randint(-128, 127, (batch_size, query_dims), dtype=torch.int8, device=device)
        batch_iterations = 20  # More iterations for stable measurements
        
        start = time.perf_counter()
        for _ in range(batch_iterations):
            scores = torch.matmul(queries.float(), db.float().T)
            values, indices = torch.topk(scores, 10, dim=1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        batch_latency = (elapsed / batch_iterations) * 1000
        batch_qps = (batch_size * batch_iterations * 1000) / (elapsed * 1000)
        
        results['batch'] = {
            'batch_size': batch_size,
            'batch_latency_ms': batch_latency,
            'qps': batch_qps,
            'iterations': batch_iterations
        }
        
        results['database'] = {
            'size': len(db),
            'dimensions': db.shape[1],
            'device': str(device),
            'memory_mb': db.element_size() * db.nelement() / 1e6
        }
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize GPU
    init_gpu()
    
    # Create default test database
    if device is not None:
        logger.info("Creating default test database...")
        test_db = torch.randint(-128, 127, (10000, 512), dtype=torch.int8, device=device)
        db = test_db
        logger.info(f"Loaded test database with {len(db)} vectors")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)