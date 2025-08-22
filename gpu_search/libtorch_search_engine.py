#!/usr/bin/env python3
"""
High-performance search engine using LibTorch and custom CUDA operations.
Provides production-ready search with batch processing and memory optimization.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
import logging

# Load our custom CUDA operations
torch.ops.load_library('./cuda_ops/build/libgobed_ann_ops.so')

@dataclass
class SearchResult:
    """Single search result"""
    id: int
    score: float
    text: Optional[str] = None

@dataclass
class BatchSearchRequest:
    """Batch search request"""
    queries: List[torch.Tensor]
    k: int = 10
    request_id: Optional[str] = None

@dataclass
class SearchEngineConfig:
    """Configuration for the search engine"""
    device: str = "cuda:0"
    max_batch_size: int = 32
    max_queue_size: int = 1000
    num_worker_threads: int = 4
    enable_caching: bool = True
    cache_size: int = 10000
    
class SearchCache:
    """LRU cache for search results"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, query_hash: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached result"""
        with self.lock:
            if query_hash in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(query_hash)
                self.access_order.append(query_hash)
                return self.cache[query_hash]
            return None
    
    def put(self, query_hash: str, result: Tuple[torch.Tensor, torch.Tensor]):
        """Put result in cache"""
        with self.lock:
            if query_hash in self.cache:
                # Update existing
                self.access_order.remove(query_hash)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[query_hash] = result
            self.access_order.append(query_hash)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

class LibTorchSearchEngine(nn.Module):
    """High-performance search engine using LibTorch and custom CUDA ops"""
    
    def __init__(self, indexer, config: SearchEngineConfig):
        super().__init__()
        self.indexer = indexer
        self.config = config
        self.device = torch.device(config.device)
        
        # Threading for batch processing
        self.request_queue = queue.Queue(maxsize=config.max_queue_size)
        self.workers = []
        self.running = False
        
        # Caching
        self.cache = SearchCache(config.cache_size) if config.enable_caching else None
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_searches': 0,
            'avg_batch_size': 0.0,
            'total_latency_ms': 0.0
        }
        self.stats_lock = threading.Lock()
        
        print(f"🔍 Initialized LibTorchSearchEngine")
        print(f"   Device: {self.device}")
        print(f"   Max batch size: {config.max_batch_size}")
        print(f"   Worker threads: {config.num_worker_threads}")
        print(f"   Caching: {'enabled' if config.enable_caching else 'disabled'}")
    
    def start_workers(self):
        """Start background worker threads for batch processing"""
        self.running = True
        
        for i in range(self.config.num_worker_threads):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"SearchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"🚀 Started {len(self.workers)} search worker threads")
    
    def stop_workers(self):
        """Stop background worker threads"""
        self.running = False
        
        # Signal workers to stop
        for _ in self.workers:
            self.request_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        print("🛑 Stopped search worker threads")
    
    def _worker_loop(self):
        """Main loop for worker threads"""
        while self.running:
            try:
                # Get batch of requests
                requests = []
                timeout = 0.01  # 10ms timeout for batching
                
                # Get first request (blocking)
                first_request = self.request_queue.get(timeout=1.0)
                if first_request is None:  # Shutdown signal
                    break
                requests.append(first_request)
                
                # Try to collect more requests for batching
                batch_deadline = time.time() + timeout
                while (len(requests) < self.config.max_batch_size and 
                       time.time() < batch_deadline):
                    try:
                        req = self.request_queue.get_nowait()
                        if req is None:  # Shutdown signal
                            break
                        requests.append(req)
                    except queue.Empty:
                        break
                
                if requests and requests[0] is not None:
                    self._process_batch(requests)
                    
            except Exception as e:
                logging.error(f"Worker error: {e}")
    
    def _process_batch(self, requests: List[Dict]):
        """Process a batch of search requests"""
        start_time = time.time()
        
        # Group requests by k value for efficient batching
        k_groups = {}
        for req in requests:
            k = req['k']
            if k not in k_groups:
                k_groups[k] = []
            k_groups[k].append(req)
        
        # Process each k group
        for k, group_requests in k_groups.items():
            try:
                # Extract queries and futures
                queries = []
                futures = []
                
                for req in group_requests:
                    queries.append(req['query'])
                    futures.append(req['future'])
                
                # Batch search
                if len(queries) == 1:
                    # Single query
                    ids, scores = self._search_single(queries[0], k)
                    results = [(ids, scores)]
                else:
                    # Batch search
                    results = self._search_batch(queries, k)
                
                # Set results
                for i, future in enumerate(futures):
                    if i < len(results):
                        future.set_result(results[i])
                    else:
                        future.set_exception(RuntimeError("Batch result missing"))
                        
            except Exception as e:
                # Set exception for all futures in this group
                for req in group_requests:
                    req['future'].set_exception(e)
        
        # Update statistics
        batch_time = (time.time() - start_time) * 1000
        with self.stats_lock:
            self.stats['batch_searches'] += 1
            self.stats['total_latency_ms'] += batch_time
            
            # Update average batch size
            prev_avg = self.stats['avg_batch_size']
            count = self.stats['batch_searches']
            self.stats['avg_batch_size'] = (prev_avg * (count - 1) + len(requests)) / count
    
    def search(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search for k nearest neighbors (synchronous)"""
        # Check cache first
        query_hash = None
        if self.cache:
            query_hash = self._hash_query(query, k)
            cached_result = self.cache.get(query_hash)
            if cached_result is not None:
                with self.stats_lock:
                    self.stats['cache_hits'] += 1
                    self.stats['total_searches'] += 1
                return cached_result
        
        # Perform search
        start_time = time.time()
        ids, scores = self._search_single(query, k)
        search_time = (time.time() - start_time) * 1000
        
        # Update cache
        if self.cache and query_hash:
            self.cache.put(query_hash, (ids.clone(), scores.clone()))
            with self.stats_lock:
                self.stats['cache_misses'] += 1
        
        # Update statistics
        with self.stats_lock:
            self.stats['total_searches'] += 1
            self.stats['total_latency_ms'] += search_time
        
        return ids, scores
    
    def search_async(self, query: torch.Tensor, k: int = 10) -> 'concurrent.futures.Future':
        """Search for k nearest neighbors (asynchronous)"""
        from concurrent.futures import Future
        
        # Check cache first
        if self.cache:
            query_hash = self._hash_query(query, k)
            cached_result = self.cache.get(query_hash)
            if cached_result is not None:
                future = Future()
                future.set_result(cached_result)
                with self.stats_lock:
                    self.stats['cache_hits'] += 1
                    self.stats['total_searches'] += 1
                return future
        
        # Queue for async processing
        future = Future()
        request = {
            'query': query.clone(),
            'k': k,
            'future': future
        }
        
        try:
            self.request_queue.put(request, timeout=5.0)
        except queue.Full:
            future.set_exception(RuntimeError("Search queue is full"))
        
        return future
    
    def batch_search(self, queries: List[torch.Tensor], k: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Batch search for multiple queries"""
        if not queries:
            return []
        
        if len(queries) == 1:
            result = self.search(queries[0], k)
            return [result]
        
        return self._search_batch(queries, k)
    
    def _search_single(self, query: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single query search implementation"""
        query = query.to(self.device, dtype=torch.int8)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        return self.indexer.search(query[0], k)
    
    def _search_batch(self, queries: List[torch.Tensor], k: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Batch search implementation"""
        # Stack queries into batch tensor
        batch_queries = torch.stack(queries).to(self.device, dtype=torch.int8)
        
        results = []
        for i in range(len(batch_queries)):
            ids, scores = self.indexer.search(batch_queries[i], k)
            results.append((ids, scores))
        
        return results
    
    def _hash_query(self, query: torch.Tensor, k: int) -> str:
        """Create hash for query caching"""
        query_bytes = query.cpu().numpy().tobytes()
        import hashlib
        hash_obj = hashlib.md5(query_bytes + str(k).encode())
        return hash_obj.hexdigest()
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Calculate derived metrics
        total_searches = stats['total_searches']
        if total_searches > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / total_searches
            stats['cache_hit_rate'] = stats['cache_hits'] / total_searches
        else:
            stats['avg_latency_ms'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Add system stats
        stats['queue_size'] = self.request_queue.qsize()
        stats['worker_threads'] = len(self.workers)
        stats['cache_size'] = len(self.cache.cache) if self.cache else 0
        
        return stats
    
    def clear_cache(self):
        """Clear search cache"""
        if self.cache:
            self.cache.clear()
            print("🗑️  Search cache cleared")
    
    def warm_up(self, num_queries: int = 100):
        """Warm up the search engine with random queries"""
        print(f"🔥 Warming up search engine with {num_queries} queries...")
        
        # Generate random queries
        query_dim = self.indexer.config.vector_dim
        warm_queries = torch.randint(-128, 127, (num_queries, query_dim), 
                                   device=self.device, dtype=torch.int8)
        
        start_time = time.time()
        for query in warm_queries:
            self.search(query, k=10)
        
        warmup_time = time.time() - start_time
        qps = num_queries / warmup_time
        
        print(f"   Warmup completed in {warmup_time:.2f}s")
        print(f"   Warmup QPS: {qps:.0f}")

class SearchBenchmark:
    """Comprehensive benchmarking for the search engine"""
    
    def __init__(self, search_engine: LibTorchSearchEngine):
        self.search_engine = search_engine
    
    def run_throughput_test(self, num_queries: int = 1000, k: int = 10):
        """Test search throughput"""
        print(f"\n🚀 Throughput test: {num_queries} queries, k={k}")
        
        # Generate test queries
        query_dim = self.search_engine.indexer.config.vector_dim
        queries = torch.randint(-128, 127, (num_queries, query_dim),
                              device=self.search_engine.device, dtype=torch.int8)
        
        # Synchronous test
        print("   Synchronous search...")
        start_time = time.time()
        for query in queries:
            self.search_engine.search(query, k)
        sync_time = time.time() - start_time
        sync_qps = num_queries / sync_time
        
        print(f"   Sync QPS: {sync_qps:.0f}")
        print(f"   Avg latency: {sync_time * 1000 / num_queries:.2f}ms")
        
        # Batch test
        print("   Batch search...")
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            if batch_size > num_queries:
                continue
                
            start_time = time.time()
            for i in range(0, num_queries, batch_size):
                end_idx = min(i + batch_size, num_queries)
                batch = queries[i:end_idx].unbind(0)
                self.search_engine.batch_search(batch, k)
            
            batch_time = time.time() - start_time
            batch_qps = num_queries / batch_time
            
            print(f"   Batch {batch_size:2d} QPS: {batch_qps:.0f}")
    
    def run_latency_test(self, num_queries: int = 100):
        """Test search latency distribution"""
        print(f"\n⏱️  Latency test: {num_queries} queries")
        
        query_dim = self.search_engine.indexer.config.vector_dim
        queries = torch.randint(-128, 127, (num_queries, query_dim),
                              device=self.search_engine.device, dtype=torch.int8)
        
        latencies = []
        for query in queries:
            start_time = time.time()
            self.search_engine.search(query, k=10)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        print(f"   Mean latency: {np.mean(latencies):.2f}ms")
        print(f"   Median latency: {np.median(latencies):.2f}ms")
        print(f"   95th percentile: {np.percentile(latencies, 95):.2f}ms")
        print(f"   99th percentile: {np.percentile(latencies, 99):.2f}ms")
        print(f"   Max latency: {np.max(latencies):.2f}ms")
    
    def run_accuracy_test(self, num_queries: int = 50):
        """Test search accuracy by comparing with brute force"""
        print(f"\n🎯 Accuracy test: {num_queries} queries")
        
        indexer = self.search_engine.indexer
        query_dim = indexer.config.vector_dim
        
        # Use a subset of indexed vectors as queries for ground truth
        if indexer.database is None or indexer.num_vectors == 0:
            print("   No vectors in database for accuracy test")
            return
        
        # Sample queries from database
        query_indices = torch.randperm(indexer.num_vectors)[:num_queries]
        queries = indexer.database[query_indices]
        
        k_values = [1, 5, 10, 20]
        recalls = {k: [] for k in k_values}
        
        for i, query in enumerate(queries):
            # Get ground truth (brute force search)
            all_scores = torch.ops.gobed_ann.i8dot512_scores(query, indexer.database)
            _, true_top_k = torch.topk(all_scores, max(k_values), largest=True)
            
            # Get approximate results
            approx_ids, _ = self.search_engine.search(query, k=max(k_values))
            
            # Calculate recall@k for each k
            for k in k_values:
                true_set = set(true_top_k[:k].cpu().numpy())
                approx_set = set(approx_ids[:k].cpu().numpy())
                recall = len(true_set & approx_set) / len(true_set)
                recalls[k].append(recall)
        
        # Print results
        for k in k_values:
            avg_recall = np.mean(recalls[k])
            print(f"   Recall@{k:2d}: {avg_recall:.3f}")

def main():
    """Main benchmark and test function"""
    print("🔥 LibTorch Search Engine Test")
    print("=" * 40)
    
    # Import and create indexer
    from libtorch_indexer import LibTorchIndexer, IndexConfig, create_test_vectors
    
    # Configuration
    index_config = IndexConfig(
        device="cuda:0",
        batch_size=1024,
        num_subquantizers=64,
        codebook_size=256,
        vector_dim=512,
        ivf_clusters=512,  # Smaller for faster testing
        probe_lists=8,
        rerank_k=100
    )
    
    search_config = SearchEngineConfig(
        device="cuda:0",
        max_batch_size=16,
        max_queue_size=1000,
        num_worker_threads=2,
        enable_caching=True,
        cache_size=1000
    )
    
    # Create and train indexer
    print("🔧 Setting up indexer...")
    indexer = LibTorchIndexer(index_config)
    
    # Create test data
    train_vectors = create_test_vectors(5000, 512)
    index_vectors = create_test_vectors(50000, 512)
    
    # Train and build index
    indexer.train_index(train_vectors)
    indexer.add_vectors(index_vectors)
    
    # Create search engine
    print("\n🔍 Setting up search engine...")
    search_engine = LibTorchSearchEngine(indexer, search_config)
    search_engine.start_workers()
    
    try:
        # Warm up
        search_engine.warm_up(100)
        
        # Run benchmarks
        benchmark = SearchBenchmark(search_engine)
        benchmark.run_throughput_test(1000)
        benchmark.run_latency_test(100)
        benchmark.run_accuracy_test(20)
        
        # Print final statistics
        print("\n📊 Final Statistics:")
        stats = search_engine.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
    finally:
        search_engine.stop_workers()
    
    print("\n✅ LibTorch search engine test completed!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available")
        exit(1)
    
    main()