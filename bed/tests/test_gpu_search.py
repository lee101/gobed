#!/usr/bin/env python3
"""
Comprehensive tests for GPU-accelerated filesystem search
Tests indexing, search accuracy, performance, and edge cases
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import tempfile
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from gpu_filesystem_search import GPUFilesystemSearch, SearchConfig, ChunkInfo


class TestGPUFilesystemSearch(unittest.TestCase):
    """Test suite for GPU filesystem search"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config = SearchConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            chunk_size=128,
            ivf_clusters=4,
            probe_lists=2,
            use_int8=True
        )
        self.searcher = GPUFilesystemSearch(self.config)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_files(self):
        """Create test files with known content"""
        test_files = {
            "code.py": """
def calculate_similarity(vec1, vec2):
    # Calculate cosine similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def search_database(query, database, k=10):
    # Search for similar vectors in database
    scores = []
    for idx, vec in enumerate(database):
        score = calculate_similarity(query, vec)
        scores.append((idx, score))

    # Return top-k results
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
""",
            "readme.md": """
# GPU-Accelerated Search System

This system provides fast similarity search using GPU acceleration.

## Features
- Hierarchical IVF indexing for large-scale search
- Int8 quantization for memory efficiency
- Batch processing for high throughput
- File chunking for handling large documents

## Performance
- 100,000+ queries per second on RTX 3090
- Sub-millisecond latency for small batches
- Linear scaling with multiple GPUs
""",
            "data.txt": """
The quick brown fox jumps over the lazy dog.
Machine learning models can learn patterns from data.
Deep neural networks have revolutionized AI.
GPU acceleration enables faster computation.
Vector similarity search is fundamental to many applications.
Embeddings capture semantic meaning in numerical form.
""",
        }

        for filename, content in test_files.items():
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)

        return list(test_files.keys())

    def test_file_chunking(self):
        """Test file chunking functionality"""
        # Create a test file
        test_file = os.path.join(self.test_dir, "test.txt")
        lines = [f"Line {i}: " + "word " * 20 for i in range(100)]
        with open(test_file, 'w') as f:
            f.write('\n'.join(lines))

        # Chunk the file
        chunks = self.searcher.chunk_file(test_file)

        # Verify chunks
        self.assertGreater(len(chunks), 0)

        # Check chunk properties
        for chunk in chunks:
            self.assertEqual(chunk.file_path, test_file)
            self.assertGreaterEqual(chunk.line_end, chunk.line_start)
            self.assertGreater(len(chunk.chunk_text), 0)
            self.assertIsNotNone(chunk.chunk_hash)

        # Check for overlap between consecutive chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # There should be some overlap
                chunk1_end = chunks[i].line_end
                chunk2_start = chunks[i + 1].line_start
                self.assertLessEqual(chunk2_start, chunk1_end + 5)  # Allow small gap

    def test_embedding_generation(self):
        """Test embedding generation for chunks"""
        test_chunks = [
            ChunkInfo(
                file_path="test1.txt",
                start_offset=0,
                end_offset=100,
                line_start=0,
                line_end=5,
                chunk_text="This is a test chunk for embedding generation.",
                chunk_hash="hash1"
            ),
            ChunkInfo(
                file_path="test2.txt",
                start_offset=0,
                end_offset=150,
                line_start=0,
                line_end=7,
                chunk_text="Another test chunk with different content.",
                chunk_hash="hash2"
            )
        ]

        embeddings = self.searcher.embed_chunks_batch(test_chunks)

        # Verify embeddings
        self.assertEqual(embeddings.shape[0], len(test_chunks))
        self.assertEqual(embeddings.shape[1], self.config.embedding_dim)

        # Check embedding properties
        if self.config.use_int8:
            self.assertEqual(embeddings.dtype, torch.int8)
            self.assertTrue((embeddings >= -128).all())
            self.assertTrue((embeddings <= 127).all())
        else:
            self.assertIn(embeddings.dtype, [torch.float32, torch.float16])

    def test_ivf_training(self):
        """Test IVF centroid training"""
        # Create random embeddings
        n_embeddings = 1000
        embeddings = torch.randn(n_embeddings, self.config.embedding_dim)

        if self.config.use_int8:
            embeddings = (embeddings * 127).round().clamp(-128, 127).to(torch.int8)

        # Train IVF centroids
        centroids = self.searcher.train_ivf_centroids(embeddings)

        # Verify centroids
        expected_clusters = min(self.config.ivf_clusters, n_embeddings // 10)
        self.assertEqual(centroids.shape[0], expected_clusters)
        self.assertEqual(centroids.shape[1], self.config.embedding_dim)

        if self.config.use_int8:
            self.assertEqual(centroids.dtype, torch.int8)

    def test_ivf_assignment(self):
        """Test IVF list building and assignment"""
        # Create embeddings and centroids
        n_embeddings = 500
        n_clusters = 10

        embeddings = torch.randn(n_embeddings, self.config.embedding_dim)
        centroids = torch.randn(n_clusters, self.config.embedding_dim)

        if self.config.use_int8:
            embeddings = (embeddings * 127).round().clamp(-128, 127).to(torch.int8)
            centroids = (centroids * 127).round().clamp(-128, 127).to(torch.int8)

        # Build IVF lists
        ivf_lists = self.searcher.build_ivf_lists(embeddings, centroids)

        # Verify assignments
        self.assertEqual(len(ivf_lists), n_clusters)

        # Check that all points are assigned
        total_assigned = sum(len(points) for points in ivf_lists.values())
        self.assertEqual(total_assigned, n_embeddings)

        # Check that assignments are unique
        all_indices = []
        for indices in ivf_lists.values():
            all_indices.extend(indices)
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_directory_indexing(self):
        """Test indexing a directory of files"""
        # Create test files
        self.create_test_files()

        # Index directory
        self.searcher.index_directory(self.test_dir, extensions=['.py', '.md', '.txt'])

        # Verify indexing
        self.assertGreater(self.searcher.num_chunks, 0)
        self.assertGreater(self.searcher.num_files, 0)
        self.assertIsNotNone(self.searcher.embeddings)

        # Check file index
        self.assertGreater(len(self.searcher.file_index), 0)

        # Verify each file has chunks
        for file_path, chunk_indices in self.searcher.file_index.items():
            self.assertGreater(len(chunk_indices), 0)
            for idx in chunk_indices:
                self.assertLess(idx, len(self.searcher.chunk_database))

    def test_search_accuracy(self):
        """Test search accuracy and relevance"""
        # Create and index test files
        self.create_test_files()
        self.searcher.index_directory(self.test_dir)

        # Test searches
        test_queries = [
            ("calculate similarity vectors", ["code.py"]),
            ("GPU acceleration performance", ["readme.md"]),
            ("quick brown fox", ["data.txt"]),
        ]

        for query, expected_files in test_queries:
            results = self.searcher.search(query, k=3)

            self.assertGreater(len(results), 0)

            # Check if expected files appear in top results
            result_files = [Path(r[0].file_path).name for r in results]
            for expected_file in expected_files:
                self.assertIn(expected_file, result_files[:2],
                            f"Expected {expected_file} in top results for query '{query}'")

            # Verify scores are in descending order
            scores = [r[1] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ivf_vs_brute_force(self):
        """Compare IVF search with brute force search"""
        # Create and index test files
        self.create_test_files()
        self.searcher.index_directory(self.test_dir)

        query = "machine learning GPU acceleration"

        # Brute force search
        brute_force_results = self.searcher.search(query, k=5, use_ivf=False)

        # IVF search (if available)
        if self.searcher.ivf_centroids is not None:
            ivf_results = self.searcher.search(query, k=5, use_ivf=True)

            # Compare top results - IVF should find similar results
            bf_top = set(r[0].chunk_hash for r in brute_force_results[:3])
            ivf_top = set(r[0].chunk_hash for r in ivf_results[:3])

            # At least some overlap expected in top results
            overlap = len(bf_top & ivf_top)
            self.assertGreater(overlap, 0,
                             "IVF search should find some of the same top results as brute force")

    def test_save_and_load_index(self):
        """Test saving and loading index to/from disk"""
        # Create and index test files
        self.create_test_files()
        self.searcher.index_directory(self.test_dir)

        # Save index
        index_path = os.path.join(self.test_dir, "test_index")
        self.searcher.save_index(index_path)

        # Verify index files exist
        self.assertTrue(os.path.exists(os.path.join(index_path, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(index_path, "chunks.pkl")))
        self.assertTrue(os.path.exists(os.path.join(index_path, "embeddings.pt")))

        # Create new searcher and load index
        new_searcher = GPUFilesystemSearch(self.config)
        new_searcher.load_index(index_path)

        # Verify loaded index matches original
        self.assertEqual(new_searcher.num_chunks, self.searcher.num_chunks)
        self.assertEqual(new_searcher.num_files, self.searcher.num_files)
        self.assertEqual(len(new_searcher.file_index), len(self.searcher.file_index))

        # Test search on loaded index
        results = new_searcher.search("test query", k=5)
        self.assertIsNotNone(results)

    def test_performance_benchmark(self):
        """Benchmark search performance"""
        # Create larger dataset
        num_files = 10
        for i in range(num_files):
            content = f"File {i}\n" + "\n".join([
                f"Line {j}: " + " ".join([f"word{k}" for k in range(50)])
                for j in range(100)
            ])
            filepath = os.path.join(self.test_dir, f"file_{i}.txt")
            with open(filepath, 'w') as f:
                f.write(content)

        # Index files
        start_time = time.time()
        self.searcher.index_directory(self.test_dir)
        index_time = time.time() - start_time

        print(f"\nPerformance Benchmark:")
        print(f"  Indexed {self.searcher.num_chunks} chunks in {index_time:.2f}s")
        print(f"  Rate: {self.searcher.num_chunks / index_time:.0f} chunks/sec")

        # Benchmark search
        queries = ["test query " + str(i) for i in range(100)]

        start_time = time.time()
        for query in queries:
            _ = self.searcher.search(query, k=10)
        search_time = time.time() - start_time

        qps = len(queries) / search_time
        latency_ms = (search_time / len(queries)) * 1000

        print(f"  Search: {len(queries)} queries in {search_time:.2f}s")
        print(f"  QPS: {qps:.0f}")
        print(f"  Latency: {latency_ms:.2f}ms")

        # Verify performance meets expectations
        self.assertGreater(qps, 10, "Should achieve at least 10 QPS")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty directory
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        self.searcher.index_directory(empty_dir)
        self.assertEqual(self.searcher.num_chunks, 0)

        # Search on empty index
        results = self.searcher.search("test query", k=5)
        self.assertEqual(len(results), 0)

        # Very large file (should be skipped)
        large_file = os.path.join(self.test_dir, "large.txt")
        with open(large_file, 'w') as f:
            f.write("x" * (self.config.max_file_size + 1))

        chunks = self.searcher.chunk_file(large_file)
        self.assertEqual(len(chunks), 0)

        # File with no content
        empty_file = os.path.join(self.test_dir, "empty.txt")
        open(empty_file, 'w').close()
        chunks = self.searcher.chunk_file(empty_file)
        # Should handle gracefully

    def test_statistics(self):
        """Test statistics reporting"""
        # Index some files
        self.create_test_files()
        self.searcher.index_directory(self.test_dir)

        # Get stats
        stats = self.searcher.get_stats()

        # Verify stats
        self.assertIn('num_chunks', stats)
        self.assertIn('num_files', stats)
        self.assertIn('embedding_dim', stats)
        self.assertIn('device', stats)
        self.assertIn('use_ivf', stats)

        self.assertGreater(stats['num_chunks'], 0)
        self.assertGreater(stats['num_files'], 0)
        self.assertEqual(stats['embedding_dim'], self.config.embedding_dim)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete search pipeline"""

    def test_end_to_end_search(self):
        """Test complete end-to-end search workflow"""
        with tempfile.TemporaryDirectory() as test_dir:
            # Setup
            config = SearchConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
                chunk_size=256,
                ivf_clusters=16,
                probe_lists=4
            )
            searcher = GPUFilesystemSearch(config)

            # Create realistic test corpus
            test_files = {
                "neural_networks.py": """
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc(output)
""",
                "optimization.py": """
def gradient_descent(loss_fn, params, learning_rate=0.01):
    gradients = compute_gradients(loss_fn, params)
    for param, grad in zip(params, gradients):
        param.data -= learning_rate * grad
    return params

def adam_optimizer(params, gradients, m, v, t, lr=0.001, beta1=0.9, beta2=0.999):
    for i, (param, grad) in enumerate(zip(params, gradients)):
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * grad**2
        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)
        param.data -= lr * m_hat / (v_hat**0.5 + 1e-8)
""",
                "database.py": """
class VectorDatabase:
    def __init__(self, dimension, metric='cosine'):
        self.dimension = dimension
        self.metric = metric
        self.vectors = []
        self.metadata = []

    def add_vector(self, vector, metadata=None):
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch")
        self.vectors.append(vector)
        self.metadata.append(metadata)

    def search(self, query_vector, k=10):
        scores = []
        for i, vec in enumerate(self.vectors):
            score = self.compute_similarity(query_vector, vec)
            scores.append((i, score, self.metadata[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
"""
            }

            for filename, content in test_files.items():
                filepath = os.path.join(test_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)

            # Index
            searcher.index_directory(test_dir)

            # Search tests
            test_cases = [
                ("transformer neural network architecture", "neural_networks.py"),
                ("adam optimizer learning rate", "optimization.py"),
                ("vector similarity search database", "database.py"),
            ]

            for query, expected_file in test_cases:
                results = searcher.search(query, k=3)
                self.assertGreater(len(results), 0)

                # Check if expected file is in top result
                top_result = results[0][0]
                self.assertIn(expected_file, top_result.file_path)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)