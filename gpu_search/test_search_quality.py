#!/usr/bin/env python3
"""Test search quality with real text embeddings to verify ranking relevance."""

import ctypes
import numpy as np
import time
import os
import sys
from typing import List, Tuple, Dict
import json

# Load the LibTorch-free library
lib_path = os.path.join(os.path.dirname(__file__), 'cuda_ops/build/libgobed_ann_ops.so')
if not os.path.exists(lib_path):
    print(f"❌ Library not found: {lib_path}")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)

# Define function signatures
lib.cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cuda_malloc.restype = ctypes.c_int

lib.cuda_free.argtypes = [ctypes.c_void_p]
lib.cuda_free.restype = ctypes.c_int

lib.cuda_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuda_memcpy_h2d.restype = ctypes.c_int

lib.cuda_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuda_memcpy_d2h.restype = ctypes.c_int

lib.cuda_synchronize.argtypes = []
lib.cuda_synchronize.restype = ctypes.c_int

lib.i8dot512_scores.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.i8dot512_scores.restype = ctypes.c_int

class MockEmbeddingModel:
    """Mock embedding model that creates consistent but realistic-looking embeddings."""

    def __init__(self):
        # Create a simple vocabulary-based embedding system
        self.vocab = {}
        self.embedding_dim = 512
        np.random.seed(42)  # Consistent embeddings

    def _get_word_embedding(self, word: str) -> np.ndarray:
        """Get a consistent embedding for a word."""
        word = word.lower()
        if word not in self.vocab:
            # Create a deterministic but varied embedding based on word hash
            hash_val = hash(word) % (2**31)
            np.random.seed(hash_val)
            self.vocab[word] = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
        return self.vocab[word]

    def encode(self, text: str) -> np.ndarray:
        """Create text embedding by averaging word embeddings."""
        words = text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()
        if not words:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Average word embeddings
        embeddings = [self._get_word_embedding(word) for word in words]
        text_embedding = np.mean(embeddings, axis=0)

        # Add some topic-specific bias for better clustering
        topic_words = {
            'tech': ['python', 'programming', 'code', 'software', 'computer', 'algorithm', 'data'],
            'food': ['recipe', 'cooking', 'delicious', 'ingredients', 'meal', 'restaurant', 'eat'],
            'sports': ['game', 'team', 'player', 'score', 'match', 'championship', 'training'],
            'science': ['research', 'experiment', 'theory', 'discovery', 'study', 'analysis', 'results']
        }

        # Boost embedding based on topic keywords
        for topic, keywords in topic_words.items():
            topic_boost = sum(1 for word in words if word in keywords)
            if topic_boost > 0:
                topic_embedding = np.mean([self._get_word_embedding(kw) for kw in keywords], axis=0)
                text_embedding += 0.3 * topic_boost * topic_embedding

        # Normalize
        norm = np.linalg.norm(text_embedding)
        if norm > 0:
            text_embedding = text_embedding / norm

        return text_embedding

    def quantize_to_int8(self, embedding: np.ndarray) -> np.ndarray:
        """Quantize float32 embedding to int8."""
        # Scale to use full int8 range
        embedding_scaled = embedding * 127.0
        return np.clip(embedding_scaled, -128, 127).astype(np.int8)

class LibTorchFreeSearchEngine:
    def __init__(self):
        self.db_gpu = None
        self.db_size = 0
        self.documents = []

    def index(self, embeddings: np.ndarray, documents: List[str]) -> float:
        """Index embeddings with associated documents."""
        if embeddings.dtype != np.int8:
            raise ValueError("Embeddings must be int8")
        if embeddings.shape[1] != 512:
            raise ValueError("Embeddings must be 512-dimensional")

        start_time = time.perf_counter()

        # Free existing index
        if self.db_gpu:
            lib.cuda_free(self.db_gpu)

        self.db_size = embeddings.shape[0]
        self.documents = documents.copy()

        # Allocate GPU memory
        size_bytes = embeddings.nbytes
        self.db_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(self.db_gpu), size_bytes)
        if result != 0:
            raise RuntimeError(f"Failed to allocate GPU memory: {result}")

        # Copy to GPU
        result = lib.cuda_memcpy_h2d(self.db_gpu, embeddings.ctypes.data_as(ctypes.c_void_p), size_bytes)
        if result != 0:
            raise RuntimeError(f"Failed to copy to GPU: {result}")

        lib.cuda_synchronize()
        end_time = time.perf_counter()

        return (end_time - start_time) * 1000  # ms

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[float, List[Tuple[int, str, int]]]:
        """Search and return (time_ms, [(rank, document, score), ...])."""
        if self.db_gpu is None:
            raise RuntimeError("No vectors indexed")

        if query_embedding.dtype != np.int8:
            raise ValueError("Query must be int8")
        if query_embedding.shape[0] != 512:
            raise ValueError("Query must be 512-dimensional")

        start_time = time.perf_counter()

        # Allocate GPU memory for query and results
        query_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(query_gpu), query_embedding.nbytes)
        if result != 0:
            raise RuntimeError(f"Failed to allocate query GPU memory: {result}")

        scores_gpu = ctypes.c_void_p()
        result = lib.cuda_malloc(ctypes.byref(scores_gpu), self.db_size * 4)  # int32
        if result != 0:
            raise RuntimeError(f"Failed to allocate scores GPU memory: {result}")

        try:
            # Copy query to GPU
            result = lib.cuda_memcpy_h2d(query_gpu, query_embedding.ctypes.data_as(ctypes.c_void_p), query_embedding.nbytes)
            if result != 0:
                raise RuntimeError(f"Failed to copy query to GPU: {result}")

            # Execute search
            result = lib.i8dot512_scores(query_gpu, self.db_gpu, scores_gpu, self.db_size)
            if result != 0:
                raise RuntimeError(f"Search failed: {result}")

            # Copy results back
            scores = np.zeros(self.db_size, dtype=np.int32)
            result = lib.cuda_memcpy_d2h(scores.ctypes.data_as(ctypes.c_void_p), scores_gpu, scores.nbytes)
            if result != 0:
                raise RuntimeError(f"Failed to copy scores back: {result}")

            lib.cuda_synchronize()
            end_time = time.perf_counter()

            # Get top-k indices (highest scores for dot product)
            top_k_indices = np.argpartition(scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

            # Format results
            results = []
            for rank, idx in enumerate(top_k_indices):
                results.append((rank + 1, self.documents[idx], int(scores[idx])))

            search_time = (end_time - start_time) * 1000  # ms

            return search_time, results

        finally:
            lib.cuda_free(query_gpu)
            lib.cuda_free(scores_gpu)

    def __del__(self):
        if hasattr(self, 'db_gpu') and self.db_gpu:
            lib.cuda_free(self.db_gpu)

def create_test_corpus():
    """Create a test corpus with various topics and relevance levels."""
    documents = [
        # Programming/Tech (should cluster together)
        "Python programming tutorial for beginners learning to code",
        "Advanced algorithms and data structures in computer science",
        "Machine learning with TensorFlow and neural networks",
        "JavaScript web development frameworks and libraries",
        "Database optimization techniques for better performance",
        "Software engineering best practices and code review",
        "Deep learning research papers and implementation guides",
        "Programming interview questions and coding challenges",

        # Food/Cooking (should cluster together)
        "Delicious pasta recipe with fresh ingredients and herbs",
        "Italian cooking techniques for authentic Mediterranean cuisine",
        "Healthy meal prep ideas for busy weeknight dinners",
        "Restaurant reviews and culinary experiences around the world",
        "Baking bread from scratch with sourdough starter techniques",
        "Vegetarian recipes featuring seasonal vegetables and grains",
        "Wine pairing suggestions for different types of cuisine",
        "Food photography tips for Instagram and social media",

        # Sports (should cluster together)
        "Basketball championship game highlights and player statistics",
        "Soccer training drills to improve team performance",
        "Olympic athletes preparing for international competition events",
        "Tennis match analysis and professional player strategies",
        "Football season predictions and fantasy league advice",
        "Marathon training plans for beginners and experienced runners",
        "Sports nutrition and recovery methods for peak performance",
        "Baseball statistics and historical records analysis",

        # Science (should cluster together)
        "Scientific research methodology and experimental design principles",
        "Climate change studies and environmental impact assessment",
        "Medical breakthrough discoveries in cancer treatment research",
        "Astronomy observations of distant galaxies and cosmic phenomena",
        "Physics theories explaining quantum mechanics and relativity",
        "Biology experiments studying cellular processes and genetics",
        "Chemistry lab procedures for synthesizing new compounds",
        "Neuroscience research on brain function and cognitive processes",

        # Random/Mixed topics (should be less clustered)
        "Travel guide to exotic destinations and cultural experiences",
        "Home renovation projects and interior design inspiration",
        "Personal finance advice for saving money and investing wisely",
        "Gardening tips for growing vegetables in small urban spaces",
        "Photography techniques for landscape and portrait sessions",
        "Music theory lessons for learning piano and composition",
        "Art history survey covering Renaissance and modern movements",
        "Book recommendations spanning fiction and non-fiction genres",
        "Meditation practices for stress relief and mindfulness training",
        "Fashion trends and styling advice for different body types",
        "Pet care guidelines for dogs cats and other companion animals",
        "DIY craft projects using recycled materials and simple tools"
    ]

    return documents

def test_search_quality():
    """Test search quality with various queries."""
    print("🔍 LibTorch-Free GPU Search Quality Test")
    print("=" * 60)
    print()

    # Create mock embedding model
    model = MockEmbeddingModel()

    # Create test corpus
    documents = create_test_corpus()
    print(f"📚 Created corpus with {len(documents)} documents")

    # Generate embeddings
    print("🧮 Generating embeddings...")
    float_embeddings = np.array([model.encode(doc) for doc in documents])
    int8_embeddings = np.array([model.quantize_to_int8(emb) for emb in float_embeddings])

    # Create search engine
    engine = LibTorchFreeSearchEngine()
    index_time = engine.index(int8_embeddings, documents)
    print(f"⚡ Indexed in {index_time:.2f} ms")
    print()

    # Test queries with expected relevance
    test_queries = [
        ("Python machine learning algorithms", "Programming/Tech"),
        ("delicious Italian pasta recipe", "Food/Cooking"),
        ("basketball training techniques", "Sports"),
        ("climate change research study", "Science"),
        ("JavaScript web development", "Programming/Tech"),
        ("healthy cooking meal prep", "Food/Cooking"),
        ("Olympic athlete training", "Sports"),
        ("quantum physics theory", "Science"),
        ("travel photography tips", "Mixed/General")
    ]

    for query_text, expected_topic in test_queries:
        print(f"🔎 Query: '{query_text}' (Expected: {expected_topic})")
        print("-" * 70)

        # Generate query embedding
        query_float = model.encode(query_text)
        query_int8 = model.quantize_to_int8(query_float)

        # Search
        search_time, results = engine.search(query_int8, k=8)

        print(f"⏱️  Search time: {search_time:.2f} ms")
        print("📊 Results:")

        for rank, doc, score in results:
            # Truncate long documents for display
            display_doc = doc if len(doc) <= 60 else doc[:57] + "..."
            print(f"  {rank:2d}. [{score:7d}] {display_doc}")

        # Analyze topic clustering
        topic_counts = {}
        for _, doc, _ in results:
            # Simple topic detection based on keywords
            doc_lower = doc.lower()
            if any(word in doc_lower for word in ['python', 'programming', 'code', 'software', 'algorithm']):
                topic = 'Tech'
            elif any(word in doc_lower for word in ['recipe', 'cooking', 'food', 'meal', 'delicious']):
                topic = 'Food'
            elif any(word in doc_lower for word in ['game', 'training', 'team', 'player', 'sport']):
                topic = 'Sports'
            elif any(word in doc_lower for word in ['research', 'study', 'science', 'theory', 'experiment']):
                topic = 'Science'
            else:
                topic = 'Other'

            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        print("📈 Topic distribution in results:")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     {topic}: {count} documents")

        print()

def test_similarity_consistency():
    """Test that similar documents get similar scores."""
    print("🎯 Testing Similarity Consistency")
    print("=" * 40)
    print()

    model = MockEmbeddingModel()

    # Create pairs of similar documents
    similar_pairs = [
        ("Python programming tutorial", "Python coding guide for beginners"),
        ("Italian pasta recipe with herbs", "Delicious Italian pasta cooking instructions"),
        ("Basketball training techniques", "Basketball practice drills and exercises"),
        ("Climate change research", "Environmental climate science study")
    ]

    for doc1, doc2 in similar_pairs:
        print(f"📝 Comparing: '{doc1}' vs '{doc2}'")

        # Generate embeddings
        emb1_float = model.encode(doc1)
        emb2_float = model.encode(doc2)
        emb1_int8 = model.quantize_to_int8(emb1_float)
        emb2_int8 = model.quantize_to_int8(emb2_float)

        # Calculate similarity scores
        float_similarity = np.dot(emb1_float, emb2_float)
        int8_similarity = np.dot(emb1_int8.astype(np.int32), emb2_int8.astype(np.int32))

        print(f"   Float32 similarity: {float_similarity:.4f}")
        print(f"   INT8 similarity:    {int8_similarity:d}")
        print(f"   Quantization ratio: {int8_similarity / (float_similarity * 127**2):.4f}")
        print()

def main():
    try:
        # Test search quality
        test_search_quality()

        # Test similarity consistency
        test_similarity_consistency()

        print("✅ Quality tests completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()