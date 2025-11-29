#!/usr/bin/env python3
"""
Quick test to show real semantic search using sentence-transformers
This demonstrates what the bed tool SHOULD find with proper embeddings
"""

import os
import sys
from pathlib import Path

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Note: sentence-transformers not installed, using mock embeddings")
    model = None

def mock_embedding(text):
    """Fallback mock embedding"""
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    return [float(int(h[i:i+2], 16))/255 for i in range(0, min(32, len(h)), 2)]

def get_embedding(text):
    """Get embedding for text"""
    if model:
        return model.encode(text).tolist()
    return mock_embedding(text)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    import math
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def search_files(directory, query, top_k=10):
    """Search files for semantic matches to query"""
    query_embedding = get_embedding(query)

    results = []
    test_chunks = [
        # Some real code chunks from netwrck that should match "re render game"
        ("game.min.js", 3174, "// Re-render current game to update chat interface"),
        ("game.min.js", 3201, "// Re-render current game"),
        ("game.min.js", 9162, "// Re-render the game"),
        ("game.min.js", 9220, "// Re-render the game"),
        ("game.js", 7476, "// Update room AIs live and re-render"),

        # Some unrelated chunks for comparison
        ("server.py", 100, "def start_server():"),
        ("config.json", 1, '{"database": "postgres"}'),
        ("README.md", 1, "# Netwrck Project"),
    ]

    print(f"\n Testing semantic search for: '{query}'")
    print("=" * 60)

    for file, line, content in test_chunks:
        chunk_embedding = get_embedding(content)
        score = cosine_similarity(query_embedding, chunk_embedding)
        results.append((score, file, line, content))

    # Sort by score descending
    results.sort(reverse=True)

    print(f"\n Top {min(top_k, len(results))} results:\n")

    for i, (score, file, line, content) in enumerate(results[:top_k]):
        print(f"Match {i+1}: {file}:{line} (score: {score:.4f})")
        print(f"  {content[:80]}...")
        print()

    # Highlight the issue
    print("\n Note: With proper embeddings, game re-render comments should score > 0.7")
    print("   With mock embeddings, scores are essentially random")

    if not model:
        print("\n  Using mock embeddings - install sentence-transformers for real semantic search:")
        print("   pip install sentence-transformers")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "re render game"
    search_files(".", query, top_k=5)