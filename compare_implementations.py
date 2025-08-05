#!/usr/bin/env python3
"""
Compare Go and Python implementations of multilingual E5 embeddings.
This script loads embeddings from both implementations and analyzes:
1. Cosine similarity between corresponding embeddings
2. Distance differences 
3. Performance comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

def load_embeddings():
    """Load embeddings from both implementations."""
    print("Loading embeddings for comparison...")
    
    # Load Python embeddings
    if os.path.exists("python_embeddings.npy"):
        python_embeddings = np.load("python_embeddings.npy")
        print(f"Loaded Python embeddings: {python_embeddings.shape}")
    else:
        print("Python embeddings not found! Run python_comparison.py first.")
        return None, None
    
    # Load texts
    texts = []
    if os.path.exists("python_embeddings_texts.txt"):
        with open("python_embeddings_texts.txt", 'r') as f:
            texts = [line.strip() for line in f.readlines()]
    
    # Check if Go embeddings exist (would need to be saved from Go implementation)
    go_embeddings = None
    if os.path.exists("go_embeddings.npy"):
        go_embeddings = np.load("go_embeddings.npy")
        print(f"Loaded Go embeddings: {go_embeddings.shape}")
    else:
        print("Go embeddings not found. You'll need to modify the Go code to save embeddings.")
    
    return python_embeddings, go_embeddings, texts

def compare_embeddings(py_embeddings, go_embeddings, texts):
    """Compare embeddings from both implementations."""
    if go_embeddings is None:
        print("Cannot compare - Go embeddings not available")
        return
    
    print("\nComparing Go vs Python embeddings...")
    print("=" * 40)
    
    # Calculate cosine similarities between corresponding embeddings
    similarities = []
    for i in range(len(py_embeddings)):
        sim = cosine_similarity([py_embeddings[i]], [go_embeddings[i]])[0, 0]
        similarities.append(sim)
        print(f"'{texts[i]}': similarity = {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    print(f"\nAverage similarity: {avg_similarity:.4f}")
    
    if avg_similarity > 0.9:
        print("✓ EXCELLENT: Very high similarity between implementations")
    elif avg_similarity > 0.8:
        print("✓ GOOD: High similarity between implementations")
    elif avg_similarity > 0.7:
        print("⚠ MODERATE: Some differences between implementations")
    else:
        print("✗ LOW: Significant differences between implementations")
    
    return similarities

def analyze_distance_preservation(py_embeddings, go_embeddings, texts):
    """Analyze if distance relationships are preserved."""
    if go_embeddings is None:
        return
    
    print("\nAnalyzing distance preservation...")
    print("=" * 40)
    
    # Calculate all pairwise distances for both implementations
    py_distances = cosine_similarity(py_embeddings)
    go_distances = cosine_similarity(go_embeddings)
    
    # Compare specific relationships
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            py_sim = py_distances[i, j]
            go_sim = go_distances[i, j]
            diff = abs(py_sim - go_sim)
            
            print(f"'{texts[i]}' vs '{texts[j]}': Python={py_sim:.4f}, Go={go_sim:.4f}, diff={diff:.4f}")
    
    # Overall correlation between distance matrices
    py_flat = py_distances.flatten()
    go_flat = go_distances.flatten()
    correlation = np.corrcoef(py_flat, go_flat)[0, 1]
    
    print(f"\nDistance matrix correlation: {correlation:.4f}")
    
    if correlation > 0.95:
        print("✓ EXCELLENT: Distance relationships very well preserved")
    elif correlation > 0.9:
        print("✓ GOOD: Distance relationships well preserved")
    else:
        print("⚠ Moderate distance preservation")

def visualize_embeddings(py_embeddings, go_embeddings, texts):
    """Create visualizations of the embeddings."""
    print("\nCreating visualization...")
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(1, 2 if go_embeddings is not None else 1, figsize=(12, 5))
    if go_embeddings is None:
        axes = [axes]
    
    # Plot Python embeddings
    pca = PCA(n_components=2)
    py_2d = pca.fit_transform(py_embeddings)
    
    axes[0].scatter(py_2d[:, 0], py_2d[:, 1], c=['red', 'blue', 'green'], s=100)
    for i, txt in enumerate(texts):
        axes[0].annotate(txt, (py_2d[i, 0], py_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
    axes[0].set_title('Python Embeddings (PCA)')
    axes[0].grid(True)
    
    # Plot Go embeddings if available
    if go_embeddings is not None:
        go_2d = pca.fit_transform(go_embeddings)
        axes[1].scatter(go_2d[:, 0], go_2d[:, 1], c=['red', 'blue', 'green'], s=100)
        for i, txt in enumerate(texts):
            axes[1].annotate(txt, (go_2d[i, 0], go_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
        axes[1].set_title('Go Embeddings (PCA)')
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualization saved as 'embedding_comparison.png'")

def main():
    print("Embedding Implementation Comparison")
    print("=" * 50)
    
    # Load embeddings
    py_embeddings, go_embeddings, texts = load_embeddings()
    
    if py_embeddings is None:
        print("Error: Could not load embeddings for comparison")
        return
    
    # Analyze Python embeddings
    print(f"\nPython embeddings analysis:")
    print(f"Shape: {py_embeddings.shape}")
    print(f"Mean norm: {np.mean(np.linalg.norm(py_embeddings, axis=1)):.4f}")
    print(f"Std norm: {np.std(np.linalg.norm(py_embeddings, axis=1)):.4f}")
    
    # Calculate similarities within Python implementation
    py_similarities = cosine_similarity(py_embeddings)
    print(f"\nPython implementation similarities:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            print(f"'{texts[i]}' vs '{texts[j]}': {py_similarities[i, j]:.4f}")
    
    # Compare with Go if available
    if go_embeddings is not None:
        similarities = compare_embeddings(py_embeddings, go_embeddings, texts)
        analyze_distance_preservation(py_embeddings, go_embeddings, texts)
    
    # Create visualization
    try:
        visualize_embeddings(py_embeddings, go_embeddings, texts)
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    
    print("\nComparison completed!")

if __name__ == "__main__":
    main()