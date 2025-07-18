#!/usr/bin/env python3
"""
Comprehensive diagnostic: Compare Python SentenceTransformers vs ONNX vs Go outputs
to identify where the conversion process breaks down.
"""

import torch
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import numpy as np
import json

def comprehensive_diagnostic():
    print("="*80)
    print("COMPREHENSIVE EMBEDDING PIPELINE DIAGNOSTIC")
    print("="*80)
    
    # Test sentences with expected diversity
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.", 
        "The weather is nice today.",
        "Hello world",
        "Python is a programming language."
    ]
    
    print("Test sentences:")
    for i, sentence in enumerate(test_sentences):
        print(f"  {i+1}. {sentence}")
    print()
    
    # Step 1: Load original SentenceTransformer model
    print("STEP 1: PYTHON SENTENCETRANSFORMERS")
    print("-" * 50)
    
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    model.cpu()
    
    # Get embeddings
    python_embeddings = model.encode(test_sentences, convert_to_tensor=True)
    print(f"Python embeddings shape: {python_embeddings.shape}")
    
    for i, sentence in enumerate(test_sentences):
        emb = python_embeddings[i]
        print(f"  S{i+1}: [{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, {emb[3]:.3f}, {emb[4]:.3f}] (norm: {torch.norm(emb):.3f})")
    
    # Calculate Python similarities
    python_similarities = torch.nn.functional.cosine_similarity(
        python_embeddings.unsqueeze(1), python_embeddings.unsqueeze(0), dim=2
    )
    print("\nPython similarity matrix:")
    print(python_similarities.numpy())
    print()
    
    # Step 2: Test ONNX model
    print("STEP 2: ONNX MODEL")
    print("-" * 50)
    
    try:
        session = ort.InferenceSession("model/production_embedding_model.onnx")
        print("✓ ONNX model loaded")
        
        # Get tokenizer
        tokenizer = model.tokenizer
        
        # Tokenize each sentence
        onnx_embeddings = []
        token_data = {}
        
        for i, sentence in enumerate(test_sentences):
            # Tokenize
            encoding = tokenizer.encode(sentence)
            tokens = encoding.ids
            token_data[sentence] = tokens
            
            print(f"  S{i+1} tokens: {tokens}")
            
            # Prepare ONNX input (single sentence)
            input_ids = np.array([tokens], dtype=np.int64)
            
            # Run ONNX inference
            outputs = session.run(['embeddings'], {'input_ids': input_ids})
            embedding = outputs[0][0]  # Get first (and only) embedding
            onnx_embeddings.append(embedding)
            
            print(f"  S{i+1} ONNX: [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, {embedding[3]:.3f}, {embedding[4]:.3f}] (norm: {np.linalg.norm(embedding):.3f})")
        
        # Convert to numpy for similarity calculation
        onnx_embeddings = np.array(onnx_embeddings)
        
        # Calculate ONNX similarities
        onnx_similarities = np.zeros((len(test_sentences), len(test_sentences)))
        for i in range(len(test_sentences)):
            for j in range(len(test_sentences)):
                dot_product = np.dot(onnx_embeddings[i], onnx_embeddings[j])
                norm_i = np.linalg.norm(onnx_embeddings[i])
                norm_j = np.linalg.norm(onnx_embeddings[j])
                onnx_similarities[i][j] = dot_product / (norm_i * norm_j)
        
        print("\nONNX similarity matrix:")
        print(onnx_similarities)
        print()
        
        # Save token data for Go comparison
        with open("model/debug_tokens.json", "w") as f:
            json.dump(token_data, f, indent=2)
        print("✓ Token data saved to model/debug_tokens.json")
        
    except Exception as e:
        print(f"✗ ONNX model failed: {e}")
        return
    
    # Step 3: Compare Python vs ONNX
    print("STEP 3: PYTHON vs ONNX COMPARISON")
    print("-" * 50)
    
    # Convert python embeddings to numpy
    python_embeddings_np = python_embeddings.numpy()
    
    print("Embedding differences (Python vs ONNX):")
    for i, sentence in enumerate(test_sentences):
        diff = np.abs(python_embeddings_np[i] - onnx_embeddings[i])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  S{i+1}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    print("\nSimilarity matrix differences:")
    sim_diff = np.abs(python_similarities.numpy() - onnx_similarities)
    print(f"  Max similarity difference: {np.max(sim_diff):.6f}")
    print(f"  Mean similarity difference: {np.mean(sim_diff):.6f}")
    
    if np.max(sim_diff) < 0.001:
        print("✓ Python and ONNX outputs match very well!")
    elif np.max(sim_diff) < 0.01:
        print("✓ Python and ONNX outputs match reasonably well")
    else:
        print("✗ Python and ONNX outputs differ significantly!")
    
    print()
    
    # Step 4: Analyze embedding diversity
    print("STEP 4: EMBEDDING DIVERSITY ANALYSIS")
    print("-" * 50)
    
    def analyze_diversity(embeddings, name):
        print(f"\n{name} diversity analysis:")
        
        # Calculate all pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if len(embeddings.shape) == 2:  # numpy
                    sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                else:  # torch
                    sim = torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                similarities.append(sim)
        
        similarities = np.array(similarities)
        print(f"  Min similarity: {np.min(similarities):.6f}")
        print(f"  Max similarity: {np.max(similarities):.6f}")
        print(f"  Mean similarity: {np.mean(similarities):.6f}")
        print(f"  Std similarity: {np.std(similarities):.6f}")
        
        if np.std(similarities) > 0.05:
            print("  ✓ Good diversity - embeddings are different")
        else:
            print("  ✗ Poor diversity - embeddings are too similar")
    
    analyze_diversity(python_embeddings, "Python")
    analyze_diversity(onnx_embeddings, "ONNX")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    # Check if the issue is in the model export or Go implementation
    if np.std([np.linalg.norm(emb) for emb in onnx_embeddings]) < 0.1:
        print("🚨 ISSUE FOUND: ONNX embeddings have very similar norms - possible model export issue")
    elif np.max(sim_diff) > 0.01:
        print("🚨 ISSUE FOUND: Python and ONNX outputs differ significantly - model export issue")
    else:
        print("✓ Python and ONNX models work correctly")
        print("  → Issue is likely in Go implementation (tokenization or inference)")
    
    return {
        "python_embeddings": python_embeddings_np,
        "onnx_embeddings": onnx_embeddings,
        "python_similarities": python_similarities.numpy(),
        "onnx_similarities": onnx_similarities,
        "token_data": token_data
    }

if __name__ == "__main__":
    results = comprehensive_diagnostic()
