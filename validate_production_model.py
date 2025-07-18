#!/usr/bin/env python3
"""
Validate the exported production ONNX model against the Python reference.
"""

import torch
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import numpy as np

def validate_production_model():
    print("="*60)
    print("PRODUCTION MODEL VALIDATION")
    print("="*60)
    
    # Load original model
    print("Loading original model...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    # Load ONNX model
    print("Loading ONNX model...")
    onnx_path = "model/production_embedding_model.onnx"
    session = ort.InferenceSession(onnx_path)
    
    # Test sentences
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today.",
        "Python is a programming language.",
        "Artificial intelligence will change the world."
    ]
    
    print(f"\nTesting with {len(test_sentences)} sentences:")
    for i, sentence in enumerate(test_sentences):
        print(f"  {i+1}. {sentence}")
    
    # Get reference embeddings
    print("\nComputing reference embeddings...")
    reference_embeddings = model.encode(test_sentences, convert_to_tensor=True)
    print(f"Reference embeddings shape: {reference_embeddings.shape}")
    
    # Get tokenizer
    tokenizer = model.tokenizer
    
    # Tokenize sentences
    print("\nTokenizing sentences...")
    inputs = tokenizer.encode_batch(test_sentences)
    
    # Convert to format expected by ONNX model
    input_ids_list = []
    max_len = max(len(inp.ids) for inp in inputs)
    
    for inp in inputs:
        # Pad to max length  
        ids = inp.ids + [0] * (max_len - len(inp.ids))
        input_ids_list.append(ids)
    
    input_ids = np.array(input_ids_list, dtype=np.int64)
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Run ONNX inference
    print("\nRunning ONNX inference...")
    onnx_outputs = session.run(['embeddings'], {'input_ids': input_ids})
    onnx_embeddings = torch.from_numpy(onnx_outputs[0])
    print(f"ONNX embeddings shape: {onnx_embeddings.shape}")
    
    # Compare embeddings
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    cosine_similarities = []
    l2_distances = []
    
    for i in range(len(test_sentences)):
        ref_emb = reference_embeddings[i]
        onnx_emb = onnx_embeddings[i]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), onnx_emb.unsqueeze(0)
        ).item()
        
        # L2 distance
        l2_dist = torch.norm(ref_emb - onnx_emb).item()
        
        cosine_similarities.append(cos_sim)
        l2_distances.append(l2_dist)
        
        print(f"Sentence {i+1}:")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  L2 distance: {l2_dist:.6f}")
        print(f"  Reference (first 5): {ref_emb[:5]}")
        print(f"  ONNX (first 5): {onnx_emb[:5]}")
        print()
    
    # Summary statistics
    avg_cos_sim = np.mean(cosine_similarities)
    avg_l2_dist = np.mean(l2_distances)
    min_cos_sim = np.min(cosine_similarities)
    max_l2_dist = np.max(l2_distances)
    
    print("SUMMARY:")
    print(f"  Average cosine similarity: {avg_cos_sim:.6f}")
    print(f"  Minimum cosine similarity: {min_cos_sim:.6f}")
    print(f"  Average L2 distance: {avg_l2_dist:.6f}")
    print(f"  Maximum L2 distance: {max_l2_dist:.6f}")
    
    # Quality assessment
    print("\nQUALITY ASSESSMENT:")
    if min_cos_sim > 0.999:
        print("  ✓ EXCELLENT: ONNX embeddings match reference almost perfectly!")
    elif min_cos_sim > 0.99:
        print("  ✓ VERY GOOD: ONNX embeddings match reference very well")
    elif min_cos_sim > 0.95:
        print("  ✓ GOOD: ONNX embeddings match reference reasonably well")
    else:
        print("  ✗ POOR: ONNX embeddings do not match reference well")
    
    # Test similarity computation
    print("\n" + "="*60)
    print("SIMILARITY COMPUTATION TEST")
    print("="*60)
    
    # Compute pairwise similarities
    ref_similarities = torch.nn.functional.cosine_similarity(
        reference_embeddings.unsqueeze(1), reference_embeddings.unsqueeze(0), dim=2
    )
    onnx_similarities = torch.nn.functional.cosine_similarity(
        onnx_embeddings.unsqueeze(1), onnx_embeddings.unsqueeze(0), dim=2
    )
    
    print("Reference similarity matrix:")
    print(ref_similarities.numpy())
    print("\nONNX similarity matrix:")
    print(onnx_similarities.numpy())
    
    # Check if similarity patterns match
    similarity_diff = torch.abs(ref_similarities - onnx_similarities)
    max_sim_diff = torch.max(similarity_diff).item()
    print(f"\nMaximum similarity difference: {max_sim_diff:.6f}")
    
    if max_sim_diff < 0.001:
        print("✓ EXCELLENT: Similarity patterns match perfectly!")
    elif max_sim_diff < 0.01:
        print("✓ VERY GOOD: Similarity patterns match very well")
    else:
        print("✗ POOR: Similarity patterns differ significantly")
    
    return {
        "avg_cos_sim": avg_cos_sim,
        "min_cos_sim": min_cos_sim,
        "avg_l2_dist": avg_l2_dist,
        "max_sim_diff": max_sim_diff
    }

if __name__ == "__main__":
    validate_production_model()
