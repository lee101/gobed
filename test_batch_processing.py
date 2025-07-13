#!/usr/bin/env python3
"""
Test batch processing in Python to understand expected behavior before fixing Go.
"""

import numpy as np
import onnxruntime as ort
import json
from sentence_transformers import SentenceTransformer

def test_batch_processing():
    print("🧪 Testing Batch Processing")
    print("=" * 40)
    
    # Load models
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    onnx_session = ort.InferenceSession("model/embedding_model.onnx")
    
    # Load reference tokens
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    test_sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful"
    ]
    
    print("🔍 Testing individual vs batch processing...")
    
    # Test 1: Individual processing (current Go approach)
    print("\n1️⃣ Individual Processing:")
    individual_embeddings = []
    for sentence in test_sentences:
        # SentenceTransformer
        st_embedding = st_model.encode([sentence])[0]  # Note: still batch of 1
        
        # ONNX
        token_ids = ref_tokens[sentence]["token_ids"]
        if len(token_ids) < 512:
            token_ids = token_ids + [0] * (512 - len(token_ids))
        
        # Single item batch
        input_tensor = np.array([token_ids], dtype=np.int64)  # Shape: [1, 512]
        onnx_output = onnx_session.run(None, {'input_ids': input_tensor})[0]
        onnx_embedding = onnx_output[0]  # Extract from batch dimension
        
        individual_embeddings.append(onnx_embedding)
        
        print(f"  '{sentence}':")
        print(f"    Input shape: {input_tensor.shape}")
        print(f"    Output shape: {onnx_output.shape}")
        print(f"    Embedding shape: {onnx_embedding.shape}")
        print(f"    ST similarity: {np.dot(st_embedding, onnx_embedding) / (np.linalg.norm(st_embedding) * np.linalg.norm(onnx_embedding)):.6f}")
    
    # Test 2: True batch processing
    print("\n2️⃣ Batch Processing:")
    
    # Prepare all token IDs
    all_token_ids = []
    for sentence in test_sentences:
        token_ids = ref_tokens[sentence]["token_ids"]
        if len(token_ids) < 512:
            token_ids = token_ids + [0] * (512 - len(token_ids))
        all_token_ids.append(token_ids)
    
    # Batch input
    batch_input = np.array(all_token_ids, dtype=np.int64)  # Shape: [3, 512]
    print(f"  Batch input shape: {batch_input.shape}")
    
    # SentenceTransformer batch
    st_batch_embeddings = st_model.encode(test_sentences)
    print(f"  ST batch output shape: {st_batch_embeddings.shape}")
    
    # ONNX batch (this might fail if our model doesn't support it properly)
    try:
        onnx_batch_output = onnx_session.run(None, {'input_ids': batch_input})[0]
        print(f"  ONNX batch output shape: {onnx_batch_output.shape}")
        
        # Compare batch vs individual
        print("\n🔍 Batch vs Individual Comparison:")
        for i, sentence in enumerate(test_sentences):
            individual_emb = individual_embeddings[i]
            batch_emb = onnx_batch_output[i]
            
            similarity = np.dot(individual_emb, batch_emb) / (np.linalg.norm(individual_emb) * np.linalg.norm(batch_emb))
            print(f"  '{sentence}': {similarity:.8f}")
            
            if similarity < 0.9999:
                print(f"    ⚠️  Batch vs individual differs: {similarity:.8f}")
        
        return True, onnx_batch_output
        
    except Exception as e:
        print(f"  ❌ ONNX batch processing failed: {e}")
        return False, None

def test_different_batch_sizes():
    print("\n🧪 Testing Different Batch Sizes")
    print("=" * 40)
    
    onnx_session = ort.InferenceSession("model/embedding_model.onnx")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 3, 5]
    
    # Use a simple repeated sentence for testing
    test_sentence = "hello world"
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    token_ids = ref_tokens[test_sentence]["token_ids"]
    if len(token_ids) < 512:
        token_ids = token_ids + [0] * (512 - len(token_ids))
    
    for batch_size in batch_sizes:
        try:
            # Create batch input
            batch_input = np.array([token_ids] * batch_size, dtype=np.int64)
            print(f"  Batch size {batch_size}: input shape {batch_input.shape}")
            
            # Run inference
            output = onnx_session.run(None, {'input_ids': batch_input})[0]
            print(f"    Output shape: {output.shape}")
            
            # Check if all embeddings in batch are identical (they should be)
            if batch_size > 1:
                first_emb = output[0]
                all_identical = True
                for i in range(1, batch_size):
                    similarity = np.dot(first_emb, output[i]) / (np.linalg.norm(first_emb) * np.linalg.norm(output[i]))
                    if similarity < 0.9999:
                        all_identical = False
                        break
                
                print(f"    All embeddings identical: {all_identical}")
                
        except Exception as e:
            print(f"    ❌ Failed: {e}")

if __name__ == "__main__":
    success, batch_output = test_batch_processing()
    test_different_batch_sizes()
    
    if success:
        print("\n✅ Batch processing works correctly!")
    else:
        print("\n❌ Batch processing has issues!")
