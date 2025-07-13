#!/usr/bin/env python3
"""
Generate reference tokens for all the benchmark sentences used in Go.
"""

import json
from sentence_transformers import SentenceTransformer

def generate_comprehensive_tokens():
    print("🔧 Generating Comprehensive Reference Tokens")
    print("=" * 50)
    
    # Load the model
    model = SentenceTransformer("model/sentence_transformer", device='cpu')
    tokenizer = model.tokenizer
    
    # All sentences used in Go benchmarks
    all_sentences = [
        # Original test sentences
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful",
        
        # Benchmark sentences
        "machine learning is fascinating",
        "artificial intelligence and deep learning",
        "natural language processing",
        "computer vision and image recognition",
        "data science and analytics", 
        "software engineering best practices",
        "distributed systems architecture",
        "cloud computing and microservices",
        "performance optimization techniques",
        
        # Additional test sentence from benchmarkModels
        "machine learning and artificial intelligence for performance optimization"
    ]
    
    reference_tokens = {}
    
    print("📝 Tokenizing sentences...")
    for sentence in all_sentences:
        print(f"  Processing: '{sentence}'")
        
        # Tokenize
        encoded = tokenizer.encode(sentence, add_special_tokens=True)
        token_ids = encoded.ids
        
        # Pad to 512 if needed
        max_len = 512
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        
        reference_tokens[sentence] = {
            "token_ids": token_ids,
            "length": len([t for t in token_ids if t != 0])
        }
        
        print(f"    Tokens: {len([t for t in token_ids if t != 0])}")
    
    # Save to file
    output_file = "model/reference_tokens.json"
    with open(output_file, "w") as f:
        json.dump(reference_tokens, f, indent=2)
    
    print(f"\n✅ Saved {len(reference_tokens)} reference token mappings to {output_file}")
    
    # Test a few embeddings to make sure they work
    print("\n🧪 Testing a few embeddings...")
    
    test_sentences = all_sentences[:3]  # Test first 3
    original_embeddings = model.encode(test_sentences)
    
    import numpy as np
    import onnxruntime as ort
    
    session = ort.InferenceSession("model/embedding_model.onnx")
    
    for i, sentence in enumerate(test_sentences):
        token_ids = reference_tokens[sentence]["token_ids"]
        input_tensor = np.array([token_ids], dtype=np.int64)
        onnx_output = session.run(None, {'input_ids': input_tensor})[0][0]
        
        similarity = np.dot(original_embeddings[i], onnx_output) / (
            np.linalg.norm(original_embeddings[i]) * np.linalg.norm(onnx_output)
        )
        
        print(f"  '{sentence}': similarity = {similarity:.6f}")
    
    print("\n🎉 Reference tokens generation complete!")

if __name__ == "__main__":
    generate_comprehensive_tokens()
