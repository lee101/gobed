#!/usr/bin/env python3
"""
Validate tokenization and ONNX model export for sentence embeddings.
This script ensures that our Go implementation can match Python's tokenization and inference.
"""

import json
import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from pathlib import Path

def main():
    print("🔍 Validating Tokenization and ONNX Export")
    print("=" * 50)
    
    # Load the SentenceTransformer model
    model_path = "model/sentence_transformer"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📂 Loading SentenceTransformer from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Test sentences
    test_sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful"
    ]
    
    print(f"\n🔤 Testing tokenization for {len(test_sentences)} sentences...")
    
    # Get tokenizer from the model
    tokenizer = model.tokenizer
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nSentence {i+1}: '{sentence}'")
        
        # Tokenize using SentenceTransformer's tokenizer
        encoded = tokenizer.encode(sentence, add_special_tokens=True)
        
        # Pad to max length
        max_len = 512
        if len(encoded.ids) > max_len:
            token_ids = encoded.ids[:max_len]
        else:
            token_ids = encoded.ids + [tokenizer.token_to_id("[PAD]") or 0] * (max_len - len(encoded.ids))
            
        print(f"  Token IDs: {token_ids[:10]}... (showing first 10)")
        print(f"  Shape: ({len(token_ids)},)")
        
        # Decode back to check
        decoded = tokenizer.decode(token_ids[:len(encoded.ids)])
        print(f"  Decoded: '{decoded[:50]}...'")
        
    print("\n🧮 Testing SentenceTransformer embedding generation...")
    
    # Generate embeddings using SentenceTransformer
    st_embeddings = model.encode(test_sentences)
    print(f"SentenceTransformer embeddings shape: {st_embeddings.shape}")
    print(f"Sample values: {st_embeddings[0][:5]}")
    
    # Test ONNX model if it exists
    onnx_path = "model/embedding_model.onnx"
    if Path(onnx_path).exists():
        print(f"\n🔧 Testing ONNX model at {onnx_path}")
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"  Input name: {input_name}")
        print(f"  Output name: {output_name}")
        print(f"  Input shape: {session.get_inputs()[0].shape}")
        print(f"  Output shape: {session.get_outputs()[0].shape}")
        
        # Test inference with first sentence
        test_sentence = test_sentences[0]
        encoded = tokenizer.encode(test_sentence, add_special_tokens=True)
        
        # Pad to max length
        max_len = 512
        if len(encoded.ids) > max_len:
            token_ids = encoded.ids[:max_len]
        else:
            token_ids = encoded.ids + [tokenizer.token_to_id("[PAD]") or 0] * (max_len - len(encoded.ids))
        
        # Convert to numpy array and reshape for ONNX
        token_array = np.array(token_ids, dtype=np.int64).reshape(1, -1)
        
        # Run ONNX inference
        onnx_output = session.run([output_name], {input_name: token_array})
        onnx_embedding = onnx_output[0].squeeze()
        
        print(f"  ONNX output shape: {onnx_embedding.shape}")
        print(f"  ONNX sample values: {onnx_embedding[:5]}")
        
        # Compare with SentenceTransformer
        st_embedding = model.encode([test_sentence])[0]
        
        # Calculate similarity
        cosine_sim = np.dot(onnx_embedding, st_embedding) / (
            np.linalg.norm(onnx_embedding) * np.linalg.norm(st_embedding)
        )
        
        print(f"  Cosine similarity (ONNX vs SentenceTransformer): {cosine_sim:.6f}")
        
        if cosine_sim > 0.95:
            print("  ✅ ONNX model matches SentenceTransformer closely!")
        elif cosine_sim > 0.5:
            print("  ⚠️  ONNX model has moderate similarity to SentenceTransformer")
        else:
            print("  ❌ ONNX model differs significantly from SentenceTransformer")
            
        # Save token IDs for Go reference
        print("\n💾 Saving tokenization reference for Go...")
        token_data = {}
        for i, sentence in enumerate(test_sentences):
            encoded = tokenizer.encode(sentence, add_special_tokens=True)
            
            # Pad to max length
            max_len = 512
            if len(encoded.ids) > max_len:
                token_ids = encoded.ids[:max_len]
            else:
                token_ids = encoded.ids + [tokenizer.token_to_id("[PAD]") or 0] * (max_len - len(encoded.ids))
                
            token_data[sentence] = {
                "token_ids": token_ids,
                "length": len(encoded.ids)
            }
        
        with open("model/reference_tokens.json", "w") as f:
            json.dump(token_data, f, indent=2)
        
        print("  ✅ Saved reference tokens to model/reference_tokens.json")
        
    else:
        print(f"❌ ONNX model not found at {onnx_path}")
    
    print("\n✨ Validation complete!")

if __name__ == "__main__":
    main()
