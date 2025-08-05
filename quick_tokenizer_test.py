#!/usr/bin/env python3
"""
Quick test to validate our tokenization hypothesis.
"""

import json
import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer

def main():
    print("🔍 TOKENIZATION VALIDATION TEST")
    print("=" * 40)
    
    # Load models
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    onnx_session = ort.InferenceSession("model/embedding_model.onnx")
    tokenizer = st_model.tokenizer
    
    sentence = "hello world"
    print(f"Testing: '{sentence}'")
    
    # Method 1: SentenceTransformer (reference)
    st_embedding = st_model.encode([sentence])
    print(f"\\n1️⃣ SentenceTransformer:")
    print(f"   Shape: {st_embedding.shape}")
    print(f"   Values: {st_embedding[0][:5]}")
    
    # Method 2: Current ONNX (with reference tokens)
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    current_tokens = ref_tokens[sentence]["token_ids"]
    # Pad to 512 as expected by ONNX model
    padded_tokens = current_tokens + [0] * (512 - len(current_tokens))
    input_tensor = np.array([padded_tokens], dtype=np.int64)
    
    onnx_output = onnx_session.run(None, {'input_ids': input_tensor})[0][0]
    print(f"\\n2️⃣ Current ONNX (with reference tokens):")
    print(f"   Input tokens: {current_tokens}")
    print(f"   Shape: {onnx_output.shape}")
    print(f"   Values: {onnx_output[:5]}")
    
    # Method 3: Proper tokenization
    encoded = tokenizer.encode(sentence)
    proper_tokens = encoded.ids
    print(f"\\n3️⃣ Proper tokenizer output:")
    print(f"   Tokens: {encoded.tokens}")
    print(f"   IDs: {proper_tokens}")
    
    # Check if they match
    if current_tokens[:len(proper_tokens)] == proper_tokens:
        print("   ✅ Reference tokens match proper tokenizer!")
    else:
        print("   ⚠️ Reference tokens differ from proper tokenizer:")
        print(f"      Reference: {current_tokens[:len(proper_tokens)]}")
        print(f"      Proper: {proper_tokens}")
    
    # Method 4: Test ONNX with proper tokens
    proper_padded = proper_tokens + [0] * (512 - len(proper_tokens))
    proper_input = np.array([proper_padded], dtype=np.int64)
    proper_onnx_output = onnx_session.run(None, {'input_ids': proper_input})[0][0]
    
    print(f"\\n4️⃣ ONNX with proper tokens:")
    print(f"   Values: {proper_onnx_output[:5]}")
    
    # Compare outputs
    diff_current = np.max(np.abs(st_embedding[0] - onnx_output))
    diff_proper = np.max(np.abs(st_embedding[0] - proper_onnx_output))
    
    print(f"\\n📊 Comparison:")
    print(f"   SentenceTransformer vs Current ONNX: {diff_current:.6f}")
    print(f"   SentenceTransformer vs Proper ONNX: {diff_proper:.6f}")
    
    if diff_proper < diff_current:
        print("   ✅ Proper tokenization gives better match!")
    elif current_tokens[:len(proper_tokens)] == proper_tokens:
        print("   ✅ Tokenization is already correct")
    else:
        print("   ⚠️ Issue might be elsewhere")

if __name__ == "__main__":
    main()
