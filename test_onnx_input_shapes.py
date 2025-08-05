#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import json

# Load the ONNX model
session = ort.InferenceSession("model/production_embedding_model.onnx")

# Check input specifications
print("ONNX Model Input Specifications:")
for input_spec in session.get_inputs():
    print(f"  Name: {input_spec.name}")
    print(f"  Shape: {input_spec.shape}")
    print(f"  Type: {input_spec.type}")

print("\nONNX Model Output Specifications:")
for output_spec in session.get_outputs():
    print(f"  Name: {output_spec.name}")
    print(f"  Shape: {output_spec.shape}")
    print(f"  Type: {output_spec.type}")

# Load reference tokens
with open('model/debug_tokens.json', 'r') as f:
    tokens = json.load(f)

print("\n" + "="*50)
print("TESTING DIFFERENT INPUT SHAPES")
print("="*50)

# Test with exact token lengths (no padding)
for sentence, token_ids in tokens.items():
    print(f"\nTesting: {sentence}")
    print(f"Token IDs: {token_ids}")
    print(f"Length: {len(token_ids)}")
    
    # Try exact length (no padding)
    try:
        input_ids = np.array([token_ids], dtype=np.int64)
        print(f"Input shape: {input_ids.shape}")
        
        result = session.run(None, {"input_ids": input_ids})
        output = result[0]
        print(f"✓ SUCCESS! Output shape: {output.shape}")
        print(f"Output first 5: [{output[0][0]:.3f}, {output[0][1]:.3f}, {output[0][2]:.3f}, {output[0][3]:.3f}, {output[0][4]:.3f}]")
        
        # Calculate norm
        norm = np.linalg.norm(output[0])
        print(f"Norm: {norm:.3f}")
        
    except Exception as e:
        print(f"❌ Failed with exact length: {e}")
        
        # Try with padding to various lengths
        for pad_length in [16, 32, 64, 128, 256, 512]:
            try:
                padded_ids = token_ids + [0] * (pad_length - len(token_ids))
                input_ids = np.array([padded_ids], dtype=np.int64)
                
                result = session.run(None, {"input_ids": input_ids})
                output = result[0]
                print(f"✓ SUCCESS with padding to {pad_length}! Output shape: {output.shape}")
                print(f"Output first 5: [{output[0][0]:.3f}, {output[0][1]:.3f}, {output[0][2]:.3f}, {output[0][3]:.3f}, {output[0][4]:.3f}]")
                
                # Calculate norm
                norm = np.linalg.norm(output[0])
                print(f"Norm: {norm:.3f}")
                break
            except Exception as e2:
                continue
        else:
            print(f"❌ Failed with all padding lengths")
    
    print("-" * 40)

print("\n" + "="*50)
print("COMPARING DIFFERENT PADDING STRATEGIES")  
print("="*50)

# Take the first sentence and test different padding strategies
sentence = "This is a test sentence."
token_ids = tokens[sentence]

print(f"Original tokens: {token_ids} (length: {len(token_ids)})")

strategies = [
    ("No padding", token_ids),
    ("Pad to 16", token_ids + [0] * (16 - len(token_ids))),
    ("Pad to 512", token_ids + [0] * (512 - len(token_ids))),
]

results = []
for name, padded_tokens in strategies:
    try:
        input_ids = np.array([padded_tokens], dtype=np.int64)
        result = session.run(None, {"input_ids": input_ids})
        output = result[0]
        results.append((name, output[0]))
        print(f"{name}: SUCCESS - shape {output.shape}, norm {np.linalg.norm(output[0]):.3f}")
    except Exception as e:
        print(f"{name}: FAILED - {e}")

# Compare results
if len(results) >= 2:
    print(f"\nComparing embeddings:")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            name1, emb1 = results[i]
            name2, emb2 = results[j]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"  {name1} vs {name2}: similarity = {similarity:.6f}")
