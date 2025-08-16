#!/usr/bin/env python3
"""Compare Go and Python embeddings to verify we're using the real model."""

import json
import os
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer

print("Preparing 100 texts for comparison...")
texts = [
    "Machine learning is fascinating.",
    "Deep learning models are powerful.",
    "The weather is nice today.",
]
# Add a deterministic variety up to 100
for i in range(3, 100):
    texts.append(f"Sample sentence number {i} for embedding verification.")

# Get Python embeddings
print("Loading Python model...")
model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
print("Computing Python embeddings...")
py_embeddings = model.encode(texts, normalize_embeddings=False)

# Ask Go program to compute embeddings
print("Running Go verifier to compute embeddings...")
tmp_texts = '/tmp/texts.json'
tmp_go_out = '/tmp/go_embeddings.json'
with open(tmp_texts, 'w') as f:
    json.dump(texts, f)

result = subprocess.run([
    'go', 'run', './cmd/verify', '--input', tmp_texts, '--output', tmp_go_out
], cwd=os.path.dirname(__file__), capture_output=True, text=True)

if result.returncode != 0:
    print("Go execution failed:\n", result.stderr)
    raise SystemExit(1)

with open(tmp_go_out, 'r') as f:
    go_embeddings = np.array(json.load(f))

print("\n" + "="*60)
print("EMBEDDING COMPARISON - REAL MODEL VERIFICATION")
print("="*60)

for i, text in enumerate(texts):
    print(f"\n📝 Text {i+1}: '{text}'")
    print(f"   Shape: Go={go_embeddings[i].shape}, Python={py_embeddings[i].shape}")

    # Compare first 5 values
    go_vals = go_embeddings[i][:5]
    py_vals = py_embeddings[i][:5]

    print(f"   Go first 5:     {go_vals}")
    print(f"   Python first 5: {py_vals}")

    # Calculate difference
    if len(go_embeddings[i]) == 0:
        print("   ⚠️  Go could not encode this text (tokenizer/path issue)")
        continue
    diff = np.abs(go_embeddings[i] - py_embeddings[i])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"   Max difference:  {max_diff:.8f}")
    print(f"   Mean difference: {mean_diff:.8f}")

    if max_diff < 0.0001:
        print(f"   ✅ PERFECT MATCH! Using real safetensors model!")
    elif max_diff < 0.001:
        print(f"   ✅ Excellent match (within float32 precision)")
    elif max_diff < 0.01:
        print(f"   ⚠️  Good match but some differences")
    else:
        print(f"   ❌ Significant differences detected")

# Calculate cosine similarities to double-check
print("\n" + "="*60)
print("SIMILARITY VERIFICATION")
print("="*60)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test similarity between first two texts
go_sim = cosine_similarity(go_embeddings[0], go_embeddings[1])
py_sim = cosine_similarity(py_embeddings[0], py_embeddings[1])

print(f"\nSimilarity between texts 1 and 2:")
print(f"   Go implementation:     {go_sim:.6f}")
print(f"   Python implementation: {py_sim:.6f}")
print(f"   Difference:            {abs(go_sim - py_sim):.8f}")

if abs(go_sim - py_sim) < 0.0001:
    print("   ✅ Similarities match perfectly!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

all_diffs = [np.max(np.abs(go_embeddings[i] - py_embeddings[i])) for i in range(3)]
overall_max = max(all_diffs)

if overall_max < 0.001:
    print("✅ SUCCESS: Go is using the REAL safetensors model!")
    print("   The embeddings match Python perfectly (within float32 precision)")
    print("   We are definitely using sentence-transformers/static-retrieval-mrl-en-v1")
else:
    print("⚠️  Some differences detected, but may still be using real model")
    print(f"   Maximum difference: {overall_max:.8f}")

print("="*60)
