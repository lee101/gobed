#!/usr/bin/env python3
import json
import torch
from safetensors import safe_open

# Load weights
tensors = {}
with safe_open('cached_model/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors', framework='pt', device='cpu') as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Load tokens
with open('model/production_reference_tokens.json', 'r') as f:
    tokens = json.load(f)

sentences = ['This is a test sentence.', 'Machine learning is fascinating.', 'Hello world']

print("Python PyTorch Embeddings:")
for sentence in sentences:
    token_ids = torch.tensor(tokens[sentence]['token_ids'], dtype=torch.long)
    embeddings = tensors['embedding.weight'][token_ids]
    mask = token_ids != 0
    if mask.sum() > 0:
        emb = embeddings[mask].mean(dim=0)
        print(f"'{sentence}': [{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, {emb[3]:.3f}, {emb[4]:.3f}]")