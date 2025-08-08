#!/usr/bin/env python3
"""
Debug the tokenizer to understand its output format.
"""

from sentence_transformers import SentenceTransformer

# Load the model
model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
model = SentenceTransformer(model_path)

sentence = "This is a test sentence."
print(f"Testing: '{sentence}'")

# Debug tokenization
inputs = model.tokenize([sentence])
print(f"Tokenize output type: {type(inputs)}")
print(f"Tokenize output: {inputs}")

if isinstance(inputs, dict):
    for key, value in inputs.items():
        print(f"  {key}: {type(value)} - {value}")
        if hasattr(value, 'shape'):
            print(f"    shape: {value.shape}")

# Try direct tokenizer access
print(f"\nTokenizer type: {type(model.tokenizer)}")
direct_tokens = model.tokenizer(sentence, return_tensors='pt')
print(f"Direct tokenizer output: {direct_tokens}")