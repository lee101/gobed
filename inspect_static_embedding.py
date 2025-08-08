#\!/usr/bin/env python3
from sentence_transformers import SentenceTransformer

model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
model = SentenceTransformer(model_path)

static_embedding = model[0]
print(f"StaticEmbedding: {type(static_embedding)}")

sentence = "This is a test sentence."
inputs = model.tokenize([sentence])
print(f"Tokenized inputs: {inputs}")

try:
    output = static_embedding(inputs)
    print(f"Static embedding output shape: {output['sentence_embedding'].shape}")
    print(f"Sample: [{output['sentence_embedding'][0][0]:.3f}, {output['sentence_embedding'][0][1]:.3f}, {output['sentence_embedding'][0][2]:.3f}, {output['sentence_embedding'][0][3]:.3f}, {output['sentence_embedding'][0][4]:.3f}]")
except Exception as e:
    print(f"Error: {e}")

full_output = model.encode([sentence])
print(f"Full model sample: [{full_output[0][0]:.3f}, {full_output[0][1]:.3f}, {full_output[0][2]:.3f}, {full_output[0][3]:.3f}, {full_output[0][4]:.3f}]")
