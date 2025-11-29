#!/usr/bin/env python3
"""
Generate reference tokens and embeddings from the real model.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print(" Generating reference data from real model...")
    
    # Load the model
    model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
    model = SentenceTransformer(model_path)
    
    # Test sentences
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.", 
        "Hello world"
    ]
    
    print(f" Processing {len(test_sentences)} sentences...")
    
    # Generate reference tokens and embeddings
    reference_tokens = {}
    embeddings = []
    
    for sentence in test_sentences:
        print(f"   Processing: '{sentence}'")
        
        # Get tokens using the model's tokenizer
        inputs = model.tokenize([sentence])
        token_ids = inputs['input_ids'].tolist()
        
        # Get embedding
        embedding = model.encode([sentence])[0]
        
        reference_tokens[sentence] = {
            "token_ids": token_ids,
            "length": len(token_ids)
        }
        embeddings.append(embedding)
        
        print(f"     Tokens: {len(token_ids)}, Embedding shape: {embedding.shape}")
    
    # Save reference tokens
    tokens_path = "./model/real_reference_tokens.json"
    with open(tokens_path, 'w') as f:
        json.dump(reference_tokens, f, indent=2)
    print(f" Saved tokens: {tokens_path}")
    
    # Save expected embeddings
    embeddings_array = np.array(embeddings)
    np.save("./model/expected_embeddings.npy", embeddings_array)
    
    # Save sentences
    with open("./model/expected_sentences.txt", 'w') as f:
        for sentence in test_sentences:
            f.write(f"{sentence}\n")
    
    print(f" Saved embeddings: shape {embeddings_array.shape}")
    print(f" Sample embedding for '{test_sentences[0]}': [{embeddings[0][0]:.3f}, {embeddings[0][1]:.3f}, {embeddings[0][2]:.3f}, {embeddings[0][3]:.3f}, {embeddings[0][4]:.3f}]")
    
    print(" Real reference data generated successfully!")

if __name__ == "__main__":
    main()