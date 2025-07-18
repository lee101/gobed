#!/usr/bin/env python3
"""
Generate reference tokens for test sentences using the production tokenizer.
"""

from sentence_transformers import SentenceTransformer
import json

def generate_production_tokens():
    print("Loading production model...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    tokenizer = model.tokenizer
    
    # Test sentences for validation
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today.",
        "Python is a programming language.",
        "Artificial intelligence will change the world.",
        "Hello world",
        "The cat sits on the mat",
        "Technology is advancing rapidly",
        "Natural language processing",
        "Deep learning models"
    ]
    
    print(f"Generating tokens for {len(test_sentences)} sentences...")
    
    reference_tokens = {}
    
    for sentence in test_sentences:
        encoding = tokenizer.encode(sentence)
        tokens = encoding.ids  # Extract the actual token IDs
        reference_tokens[sentence] = {
            "token_ids": tokens,
            "length": len(tokens)
        }
        print(f"'{sentence}' -> {len(tokens)} tokens: {tokens}")
    
    # Save to file
    output_file = "model/production_reference_tokens.json"
    with open(output_file, "w") as f:
        json.dump(reference_tokens, f, indent=2)
    
    print(f"\nReference tokens saved to: {output_file}")
    print(f"Total sentences: {len(reference_tokens)}")
    
    return reference_tokens

if __name__ == "__main__":
    generate_production_tokens()
