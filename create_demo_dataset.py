#!/usr/bin/env python3
"""
Create a diverse dataset for demonstrating semantic similarity relationships.
"""

import json
from sentence_transformers import SentenceTransformer

def main():
    print(" Creating demo dataset with diverse semantic relationships...")
    
    # Load the model
    model_path = "./real_model_cache/models--sentence-transformers--static-retrieval-mrl-en-v1/snapshots/f60985c706f192d45d218078e49e5a8b6f15283a"
    model = SentenceTransformer(model_path)
    
    # Create a diverse set of sentences showing different semantic relationships
    demo_sentences = [
        # Technology cluster
        "Machine learning is fascinating.",
        "Artificial intelligence will change the world.",
        "Deep learning models are powerful.",
        "Neural networks process information.",
        
        # Greetings cluster  
        "Hello world",
        "Good morning everyone",
        "Hi there friend",
        
        # Nature cluster
        "The weather is nice today.",
        "Birds are singing beautifully.",
        "Trees grow tall in the forest.",
        
        # Programming cluster
        "Python is a programming language.", 
        "JavaScript runs in browsers.",
        "Code should be readable.",
        
        # Random/unrelated
        "The cat sits on the mat",
        "Pizza tastes delicious.",
        "Mathematics requires practice.",
        
        # Original test sentences
        "This is a test sentence.",
        "Technology is advancing rapidly",
        "Natural language processing"
    ]
    
    print(f" Processing {len(demo_sentences)} sentences...")
    
    # Generate tokens and embeddings for all sentences
    reference_tokens = {}
    embeddings = []
    
    for i, sentence in enumerate(demo_sentences):
        print(f"   {i+1:2d}. Processing: '{sentence}'")
        
        try:
            # Get tokens
            inputs = model.tokenize([sentence])
            token_ids = inputs['input_ids'].tolist()
            
            # Get embedding
            embedding = model.encode([sentence])[0]
            
            reference_tokens[sentence] = {
                "token_ids": token_ids,
                "length": len(token_ids)
            }
            embeddings.append(embedding)
            
        except Exception as e:
            print(f"      Error processing '{sentence}': {e}")
    
    # Save reference tokens
    tokens_path = "./model/real_reference_tokens.json"
    with open(tokens_path, 'w') as f:
        json.dump(reference_tokens, f, indent=2)
    print(f" Saved {len(reference_tokens)} reference tokens to: {tokens_path}")
    
    # Save embeddings for verification
    import numpy as np
    embeddings_array = np.array(embeddings)
    np.save("./model/expected_embeddings.npy", embeddings_array)
    
    with open("./model/expected_sentences.txt", 'w') as f:
        for sentence in reference_tokens.keys():
            f.write(f"{sentence}\n")
    
    print(f" Saved embeddings: shape {embeddings_array.shape}")
    
    # Preview some similarity relationships
    print(f"\n Preview of semantic relationships:")
    tech_sentences = [s for s in demo_sentences if any(word in s.lower() for word in ['machine', 'artificial', 'deep', 'neural'])]
    greeting_sentences = [s for s in demo_sentences if any(word in s.lower() for word in ['hello', 'good', 'hi'])]
    
    if len(tech_sentences) >= 2:
        emb1 = model.encode([tech_sentences[0]])[0]
        emb2 = model.encode([tech_sentences[1]])[0]
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   Tech similarity: '{tech_sentences[0]}' ↔ '{tech_sentences[1]}' = {sim:.4f}")
    
    if len(greeting_sentences) >= 2:
        emb1 = model.encode([greeting_sentences[0]])[0]
        emb2 = model.encode([greeting_sentences[1]])[0]
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   Greeting similarity: '{greeting_sentences[0]}' ↔ '{greeting_sentences[1]}' = {sim:.4f}")
    
    print(f"\n Demo dataset created successfully!")
    print(f" {len(reference_tokens)} sentences ready for Go demonstration")

if __name__ == "__main__":
    main()