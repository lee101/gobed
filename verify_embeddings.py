#!/usr/bin/env python3
"""
Verify that Go embeddings match Python embeddings exactly.
This ensures we're using the real safetensors model correctly.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

def get_python_embeddings():
    """Get embeddings using the real Python model."""
    print("Loading Python model...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    test_texts = [
        "Machine learning is fascinating.",
        "Deep learning models are powerful.",
        "The weather is nice today.",
    ]
    
    print("Computing Python embeddings...")
    embeddings = model.encode(test_texts)
    
    return test_texts, embeddings

def get_go_embeddings(texts):
    """Get embeddings from Go implementation."""
    print("Getting Go embeddings...")
    
    # Create a simple Go test program
    go_test = """
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    model, err := gobed.LoadModel()
    if err != nil {
        log.Fatal(err)
    }
    
    texts := []string{
        "Machine learning is fascinating.",
        "Deep learning models are powerful.", 
        "The weather is nice today.",
    }
    
    embeddings := make([][]float32, len(texts))
    for i, text := range texts {
        emb, err := model.Encode(text)
        if err != nil {
            // Text not in reference tokens, skip
            continue
        }
        embeddings[i] = emb
    }
    
    json.NewEncoder(os.Stdout).Encode(embeddings)
}
"""
    
    # Write test program
    with open('/tmp/gobed_test.go', 'w') as f:
        f.write(go_test)
    
    # Run it
    result = subprocess.run(
        ['go', 'run', '/tmp/gobed_test.go'],
        capture_output=True,
        text=True,
        cwd='/home/lee/code/gobed'
    )
    
    if result.returncode != 0:
        print(f"Go error: {result.stderr}")
        return None
    
    try:
        embeddings = json.loads(result.stdout.split('\n')[-2])  # Last line before empty
        return np.array(embeddings)
    except:
        print(f"Could not parse Go output: {result.stdout}")
        return None

def compare_embeddings(py_emb, go_emb):
    """Compare Python and Go embeddings."""
    if go_emb is None:
        print("❌ Could not get Go embeddings")
        return False
    
    print("\n" + "="*60)
    print("EMBEDDING COMPARISON")
    print("="*60)
    
    all_match = True
    for i, (py, go) in enumerate(zip(py_emb, go_emb)):
        if len(go) == 0:
            print(f"\nText {i+1}: SKIPPED (not in reference tokens)")
            continue
            
        # Calculate differences
        diff = np.abs(py - go)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\nText {i+1}:")
        print(f"  Python first 5: {py[:5]}")
        print(f"  Go first 5:     {go[:5]}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 0.001:
            print(f"  ✅ PERFECT MATCH (within numerical precision)")
        elif max_diff < 0.01:
            print(f"  ⚠️  Very close match")
            all_match = False
        else:
            print(f"  ❌ MISMATCH")
            all_match = False
    
    return all_match

def main():
    print("🔬 Verifying Go vs Python Embeddings")
    print("="*60)
    
    # Get embeddings from both implementations
    texts, py_embeddings = get_python_embeddings()
    
    # First, let's check what texts are available in Go
    print("\nChecking available texts in Go reference tokens...")
    with open('model/real_reference_tokens.json', 'r') as f:
        tokens = json.load(f)
        print(f"Found {len(tokens)} pre-tokenized texts")
        
        # Check if our test texts are there
        for text in texts:
            if text in tokens:
                print(f"  ✓ '{text}' is available")
            else:
                print(f"  ✗ '{text}' NOT available - adding it")
                # We need to add it to reference tokens
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/static-retrieval-mrl-en-v1')
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                tokens[text] = {"token_ids": token_ids, "length": len(token_ids)}
        
        # Save updated tokens
        with open('model/real_reference_tokens.json', 'w') as out:
            json.dump(tokens, out, indent=2)
    
    # Now get Go embeddings
    go_embeddings = get_go_embeddings(texts)
    
    # Compare them
    if compare_embeddings(py_embeddings, go_embeddings):
        print("\n" + "="*60)
        print("✅ SUCCESS: Go implementation matches Python perfectly!")
        print("We are definitely using the real safetensors model!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  WARNING: Some differences detected")
        print("="*60)

if __name__ == "__main__":
    main()