#!/usr/bin/env python3
"""
Final validation: Compare Go output with Python SentenceTransformer to ensure they match.
"""

import json
import numpy as np
import subprocess
import tempfile
import os
from sentence_transformers import SentenceTransformer

def test_go_vs_python():
    print("🔍 Final Validation: Go vs Python Comparison")
    print("=" * 60)
    
    # Load the Python model
    model = SentenceTransformer("model/sentence_transformer", device='cpu')
    
    test_sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful"
    ]
    
    # Get Python embeddings
    print("📊 Getting Python embeddings...")
    python_embeddings = model.encode(test_sentences)
    
    # Get Go embeddings by running the Go program and parsing output
    print("📊 Getting Go embeddings...")
    
    # Create a simple Go program that just outputs embeddings
    go_test_code = '''package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	onnxruntime "github.com/yalue/onnxruntime_go"
)

// Include the same types and functions from main.go
''' + open('main.go').read().split('func main()')[0] + '''

func main() {
	// Create the model
	model, err := NewEmbeddingModel("model/embedding_model.onnx", "model/reference_tokens.json", false)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()
	
	testSentences := []string{
		"hello world",
		"the weather is nice today", 
		"machine learning algorithms are powerful",
	}
	
	results := make(map[string][]float32)
	
	for _, sentence := range testSentences {
		embedding, err := model.Encode(sentence)
		if err != nil {
			log.Fatalf("Failed to encode '%s': %v", sentence, err)
		}
		results[sentence] = embedding
	}
	
	// Output as JSON
	jsonData, _ := json.Marshal(results)
	fmt.Print(string(jsonData))
}
'''
    
    # Write to temporary file and compile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
        f.write(go_test_code)
        temp_go_file = f.name
    
    try:
        # Compile the Go program
        temp_binary = temp_go_file.replace('.go', '')
        compile_result = subprocess.run(['go', 'build', '-o', temp_binary, temp_go_file], 
                                      capture_output=True, text=True, cwd='/home/lee/code/gobed')
        
        if compile_result.returncode != 0:
            print(f"❌ Go compilation failed: {compile_result.stderr}")
            return False
        
        # Run the Go program
        result = subprocess.run([temp_binary], capture_output=True, text=True, cwd='/home/lee/code/gobed')
        
        if result.returncode != 0:
            print(f"❌ Go execution failed: {result.stderr}")
            return False
        
        # Parse the JSON output
        try:
            go_embeddings_dict = json.loads(result.stdout)
            go_embeddings = np.array([go_embeddings_dict[sentence] for sentence in test_sentences])
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Go output: {e}")
            print(f"Go stdout: {result.stdout}")
            print(f"Go stderr: {result.stderr}")
            return False
            
    finally:
        # Clean up
        if os.path.exists(temp_go_file):
            os.unlink(temp_go_file)
        if os.path.exists(temp_binary):
            os.unlink(temp_binary)
    
    # Compare embeddings
    print("📊 Comparison Results:")
    print("-" * 40)
    
    all_similarities = []
    for i, sentence in enumerate(test_sentences):
        # Calculate cosine similarity between Python and Go embeddings
        py_emb = python_embeddings[i]
        go_emb = go_embeddings[i]
        
        similarity = np.dot(py_emb, go_emb) / (np.linalg.norm(py_emb) * np.linalg.norm(go_emb))
        all_similarities.append(similarity)
        
        print(f"'{sentence}':")
        print(f"  Python sample: {py_emb[:5]}")
        print(f"  Go sample:     {go_emb[:5]}")
        print(f"  Similarity:    {similarity:.6f}")
        print()
    
    # Calculate distance comparisons
    print("📏 Distance Comparison:")
    print("-" * 40)
    
    # Python distances
    py_dist1 = np.sum((python_embeddings[0] - python_embeddings[1]) ** 2)
    py_dist2 = np.sum((python_embeddings[0] - python_embeddings[2]) ** 2)
    py_dist3 = np.sum((python_embeddings[1] - python_embeddings[2]) ** 2)
    
    # Go distances
    go_dist1 = np.sum((go_embeddings[0] - go_embeddings[1]) ** 2)
    go_dist2 = np.sum((go_embeddings[0] - go_embeddings[2]) ** 2)
    go_dist3 = np.sum((go_embeddings[1] - go_embeddings[2]) ** 2)
    
    print("Squared Euclidean Distances:")
    print(f"  Sentence 1 vs 2:")
    print(f"    Python: {py_dist1:.6f}")
    print(f"    Go:     {go_dist1:.6f}")
    print(f"    Diff:   {abs(py_dist1 - go_dist1):.6f}")
    print()
    
    print(f"  Sentence 1 vs 3:")
    print(f"    Python: {py_dist2:.6f}")
    print(f"    Go:     {go_dist2:.6f}")
    print(f"    Diff:   {abs(py_dist2 - go_dist2):.6f}")
    print()
    
    print(f"  Sentence 2 vs 3:")
    print(f"    Python: {py_dist3:.6f}")
    print(f"    Go:     {go_dist3:.6f}")
    print(f"    Diff:   {abs(py_dist3 - go_dist3):.6f}")
    print()
    
    # Validation
    avg_similarity = np.mean(all_similarities)
    max_distance_diff = max(abs(py_dist1 - go_dist1), abs(py_dist2 - go_dist2), abs(py_dist3 - go_dist3))
    
    print("🎯 Validation Results:")
    print("-" * 40)
    print(f"Average embedding similarity: {avg_similarity:.6f}")
    print(f"Max distance difference:      {max_distance_diff:.6f}")
    
    if avg_similarity > 0.999:
        print("✅ EXCELLENT: Go and Python embeddings are nearly identical!")
    elif avg_similarity > 0.95:
        print("✅ GOOD: Go and Python embeddings are very close!")
    else:
        print("❌ PROBLEM: Go and Python embeddings differ significantly!")
        return False
    
    if max_distance_diff < 0.01:
        print("✅ EXCELLENT: Distance calculations match perfectly!")
    elif max_distance_diff < 1.0:
        print("✅ GOOD: Distance calculations are very close!")
    else:
        print("❌ PROBLEM: Distance calculations differ significantly!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_go_vs_python()
    if success:
        print("\n🎉 SUCCESS: Go implementation matches Python reference!")
        print("✅ The embedding similarity issue has been resolved!")
    else:
        print("\n💥 FAILURE: Go implementation still differs from Python!")
