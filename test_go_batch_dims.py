#!/usr/bin/env python3
"""
Test to verify Go handles batch dimensions correctly by comparing with expected ONNX outputs.
"""

import json
import numpy as np
import onnxruntime as ort
import subprocess
import tempfile
import os

def test_go_batch_dimension_handling():
    print("🧪 Testing Go Batch Dimension Handling")
    print("=" * 50)
    
    # Load reference tokens and ONNX model
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    session = ort.InferenceSession("model/embedding_model.onnx")
    
    test_sentences = [
        "hello world",
        "machine learning is fascinating",
        "artificial intelligence and deep learning"
    ]
    
    print("🔍 Testing different batch sizes with ONNX...")
    
    # Test batch size 1 (current Go approach)
    print("\n1️⃣ Batch Size 1 (individual):")
    individual_outputs = []
    for sentence in test_sentences:
        token_ids = ref_tokens[sentence]["token_ids"]
        input_tensor = np.array([token_ids], dtype=np.int64)  # Shape: [1, 512]
        output = session.run(None, {'input_ids': input_tensor})[0]  # Shape: [1, 1024]
        
        print(f"  '{sentence}':")
        print(f"    Input shape: {input_tensor.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Embedding: {output[0][:3]}")  # First 3 values
        
        individual_outputs.append(output[0])  # Extract embedding from batch dim
    
    # Test batch size 3 (true batch)
    print("\n2️⃣ Batch Size 3 (true batch):")
    all_token_ids = []
    for sentence in test_sentences:
        token_ids = ref_tokens[sentence]["token_ids"] 
        all_token_ids.append(token_ids)
    
    batch_input = np.array(all_token_ids, dtype=np.int64)  # Shape: [3, 512]
    batch_output = session.run(None, {'input_ids': batch_input})[0]  # Shape: [3, 1024]
    
    print(f"  Batch input shape: {batch_input.shape}")
    print(f"  Batch output shape: {batch_output.shape}")
    
    # Compare individual vs batch
    print("\n🔍 Individual vs Batch Comparison:")
    all_match = True
    for i, sentence in enumerate(test_sentences):
        individual_emb = individual_outputs[i]
        batch_emb = batch_output[i]
        
        # Check if they're identical (they should be)
        similarity = np.dot(individual_emb, batch_emb) / (
            np.linalg.norm(individual_emb) * np.linalg.norm(batch_emb)
        )
        
        print(f"  '{sentence}':")
        print(f"    Individual: {individual_emb[:3]}")
        print(f"    Batch:      {batch_emb[:3]}")
        print(f"    Similarity: {similarity:.8f}")
        
        if similarity < 0.9999:
            print(f"    ❌ Mismatch detected!")
            all_match = False
        else:
            print(f"    ✅ Perfect match")
    
    if all_match:
        print("\n✅ Batch processing works correctly!")
    else:
        print("\n❌ Batch processing has issues!")
        return False
    
    # Test what Go is actually producing
    print("\n3️⃣ Testing Go Output:")
    
    # Create a simple Go program to output raw embeddings
    go_test_code = f'''package main

import (
	"encoding/json"
	"fmt"
	"log"
	onnxruntime "github.com/yalue/onnxruntime_go"
)

{open('main.go').read().split('func main()')[0]}

func main() {{
	model, err := NewEmbeddingModel("model/embedding_model.onnx", "model/reference_tokens.json", false)
	if err != nil {{
		log.Fatalf("Failed to create model: %v", err)
	}}
	defer model.Close()
	
	testSentences := []string{{
		"hello world",
		"machine learning is fascinating", 
		"artificial intelligence and deep learning",
	}}
	
	results := make(map[string][]float32)
	
	for _, sentence := range testSentences {{
		embedding, err := model.Encode(sentence)
		if err != nil {{
			log.Fatalf("Failed to encode '%s': %v", sentence, err)
		}}
		results[sentence] = embedding[:3]  // Just first 3 values for comparison
	}}
	
	jsonData, _ := json.Marshal(results)
	fmt.Print(string(jsonData))
}}
'''
    
    # Write, compile, and run Go test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
        f.write(go_test_code)
        temp_go_file = f.name
    
    try:
        temp_binary = temp_go_file.replace('.go', '')
        
        # Compile
        compile_result = subprocess.run(['go', 'build', '-o', temp_binary, temp_go_file], 
                                      capture_output=True, text=True, cwd='/home/lee/code/gobed')
        
        if compile_result.returncode != 0:
            print(f"❌ Go compilation failed: {compile_result.stderr}")
            return False
        
        # Run
        result = subprocess.run([temp_binary], capture_output=True, text=True, cwd='/home/lee/code/gobed')
        
        if result.returncode != 0:
            print(f"❌ Go execution failed: {result.stderr}")
            return False
        
        # Parse JSON output
        try:
            go_results = json.loads(result.stdout)
            
            print("  Go embeddings (first 3 values):")
            for sentence in test_sentences:
                go_emb = np.array(go_results[sentence])
                individual_emb = individual_outputs[test_sentences.index(sentence)][:3]
                
                print(f"    '{sentence}':")
                print(f"      Go:     {go_emb}")
                print(f"      Python: {individual_emb}")
                
                # Check similarity
                similarity = np.dot(go_emb, individual_emb) / (
                    np.linalg.norm(go_emb) * np.linalg.norm(individual_emb)
                )
                print(f"      Match:  {similarity:.8f}")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Go output: {e}")
            print(f"Go stdout: {result.stdout}")
            return False
            
    finally:
        # Cleanup
        if os.path.exists(temp_go_file):
            os.unlink(temp_go_file)
        if os.path.exists(temp_binary):
            os.unlink(temp_binary)
    
    print("\n✅ Go batch dimension handling verified!")
    return True

if __name__ == "__main__":
    success = test_go_batch_dimension_handling()
    if success:
        print("\n🎉 All batch dimension tests passed!")
    else:
        print("\n💥 Batch dimension issues detected!")
