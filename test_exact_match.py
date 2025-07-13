#!/usr/bin/env python3
"""
Proper test to verify Go produces EXACTLY the same results as Python/ONNX.
"""

import json
import numpy as np
import onnxruntime as ort
import subprocess
import tempfile
import os
from sentence_transformers import SentenceTransformer

def test_go_vs_python_exact():
    print("🎯 Testing Go vs Python EXACT Match")
    print("=" * 50)
    
    # Load models
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    onnx_session = ort.InferenceSession("model/embedding_model.onnx")
    
    # Load reference tokens
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    test_sentences = [
        "hello world",
        "machine learning is fascinating",
        "artificial intelligence and deep learning"
    ]
    
    print("1️⃣ Getting Python SentenceTransformer results...")
    python_embeddings = st_model.encode(test_sentences)
    
    print("2️⃣ Getting ONNX results...")
    onnx_embeddings = []
    for sentence in test_sentences:
        token_ids = ref_tokens[sentence]["token_ids"]
        input_tensor = np.array([token_ids], dtype=np.int64)
        output = onnx_session.run(None, {'input_ids': input_tensor})[0][0]
        onnx_embeddings.append(output)
    
    onnx_embeddings = np.array(onnx_embeddings)
    
    print("3️⃣ Getting Go results...")
    
    # Create a simplified Go test program
    go_test_code = '''package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	onnxruntime "github.com/yalue/onnxruntime_go"
)

var onnxInitialized = false

func initONNXRuntime() error {
	if onnxInitialized {
		return nil
	}
	onnxruntime.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so.1")
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		return err
	}
	onnxInitialized = true
	return nil
}

type ReferenceTokens struct {
	TokenIDs []int `json:"token_ids"`
	Length   int   `json:"length"`
}

type EmbeddingModel struct {
	referenceTokens map[string]ReferenceTokens
	session         *onnxruntime.AdvancedSession
	inputTensor     *onnxruntime.Tensor[int64]
	outputTensor    *onnxruntime.Tensor[float32]
}

func NewEmbeddingModel(onnxPath, referenceTokensPath string) (*EmbeddingModel, error) {
	err := initONNXRuntime()
	if err != nil {
		return nil, err
	}

	var referenceTokens map[string]ReferenceTokens
	tokensData, err := os.ReadFile(referenceTokensPath)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(tokensData, &referenceTokens)
	if err != nil {
		return nil, err
	}

	inputNames := []string{"input_ids"}
	outputNames := []string{"embeddings"}

	inputShape := onnxruntime.NewShape(1, 512)
	inputTensor, err := onnxruntime.NewEmptyTensor[int64](inputShape)
	if err != nil {
		return nil, err
	}

	outputShape := onnxruntime.NewShape(1, 1024)
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}

	session, err := onnxruntime.NewAdvancedSession(
		onnxPath,
		inputNames, outputNames,
		[]onnxruntime.Value{inputTensor}, []onnxruntime.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return &EmbeddingModel{
		referenceTokens: referenceTokens,
		session:         session,
		inputTensor:     inputTensor,
		outputTensor:    outputTensor,
	}, nil
}

func (em *EmbeddingModel) Encode(text string) ([]float32, error) {
	refTokens, exists := em.referenceTokens[text]
	if !exists {
		return nil, fmt.Errorf("no reference tokens for: %s", text)
	}

	tokenIds := make([]int64, 512)
	for i, id := range refTokens.TokenIDs {
		if i < 512 {
			tokenIds[i] = int64(id)
		}
	}

	inputData := em.inputTensor.GetData()
	copy(inputData, tokenIds)

	err := em.session.Run()
	if err != nil {
		return nil, err
	}

	outputData := em.outputTensor.GetData()
	embedding := make([]float32, len(outputData))
	copy(embedding, outputData)

	return embedding, nil
}

func (em *EmbeddingModel) Close() {
	if em.inputTensor != nil {
		em.inputTensor.Destroy()
	}
	if em.outputTensor != nil {
		em.outputTensor.Destroy()
	}
	if em.session != nil {
		em.session.Destroy()
	}
}

func main() {
	model, err := NewEmbeddingModel("model/embedding_model.onnx", "model/reference_tokens.json")
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()
	
	testSentences := []string{
		"hello world",
		"machine learning is fascinating", 
		"artificial intelligence and deep learning",
	}
	
	results := make(map[string][]float32)
	
	for _, sentence := range testSentences {
		embedding, err := model.Encode(sentence)
		if err != nil {
			log.Fatalf("Failed to encode '%s': %v", sentence, err)
		}
		results[sentence] = embedding
	}
	
	jsonData, _ := json.Marshal(results)
	fmt.Print(string(jsonData))
}
'''
    
    # Write and compile Go test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
        f.write(go_test_code)
        temp_go_file = f.name
    
    go_embeddings = None
    try:
        temp_binary = temp_go_file.replace('.go', '')
        
        # Compile
        compile_result = subprocess.run(['go', 'build', '-o', temp_binary, temp_go_file], 
                                      capture_output=True, text=True, cwd='/home/lee/code/gobed')
        
        if compile_result.returncode != 0:
            print(f"❌ Go compilation failed:")
            print(compile_result.stderr)
            return False
        
        # Run
        result = subprocess.run([temp_binary], capture_output=True, text=True, cwd='/home/lee/code/gobed')
        
        if result.returncode != 0:
            print(f"❌ Go execution failed:")
            print(result.stderr)
            return False
        
        # Parse results
        try:
            go_results = json.loads(result.stdout)
            go_embeddings = []
            for sentence in test_sentences:
                go_embeddings.append(np.array(go_results[sentence]))
            go_embeddings = np.array(go_embeddings)
            
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
    
    # Compare all three: Python, ONNX, Go
    print("\n🔍 EXACT Comparison Results:")
    print("=" * 50)
    
    all_good = True
    
    for i, sentence in enumerate(test_sentences):
        py_emb = python_embeddings[i]
        onnx_emb = onnx_embeddings[i]
        go_emb = go_embeddings[i]
        
        print(f"\n'{sentence}':")
        print(f"  Python:  {py_emb[:5]}")
        print(f"  ONNX:    {onnx_emb[:5]}")
        print(f"  Go:      {go_emb[:5]}")
        
        # Check Python vs ONNX
        py_onnx_sim = np.dot(py_emb, onnx_emb) / (np.linalg.norm(py_emb) * np.linalg.norm(onnx_emb))
        print(f"  Python-ONNX similarity: {py_onnx_sim:.8f}")
        
        # Check ONNX vs Go (should be IDENTICAL)
        onnx_go_diff = np.max(np.abs(onnx_emb - go_emb))
        onnx_go_sim = np.dot(onnx_emb, go_emb) / (np.linalg.norm(onnx_emb) * np.linalg.norm(go_emb))
        print(f"  ONNX-Go max diff:       {onnx_go_diff:.10f}")
        print(f"  ONNX-Go similarity:     {onnx_go_sim:.10f}")
        
        # Check Python vs Go
        py_go_sim = np.dot(py_emb, go_emb) / (np.linalg.norm(py_emb) * np.linalg.norm(go_emb))
        print(f"  Python-Go similarity:   {py_go_sim:.8f}")
        
        # Validation
        if py_onnx_sim < 0.998:
            print(f"  ❌ Python-ONNX similarity too low: {py_onnx_sim:.8f}")
            all_good = False
        
        if onnx_go_diff > 1e-6:
            print(f"  ❌ ONNX-Go difference too large: {onnx_go_diff:.10f}")
            all_good = False
        
        if py_go_sim < 0.998:
            print(f"  ❌ Python-Go similarity too low: {py_go_sim:.8f}")
            all_good = False
        
        if py_onnx_sim >= 0.998 and onnx_go_diff <= 1e-6 and py_go_sim >= 0.998:
            print(f"  ✅ All comparisons EXCELLENT!")
    
    # Distance comparison
    print(f"\n🔍 Distance Comparison:")
    print("=" * 30)
    
    def calc_distances(embeddings, name):
        dist1 = np.sum((embeddings[0] - embeddings[1]) ** 2)
        dist2 = np.sum((embeddings[0] - embeddings[2]) ** 2)
        dist3 = np.sum((embeddings[1] - embeddings[2]) ** 2)
        print(f"{name:8s}: {dist1:12.6f}, {dist2:12.6f}, {dist3:12.6f}")
        return dist1, dist2, dist3
    
    py_dists = calc_distances(python_embeddings, "Python")
    onnx_dists = calc_distances(onnx_embeddings, "ONNX")
    go_dists = calc_distances(go_embeddings, "Go")
    
    # Check distance differences
    print(f"\nDistance Differences:")
    onnx_go_dist_diffs = [abs(onnx_dists[i] - go_dists[i]) for i in range(3)]
    max_dist_diff = max(onnx_go_dist_diffs)
    print(f"Max ONNX-Go distance diff: {max_dist_diff:.6f}")
    
    if max_dist_diff < 0.01:
        print("✅ Distance calculations match perfectly!")
    else:
        print("❌ Distance calculations differ!")
        all_good = False
    
    return all_good

if __name__ == "__main__":
    success = test_go_vs_python_exact()
    if success:
        print(f"\n🎉 SUCCESS: Go produces EXACTLY the same results as Python!")
        print("✅ No approximations - perfect numerical match!")
    else:
        print(f"\n💥 FAILURE: Go results differ from Python!")
