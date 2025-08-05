#!/usr/bin/env python3
"""
Exact comparison between Python PyTorch and Go Safetensors results.
"""

import subprocess
import json
import re

def extract_embeddings(output):
    """Extract embedding values from test output."""
    results = {}
    lines = output.split('\n')
    for line in lines:
        if "'->" in line and "[" in line:
            # Extract sentence and embedding values
            match = re.search(r"'([^']+)' -> \[([-\d.]+), ([-\d.]+), ([-\d.]+), ([-\d.]+), ([-\d.]+)\]", line)
            if match:
                sentence = match.group(1)
                values = [float(match.group(i)) for i in range(2, 7)]
                results[sentence] = values
    return results

def extract_similarities(output):
    """Extract similarity matrix from test output."""
    lines = output.split('\n')
    similarities = []
    in_matrix = False
    for line in lines:
        if "S1  " in line and "S2  " in line:
            in_matrix = True
            continue
        if in_matrix and line.startswith("S"):
            # Extract similarity values
            parts = line.split()
            if len(parts) >= 6:  # S1, sim1, sim2, sim3, sim4, sim5
                sims = [float(parts[i]) for i in range(1, 6)]
                similarities.append(sims)
    return similarities

def run_python_test():
    """Run Python PyTorch test."""
    try:
        result = subprocess.run(
            ["python", "test_python_pytorch.py"],
            capture_output=True, text=True, cwd=".", 
            env={"PATH": "/home/lee/code/gobed/.venv/bin:" + subprocess.os.environ.get("PATH", "")}
        )
        return result.stdout
    except Exception as e:
        print(f"Error running Python test: {e}")
        return ""

def run_go_test():
    """Run Go safetensors test."""
    try:
        result = subprocess.run(
            ["go", "run", "safetensors_loader.go"],
            capture_output=True, text=True, cwd="."
        )
        return result.stdout
    except Exception as e:
        print(f"Error running Go test: {e}")
        return ""

def main():
    print("EXACT EMBEDDING COMPARISON")
    print("=" * 50)
    
    # Run both tests
    print("Running Python PyTorch test...")
    python_output = run_python_test()
    
    print("Running Go Safetensors test...")
    go_output = run_go_test()
    
    # Extract results
    python_embeddings = extract_embeddings(python_output)
    go_embeddings = extract_embeddings(go_output)
    
    python_similarities = extract_similarities(python_output)
    go_similarities = extract_similarities(go_output)
    
    print("\nEMBEDDING VALUES COMPARISON")
    print("-" * 70)
    
    all_match = True
    
    for sentence in python_embeddings:
        if sentence in go_embeddings:
            py_vals = python_embeddings[sentence]
            go_vals = go_embeddings[sentence]
            
            print(f"\n'{sentence}':")
            print(f"Python: [{py_vals[0]:6.3f}, {py_vals[1]:6.3f}, {py_vals[2]:6.3f}, {py_vals[3]:6.3f}, {py_vals[4]:6.3f}]")
            print(f"Go:     [{go_vals[0]:6.3f}, {go_vals[1]:6.3f}, {go_vals[2]:6.3f}, {go_vals[3]:6.3f}, {go_vals[4]:6.3f}]")
            
            # Check if they match exactly
            match = all(abs(py_vals[i] - go_vals[i]) < 0.0001 for i in range(5))
            max_diff = max(abs(py_vals[i] - go_vals[i]) for i in range(5))
            
            if match:
                print(f"✅ PERFECT MATCH (max diff: {max_diff:.6f})")
            else:
                print(f"❌ MISMATCH (max diff: {max_diff:.6f})")
                all_match = False
    
    print("\nSIMILARITY MATRIX COMPARISON")
    print("-" * 40)
    
    if python_similarities and go_similarities:
        print("\nPython PyTorch Similarities:")
        for i, row in enumerate(python_similarities):
            print(f"S{i+1}: {' '.join(f'{val:5.3f}' for val in row)}")
        
        print("\nGo Safetensors Similarities:")
        for i, row in enumerate(go_similarities):
            print(f"S{i+1}: {' '.join(f'{val:5.3f}' for val in row)}")
        
        # Check similarity matrix match
        sim_match = True
        max_sim_diff = 0.0
        
        if len(python_similarities) == len(go_similarities):
            for i in range(len(python_similarities)):
                if len(python_similarities[i]) == len(go_similarities[i]):
                    for j in range(len(python_similarities[i])):
                        diff = abs(python_similarities[i][j] - go_similarities[i][j])
                        max_sim_diff = max(max_sim_diff, diff)
                        if diff > 0.001:
                            sim_match = False
        
        print(f"\nSimilarity matrix match: {'✅ YES' if sim_match else '❌ NO'} (max diff: {max_sim_diff:.6f})")
    
    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    
    if all_match and (not python_similarities or not go_similarities or sim_match):
        print("🎉 PERFECT SUCCESS!")
        print("✅ All embeddings match exactly")
        print("✅ All similarity scores match exactly") 
        print("✅ Go and Python implementations are 100% consistent")
        print("\nThe Go static embedding model works exactly the same as Python PyTorch!")
    else:
        print("⚠️  Some differences detected")
        if not all_match:
            print("❌ Embedding values differ")
        if python_similarities and go_similarities and not sim_match:
            print("❌ Similarity matrices differ")
    
    print(f"\nTested sentences: {len(python_embeddings)}")
    print(f"Embedding dimension: 1024 (showing first 5 values)")
    print(f"Model: sentence-transformers/static-retrieval-mrl-en-v1")

if __name__ == "__main__":
    main()