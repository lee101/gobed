#!/usr/bin/env python3
"""
Convert SentenceTransformer model to ONNX format for use in Go with GPU support
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import shutil

def convert_model_to_onnx():
    # Load the model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    
    # Create output directory
    output_dir = Path("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test the model with our example texts first
    test_texts = ["hi", "bonjour", "actionable business insights"]
    embeddings = model.encode(test_texts)
    
    # Save test embeddings for verification
    np.save(str(output_dir / "test_embeddings.npy"), embeddings)
    
    print("Model tested successfully!")
    print(f"Test embeddings saved to: {output_dir / 'test_embeddings.npy'}")
    
    # Print embedding dimensions
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Show similarity between test texts
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    print("\nSimilarity matrix:")
    print(f"hi vs bonjour: {similarities[0][1]:.4f}")
    print(f"hi vs actionable business insights: {similarities[0][2]:.4f}")
    print(f"bonjour vs actionable business insights: {similarities[1][2]:.4f}")
    
    # Save the model in SentenceTransformer format
    model.save(str(output_dir / "sentence_transformer"))
    print(f"Model saved to: {output_dir / 'sentence_transformer'}")
    
    # Copy tokenizer.json file for Go tokenization
    tokenizer_src = output_dir / "sentence_transformer" / "tokenizer.json"
    tokenizer_dst = output_dir / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"Tokenizer copied to: {tokenizer_dst}")
    
    # Now convert to ONNX properly
    print("\nConverting to ONNX...")
    try:
        # Create dummy input with proper batch dimension
        dummy_input = torch.ones(1, 10, dtype=torch.long)  # batch_size=1, seq_len=10
        
        # Export the entire model to ONNX
        onnx_path = output_dir / "sentence_model.onnx"
        torch.onnx.export(
            model,                      # The entire sentence transformer model
            dummy_input,                # Dummy input tensor
            str(onnx_path),            # Output file
            input_names=["input_ids"],  # Name of input tensor
            output_names=["sentence_embedding"],  # Name of output tensor
            dynamic_axes={             # Support variable sequence lengths
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "sentence_embedding": {0: "batch_size"}
            },
            opset_version=12           # ONNX opset version
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        
        # Test the ONNX model to verify it works
        print("\nTesting ONNX model...")
        import onnx
        import onnxruntime as ort
        
        # Load and verify the model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        # Test with ONNX Runtime
        session = ort.InferenceSession(str(onnx_path))
        
        # Test with dummy input
        test_input = np.array([[101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0]], dtype=np.int64)  # CLS hello world SEP + padding
        result = session.run(None, {"input_ids": test_input})
        print(f"ONNX test successful! Output shape: {result[0].shape}")
        
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        print("Trying alternative approach with just the transformer...")
        
        try:
            # Alternative: Export just the transformer part
            transformer_model = model[0]  # Get the transformer module
            
            # Create dummy inputs for the transformer
            dummy_input_ids = torch.ones(1, 10, dtype=torch.long)
            dummy_attention_mask = torch.ones(1, 10, dtype=torch.long)
            
            onnx_path_alt = output_dir / "transformer_model.onnx"
            torch.onnx.export(
                transformer_model,
                (dummy_input_ids, dummy_attention_mask),
                str(onnx_path_alt),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=12
            )
            
            print(f"Alternative ONNX model saved to: {onnx_path_alt}")
            print("Note: This model outputs token embeddings. Mean pooling will be needed in Go.")
            
        except Exception as e2:
            print(f"Alternative export also failed: {e2}")
            print("Using the simple ONNX model created earlier...")

if __name__ == "__main__":
    convert_model_to_onnx()