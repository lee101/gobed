#!/usr/bin/env python3
"""
Download and convert multilingual-e5-base model to ONNX format for Go implementation.
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import numpy as np

def download_and_convert_model():
    """Download E5 model and convert to ONNX."""
    model_name = "intfloat/multilingual-e5-base"
    output_dir = "./model"
    
    print(f"Downloading {model_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and convert model to ONNX using optimum
        print("Converting model to ONNX format...")
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_name, 
            export=True,
            provider="CPUExecutionProvider"
        )
        
        # Save ONNX model
        onnx_path = os.path.join(output_dir, "embedding_model.onnx")
        ort_model.save_pretrained(output_dir)
        
        # The model is saved as model.onnx, rename to embedding_model.onnx
        if os.path.exists(os.path.join(output_dir, "model.onnx")):
            os.rename(os.path.join(output_dir, "model.onnx"), onnx_path)
        
        print(f"ONNX model saved to: {onnx_path}")
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        
        # Save tokenizer in the format Go library expects
        if hasattr(tokenizer, "save"):
            tokenizer.save_pretrained(output_dir)
        
        print(f"Tokenizer saved to: {output_dir}")
        
        # Test the ONNX model
        print("Testing ONNX model...")
        test_texts = ["query: hi", "query: bonjour", "query: actionable business insights"]
        
        inputs = tokenizer(test_texts, return_tensors="np", padding=True, truncation=True, max_length=512)
        
        # Test with ONNX model
        outputs = ort_model(**inputs)
        embeddings = outputs.last_hidden_state
        
        # Apply average pooling
        attention_mask = inputs['attention_mask']
        embeddings = embeddings * attention_mask[:, :, None]
        embeddings = embeddings.sum(axis=1) / attention_mask.sum(axis=1)[:, None]
        
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        print(f"Test embeddings shape: {embeddings.shape}")
        print(f"Sample embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
        
        # Save test embeddings
        np.save(os.path.join(output_dir, "test_embeddings.npy"), embeddings)
        
        print("✅ Model conversion completed successfully!")
        return onnx_path, tokenizer_path
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("Trying alternative approach...")
        
        # Alternative: manual conversion
        try:
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create dummy input
            dummy_input = tokenizer("query: hello world", return_tensors="pt", padding=True, truncation=True)
            
            # Export to ONNX
            onnx_path = os.path.join(output_dir, "embedding_model.onnx")
            
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                },
                opset_version=11
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            print("✅ Manual conversion completed!")
            return onnx_path, os.path.join(output_dir, "tokenizer.json")
            
        except Exception as e2:
            print(f"❌ Manual conversion also failed: {e2}")
            return None, None

if __name__ == "__main__":
    print("E5 Model Download and Conversion")
    print("=" * 40)
    
    # Check if optimum is available
    try:
        import optimum
        print("Optimum library found - using optimized conversion")
    except ImportError:
        print("Optimum library not found. Installing...")
        os.system("pip install optimum[onnxruntime]")
    
    onnx_path, tokenizer_path = download_and_convert_model()
    
    if onnx_path and tokenizer_path:
        print(f"\n🎉 Success! Model files ready:")
        print(f"   ONNX Model: {onnx_path}")
        print(f"   Tokenizer: {tokenizer_path}")
        print(f"\nYou can now run the Go implementation!")
    else:
        print(f"\n❌ Conversion failed. Check the error messages above.")