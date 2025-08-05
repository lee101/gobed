#!/usr/bin/env python3
"""
Simple ONNX model validation - test if the exported model works.
"""

import numpy as np
import onnxruntime as ort

def test_onnx_model():
    print("Testing ONNX model directly...")
    
    # Load ONNX model
    try:
        session = ort.InferenceSession("model/production_embedding_model.onnx")
        print("✓ ONNX model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load ONNX model: {e}")
        return
    
    # Check input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
    
    # Test with simple input
    test_tokens = np.array([[101, 2023, 2003, 1037, 3231, 6251, 1012, 102]], dtype=np.int64)
    print(f"Test input shape: {test_tokens.shape}")
    
    try:
        outputs = session.run([output_info.name], {input_info.name: test_tokens})
        embedding = outputs[0]
        print("✓ Inference successful!")
        print(f"Output shape: {embedding.shape}")
        print(f"Output first 5 values: {embedding[0][:5]}")
        print(f"Output norm: {np.linalg.norm(embedding[0]):.3f}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")

if __name__ == "__main__":
    test_onnx_model()
