#!/usr/bin/env python3
"""
Create a properly quantized int8 ONNX model.
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import json

def create_int8_model():
    print("🔧 Creating Int8 Quantized Model")
    print("=" * 40)
    
    model_fp32 = "model/embedding_model.onnx"
    model_int8 = "model/embedding_model_int8.onnx"
    
    print(f"📦 Quantizing {model_fp32} -> {model_int8}")
    
    try:
        # Quantize the model
        quantize_dynamic(
            model_input=model_fp32,
            model_output=model_int8,
            weight_type=QuantType.QInt8
        )
        print("✅ Quantization completed!")
        
        # Test the quantized model
        print("\n🧪 Testing quantized model...")
        
        # Load reference tokens
        with open("model/reference_tokens.json", "r") as f:
            ref_tokens = json.load(f)
        
        # Test with a sample sentence
        test_sentence = "hello world"
        token_ids = ref_tokens[test_sentence]["token_ids"]
        if len(token_ids) < 512:
            token_ids = token_ids + [0] * (512 - len(token_ids))
        
        input_tensor = np.array([token_ids], dtype=np.int64)
        
        # Test original model
        session_fp32 = ort.InferenceSession(model_fp32)
        output_fp32 = session_fp32.run(None, {'input_ids': input_tensor})[0][0]
        
        # Test quantized model
        session_int8 = ort.InferenceSession(model_int8)
        output_int8 = session_int8.run(None, {'input_ids': input_tensor})[0][0]
        
        # Compare outputs
        similarity = np.dot(output_fp32, output_int8) / (
            np.linalg.norm(output_fp32) * np.linalg.norm(output_int8)
        )
        
        print(f"  Original shape: {output_fp32.shape}")
        print(f"  Quantized shape: {output_int8.shape}")
        print(f"  Similarity: {similarity:.6f}")
        print(f"  Original sample: {output_fp32[:5]}")
        print(f"  Quantized sample: {output_int8[:5]}")
        
        # Check file sizes
        import os
        fp32_size = os.path.getsize(model_fp32) / (1024 * 1024)  # MB
        int8_size = os.path.getsize(model_int8) / (1024 * 1024)  # MB
        
        print(f"\n📊 Model Comparison:")
        print(f"  FP32 size: {fp32_size:.1f} MB")
        print(f"  Int8 size: {int8_size:.1f} MB")
        print(f"  Size reduction: {fp32_size/int8_size:.1f}x")
        
        if similarity > 0.95:
            print("✅ Quantized model maintains good accuracy!")
            return True
        else:
            print("❌ Quantized model accuracy is too low!")
            return False
            
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False

def benchmark_int8_vs_fp32():
    print("\n⚡ Benchmarking Int8 vs FP32")
    print("=" * 40)
    
    # Load reference tokens
    with open("model/reference_tokens.json", "r") as f:
        ref_tokens = json.load(f)
    
    test_sentence = "hello world"
    token_ids = ref_tokens[test_sentence]["token_ids"]
    if len(token_ids) < 512:
        token_ids = token_ids + [0] * (512 - len(token_ids))
    
    input_tensor = np.array([token_ids], dtype=np.int64)
    
    # Benchmark FP32
    session_fp32 = ort.InferenceSession("model/embedding_model.onnx")
    
    # Warmup
    for _ in range(10):
        session_fp32.run(None, {'input_ids': input_tensor})
    
    import time
    iterations = 100
    
    start = time.time()
    for _ in range(iterations):
        session_fp32.run(None, {'input_ids': input_tensor})
    fp32_time = (time.time() - start) / iterations * 1000  # ms
    
    # Benchmark Int8
    try:
        session_int8 = ort.InferenceSession("model/embedding_model_int8.onnx")
        
        # Warmup
        for _ in range(10):
            session_int8.run(None, {'input_ids': input_tensor})
        
        start = time.time()
        for _ in range(iterations):
            session_int8.run(None, {'input_ids': input_tensor})
        int8_time = (time.time() - start) / iterations * 1000  # ms
        
        print(f"  FP32 average time: {fp32_time:.3f} ms")
        print(f"  Int8 average time: {int8_time:.3f} ms")
        print(f"  Speedup: {fp32_time/int8_time:.2f}x")
        
        if int8_time < fp32_time:
            print("✅ Int8 model is faster!")
        else:
            print("⚠️  Int8 model is not faster (might be due to CPU architecture)")
            
    except Exception as e:
        print(f"❌ Int8 benchmark failed: {e}")

if __name__ == "__main__":
    success = create_int8_model()
    if success:
        benchmark_int8_vs_fp32()
    else:
        print("❌ Skipping benchmark due to quantization failure")
