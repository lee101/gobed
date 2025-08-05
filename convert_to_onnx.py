#!/usr/bin/env python3
"""
Convert SentenceTransformer model to ONNX format with INT8 quantization for optimal CPU performance
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import shutil
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def convert_model_to_onnx():
    print("=" * 60)
    print("CONVERTING SENTENCETRANSFORMER TO ONNX")
    print("=" * 60)
    
    # Load the model
    print("📥 Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    
    # Create output directory
    output_dir = Path("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test the model with our example texts first
    test_texts = ["hello world", "the weather is nice today", "machine learning algorithms are powerful"]
    print(f"🧪 Testing with {len(test_texts)} texts...")
    embeddings = model.encode(test_texts)
    
    # Save test embeddings for verification
    np.save(str(output_dir / "test_embeddings.npy"), embeddings)
    print(f"✅ Test embeddings saved to: {output_dir / 'test_embeddings.npy'}")
    print(f"📊 Embedding dimension: {embeddings.shape[1]}")
    
    # Show proper similarities (should be low, not 0.999!)
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    similarities = cosine_similarity(embeddings)
    distances = euclidean_distances(embeddings)
    
    print(f"\n📈 Expected Results (Python SentenceTransformer):")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            cos_sim = similarities[i][j]
            eucl_dist = distances[i][j]
            print(f"  '{test_texts[i]}' vs '{test_texts[j]}':")
            print(f"    Cosine similarity: {cos_sim:.6f}")
            print(f"    Euclidean distance: {eucl_dist:.6f}")
    
    # Save the model in SentenceTransformer format for reference
    model.save(str(output_dir / "sentence_transformer"))
    print(f"✅ Model saved to: {output_dir / 'sentence_transformer'}")
    
    # Copy tokenizer.json file for Go tokenization
    tokenizer_src = output_dir / "sentence_transformer" / "tokenizer.json"
    tokenizer_dst = output_dir / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"Tokenizer copied to: {tokenizer_dst}")
    
    # Export to ONNX properly - get the complete pipeline
    print(f"\n🔧 Exporting to ONNX...")
    
    try:
        # Method 1: Use SentenceTransformer's built-in ONNX export if available
        if hasattr(model, 'save') and hasattr(model, '_modules'):
            print("📦 Attempting direct ONNX export...")
            
            # Get the tokenizer for proper input processing
            tokenizer = model.tokenizer
            
            # Create a sample input - fix the tokenizer call
            sample_text = "hello world"
            if hasattr(tokenizer, 'encode_plus'):
                inputs = tokenizer.encode_plus(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            else:
                # Use the tokenizer's __call__ method properly
                inputs = tokenizer([sample_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get the transformer model (first module is usually the transformer)
            transformer_model = model[0].auto_model
            
            # Set to evaluation mode
            transformer_model.eval()
            
            onnx_path = output_dir / "embedding_model.onnx"
            
            # Export the transformer with proper input names
            torch.onnx.export(
                transformer_model,
                (inputs["input_ids"], inputs["attention_mask"]),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=14,
                export_params=True,
                do_constant_folding=True
            )
            
            print(f"✅ ONNX model exported to: {onnx_path}")
            
            # Test the ONNX model
            print(f"🧪 Testing ONNX model...")
            session = ort.InferenceSession(str(onnx_path))
            
            # Print model info
            print(f"  ONNX inputs: {[inp.name for inp in session.get_inputs()]}")
            print(f"  ONNX outputs: {[out.name for out in session.get_outputs()]}")
            
            # Test with the same input
            onnx_result = session.run(None, {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            })
            
            # The output will be [batch_size, seq_len, hidden_size]
            # We need mean pooling to get sentence embeddings
            last_hidden_state = onnx_result[0]  # Shape: [1, seq_len, 768 or 1024]
            attention_mask = inputs["attention_mask"].numpy()
            
            # Mean pooling with attention mask
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.sum(input_mask_expanded, axis=1)
            mean_pooled = sum_embeddings / sum_mask
            
            print(f"  ONNX output shape: {last_hidden_state.shape}")
            print(f"  Mean pooled shape: {mean_pooled.shape}")
            print(f"  Mean pooled first 5: {mean_pooled[0][:5]}")
            
            # Compare with SentenceTransformer output
            sentence_embedding = model.encode([sample_text])[0]
            print(f"  SentenceTransformer first 5: {sentence_embedding[:5]}")
            
            # Check similarity
            onnx_flat = mean_pooled[0]
            cos_sim = np.dot(onnx_flat, sentence_embedding) / (np.linalg.norm(onnx_flat) * np.linalg.norm(sentence_embedding))
            print(f"  Cross-similarity: {cos_sim:.6f}")
            
            if cos_sim > 0.95:
                print(f"✅ ONNX export successful! High similarity: {cos_sim:.6f}")
                return True
            else:
                print(f"⚠️  ONNX export may have issues. Low similarity: {cos_sim:.6f}")
                return False
                
        else:
            print("❌ Direct ONNX export not supported")
            return False
            
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        return False

def quantize_model_to_int8():
    """
    Quantize the ONNX model to INT8 for faster CPU inference
    """
    print("\n" + "="*60)
    print("QUANTIZING MODEL TO INT8 FOR OPTIMAL CPU PERFORMANCE")
    print("="*60)
    
    model_dir = Path("model")
    onnx_model_path = model_dir / "embedding_model.onnx"
    quantized_model_path = model_dir / "embedding_model_int8.onnx"
    
    if not onnx_model_path.exists():
        print(f"Error: {onnx_model_path} not found!")
        print("Please run the model conversion first.")
        return False
    
    try:
        print(f"Loading model from: {onnx_model_path}")
        
        # Quantize the model using dynamic quantization (INT8)
        quantize_dynamic(
            str(onnx_model_path),
            str(quantized_model_path),
            weight_type=QuantType.QInt8,  # Quantize weights to INT8
        )
        
        print(f"✅ INT8 quantized model saved to: {quantized_model_path}")
        
        # Compare model sizes
        original_size = onnx_model_path.stat().st_size / (1024 * 1024)
        quantized_size = quantized_model_path.stat().st_size / (1024 * 1024)
        compression_ratio = original_size / quantized_size
        
        print(f"\n📊 Model Size Comparison:")
        print(f"   Original:  {original_size:.2f} MB")
        print(f"   Quantized: {quantized_size:.2f} MB")
        print(f"   Compression: {compression_ratio:.2f}x smaller")
        
        # Test the quantized model
        print(f"\n🧪 Testing quantized model...")
        session = ort.InferenceSession(str(quantized_model_path))
        
        # Create a sample input (batch_size=1, sequence_length=512)
        sample_input = np.random.randint(0, 30522, (1, 512), dtype=np.int64)
        sample_input[0, 0] = 101  # CLS token
        sample_input[0, -1] = 102  # SEP token
        
        # Run inference
        result = session.run(None, {"input_ids": sample_input})
        
        print(f"✅ Quantized model test successful!")
        print(f"   Input shape:  {sample_input.shape}")
        print(f"   Output shape: {result[0].shape}")
        print(f"   Output dtype: {result[0].dtype}")
        
        # Benchmark speed difference
        print(f"\n⚡ Performance Benchmark:")
        
        import time
        
        # Original model
        original_session = ort.InferenceSession(str(onnx_model_path))
        
        # Warmup
        for _ in range(5):
            original_session.run(None, {"input_ids": sample_input})
            session.run(None, {"input_ids": sample_input})
        
        # Benchmark original
        start_time = time.time()
        for _ in range(100):
            original_session.run(None, {"input_ids": sample_input})
        original_time = time.time() - start_time
        
        # Benchmark quantized
        start_time = time.time()
        for _ in range(100):
            session.run(None, {"input_ids": sample_input})
        quantized_time = time.time() - start_time
        
        speedup = original_time / quantized_time
        
        print(f"   Original model:  {original_time*10:.2f}ms per inference")
        print(f"   Quantized model: {quantized_time*10:.2f}ms per inference")
        print(f"   Speedup: {speedup:.2f}x faster")
        
        # Replace the original model with the quantized version for production use
        backup_path = model_dir / "embedding_model_fp32_backup.onnx"
        shutil.copy(onnx_model_path, backup_path)
        shutil.copy(quantized_model_path, onnx_model_path)
        
        print(f"\n🔄 Model Replacement:")
        print(f"   Original model backed up to: {backup_path}")
        print(f"   Quantized model is now the default: {onnx_model_path}")
        print(f"\n✅ INT8 quantization complete! Your Go app will now use the faster model.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quantize":
        # Only run quantization
        quantize_model_to_int8()
    elif len(sys.argv) > 1 and sys.argv[1] == "both":
        # Run conversion then quantization
        convert_model_to_onnx()
        quantize_model_to_int8()
    else:
        # Default: just conversion
        print("Use 'python convert_to_onnx.py quantize' to quantize existing model")
        print("Use 'python convert_to_onnx.py both' to convert and quantize")
        convert_model_to_onnx()
