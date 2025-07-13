#!/usr/bin/env python3
"""
Export the StaticEmbedding model correctly to ONNX.
"""

import torch
import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def export_static_embedding_to_onnx():
    print("🚀 Exporting StaticEmbedding model to ONNX")
    print("=" * 50)
    
    # Load the model
    model_path = "model/sentence_transformer"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return False
        
    print(f"📂 Loading model from {model_path}")
    model = SentenceTransformer(model_path, device='cpu')  # Force CPU
    
    # Get the StaticEmbedding module
    static_embedding = model[0]  # First module should be StaticEmbedding
    print(f"📝 Model type: {type(static_embedding)}")
    print(f"📝 Embedding dim: {static_embedding.get_sentence_embedding_dimension()}")
    print(f"📝 Vocab size: {static_embedding.embedding.num_embeddings}")
    
    # Create a wrapper that handles the StaticEmbedding correctly
    class StaticEmbeddingONNX(torch.nn.Module):
        def __init__(self, static_embedding_model):
            super().__init__()
            self.embedding = static_embedding_model.embedding.cpu()  # Ensure CPU
            
        def forward(self, input_ids):
            # For StaticEmbedding, we need to:
            # 1. Look up embeddings for each token
            # 2. Average them (mean pooling)
            
            batch_size, seq_len = input_ids.shape
            
            # Get embeddings for all tokens
            token_embeddings = self.embedding(input_ids)  # [batch, seq, dim]
            
            # Create mask for non-padding tokens (assuming 0 is padding)
            mask = (input_ids != 0).float().unsqueeze(-1)  # [batch, seq, 1]
            
            # Apply mask and sum
            masked_embeddings = token_embeddings * mask  # [batch, seq, dim]
            sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, dim]
            
            # Count non-padding tokens
            sum_mask = mask.sum(dim=1)  # [batch, 1]
            
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # Mean pooling
            mean_embeddings = sum_embeddings / sum_mask  # [batch, dim]
            
            return mean_embeddings
    
    # Create the wrapper
    onnx_model = StaticEmbeddingONNX(static_embedding)
    onnx_model.eval()
    
    # Test with a simple sentence
    test_sentence = "hello world"
    print(f"🧪 Testing with sentence: '{test_sentence}'")
    
    # Tokenize using the model's tokenizer
    tokenizer = model.tokenizer
    encoded = tokenizer.encode(test_sentence, add_special_tokens=True)
    
    # Prepare input
    max_len = 512
    if len(encoded.ids) > max_len:
        token_ids = encoded.ids[:max_len]
    else:
        token_ids = encoded.ids + [0] * (max_len - len(encoded.ids))
    
    input_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    print(f"📏 Input shape: {input_ids_tensor.shape}")
    print(f"📏 Input tokens: {token_ids[:10]}")
    
    # Test the wrapper
    with torch.no_grad():
        test_output = onnx_model(input_ids_tensor)
        print(f"📊 Test output shape: {test_output.shape}")
        print(f"📊 Test output sample: {test_output[0][:5]}")
    
    # Compare with original SentenceTransformer
    original_output = model.encode([test_sentence])
    print(f"📊 Original output shape: {original_output.shape}")
    print(f"📊 Original output sample: {original_output[0][:5]}")
    
    # Check similarity
    cosine_sim = np.dot(test_output[0].numpy(), original_output[0]) / (
        np.linalg.norm(test_output[0].numpy()) * np.linalg.norm(original_output[0])
    )
    print(f"📊 Wrapper vs Original similarity: {cosine_sim:.6f}")
    
    # Export to ONNX
    output_path = "model/embedding_model.onnx"
    print(f"💾 Exporting to {output_path}")
    
    try:
        torch.onnx.export(
            onnx_model,
            input_ids_tensor,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'embeddings': {0: 'batch_size'}
            }
        )
        print("✅ ONNX export successful!")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False
    
    # Verify the exported model
    print("🔍 Verifying exported ONNX model...")
    
    try:
        session = ort.InferenceSession(output_path)
        
        # Test with same inputs
        onnx_inputs = {
            'input_ids': input_ids_tensor.numpy().astype(np.int64)
        }
        
        onnx_outputs = session.run(None, onnx_inputs)
        onnx_embedding = onnx_outputs[0][0]
        
        print(f"📊 ONNX output shape: {onnx_embedding.shape}")
        print(f"📊 ONNX output sample: {onnx_embedding[:5]}")
        
        # Compare with original
        final_similarity = np.dot(onnx_embedding, original_output[0]) / (
            np.linalg.norm(onnx_embedding) * np.linalg.norm(original_output[0])
        )
        print(f"📊 Final ONNX vs Original similarity: {final_similarity:.6f}")
        
        if final_similarity > 0.95:
            print("✅ ONNX model exported successfully and matches original!")
            return True
        else:
            print("❌ ONNX model doesn't match original closely enough")
            return False
            
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False

if __name__ == "__main__":
    success = export_static_embedding_to_onnx()
    if success:
        print("\n🎉 Export completed successfully!")
    else:
        print("\n💥 Export failed!")
