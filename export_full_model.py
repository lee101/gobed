#!/usr/bin/env python3
"""
Proper ONNX export for the SentenceTransformer model.
This will ensure we export the full transformer pipeline, not just the embedding layer.
"""

import torch
import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def export_sentence_transformer_to_onnx():
    print("🚀 Exporting SentenceTransformer to ONNX")
    print("=" * 50)
    
    # Load the model
    model_path = "model/sentence_transformer"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return False
        
    print(f"📂 Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Get the tokenizer and first module
    tokenizer = model.tokenizer
    print(f"📝 Tokenizer type: {type(tokenizer)}")
    print(f"📝 Model modules: {[type(m).__name__ for m in model]}")
    
    # Create a wrapper that takes token IDs and produces embeddings
    class SentenceTransformerONNX(torch.nn.Module):
        def __init__(self, sentence_transformer):
            super().__init__()
            self.sentence_transformer = sentence_transformer
            
        def forward(self, input_ids, attention_mask=None):
            # Create features dict as expected by SentenceTransformer
            if attention_mask is None:
                attention_mask = (input_ids != 0).long()
                
            features = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Run through SentenceTransformer pipeline
            with torch.no_grad():
                embeddings = self.sentence_transformer(features)
                
            return embeddings['sentence_embedding']
    
    # Create the wrapper
    onnx_model = SentenceTransformerONNX(model)
    onnx_model.eval()
    
    # Test sentence for export
    test_sentence = "hello world"
    print(f"🧪 Testing with sentence: '{test_sentence}'")
    
    # Tokenize
    encoded = tokenizer.encode(test_sentence, add_special_tokens=True)
    max_len = 512
    
    if len(encoded.ids) > max_len:
        token_ids = encoded.ids[:max_len]
    else:
        token_ids = encoded.ids + [0] * (max_len - len(encoded.ids))
    
    # Create attention mask
    attention_mask = [1 if tid != 0 else 0 for tid in token_ids]
    
    # Convert to tensors
    input_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
    
    print(f"📏 Input shape: {input_ids_tensor.shape}")
    print(f"📏 Attention mask shape: {attention_mask_tensor.shape}")
    
    # Test the wrapper
    with torch.no_grad():
        test_output = onnx_model(input_ids_tensor, attention_mask_tensor)
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
    
    if cosine_sim < 0.95:
        print("❌ Wrapper doesn't match original closely enough")
        return False
    
    # Export to ONNX
    output_path = "model/embedding_model.onnx"
    print(f"💾 Exporting to {output_path}")
    
    try:
        torch.onnx.export(
            onnx_model,
            (input_ids_tensor, attention_mask_tensor),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
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
            'input_ids': input_ids_tensor.numpy().astype(np.int64),
            'attention_mask': attention_mask_tensor.numpy().astype(np.int64)
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
    success = export_sentence_transformer_to_onnx()
    if success:
        print("\n🎉 Export completed successfully!")
    else:
        print("\n💥 Export failed!")
