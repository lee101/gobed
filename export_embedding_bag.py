#!/usr/bin/env python3
"""
Proper ONNX export for StaticEmbedding model with EmbeddingBag.
"""

import torch
import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer

def export_static_embedding_correct():
    print("🚀 Exporting StaticEmbedding (EmbeddingBag) to ONNX")
    print("=" * 50)
    
    # Load the model
    model_path = "model/sentence_transformer"
    print(f"📂 Loading model from {model_path}")
    model = SentenceTransformer(model_path, device='cpu')
    
    # Get the StaticEmbedding module
    static_embedding = model[0]
    embedding_bag = static_embedding.embedding
    
    print(f"📝 EmbeddingBag mode: {embedding_bag.mode}")
    print(f"📝 Vocab size: {embedding_bag.num_embeddings}")
    print(f"📝 Embedding dim: {embedding_bag.embedding_dim}")
    
    # Create a wrapper that properly handles the EmbeddingBag input format
    class StaticEmbeddingBagONNX(torch.nn.Module):
        def __init__(self, embedding_bag):
            super().__init__()
            self.embedding_bag = embedding_bag
            
        def forward(self, input_ids):
            # For 2D input to EmbeddingBag, we need to handle it properly
            batch_size, seq_len = input_ids.shape
            
            # Flatten the input and create proper offsets
            flattened_ids = input_ids.flatten()  # [batch*seq]
            
            # Create offsets for each sequence in the batch
            # offsets[i] = i * seq_len
            offsets = torch.arange(0, batch_size * seq_len, seq_len, dtype=torch.long)
            
            # Remove padding tokens from each sequence
            # We'll need to handle this differently
            embeddings_list = []
            
            for i in range(batch_size):
                seq_ids = input_ids[i]  # [seq_len]
                # Remove padding (assuming 0 is padding)
                non_pad_mask = seq_ids != 0
                non_pad_ids = seq_ids[non_pad_mask]
                
                if len(non_pad_ids) == 0:
                    # If all tokens are padding, use a zero embedding
                    embedding = torch.zeros(self.embedding_bag.embedding_dim)
                else:
                    # Use EmbeddingBag with single offset
                    embedding = self.embedding_bag(non_pad_ids, torch.tensor([0], dtype=torch.long))
                    embedding = embedding.squeeze(0)  # Remove batch dim
                
                embeddings_list.append(embedding)
            
            # Stack all embeddings
            result = torch.stack(embeddings_list, dim=0)
            return result
    
    # Create the wrapper
    onnx_model = StaticEmbeddingBagONNX(embedding_bag)
    onnx_model.eval()
    
    # Test with a sentence
    test_sentence = "hello world"
    print(f"🧪 Testing with sentence: '{test_sentence}'")
    
    # Tokenize
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
    print(f"📏 Non-zero tokens: {len([t for t in token_ids if t != 0])}")
    
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
    
    if cosine_sim < 0.95:
        print("❌ Wrapper doesn't match original closely enough")
        print("Let's try calling the StaticEmbedding directly...")
        
        # Try calling StaticEmbedding with proper features
        features = {'input_ids': input_ids_tensor}
        try:
            with torch.no_grad():
                direct_output = static_embedding(features)
                print(f"📊 Direct StaticEmbedding output: {direct_output['sentence_embedding'].shape}")
                print(f"📊 Direct sample: {direct_output['sentence_embedding'][0][:5]}")
                
                direct_sim = np.dot(direct_output['sentence_embedding'][0].numpy(), original_output[0]) / (
                    np.linalg.norm(direct_output['sentence_embedding'][0].numpy()) * np.linalg.norm(original_output[0])
                )
                print(f"📊 Direct vs Original similarity: {direct_sim:.6f}")
                
                if direct_sim > 0.95:
                    print("✅ Direct StaticEmbedding works! Let me fix the wrapper...")
                    # Replace the wrapper with the direct StaticEmbedding approach
                    onnx_model = static_embedding
                    test_output = direct_output['sentence_embedding']
                    cosine_sim = direct_sim
                
        except Exception as e:
            print(f"❌ Direct StaticEmbedding failed: {e}")
            return False
    
    if cosine_sim < 0.95:
        print(f"❌ Still not matching (similarity: {cosine_sim:.6f})")
        return False
    
    # Export to ONNX - but we need to handle the StaticEmbedding differently
    output_path = "model/embedding_model.onnx"
    print(f"💾 Exporting to {output_path}")
    
    # For StaticEmbedding, we need a different approach
    # Let's create a minimal wrapper that just does the EmbeddingBag lookup correctly
    class MinimalEmbeddingONNX(torch.nn.Module):
        def __init__(self, static_embedding_module):
            super().__init__()
            self.static_embedding = static_embedding_module
            
        def forward(self, input_ids):
            features = {'input_ids': input_ids}
            result = self.static_embedding(features)
            return result['sentence_embedding']
    
    minimal_model = MinimalEmbeddingONNX(static_embedding)
    minimal_model.eval()
    
    try:
        torch.onnx.export(
            minimal_model,
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
    success = export_static_embedding_correct()
    if success:
        print("\n🎉 Export completed successfully!")
    else:
        print("\n💥 Export failed!")
