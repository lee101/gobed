#!/usr/bin/env python3
"""
Simple ONNX export by extracting embedding weights and implementing mean pooling manually.
This avoids the EmbeddingBag's complex offset requirements in ONNX.
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model that manually implements mean pooling without EmbeddingBag."""
    
    def __init__(self, embedding_weights):
        super().__init__()
        # Create a simple Embedding layer from the EmbeddingBag weights
        vocab_size, embed_dim = embedding_weights.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data = embedding_weights.clone()
        
    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings for all tokens
        token_embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Create mask for non-padding tokens (assuming 0 is padding)
        mask = (input_ids != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply mask and sum
        masked_embeddings = token_embeddings * mask  # [batch_size, seq_len, embed_dim]
        summed_embeddings = torch.sum(masked_embeddings, dim=1)  # [batch_size, embed_dim]
        
        # Count non-padding tokens for averaging
        token_counts = torch.sum(mask, dim=1)  # [batch_size, 1]
        token_counts = torch.clamp(token_counts, min=1.0)  # Avoid division by zero
        
        # Compute mean
        mean_embeddings = summed_embeddings / token_counts  # [batch_size, embed_dim]
        
        return mean_embeddings

def export_simple_embedding():
    print("🚀 Exporting Simple Embedding Model to ONNX")
    print("=" * 50)
    
    # Load the model
    model_path = "model/sentence_transformer"
    print(f"📂 Loading model from {model_path}")
    model = SentenceTransformer(model_path, device='cpu')
    
    # Get the StaticEmbedding module and extract weights
    static_embedding = model[0]
    embedding_bag = static_embedding.embedding
    embedding_weights = embedding_bag.weight.data
    
    print(f"📝 Vocab size: {embedding_weights.shape[0]}")
    print(f"📝 Embedding dim: {embedding_weights.shape[1]}")
    
    # Create our simple model
    simple_model = SimpleEmbeddingModel(embedding_weights)
    simple_model.eval()
    
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
    print(f"📏 Token IDs: {token_ids[:10]}...")
    
    # Test our simple model
    with torch.no_grad():
        simple_output = simple_model(input_ids_tensor)
        print(f"📊 Simple model output shape: {simple_output.shape}")
        print(f"📊 Simple model sample: {simple_output[0][:5]}")
    
    # Compare with original SentenceTransformer
    original_output = model.encode([test_sentence])
    print(f"📊 Original output shape: {original_output.shape}")
    print(f"📊 Original output sample: {original_output[0][:5]}")
    
    # Check similarity
    cosine_sim = np.dot(simple_output[0].numpy(), original_output[0]) / (
        np.linalg.norm(simple_output[0].numpy()) * np.linalg.norm(original_output[0])
    )
    print(f"📊 Simple vs Original similarity: {cosine_sim:.6f}")
    
    if cosine_sim < 0.95:
        print(f"❌ Simple model doesn't match original closely enough (similarity: {cosine_sim:.6f})")
        
        # Let's debug the tokenization and see what's happening
        print("\n🔍 Debugging tokenization and embeddings...")
        
        # Check the StaticEmbedding directly
        features = {'input_ids': input_ids_tensor}
        with torch.no_grad():
            static_output = static_embedding(features)
            static_emb = static_output['sentence_embedding']
            print(f"📊 Direct StaticEmbedding output: {static_emb[0][:5]}")
            
            static_sim = np.dot(static_emb[0].numpy(), original_output[0]) / (
                np.linalg.norm(static_emb[0].numpy()) * np.linalg.norm(original_output[0])
            )
            print(f"📊 Direct StaticEmbedding vs Original similarity: {static_sim:.6f}")
        
        # Check individual token embeddings
        non_zero_tokens = [t for t in token_ids if t != 0]
        print(f"📏 Non-zero token IDs: {non_zero_tokens}")
        
        with torch.no_grad():
            # Get embeddings for each token
            for i, token_id in enumerate(non_zero_tokens[:3]):  # Just first 3 tokens
                token_emb = embedding_bag(torch.tensor([token_id]), torch.tensor([0]))
                print(f"📊 Token {i} (ID {token_id}) embedding sample: {token_emb[0][:3]}")
        
        return False
    
    # Export to ONNX
    output_path = "model/embedding_model.onnx"
    print(f"\n💾 Exporting to {output_path}")
    
    try:
        torch.onnx.export(
            simple_model,
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
            
            # Test with multiple sentences
            print("\n🧪 Testing with multiple sentences...")
            test_sentences = [
                "hello world",
                "artificial intelligence",
                "the quick brown fox",
                "machine learning models"
            ]
            
            # Get original embeddings
            original_embeds = model.encode(test_sentences)
            
            # Get ONNX embeddings
            onnx_embeds = []
            for sentence in test_sentences:
                encoded = tokenizer.encode(sentence, add_special_tokens=True)
                if len(encoded.ids) > max_len:
                    token_ids = encoded.ids[:max_len]
                else:
                    token_ids = encoded.ids + [0] * (max_len - len(encoded.ids))
                
                input_tensor = np.array([token_ids], dtype=np.int64)
                onnx_out = session.run(None, {'input_ids': input_tensor})
                onnx_embeds.append(onnx_out[0][0])
            
            onnx_embeds = np.array(onnx_embeds)
            
            # Compare similarities
            print("📊 Similarity comparison:")
            for i, sentence in enumerate(test_sentences):
                sim = np.dot(onnx_embeds[i], original_embeds[i]) / (
                    np.linalg.norm(onnx_embeds[i]) * np.linalg.norm(original_embeds[i])
                )
                print(f"  '{sentence}': {sim:.6f}")
            
            return True
        else:
            print("❌ ONNX model doesn't match original closely enough")
            return False
            
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False

if __name__ == "__main__":
    success = export_simple_embedding()
    if success:
        print("\n🎉 Export completed successfully!")
    else:
        print("\n💥 Export failed!")
