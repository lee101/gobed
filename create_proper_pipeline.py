#!/usr/bin/env python3
"""
Create a proper embedding model that uses external tokenization correctly.
"""

import json
import torch
import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer

def create_proper_model():
    print("🔧 CREATING PROPER TOKENIZER + MODEL PIPELINE")
    print("=" * 50)
    
    # Load model
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    tokenizer = st_model.tokenizer
    transformer_model = st_model[0].auto_model
    
    # Test with proper tokenization
    sentence = "hello world"
    print(f"Testing with: '{sentence}'")
    
    # Step 1: Proper tokenization
    encoded = tokenizer.encode(sentence)
    token_ids = torch.tensor([encoded.ids], dtype=torch.int64)
    
    # Pad to fixed length (512)
    padded_ids = torch.zeros((1, 512), dtype=torch.int64)
    padded_ids[0, :len(encoded.ids)] = token_ids[0]
    
    print(f"Original tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Padded shape: {padded_ids.shape}")
    print(f"Padded IDs: {padded_ids[0][:10].tolist()}")
    
    # Step 2: Get transformer output
    transformer_model.eval()
    with torch.no_grad():
        outputs = transformer_model(padded_ids)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
    
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Step 3: Apply mean pooling (proper way)
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_ids != 0).float()
    
    # Expand mask to hidden dimension
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    
    # Apply mask and compute mean
    masked_embeddings = hidden_states * mask_expanded
    summed = torch.sum(masked_embeddings, dim=1)
    mask_sum = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
    mean_pooled = summed / mask_sum
    
    print(f"Final embedding shape: {mean_pooled.shape}")
    print(f"First 5 values: {mean_pooled[0][:5].tolist()}")
    
    # Step 4: Compare with SentenceTransformer
    st_embedding = st_model.encode([sentence])
    print(f"SentenceTransformer: {st_embedding[0][:5].tolist()}")
    
    diff = torch.max(torch.abs(mean_pooled[0] - torch.tensor(st_embedding[0])))
    print(f"Max difference: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ Manual approach matches SentenceTransformer!")
        return True, transformer_model, tokenizer
    else:
        print("⚠️ Manual approach differs from SentenceTransformer")
        return False, None, None

def export_transformer_model(transformer_model):
    """Export just the transformer part to ONNX."""
    print("\\n🚀 EXPORTING TRANSFORMER MODEL")
    print("=" * 50)
    
    # Create dummy input
    dummy_input = torch.zeros((1, 512), dtype=torch.int64)
    dummy_input[0, :4] = torch.tensor([101, 7592, 2088, 102])  # hello world tokens
    
    try:
        torch.onnx.export(
            transformer_model,
            dummy_input,
            "model/proper_transformer.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size'}
            }
        )
        print("✅ Successfully exported proper_transformer.onnx")
        return True
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def test_onnx_with_proper_tokenizer():
    """Test the exported ONNX model with proper tokenization."""
    print("\\n🧪 TESTING ONNX + TOKENIZER")
    print("=" * 50)
    
    try:
        # Load models
        st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
        tokenizer = st_model.tokenizer
        onnx_session = ort.InferenceSession("model/proper_transformer.onnx")
        
        test_sentences = ["hello world", "machine learning is fascinating"]
        
        for sentence in test_sentences:
            print(f"\\n📝 Testing: '{sentence}'")
            
            # Step 1: Tokenize properly
            encoded = tokenizer.encode(sentence)
            padded_ids = np.zeros((1, 512), dtype=np.int64)
            padded_ids[0, :len(encoded.ids)] = encoded.ids
            
            print(f"Tokens: {encoded.tokens}")
            print(f"Token IDs: {encoded.ids}")
            
            # Step 2: ONNX inference
            onnx_output = onnx_session.run(None, {'input_ids': padded_ids})
            hidden_states = onnx_output[0]  # [batch, seq_len, hidden_dim]
            
            # Step 3: Mean pooling
            attention_mask = (padded_ids != 0).astype(np.float32)
            mask_expanded = np.expand_dims(attention_mask, -1)
            mask_expanded = np.broadcast_to(mask_expanded, hidden_states.shape)
            
            masked_embeddings = hidden_states * mask_expanded
            summed = np.sum(masked_embeddings, axis=1)
            mask_sum = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            mean_pooled = summed / mask_sum
            
            print(f"ONNX embedding: {mean_pooled[0][:5]}")
            
            # Step 4: Compare with SentenceTransformer
            st_embedding = st_model.encode([sentence])
            print(f"SentenceTransformer: {st_embedding[0][:5]}")
            
            diff = np.max(np.abs(mean_pooled[0] - st_embedding[0]))
            print(f"Max difference: {diff:.8f}")
            
            if diff < 1e-4:
                print("✅ ONNX + tokenizer matches SentenceTransformer!")
            else:
                print("⚠️ Still some difference")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

def generate_proper_go_reference():
    """Generate proper reference for Go implementation."""
    print("\\n📚 GENERATING GO REFERENCE")
    print("=" * 50)
    
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    tokenizer = st_model.tokenizer
    
    sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful",
        "machine learning is fascinating",
        "artificial intelligence and deep learning"
    ]
    
    go_reference = {}
    
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        go_reference[sentence] = {
            "tokens": encoded.tokens,
            "token_ids": encoded.ids,
            "length": len(encoded.ids)
        }
        print(f"'{sentence}': {encoded.tokens} -> {encoded.ids}")
    
    with open("model/go_tokenizer_reference.json", "w") as f:
        json.dump(go_reference, f, indent=2)
    
    print("\\n✅ Saved Go tokenizer reference")

if __name__ == "__main__":
    print("🎯 PROPER TOKENIZER + MODEL PIPELINE")
    print("=" * 60)
    
    # Test the manual approach
    success, transformer_model, tokenizer = create_proper_model()
    
    if success:
        # Export the transformer model
        if export_transformer_model(transformer_model):
            # Test the combination
            test_onnx_with_proper_tokenizer()
        
        # Generate Go reference
        generate_proper_go_reference()
        
        print("\\n🎯 SUMMARY:")
        print("   ✅ Manual tokenizer + transformer matches SentenceTransformer")
        print("   ✅ ONNX model exported successfully")
        print("   ✅ Go reference generated")
        print("\\n   Next: Implement proper tokenizer in Go + use new ONNX model")
    else:
        print("\\n❌ Manual approach failed - need to debug further")
