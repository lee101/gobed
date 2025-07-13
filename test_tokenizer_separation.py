#!/usr/bin/env python3
"""
Test proper tokenizer + model separation for ONNX export.
This will help identify if tokenization is the issue.
"""

import json
import torch
import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
import onnx

def test_tokenizer_separation():
    print("🔧 Testing Tokenizer + Model Separation")
    print("=" * 50)
    
    # Load the SentenceTransformer model
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    
    # Get the underlying tokenizer and model
    tokenizer = st_model.tokenizer
    model = st_model[0].auto_model  # Get the actual transformer model
    
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Model type: {type(model)}")
    
    # Test sentences
    test_sentences = [
        "hello world",
        "machine learning is fascinating",
        "artificial intelligence and deep learning"
    ]
    
    print("\\n1️⃣ STEP-BY-STEP ANALYSIS:")
    
    for sentence in test_sentences:
        print(f"\\n📝 Sentence: '{sentence}'")
        
        # Step 1: Tokenize manually
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.encode(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
        print(f"   Token IDs: {token_ids[:10]}..." if len(token_ids) > 10 else f"   Token IDs: {token_ids}")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   First 10 IDs: {input_ids[0][:10].tolist()}")
        
        # Step 2: Get model embeddings (without pooling)
        with torch.no_grad():
            outputs = model(input_ids)
            raw_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
        print(f"   Raw embeddings shape: {raw_embeddings.shape}")
        
        # Step 3: Apply mean pooling manually
        attention_mask = torch.ones_like(input_ids)
        # Expand attention mask to match embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(raw_embeddings.size()).float()
        
        # Apply mask and sum
        sum_embeddings = torch.sum(raw_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        print(f"   Pooled embeddings shape: {mean_pooled.shape}")
        print(f"   First 5 values: {mean_pooled[0][:5].tolist()}")
        
        # Step 4: Compare with SentenceTransformer output
        st_embedding = st_model.encode([sentence])
        print(f"   SentenceTransformer shape: {st_embedding.shape}")
        print(f"   ST first 5 values: {st_embedding[0][:5].tolist()}")
        
        # Calculate difference
        diff = np.max(np.abs(mean_pooled[0].numpy() - st_embedding[0]))
        print(f"   Max difference: {diff:.8f}")
        
        if diff < 1e-4:
            print("   ✅ Manual tokenizer+model matches SentenceTransformer")
        else:
            print("   ⚠️ Manual approach differs from SentenceTransformer")

def export_model_without_tokenizer():
    """Export just the transformer model (post-tokenization) to ONNX."""
    print("\\n\\n🚀 EXPORTING MODEL WITHOUT TOKENIZER")
    print("=" * 50)
    
    # Load models
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    model = st_model[0].auto_model
    tokenizer = st_model.tokenizer
    
    # Create a dummy input for ONNX export
    dummy_sentence = "hello world"
    dummy_input = tokenizer.encode(dummy_sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Export the model (without tokenizer)
    model.eval()
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "model/transformer_only.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
            }
        )
        print("✅ Successfully exported transformer_only.onnx")
        
        # Verify the exported model
        onnx_model = onnx.load("model/transformer_only.onnx")
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False
    
    return True

def test_tokenizer_plus_onnx():
    """Test using external tokenizer + ONNX model."""
    print("\\n\\n🧪 TESTING TOKENIZER + ONNX COMBINATION")
    print("=" * 50)
    
    try:
        # Load tokenizer and ONNX session
        st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
        tokenizer = st_model.tokenizer
        onnx_session = ort.InferenceSession("model/transformer_only.onnx")
        
        test_sentences = [
            "hello world",
            "machine learning is fascinating"
        ]
        
        for sentence in test_sentences:
            print(f"\\n📝 Testing: '{sentence}'")
            
            # Step 1: Tokenize
            input_ids = tokenizer.encode(sentence, return_tensors='np', padding=True, truncation=True, max_length=512)
            print(f"   Tokenized shape: {input_ids.shape}")
            
            # Step 2: Run ONNX inference
            onnx_outputs = onnx_session.run(None, {'input_ids': input_ids.astype(np.int64)})
            raw_embeddings = onnx_outputs[0]  # [batch_size, seq_len, hidden_size]
            print(f"   ONNX raw output shape: {raw_embeddings.shape}")
            
            # Step 3: Apply mean pooling
            attention_mask = np.ones_like(input_ids)
            input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones((1, 1, raw_embeddings.shape[-1]))
            
            sum_embeddings = np.sum(raw_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask
            
            print(f"   Final embedding shape: {mean_pooled.shape}")
            print(f"   First 5 values: {mean_pooled[0][:5].tolist()}")
            
            # Step 4: Compare with SentenceTransformer
            st_embedding = st_model.encode([sentence])
            print(f"   SentenceTransformer: {st_embedding[0][:5].tolist()}")
            
            diff = np.max(np.abs(mean_pooled[0] - st_embedding[0]))
            print(f"   Max difference: {diff:.8f}")
            
            if diff < 1e-4:
                print("   ✅ Tokenizer + ONNX matches SentenceTransformer!")
            else:
                print("   ⚠️ Difference detected")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

def generate_tokenizer_reference():
    """Generate comprehensive tokenizer reference for Go implementation."""
    print("\\n\\n📚 GENERATING TOKENIZER REFERENCE")
    print("=" * 50)
    
    # Load tokenizer
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    tokenizer = st_model.tokenizer
    
    # All sentences we need to test
    all_sentences = [
        "hello world",
        "the weather is nice today", 
        "machine learning algorithms are powerful",
        "machine learning is fascinating",
        "artificial intelligence and deep learning",
        "computer vision and image recognition",
        "data science and analytics", 
        "natural language processing",
        "software engineering best practices",
        "distributed systems architecture",
        "cloud computing and microservices",
        "performance optimization techniques",
        "it's a beautiful sunny day"  # This one was missing tokens
    ]
    
    tokenizer_reference = {}
    
    for sentence in all_sentences:
        # Get tokens and IDs
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence, padding=True, truncation=True, max_length=512)
        
        tokenizer_reference[sentence] = {
            "tokens": tokens,
            "token_ids": token_ids,
            "length": len(token_ids)
        }
        
        print(f"'{sentence}': {len(token_ids)} tokens")
    
    # Save the reference
    with open("model/comprehensive_tokenizer_reference.json", "w") as f:
        json.dump(tokenizer_reference, f, indent=2)
    
    print(f"\\n✅ Saved tokenizer reference for {len(all_sentences)} sentences")
    print("   File: model/comprehensive_tokenizer_reference.json")

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE TOKENIZER + MODEL ANALYSIS")
    print("=" * 60)
    
    # Step 1: Analyze current approach
    test_tokenizer_separation()
    
    # Step 2: Export model without tokenizer
    if export_model_without_tokenizer():
        # Step 3: Test the combination
        test_tokenizer_plus_onnx()
    
    # Step 4: Generate comprehensive tokenizer reference
    generate_tokenizer_reference()
    
    print("\\n🎯 NEXT STEPS:")
    print("   1. If tokenizer+ONNX matches SentenceTransformer perfectly,")
    print("      then we should use this approach in Go")
    print("   2. Implement proper tokenizer in Go using the reference")
    print("   3. Use transformer_only.onnx for inference")
    print("   4. Apply mean pooling in Go code")
