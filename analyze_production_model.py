#!/usr/bin/env python3
"""
Analyze the actual architecture of the sentence-transformers/static-retrieval-mrl-en-v1 model
to understand what we need to export to ONNX for production use.
"""

import torch
from sentence_transformers import SentenceTransformer

def analyze_model():
    print("Loading sentence-transformers/static-retrieval-mrl-en-v1 model...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    # Print model structure
    print(f"\nModel: {model}")
    print(f"Max sequence length: {model.max_seq_length}")
    print(f"Device: {model.device}")
    
    # Analyze modules
    print(f"\nNumber of modules: {len(model._modules)}")
    for i, (name, module) in enumerate(model._modules.items()):
        print(f"\nModule {i}: {name}")
        print(f"  Type: {type(module)}")
        print(f"  Module: {module}")
        
        # For transformer modules, get more details
        if hasattr(module, 'auto_model'):
            print(f"  Auto model: {module.auto_model}")
            print(f"  Config: {module.auto_model.config}")
            
        if hasattr(module, 'tokenizer'):
            print(f"  Tokenizer: {module.tokenizer}")
            
        # Check if it's a pooling module
        if 'pooling' in name.lower() or 'Pooling' in str(type(module)):
            print(f"  Pooling type: {getattr(module, 'pooling_mode', 'unknown')}")
            print(f"  Pooling config: {getattr(module, 'config_keys', 'unknown')}")
    
    # Test the model with a sample input
    print("\n" + "="*60)
    print("TESTING MODEL OUTPUT")
    print("="*60)
    
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today."
    ]
    
    print(f"Test sentences: {test_sentences}")
    
    # Get embeddings
    embeddings = model.encode(test_sentences, convert_to_tensor=True)
    print(f"Output shape: {embeddings.shape}")
    print(f"Output dtype: {embeddings.dtype}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test tokenization
    print("\n" + "="*60)
    print("TOKENIZATION ANALYSIS")
    print("="*60)
    
    # Get the tokenizer from the first module (usually the transformer)
    first_module = list(model._modules.values())[0]
    if hasattr(first_module, 'tokenizer'):
        tokenizer = first_module.tokenizer
        print(f"Tokenizer: {tokenizer}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Model max length: {tokenizer.model_max_length}")
        
        # Test tokenization
        test_text = test_sentences[0]
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"\nTest text: '{test_text}'")
        print(f"Token IDs: {tokens['input_ids']}")
        print(f"Attention mask: {tokens['attention_mask']}")
        print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
    
    # Analyze the forward pass step by step
    print("\n" + "="*60)
    print("FORWARD PASS ANALYSIS")
    print("="*60)
    
    test_text = test_sentences[0]
    print(f"Input: '{test_text}'")
    
    # Step through each module
    features = {'sentence': test_text}
    
    for i, (name, module) in enumerate(model._modules.items()):
        print(f"\nStep {i+1}: {name} ({type(module).__name__})")
        print(f"  Input keys: {list(features.keys())}")
        
        try:
            features = module(features)
            print(f"  Output keys: {list(features.keys())}")
            for key, value in features.items():
                if torch.is_tensor(value):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                    print(f"    {key}: list of {len(value)} tensors, first shape={value[0].shape}")
                else:
                    print(f"    {key}: {type(value)}")
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    return model

if __name__ == "__main__":
    model = analyze_model()
