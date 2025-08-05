#!/usr/bin/env python3
"""
Simple tokenizer analysis to identify the tokenization issue.
"""

import json
import torch
from sentence_transformers import SentenceTransformer

def analyze_tokenization():
    print("🔍 SIMPLE TOKENIZER ANALYSIS")
    print("=" * 40)
    
    # Load model
    st_model = SentenceTransformer("model/sentence_transformer", device='cpu')
    tokenizer = st_model.tokenizer
    
    # Test sentence
    sentence = "hello world"
    print(f"Analyzing: '{sentence}'")
    
    # Method 1: SentenceTransformer tokenization
    print("\\n1️⃣ SentenceTransformer internal tokenization:")
    st_embedding = st_model.encode([sentence])
    print(f"   Result shape: {st_embedding.shape}")
    print(f"   First 5 values: {st_embedding[0][:5]}")
    
    # Method 2: Manual tokenization
    print("\\n2️⃣ Manual tokenization:")
    # Use the encode method directly
    encoded = tokenizer.encode(sentence)
    token_ids = torch.tensor([encoded.ids], dtype=torch.int64)
    tokens = encoded.tokens
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs shape: {token_ids.shape}")
    print(f"   Token IDs: {token_ids[0][:10].tolist()}")
    
    # Method 3: Check what we have in reference tokens
    try:
        with open("model/reference_tokens.json", "r") as f:
            ref_tokens = json.load(f)
        
        if sentence in ref_tokens:
            print("\\n3️⃣ Current reference tokens:")
            print(f"   Token IDs: {ref_tokens[sentence]['token_ids'][:10]}")
            print(f"   Length: {ref_tokens[sentence]['length']}")
            
            # Compare
            manual_ids = token_ids[0].tolist()
            ref_ids = ref_tokens[sentence]['token_ids']
            
            if manual_ids == ref_ids:
                print("   ✅ Reference tokens match manual tokenization")
            else:
                print("   ⚠️ Reference tokens differ from manual tokenization")
                print(f"   Manual: {manual_ids[:10]}")
                print(f"   Reference: {ref_ids[:10]}")
        else:
            print(f"\\n3️⃣ No reference tokens found for '{sentence}'")
            
    except FileNotFoundError:
        print("\\n3️⃣ No reference_tokens.json file found")
    
    print("\\n4️⃣ Tokenizer details:")
    print(f"   Tokenizer type: {type(tokenizer)}")
    print(f"   Vocab size: {tokenizer.get_vocab_size()}")
    
    # Check specific tokens
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
    print("\\n5️⃣ Special tokens:")
    vocab = tokenizer.get_vocab()
    for token in special_tokens:
        if token in vocab:
            token_id = vocab[token]
            print(f"   {token}: {token_id}")
        else:
            print(f"   {token}: not found")

if __name__ == "__main__":
    analyze_tokenization()
