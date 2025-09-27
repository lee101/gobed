#!/usr/bin/env python3
"""
Download the real static-retrieval-mrl-en-v1 model to get the exact safetensors weights.
This will give us the same weights that Python uses.
"""

import os
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import shutil

def download_real_model():
    """Download the actual static-retrieval-mrl-en-v1 model."""
    model_name = "sentence-transformers/static-retrieval-mrl-en-v1"
    cache_dir = "./real_model_cache"
    
    print(f"🔄 Downloading real model: {model_name}")
    print(f"📂 Cache directory: {cache_dir}")
    
    try:
        # Download the model using HuggingFace hub
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        print(f" Model downloaded to: {model_path}")
        
        # List contents to see what we have
        print("\n Model contents:")
        for root, dirs, files in os.walk(model_path):
            level = root.replace(model_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Look for safetensors files specifically
        safetensors_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.safetensors'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, model_path)
                    safetensors_files.append((rel_path, full_path))
                    
        if safetensors_files:
            print(f"\n Found {len(safetensors_files)} safetensors files:")
            for rel_path, full_path in safetensors_files:
                size = os.path.getsize(full_path) / (1024*1024)  # MB
                print(f"   {rel_path} ({size:.2f} MB)")
                
            # Copy the main safetensors file to our working directory
            main_safetensors = None
            for rel_path, full_path in safetensors_files:
                if 'model.safetensors' in rel_path:
                    main_safetensors = full_path
                    break
            
            if not main_safetensors and safetensors_files:
                # Take the first one if no 'model.safetensors' found
                main_safetensors = safetensors_files[0][1]
                
            if main_safetensors:
                dest_path = "./model/real_model.safetensors"
                os.makedirs("./model", exist_ok=True)
                shutil.copy2(main_safetensors, dest_path)
                print(f" Copied main safetensors to: {dest_path}")
                
                return dest_path, model_path
        else:
            print(" No safetensors files found!")
            return None, model_path
            
    except Exception as e:
        print(f" Error downloading model: {e}")
        return None, None

def extract_tokenizer_info(model_path):
    """Extract tokenizer information from the downloaded model."""
    print(f"\n Extracting tokenizer information...")
    
    try:
        # Load with sentence-transformers to get the tokenizer
        model = SentenceTransformer(model_path)
        
        # Get tokenizer
        tokenizer = model.tokenizer
        
        # Test tokenization
        test_sentences = [
            "This is a test sentence.",
            "Machine learning is fascinating.",
            "Hello world"
        ]
        
        print(f" Tokenizer info:")
        print(f"   Vocab size: {len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 'Unknown'}")
        print(f"   Model max length: {getattr(tokenizer, 'model_max_length', 512)}")
        
        # Generate reference tokens for the same sentences
        reference_tokens = {}
        for sentence in test_sentences:
            # Use sentence-transformers tokenizer properly
            inputs = model.tokenize([sentence])
            token_ids = inputs['input_ids'][0].tolist()
            
            reference_tokens[sentence] = {
                "token_ids": token_ids,
                "length": len(token_ids)
            }
            
            print(f"   '{sentence}': {len(token_ids)} tokens")
        
        # Save reference tokens
        import json
        tokens_path = "./model/real_reference_tokens.json"
        with open(tokens_path, 'w') as f:
            json.dump(reference_tokens, f, indent=2)
        
        print(f" Saved reference tokens to: {tokens_path}")
        
        # Test encoding to get expected embeddings
        embeddings = model.encode(test_sentences)
        print(f" Generated embeddings shape: {embeddings.shape}")
        print(f"   Sample embedding for '{test_sentences[0]}': [{embeddings[0][0]:.3f}, {embeddings[0][1]:.3f}, {embeddings[0][2]:.3f}, {embeddings[0][3]:.3f}, {embeddings[0][4]:.3f}]")
        
        # Save expected embeddings
        import numpy as np
        np.save("./model/expected_embeddings.npy", embeddings)
        
        with open("./model/expected_sentences.txt", 'w') as f:
            for sentence in test_sentences:
                f.write(f"{sentence}\n")
                
        print(f" Saved expected embeddings for Go comparison")
        
        return reference_tokens, embeddings
        
    except Exception as e:
        print(f" Error extracting tokenizer info: {e}")
        return None, None

def main():
    print("=" * 80)
    print(" DOWNLOADING REAL MODEL WEIGHTS")
    print("=" * 80)
    print("Model: sentence-transformers/static-retrieval-mrl-en-v1")
    print("Purpose: Get exact same weights that Python uses")
    print()
    
    # Download the model
    safetensors_path, model_path = download_real_model()
    
    if safetensors_path and model_path:
        print(f"\n Successfully downloaded real model!")
        print(f"   Safetensors: {safetensors_path}")
        print(f"   Model path: {model_path}")
        
        # Extract tokenizer and reference data
        reference_tokens, embeddings = extract_tokenizer_info(model_path)
        
        if reference_tokens and embeddings is not None:
            print(f"\n Setup complete! Ready for Go implementation.")
            print(f" Next steps:")
            print(f"   1. Load {safetensors_path} in Go")
            print(f"   2. Use reference tokens from ./model/real_reference_tokens.json") 
            print(f"   3. Compare with expected embeddings")
        else:
            print(f"\n  Model downloaded but tokenizer extraction failed")
    else:
        print(f"\n Failed to download model")
    
    print("=" * 80)

if __name__ == "__main__":
    main()