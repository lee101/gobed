#!/usr/bin/env python3
"""
Download and inspect the static-retrieval-mrl-en-v1 model configuration
without loading the full model weights.
"""

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
import os
import json

def inspect_model_config():
    model_name = "sentence-transformers/static-retrieval-mrl-en-v1"
    cache_dir = "./model_cache"
    
    print(f"Downloading model configuration for {model_name}...")
    
    try:
        # Download just the config files
        repo_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.txt", "README.md", "modules.json"],
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth"]
        )
        
        print(f"Model files downloaded to: {repo_path}")
        
        # List all files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                print(f"  {rel_path}")
        
        # Read modules.json if it exists
        modules_file = os.path.join(repo_path, "modules.json")
        if os.path.exists(modules_file):
            print("\n" + "="*60)
            print("MODULES.JSON CONTENT")
            print("="*60)
            with open(modules_file, 'r') as f:
                modules = json.load(f)
            print(json.dumps(modules, indent=2))
        
        # Try to load config from each subdirectory
        for item in os.listdir(repo_path):
            item_path = os.path.join(repo_path, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, "config.json")
                if os.path.exists(config_file):
                    print("\n" + "="*60)
                    print(f"CONFIG FOR {item}")
                    print("="*60)
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    print(json.dumps(config, indent=2))
                    
                    # Try to create AutoConfig if it's a transformer config
                    try:
                        auto_config = AutoConfig.from_pretrained(item_path)
                        print(f"\nAutoConfig loaded successfully for {item}")
                        print(f"Model type: {auto_config.model_type}")
                        print(f"Hidden size: {getattr(auto_config, 'hidden_size', 'unknown')}")
                        print(f"Num attention heads: {getattr(auto_config, 'num_attention_heads', 'unknown')}")
                        print(f"Num hidden layers: {getattr(auto_config, 'num_hidden_layers', 'unknown')}")
                        print(f"Max position embeddings: {getattr(auto_config, 'max_position_embeddings', 'unknown')}")
                        print(f"Vocab size: {getattr(auto_config, 'vocab_size', 'unknown')}")
                    except Exception as e:
                        print(f"Could not load AutoConfig for {item}: {e}")
                
                # Check for tokenizer
                tokenizer_file = os.path.join(item_path, "tokenizer.json")
                if os.path.exists(tokenizer_file):
                    print(f"\nTokenizer found in {item}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(item_path)
                        print(f"Tokenizer loaded: {type(tokenizer)}")
                        print(f"Vocab size: {tokenizer.vocab_size}")
                        print(f"Model max length: {tokenizer.model_max_length}")
                    except Exception as e:
                        print(f"Could not load tokenizer: {e}")
        
        return repo_path
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    inspect_model_config()
