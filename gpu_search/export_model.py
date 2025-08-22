#!/usr/bin/env python3
"""
Export PyTorch embedding model to TorchScript for Go integration
This allows us to run GPU inference directly in Go without Python server
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import os

class EmbeddingModel(nn.Module):
    """Wrapper model for TorchScript export"""
    
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-en"):
        super().__init__()
        print(f"🔄 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        
        # Get model config
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = 512
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"✅ Model loaded:")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Max length: {self.max_length}")
        print(f"   Embedding dim: {self.embedding_dim}")
    
    def forward(self, input_ids, attention_mask):
        """Forward pass for TorchScript"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
    
    def encode_texts(self, texts):
        """Encode list of texts to embeddings"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=self.max_length
        )
        
        return self.forward(inputs['input_ids'], inputs['attention_mask'])

def export_model_to_torchscript():
    """Export the embedding model to TorchScript format"""
    
    # Create model
    model = EmbeddingModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"🎯 Using device: {device}")
    
    # Create example inputs for tracing
    batch_size = 8
    seq_length = 128
    
    example_input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length)).to(device)
    example_attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    print("🔄 Tracing model with example inputs...")
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(
            model, 
            (example_input_ids, example_attention_mask),
            strict=False
        )
        print("✅ Model traced successfully")
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print("🔄 Trying script mode...")
        # Fallback to script mode
        traced_model = torch.jit.script(model)
        print("✅ Model scripted successfully")
    
    # Test the traced model
    print("🧪 Testing traced model...")
    with torch.no_grad():
        original_output = model(example_input_ids, example_attention_mask)
        traced_output = traced_model(example_input_ids, example_attention_mask)
        
        # Check if outputs are close
        diff = torch.abs(original_output - traced_output).max().item()
        print(f"   Max difference: {diff:.6f}")
        
        if diff < 1e-5:
            print("✅ Traced model matches original")
        else:
            print("⚠️  Traced model differs from original")
    
    # Save the traced model
    output_path = "/home/lee/code/gobed/model/embedding_model.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"💾 Saving traced model to: {output_path}")
    traced_model.save(output_path)
    
    # Save tokenizer vocab for Go
    vocab_path = "/home/lee/code/gobed/model/vocab.json"
    print(f"💾 Saving vocabulary to: {vocab_path}")
    
    # Save tokenizer configuration
    tokenizer_config = {
        "vocab_size": model.vocab_size,
        "max_length": model.max_length,
        "embedding_dim": model.embedding_dim,
        "pad_token_id": model.tokenizer.pad_token_id,
        "cls_token_id": model.tokenizer.cls_token_id,
        "sep_token_id": model.tokenizer.sep_token_id,
        "unk_token_id": model.tokenizer.unk_token_id,
    }
    
    config_path = "/home/lee/code/gobed/model/tokenizer_config.json"
    with open(config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Save vocabulary
    vocab = model.tokenizer.get_vocab()
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    
    print("✅ Export completed successfully!")
    print(f"📁 Files created:")
    print(f"   - {output_path} (TorchScript model)")
    print(f"   - {vocab_path} (Vocabulary)")
    print(f"   - {config_path} (Config)")
    
    return output_path

def test_exported_model(model_path):
    """Test the exported TorchScript model"""
    print(f"\n🧪 Testing exported model: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the exported model
    model = torch.jit.load(model_path)
    model = model.to(device)
    model.eval()
    
    # Test with sample inputs
    batch_size = 4
    seq_length = 64
    vocab_size = 32000  # Approximate
    
    test_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    test_attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    print(f"📊 Input shape: {test_input_ids.shape}")
    
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        embeddings = model(test_input_ids, test_attention_mask)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    
    print(f"📊 Output shape: {embeddings.shape}")
    print(f"⚡ Inference time: {elapsed_time:.2f}ms")
    print(f"🔥 Throughput: {batch_size * 1000 / elapsed_time:.0f} texts/sec")
    
    # Check output properties
    print(f"📈 Embedding stats:")
    print(f"   Mean: {embeddings.mean().item():.6f}")
    print(f"   Std: {embeddings.std().item():.6f}")
    print(f"   Min: {embeddings.min().item():.6f}")
    print(f"   Max: {embeddings.max().item():.6f}")
    
    # Check normalization
    norms = torch.norm(embeddings, dim=1)
    print(f"   Norms: {norms.mean().item():.6f} ± {norms.std().item():.6f}")
    
    print("✅ Exported model test passed!")

if __name__ == "__main__":
    print("🚀 PyTorch Model Export for Go Integration")
    print("=" * 50)
    
    try:
        # Export model
        model_path = export_model_to_torchscript()
        
        # Test exported model
        test_exported_model(model_path)
        
        print("\n🎉 Success! Model exported and ready for Go integration")
        print("\nNext steps:")
        print("1. Install libtorch C++ library")
        print("2. Create Go bindings using CGO")
        print("3. Replace Python server with native Go calls")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()