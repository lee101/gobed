#!/usr/bin/env python3
"""
Export the native PyTorch model for libtorch loading.
"""

import torch
from sentence_transformers import SentenceTransformer

def export_pytorch_model():
    """Export the native PyTorch model for libtorch."""
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    
    # Get the embedding module
    embedding_module = model[0]  # First module is usually the transformer
    
    print("Model architecture:")
    print(embedding_module)
    
    # Save the native PyTorch model using TorchScript
    print("\nExporting PyTorch model with TorchScript...")
    
    # Create a sample input for tracing
    sample_input = torch.randint(0, 1000, (1, 8), dtype=torch.long)
    
    try:
        # Try tracing the model
        traced_model = torch.jit.trace(embedding_module, sample_input)
        traced_model.save("model/production_pytorch_model.pt")
        print("✓ TorchScript traced model saved to model/production_pytorch_model.pt")
    except Exception as e:
        print(f"✗ Tracing failed: {e}")
        
        # Try scripting instead
        try:
            scripted_model = torch.jit.script(embedding_module)
            scripted_model.save("model/production_pytorch_scripted_model.pt")
            print("✓ TorchScript scripted model saved to model/production_pytorch_scripted_model.pt")
        except Exception as e2:
            print(f"✗ Scripting also failed: {e2}")
            
            # Fallback: save the state dict
            torch.save(embedding_module.state_dict(), "model/production_pytorch_state_dict.pt")
            print("✓ Fallback: state dict saved to model/production_pytorch_state_dict.pt")
    
    # Also save the full model directly
    torch.save(embedding_module, "model/production_pytorch_full_model.pt")
    print("✓ Full model saved to model/production_pytorch_full_model.pt")
    
    print("\nPyTorch model export completed!")

if __name__ == "__main__":
    export_pytorch_model()
