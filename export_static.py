#!/usr/bin/env python3
"""
Export the StaticEmbedding model to ONNX - this should be much simpler
"""

import torch
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import numpy as np
from pathlib import Path

def export_static_embedding_to_onnx():
    print("🚀 Exporting StaticEmbedding model to ONNX...")
    
    # Load model
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    
    # Create output directory
    output_dir = Path("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the embedding model
    static_embed = model[0]
    embed_layer = static_embed.embedding
    
    print(f"📊 Embedding layer info:")
    print(f"  Type: {type(embed_layer)}")
    print(f"  Vocab size: {embed_layer.num_embeddings}")
    print(f"  Embedding dim: {embed_layer.embedding_dim}")
    print(f"  Mode: {embed_layer.mode}")
    
    # Move to CPU for export
    embed_layer = embed_layer.cpu()
    
    # Create a wrapper model that takes token IDs and returns mean-pooled embeddings
    class EmbeddingModel(torch.nn.Module):
        def __init__(self, embedding_layer):
            super().__init__()
            self.embedding = embedding_layer
        
        def forward(self, input_ids):
            # input_ids shape: [batch_size, seq_len]
            # We need to handle padding (zeros) by creating offsets
            batch_size, seq_len = input_ids.shape
            
            # Flatten input and create offsets for each sequence
            flattened = input_ids.view(-1)
            
            # Create offsets - start positions for each sequence in the batch
            offsets = torch.arange(0, batch_size * seq_len, seq_len, dtype=torch.long)
            
            # Get embeddings - this will do mean pooling automatically
            embeddings = self.embedding(flattened, offsets)
            
            return embeddings
    
    # Create the wrapper model
    wrapper_model = EmbeddingModel(embed_layer)
    wrapper_model.eval()
    
    # Test with sample input
    sample_input = torch.tensor([[101, 7592, 2088, 102]], dtype=torch.long)  # [CLS, hello, world, SEP]
    print(f"📝 Testing with sample input: {sample_input}")
    
    with torch.no_grad():
        test_output = wrapper_model(sample_input)
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output first 5: {test_output[0][:5]}")
    
    # Export to ONNX
    onnx_path = output_dir / "embedding_model.onnx"
    
    torch.onnx.export(
        wrapper_model,
        sample_input,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"}
        },
        opset_version=14,
        export_params=True,
        do_constant_folding=True
    )
    
    print(f"✅ ONNX model exported to: {onnx_path}")
    
    # Test the ONNX model
    print(f"🧪 Testing ONNX model...")
    session = ort.InferenceSession(str(onnx_path))
    
    onnx_result = session.run(None, {"input_ids": sample_input.numpy()})
    onnx_embedding = onnx_result[0][0]
    
    print(f"  ONNX output shape: {onnx_embedding.shape}")
    print(f"  ONNX first 5: {onnx_embedding[:5]}")
    
    # Compare with PyTorch
    pytorch_embedding = test_output[0].numpy()
    cos_sim = np.dot(onnx_embedding, pytorch_embedding) / (np.linalg.norm(onnx_embedding) * np.linalg.norm(pytorch_embedding))
    print(f"  PyTorch vs ONNX similarity: {cos_sim:.8f}")
    
    if cos_sim > 0.99:
        print(f"✅ ONNX export successful! High similarity: {cos_sim:.8f}")
        
        # Test with the sentences we care about
        print(f"\n🎯 Testing with real sentences...")
        
        # Get tokenizer
        tokenizer = model.tokenizer
        
        test_texts = ["hello world", "the weather is nice today", "machine learning algorithms are powerful"]
        
        for i, text in enumerate(test_texts):
            # Tokenize using the real tokenizer
            encoding = tokenizer.encode(text)
            token_ids = encoding.ids
            
            # Pad to same length (let's use 512)
            padded_ids = token_ids + [0] * (512 - len(token_ids))
            if len(padded_ids) > 512:
                padded_ids = padded_ids[:512]
            
            input_tensor = np.array([padded_ids], dtype=np.int64)
            
            # Get ONNX result
            onnx_result = session.run(None, {"input_ids": input_tensor})
            onnx_emb = onnx_result[0][0]
            
            # Get SentenceTransformer result
            st_emb = model.encode([text])[0]
            
            # Compare
            cos_sim = np.dot(onnx_emb, st_emb) / (np.linalg.norm(onnx_emb) * np.linalg.norm(st_emb))
            
            print(f"  '{text}':")
            print(f"    ONNX first 5: {onnx_emb[:5]}")
            print(f"    ST first 5:   {st_emb[:5]}")
            print(f"    Similarity: {cos_sim:.6f}")
        
        return True
    else:
        print(f"❌ ONNX export failed. Low similarity: {cos_sim:.8f}")
        return False

if __name__ == "__main__":
    export_static_embedding_to_onnx()
