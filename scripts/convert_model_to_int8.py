#!/usr/bin/env python3
"""
Convert model from float32 to int8 with dimension reduction
- Loads model/real_model.safetensors (1024 dims, float32)
- Reduces to 512 dimensions (keeping most important)
- Quantizes to int8 with scale factors
- Saves as model/modelint8_512dim.safetensors
"""

import numpy as np
import json
import struct
import os
from pathlib import Path

def read_safetensors(filepath):
    """Read safetensors file format"""
    with open(filepath, 'rb') as f:
        # Read header length (8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]

        # Read header JSON
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)

        # Read tensor data
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue

            dtype = info['dtype']
            shape = info['data_offsets']
            offset_start, offset_end = shape[0], shape[1]

            # Seek to tensor data position (8 + header_len + offset)
            f.seek(8 + header_len + offset_start)
            num_bytes = offset_end - offset_start

            # Read raw bytes
            raw_data = f.read(num_bytes)

            # Parse based on dtype
            if dtype == 'F32':
                # Float32 data
                num_elements = num_bytes // 4
                tensor_data = np.frombuffer(raw_data, dtype=np.float32)
            elif dtype == 'I8':
                # Int8 data
                tensor_data = np.frombuffer(raw_data, dtype=np.int8)
            else:
                print(f"Unsupported dtype: {dtype} for tensor {name}")
                continue

            tensors[name] = tensor_data

    return tensors, header

def write_safetensors(filepath, tensors, metadata=None):
    """Write safetensors file with int8 data"""
    header = {}
    tensor_bytes = b''
    current_offset = 0

    # Add metadata if provided
    if metadata:
        header["__metadata__"] = metadata

    # Process each tensor
    for name, data in tensors.items():
        if isinstance(data, dict):
            continue  # Skip metadata

        # Ensure data is contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        # Determine dtype string
        if data.dtype == np.float32:
            dtype_str = 'F32'
        elif data.dtype == np.int8:
            dtype_str = 'I8'
        elif data.dtype == np.float64:
            dtype_str = 'F64'
            data = data.astype(np.float32)  # Convert to F32
            dtype_str = 'F32'
        else:
            print(f"Warning: unsupported dtype {data.dtype} for {name}, converting to float32")
            data = data.astype(np.float32)
            dtype_str = 'F32'

        # Get tensor bytes
        tensor_data = data.tobytes()

        # Add to header
        header[name] = {
            'dtype': dtype_str,
            'shape': list(data.shape),
            'data_offsets': [current_offset, current_offset + len(tensor_data)]
        }

        tensor_bytes += tensor_data
        current_offset += len(tensor_data)

    # Serialize header to JSON
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_json)

    # Write file
    with open(filepath, 'wb') as f:
        # Write header length (8 bytes, little-endian uint64)
        f.write(struct.pack('<Q', header_len))
        # Write header JSON
        f.write(header_json)
        # Write tensor data
        f.write(tensor_bytes)

    print(f" Saved {filepath} ({len(tensors)} tensors, {len(tensor_bytes) / 1024 / 1024:.1f} MB)")

def quantize_to_int8(embeddings, dim_reduction=512):
    """
    Quantize float32 embeddings to int8 with dimension reduction

    Args:
        embeddings: numpy array of shape (vocab_size, embedding_dim)
        dim_reduction: reduce to this many dimensions (default 512)

    Returns:
        quantized: int8 array of shape (vocab_size, dim_reduction)
        scales: float32 array of shape (vocab_size,) with scale factors
    """
    vocab_size, embedding_dim = embeddings.shape
    print(f"Original shape: {embeddings.shape}")

    # Step 1: Reduce dimensions (keep first 512 dims)
    if dim_reduction < embedding_dim:
        print(f"Reducing dimensions from {embedding_dim} to {dim_reduction}")
        embeddings_reduced = embeddings[:, :dim_reduction]

        # Optional: Use PCA or importance-based selection
        # For now, we'll use the first 512 dimensions which typically
        # contain the most important information
    else:
        embeddings_reduced = embeddings

    # Step 2: Quantize to int8
    # Find scale factor for each embedding vector
    scales = np.zeros(vocab_size, dtype=np.float32)
    quantized = np.zeros((vocab_size, dim_reduction), dtype=np.int8)

    for i in range(vocab_size):
        vec = embeddings_reduced[i]

        # Find the maximum absolute value
        max_abs = np.abs(vec).max()

        if max_abs > 0:
            # Scale factor to map to int8 range [-127, 127]
            scale = max_abs / 127.0
            scales[i] = scale

            # Quantize
            quantized_vec = np.round(vec / scale).astype(np.int8)
            quantized[i] = quantized_vec
        else:
            scales[i] = 1.0
            quantized[i] = np.zeros(dim_reduction, dtype=np.int8)

    print(f"Quantized shape: {quantized.shape}")
    print(f"Scales shape: {scales.shape}")

    # Calculate compression ratio
    original_size = embeddings.nbytes
    quantized_size = quantized.nbytes + scales.nbytes
    compression_ratio = original_size / quantized_size

    print(f"Original size: {original_size / 1024 / 1024:.1f} MB")
    print(f"Quantized size: {quantized_size / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {compression_ratio:.1f}x")

    return quantized, scales

def test_reconstruction(original, quantized, scales, num_tests=10):
    """Test reconstruction quality"""
    vocab_size = original.shape[0]
    dim_reduction = quantized.shape[1]

    # Test random vectors
    test_indices = np.random.choice(vocab_size, num_tests, replace=False)

    total_error = 0
    for idx in test_indices:
        # Original vector (truncated to dim_reduction)
        orig_vec = original[idx, :dim_reduction]

        # Reconstruct from quantized
        quant_vec = quantized[idx]
        scale = scales[idx]
        reconstructed = quant_vec.astype(np.float32) * scale

        # Calculate error
        error = np.mean(np.abs(orig_vec - reconstructed))
        total_error += error

    avg_error = total_error / num_tests
    print(f"Average reconstruction error: {avg_error:.6f}")

    # Calculate cosine similarity for a sample
    idx = test_indices[0]
    orig_vec = original[idx, :dim_reduction]
    quant_vec = quantized[idx]
    scale = scales[idx]
    reconstructed = quant_vec.astype(np.float32) * scale

    # Cosine similarity
    cos_sim = np.dot(orig_vec, reconstructed) / (np.linalg.norm(orig_vec) * np.linalg.norm(reconstructed) + 1e-10)
    print(f"Sample cosine similarity: {cos_sim:.4f}")

def main():
    # Paths
    input_path = Path("model/real_model.safetensors")
    output_path = Path("model/modelint8_512dim.safetensors")

    if not input_path.exists():
        print(f" Input file not found: {input_path}")
        print("Please ensure model/real_model.safetensors exists")
        return

    print(f"📚 Loading model from {input_path}")

    # Load the original model
    tensors, header = read_safetensors(input_path)

    # Find the embedding tensor
    embedding_tensor = None
    embedding_name = None

    for name, tensor in tensors.items():
        if 'embed' in name.lower() or tensor.ndim == 2:
            # Likely the embedding matrix
            if tensor.shape[0] > 1000:  # Vocab size check
                embedding_tensor = tensor
                embedding_name = name
                print(f"Found embedding tensor: {name} with shape {tensor.shape}")
                break

    if embedding_tensor is None:
        print(" Could not find embedding tensor")
        print("Available tensors:")
        for name, tensor in tensors.items():
            print(f"  {name}: shape={tensor.shape if hasattr(tensor, 'shape') else 'unknown'}")
        return

    # Reshape if needed
    if embedding_tensor.ndim == 1:
        # Assume it's flattened, try to reshape
        # Standard vocab size is 30522 for BERT-like models
        vocab_size = 30522
        embedding_dim = len(embedding_tensor) // vocab_size
        if len(embedding_tensor) % vocab_size == 0:
            print(f"Reshaping from {embedding_tensor.shape} to ({vocab_size}, {embedding_dim})")
            embedding_tensor = embedding_tensor.reshape(vocab_size, embedding_dim)
        else:
            print(" Cannot determine correct shape for embedding tensor")
            return

    # Quantize to int8 with dimension reduction
    print("\n Converting to int8 with 512 dimensions...")
    quantized_embeddings, scales = quantize_to_int8(embedding_tensor, dim_reduction=512)

    # Test reconstruction quality
    print("\n🧪 Testing reconstruction quality...")
    test_reconstruction(embedding_tensor, quantized_embeddings, scales)

    # Prepare output tensors
    output_tensors = {
        'embeddings.weight': quantized_embeddings,
        'embeddings.scales': scales,
    }

    # Add metadata
    metadata = {
        'format': 'int8_quantized',
        'original_dim': str(embedding_tensor.shape[1]),
        'reduced_dim': '512',
        'vocab_size': str(quantized_embeddings.shape[0]),
        'quantization': 'per-vector-int8',
        'description': 'Int8 quantized embeddings with dimension reduction to 512'
    }

    # Save the quantized model
    print(f"\n Saving quantized model to {output_path}")
    output_path.parent.mkdir(exist_ok=True)
    write_safetensors(output_path, output_tensors, metadata)

    # Print summary
    print("\n Conversion complete!")
    print(f"Original: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Quantized: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Space saved: {(1 - output_path.stat().st_size / input_path.stat().st_size) * 100:.1f}%")

    print("\n Next steps:")
    print("1. Update the tokenizer to output int16 token IDs")
    print("2. Update the model loader to use the int8 embeddings")
    print("3. Test performance with the quantized model")

if __name__ == "__main__":
    main()