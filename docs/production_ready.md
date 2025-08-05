# GOBED - Fast Embedding Model Inference

## Overview
This project provides fast, reliable CPU-based inference for sentence embedding models using ONNX Runtime and Go. It supports both full-precision and INT8-quantized models for optimal performance-memory trade-offs.

## Current Status ✅
- **ONNX Runtime**: v1.19.2 (stable, CPU-only)
- **Go Bindings**: v1.19.0 (compatible with ONNX Runtime)
- **Model Format**: ONNX with INT8 quantization support
- **Performance**: 667-1096 embeddings/second (depending on quantization)
- **Memory Usage**: 11MB (quantized) or 44MB (full precision)

## Quick Start

### 1. Build and Run
```bash
cd /home/lee/code/gobed
go build
./gobed
```

### 2. Model Options
The application automatically uses the INT8 quantized model (`model/embedding_model_int8.onnx`) for optimal memory efficiency. To switch back to the full precision model, edit `main.go` line 311:

```go
// For full precision (faster inference, larger memory):
model, err := NewEmbeddingModel("model/embedding_model.onnx", "model/tokenizer.json", false)

// For INT8 quantization (smaller memory, slightly slower):
model, err := NewEmbeddingModel("model/embedding_model_int8.onnx", "model/tokenizer.json", false)
```

## Performance Benchmarks

### Model Comparison
| Model Type | Size | Inference Speed | Throughput | Best For |
|------------|------|----------------|------------|----------|
| **Full Precision** | 44 MB | 0.91 ms | 1,096/sec | Maximum speed |
| **INT8 Quantized** | 11 MB | 1.50 ms | 667/sec | Memory efficiency |

### Real-World Performance
- **Latency**: Sub-millisecond to ~1.5ms per embedding
- **Batch Processing**: Excellent for high-throughput applications
- **Memory**: Minimal RAM usage with quantized models
- **CPU Utilization**: Optimized for modern multi-core processors

## Model Management

### Creating Quantized Models
```bash
# Activate Python environment
source .venv/bin/activate

# Create INT8 quantized model from existing ONNX
python convert_to_onnx.py quantize

# Or convert and quantize in one step
python convert_to_onnx.py both
```

### Model Files
- `model/embedding_model.onnx` - Full precision model (44MB)
- `model/embedding_model_int8.onnx` - INT8 quantized model (11MB)
- `model/tokenizer.json` - Tokenizer configuration
- `model/vocab.json` - Vocabulary mapping

## Architecture

### ONNX Runtime Integration
- **CPU-only inference** for maximum stability
- **Global initialization** for optimal performance
- **Session reuse** to minimize overhead
- **Tensor pooling** for memory efficiency

### Go Application Structure
- `NewEmbeddingModel()` - Model initialization with ONNX
- `Encode()` - Text to embedding conversion
- `Close()` - Proper resource cleanup
- Built-in benchmarking and testing

## Development Notes

### Stability Choices
1. **CPU-only**: Eliminates GPU driver dependencies
2. **Fixed ONNX Runtime version**: Prevents version conflicts
3. **INT8 quantization**: Reduces memory pressure
4. **Go static linking**: Simplified deployment

### Performance Optimizations
- Pre-allocated tensor buffers
- Minimal memory copying
- Efficient tokenization
- Session reuse across calls

## Troubleshooting

### Common Issues
1. **ONNX Runtime version mismatch**: Ensure v1.19.2 is installed
2. **Missing model files**: Run Python conversion script
3. **Memory issues**: Use INT8 quantized model
4. **Slow inference**: Check CPU thread allocation

### Version Compatibility
- **ONNX Runtime**: 1.19.2 (recommended)
- **Go Bindings**: github.com/yalue/onnxruntime_go v1.19.0
- **Go Version**: 1.21+ recommended
- **Python**: 3.8+ for conversion scripts

## Future Improvements
- [ ] Dynamic batch processing
- [ ] Model auto-switching based on memory constraints
- [ ] Additional quantization formats (INT4, FLOAT16)
- [ ] WebAssembly deployment support
- [ ] Model serving with HTTP API

---

**Status**: Production-ready ✅  
**Last Updated**: July 2025  
**Performance**: 667-1096 embeddings/second on CPU
