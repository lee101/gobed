#!/bin/bash
# Ultra-fast bed build with optimized CUDA kernels

set -e

echo "🚀 Building ultra-fast semantic search..."

# Source CUDA environment detection
source ./cuda_env.sh

if [ "$CUDA_ENABLED" = "true" ]; then
    echo "🔥 Building with CUDA acceleration..."

    # Build optimized CUDA kernels with vectorization
    echo "Compiling CUDA similarity kernels..."
    nvcc -shared -Xcompiler -fPIC \
         --use_fast_math \
         --ptxas-options=-v \
         $GPU_ARCH \
         -O3 \
         -o libcuda_similarity.so \
         cuda_similarity.cu \
         -lcublas -lcudart

    echo "✅ CUDA kernels compiled"

    # Build Go binary with optimized flags
    echo "Building Go binary with GPU support..."
    env CC="$CUDA_GCC" \
        CXX="$CUDA_GXX" \
        CGO_ENABLED=1 \
        CGO_CFLAGS="-O3 -march=native $CGO_CFLAGS" \
        CGO_LDFLAGS="$CGO_LDFLAGS" \
        go build -ldflags="-s -w" \
        -o bed_ultra \
        bed_ultra.go tokenizer.go model.go

    echo "✅ Ultra-fast bed built with GPU acceleration"
    echo "📊 Usage: ./bed_ultra --debug \"search query\""

else
    echo "⚠️  CUDA not available, building CPU-only version"

    # Build CPU-only fallback
    go build -ldflags="-s -w" \
        -o bed_cpu \
        bed_ultra.go tokenizer.go model.go

    echo "✅ CPU-only version built as bed_cpu"
fi

echo ""
echo "🎯 Build complete!"
echo "   - Modular design: tokenizer.go + model.go + bed_ultra.go"
echo "   - Vectorized CUDA kernels for similarity"
echo "   - Zero-allocation Go tokenizer"
echo "   - Optimized memory access patterns"