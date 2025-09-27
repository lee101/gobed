#!/bin/bash
set -e

echo " Building GPU-Accelerated Search System"
echo "========================================="

# 1. Build CUDA ops
echo -e "\n Building CUDA custom ops..."
cd cuda_ops
./build.sh
cd ..

# 2. Generate TorchScript modules
echo -e "\n Creating TorchScript search modules..."
cd scripts
python3 search_module.py
cd ..

# 3. Setup Go module
echo -e "\n Setting up Go module..."
cd go_client

if [ ! -f go.mod ]; then
    go mod init gpu_search_client
    go get github.com/sugarme/gotch@v0.9.1
fi

# 4. Build Go client
echo -e "\n Building Go client..."
source ~/.secretbashrc 2>/dev/null || true
export LD_LIBRARY_PATH=$LIBTORCH/lib:/usr/local/lib:$LD_LIBRARY_PATH

go build -o gpu_search_client .

echo -e "\n Build complete!"
echo -e "\n Running test..."

# 5. Run test
./gpu_search_client

echo -e "\n GPU-accelerated search system ready!"
echo ""
echo "Components built:"
echo "  - CUDA ops: libgobed_ann_ops.so"
echo "  - TorchScript modules: gpu_search_flat.pt, gpu_search_ivf_pq.pt"
echo "  - Go client: gpu_search_client"
echo ""
echo "To use in your Go project:"
echo "  1. Link libgobed_ann_ops.so"
echo "  2. Load the .pt module with gotch"
echo "  3. Call search methods"