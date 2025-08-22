#!/bin/bash
# Fix CUDA path issues

echo "Current CUDA environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Set proper CUDA paths
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo ""
echo "Updated CUDA environment:"
echo "CUDA_HOME: $CUDA_HOME"

# Clean and rebuild
cd /home/lee/code/gobed/gpu_search/cuda_ops/build
rm -rf *
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
make -j8

echo ""
echo "Checking library dependencies:"
ldd libgobed_ann_ops.so | grep cuda