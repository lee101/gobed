#!/bin/bash
# Source this file to set up LibTorch environment
export LIBTORCH=$(pwd)/libtorch/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export CGO_CPPFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L${LIBTORCH}/lib -ltorch -ltorch_cpu"
echo "🔧 LibTorch environment configured"
echo "   LIBTORCH=$LIBTORCH"
