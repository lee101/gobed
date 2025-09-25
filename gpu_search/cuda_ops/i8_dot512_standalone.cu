// i8_dot512_standalone.cu - LibTorch-free INT8 dot product using __dp4a intrinsic
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_ops.h"

// Fast INT8 dot product with version compatibility and error checking
template<int D>
__device__ __forceinline__ int dot_dp4a(const int8_t* __restrict__ a,
                                        const int8_t* __restrict__ b) {
  int acc = 0;

  // Check for proper alignment and CUDA version
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610 && CUDA_VERSION >= 9000
  // Use __dp4a for compute capability >= 6.1 and CUDA >= 9.0
  #pragma unroll
  for (int i = 0; i < D; i += 4) {
    // Ensure 4-byte alignment
    if ((reinterpret_cast<uintptr_t>(a + i) % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(b + i) % 4 == 0)) {
      int pa = *reinterpret_cast<const int*>(a + i);
      int pb = *reinterpret_cast<const int*>(b + i);
      acc = __dp4a(pa, pb, acc);
    } else {
      // Fallback for unaligned access
      for (int j = 0; j < 4 && i + j < D; j++) {
        acc += static_cast<int>(a[i + j]) * static_cast<int>(b[i + j]);
      }
    }
  }
  #else
  // Fallback for older architectures or CUDA versions
  #pragma unroll 8
  for (int i = 0; i < D; i++) {
    acc += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  }
  #endif

  return acc;
}

// Kernel: each thread computes one database vector's dot product with query
template<int D>
__global__ void i8dot_kernel(const int8_t* __restrict__ q,        // [D]
                             const int8_t* __restrict__ db,       // [N,D]
                             int32_t* __restrict__ out,           // [N]
                             int64_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N) return;

  const int8_t* row = db + idx * D;
  out[idx] = dot_dp4a<D>(q, row);
}

// Batch kernel: each thread computes one database vector's dot product with one query
template<int D>
__global__ void i8dot_batch_kernel(const int8_t* __restrict__ queries,  // [B,D]
                                   const int8_t* __restrict__ db,        // [N,D]
                                   int32_t* __restrict__ out,            // [B,N]
                                   int64_t B, int64_t N) {
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b_idx = blockIdx.y;

  if (n_idx >= N || b_idx >= B) return;

  const int8_t* q = queries + b_idx * D;
  const int8_t* row = db + n_idx * D;

  out[b_idx * N + n_idx] = dot_dp4a<D>(q, row);
}

// C wrapper for single query
extern "C" cuda_op_result_t i8dot512_scores(
    const int8_t* q,
    const int8_t* db,
    int32_t* out,
    int64_t N) {

  // Input validation
  if (!q || !db || !out) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  if (N <= 0 || N > INT_MAX) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch parameters
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  if (blocks > 65535) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch kernel
  i8dot_kernel<512><<<blocks, threads>>>(q, db, out, N);

  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    return CUDA_OP_ERROR_CUDA_RUNTIME;
  }

  // Synchronize and check for runtime errors
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    return CUDA_OP_ERROR_CUDA_RUNTIME;
  }

  return CUDA_OP_SUCCESS;
}

// C wrapper for batch queries
extern "C" cuda_op_result_t i8dot512_batch(
    const int8_t* queries,
    const int8_t* db,
    int32_t* out,
    int64_t B,
    int64_t N) {

  // Input validation
  if (!queries || !db || !out) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  if (B <= 0 || N <= 0 || B > 1024 || N > INT_MAX) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch parameters
  dim3 threads(16, 16);  // 256 threads per block
  dim3 blocks((N + threads.x - 1) / threads.x, (B + threads.y - 1) / threads.y);
  if (blocks.x > 65535 || blocks.y > 65535) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch batch kernel
  i8dot_batch_kernel<512><<<blocks, threads>>>(queries, db, out, B, N);

  // Error checking
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    return CUDA_OP_ERROR_CUDA_RUNTIME;
  }

  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    return CUDA_OP_ERROR_CUDA_RUNTIME;
  }

  return CUDA_OP_SUCCESS;
}