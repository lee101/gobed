// pq_ops_standalone.cu - LibTorch-free PQ LUT building and ADC scanning
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_ops.h"

// Kernel to build PQ lookup table for a query
__global__ void build_pq_lut_kernel(const float* __restrict__ q_rot,    // [m, dsub]
                                    const float* __restrict__ cb,       // [m, 256, dsub]
                                    float* __restrict__ lut,            // [m, 256]
                                    int m, int dsub) {
  int sub_idx = blockIdx.x;  // which subquantizer
  int code_idx = threadIdx.x;  // which code (0-255)

  if (sub_idx >= m || code_idx >= 256) return;

  const float* q_sub = q_rot + sub_idx * dsub;
  const float* cb_sub = cb + sub_idx * 256 * dsub + code_idx * dsub;

  // Compute L2 distance or negative dot product
  float dist = 0.0f;
  for (int i = 0; i < dsub; i++) {
    float diff = q_sub[i] - cb_sub[i];
    dist += diff * diff;  // L2 distance
    // For cosine/IP: dist -= q_sub[i] * cb_sub[i];
  }

  lut[sub_idx * 256 + code_idx] = dist;
}

// Fast ADC scan using PQ lookup table
template<int M>
__global__ void adc_scan_kernel(const uint8_t* __restrict__ codes,  // [K, M]
                                const float* __restrict__ lut,      // [M, 256]
                                float* __restrict__ out,            // [K]
                                int64_t K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K) return;

  const uint8_t* row = codes + (size_t)idx * M;
  float acc = 0.0f;

  #pragma unroll
  for (int j = 0; j < M; ++j) {
    uint8_t c = row[j];
    acc += lut[j * 256 + c];
  }

  out[idx] = acc;
}

// Dynamic dispatch wrapper for ADC scan
__global__ void adc_scan_kernel_dynamic(const uint8_t* __restrict__ codes,
                                        const float* __restrict__ lut,
                                        float* __restrict__ out,
                                        int64_t K, int M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K) return;

  const uint8_t* row = codes + (size_t)idx * M;
  float acc = 0.0f;

  for (int j = 0; j < M; ++j) {
    uint8_t c = row[j];
    acc += lut[j * 256 + c];
  }

  out[idx] = acc;
}

// C wrapper for PQ LUT building
extern "C" cuda_op_result_t build_pq_lut(
    const float* q_rot,
    const float* cb,
    float* lut,
    int D,
    int M,
    int K) {

  // Input validation
  if (!q_rot || !cb || !lut) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  if (D <= 0 || M <= 0 || K != 256 || D % M != 0) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  int dsub = D / M;

  // Launch parameters
  dim3 blocks(M);
  dim3 threads(256);

  // Launch kernel
  build_pq_lut_kernel<<<blocks, threads>>>(q_rot, cb, lut, M, dsub);

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

// C wrapper for ADC scan
extern "C" cuda_op_result_t adc_scan(
    const uint8_t* codes,
    const float* lut,
    float* out,
    int64_t N,
    int M,
    int K) {

  // Input validation
  if (!codes || !lut || !out) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  if (N <= 0 || M <= 0 || K != 256) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch parameters
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  if (blocks > 65535) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Use template specialization for common M values, fallback to dynamic
  if (M == 64) {
    adc_scan_kernel<64><<<blocks, threads>>>(codes, lut, out, N);
  } else if (M == 32) {
    adc_scan_kernel<32><<<blocks, threads>>>(codes, lut, out, N);
  } else if (M == 16) {
    adc_scan_kernel<16><<<blocks, threads>>>(codes, lut, out, N);
  } else {
    adc_scan_kernel_dynamic<<<blocks, threads>>>(codes, lut, out, N, M);
  }

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