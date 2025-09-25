// search_ops_standalone.cu - LibTorch-free IVF operations for GPU search
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_ops.h"

// Kernel for gathering IVF codes and IDs based on list offsets
__global__ void gather_ivf_codes_kernel(
    const uint8_t* __restrict__ codes,      // input codes [total_size, M]
    const int64_t* __restrict__ ids,        // vector IDs [total_size]
    const int32_t* __restrict__ list_ids,   // list IDs for each vector [total_size]
    const int64_t* __restrict__ list_offsets, // offsets for each list [num_lists+1]
    uint8_t* __restrict__ out_codes,        // output codes [total_size, M]
    int64_t* __restrict__ out_ids,          // output IDs [total_size]
    int64_t total_size,
    int M) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;

  // Get list ID and offset for this vector
  int32_t list_id = list_ids[idx];
  int64_t list_start = list_offsets[list_id];
  int64_t local_idx = idx - list_start;

  // Copy codes
  for (int j = 0; j < M; j++) {
    out_codes[idx * M + j] = codes[idx * M + j];
  }

  // Copy ID
  out_ids[idx] = ids[idx];
}

// C wrapper for IVF code gathering
extern "C" cuda_op_result_t gather_ivf_codes(
    const uint8_t* codes,
    const int64_t* ids,
    const int32_t* list_ids,
    const int64_t* list_offsets,
    uint8_t* out_codes,
    int64_t* out_ids,
    int64_t total_size,
    int M,
    int num_lists) {

  // Input validation
  if (!codes || !ids || !list_ids || !list_offsets || !out_codes || !out_ids) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  if (total_size <= 0 || M <= 0 || num_lists <= 0) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch parameters
  const int threads = 256;
  const int blocks = (total_size + threads - 1) / threads;
  if (blocks > 65535) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Launch kernel
  gather_ivf_codes_kernel<<<blocks, threads>>>(
    codes, ids, list_ids, list_offsets, out_codes, out_ids, total_size, M);

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

// CUDA capability and version checking
extern "C" cuda_op_result_t check_cuda_capabilities(
    int* compute_major,
    int* compute_minor,
    int* cuda_version) {

  if (!compute_major || !compute_minor || !cuda_version) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Get device properties
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    return CUDA_OP_ERROR_DEVICE;
  }

  *compute_major = prop.major;
  *compute_minor = prop.minor;

  // Get CUDA runtime version
  int runtime_version;
  err = cudaRuntimeGetVersion(&runtime_version);
  if (err != cudaSuccess) {
    return CUDA_OP_ERROR_CUDA_RUNTIME;
  }

  *cuda_version = runtime_version;

  return CUDA_OP_SUCCESS;
}

// Benchmark kernel for version compatibility testing
__global__ void benchmark_kernel(
    const int8_t* __restrict__ db,       // test database [1024, 512]
    const int8_t* __restrict__ query,    // test query [512]
    int32_t* __restrict__ result) {      // output result [1024]

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1024) return;

  const int8_t* row = db + idx * 512;

  // Simple dot product
  int acc = 0;
  #pragma unroll 8
  for (int i = 0; i < 512; i++) {
    acc += static_cast<int>(query[i]) * static_cast<int>(row[i]);
  }

  result[idx] = acc;
}

// Version compatibility test with benchmarking
extern "C" cuda_op_result_t test_version_compatibility(
    const int8_t* db,
    const int8_t* query,
    int32_t* result,
    float* benchmark_time) {

  if (!db || !query || !result || !benchmark_time) {
    return CUDA_OP_ERROR_INVALID_ARGS;
  }

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Launch parameters
  const int threads = 256;
  const int blocks = (1024 + threads - 1) / threads;

  // Start timing
  cudaEventRecord(start);

  // Launch kernel
  benchmark_kernel<<<blocks, threads>>>(db, query, result);

  // Stop timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  *benchmark_time = milliseconds;

  // Cleanup events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

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