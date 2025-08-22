// i8_dot512.cu - Fast INT8 dot product using __dp4a intrinsic
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDAGuard.h>

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
  
  const int8_t* row = db + (size_t)idx * D;
  out[idx] = dot_dp4a<D>(q, row);
}

// Batched version for multiple queries
template<int D>
__global__ void i8dot_batch_kernel(const int8_t* __restrict__ queries,  // [B,D]
                                   const int8_t* __restrict__ db,       // [N,D]
                                   int32_t* __restrict__ out,           // [B,N]
                                   int64_t B, int64_t N) {
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b_idx = blockIdx.y;
  
  if (n_idx >= N || b_idx >= B) return;
  
  const int8_t* q = queries + b_idx * D;
  const int8_t* row = db + (size_t)n_idx * D;
  out[b_idx * N + n_idx] = dot_dp4a<D>(q, row);
}

// C++ wrapper for single query with comprehensive error handling
at::Tensor i8dot512_scores_cuda(const at::Tensor& q, const at::Tensor& db) {
  // Input validation
  TORCH_CHECK(q.dtype() == at::kChar && db.dtype() == at::kChar, 
    "Tensors must be int8, got query: ", q.dtype(), " db: ", db.dtype());
  TORCH_CHECK(q.is_cuda() && db.is_cuda(), 
    "Tensors must be on CUDA, got query device: ", q.device(), " db device: ", db.device());
  TORCH_CHECK(q.numel() == 512, 
    "Query must be 512-dimensional, got: ", q.numel());
  TORCH_CHECK(db.size(1) == 512, 
    "Database vectors must be 512-dimensional, got: ", db.size(1));
  TORCH_CHECK(db.size(0) > 0, 
    "Database must have at least one vector, got: ", db.size(0));
  TORCH_CHECK(q.is_contiguous() && db.is_contiguous(),
    "Tensors must be contiguous");
  
  // Device guard for multi-GPU safety
  const at::cuda::CUDAGuard device_guard(q.device());
  
  auto N = db.size(0);
  TORCH_CHECK(N <= INT_MAX, "Database too large: ", N, " > ", INT_MAX);
  
  // Create output tensor
  auto out = at::empty({N}, db.options().dtype(at::kInt));
  
  // Check for empty input
  if (N == 0) {
    return out;
  }
  
  // Launch parameters
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  TORCH_CHECK(blocks <= 65535, "Too many blocks needed: ", blocks);
  
  // Launch kernel
  i8dot_kernel<512><<<blocks, threads>>>(
    q.data_ptr<int8_t>(),
    db.data_ptr<int8_t>(),
    out.data_ptr<int32_t>(),
    N
  );
  
  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, 
    "CUDA kernel launch failed: ", cudaGetErrorString(launch_err));
  
  // Synchronize and check for execution errors
  cudaError_t sync_err = cudaDeviceSynchronize();
  TORCH_CHECK(sync_err == cudaSuccess,
    "CUDA kernel execution failed: ", cudaGetErrorString(sync_err));
  
  return out;
}

// C++ wrapper for batch queries with comprehensive error handling
at::Tensor i8dot512_batch_cuda(const at::Tensor& queries, const at::Tensor& db) {
  // Input validation
  TORCH_CHECK(queries.dtype() == at::kChar && db.dtype() == at::kChar, 
    "Tensors must be int8, got queries: ", queries.dtype(), " db: ", db.dtype());
  TORCH_CHECK(queries.is_cuda() && db.is_cuda(), 
    "Tensors must be on CUDA, got queries device: ", queries.device(), " db device: ", db.device());
  TORCH_CHECK(queries.size(1) == 512 && db.size(1) == 512, 
    "Vectors must be 512-dimensional, got queries: ", queries.size(1), " db: ", db.size(1));
  TORCH_CHECK(queries.size(0) > 0 && db.size(0) > 0,
    "Must have at least one query and one database vector");
  TORCH_CHECK(queries.is_contiguous() && db.is_contiguous(),
    "Tensors must be contiguous");
  
  // Device guard for multi-GPU safety
  const at::cuda::CUDAGuard device_guard(queries.device());
  
  auto B = queries.size(0);
  auto N = db.size(0);
  TORCH_CHECK(B <= 65535 && N <= INT_MAX, 
    "Batch size or database size too large: B=", B, " N=", N);
  
  // Create output tensor
  auto out = at::empty({B, N}, db.options().dtype(at::kInt));
  
  // Check for empty input
  if (B == 0 || N == 0) {
    return out;
  }
  
  // Launch parameters
  const int threads = 256;
  const int blocks_x = (N + threads - 1) / threads;
  TORCH_CHECK(blocks_x <= 65535, "Too many X blocks needed: ", blocks_x);
  
  dim3 blocks(blocks_x, B);
  
  // Launch kernel
  i8dot_batch_kernel<512><<<blocks, threads>>>(
    queries.data_ptr<int8_t>(),
    db.data_ptr<int8_t>(),
    out.data_ptr<int32_t>(),
    B, N
  );
  
  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, 
    "CUDA batch kernel launch failed: ", cudaGetErrorString(launch_err));
  
  // Synchronize and check for execution errors
  cudaError_t sync_err = cudaDeviceSynchronize();
  TORCH_CHECK(sync_err == cudaSuccess,
    "CUDA batch kernel execution failed: ", cudaGetErrorString(sync_err));
  
  return out;
}