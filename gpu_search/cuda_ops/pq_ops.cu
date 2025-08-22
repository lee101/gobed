// pq_ops.cu - PQ LUT building and ADC scanning
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDAGuard.h>

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

// Generic ADC kernel for variable M
__global__ void adc_scan_generic_kernel(const uint8_t* __restrict__ codes,
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

// Build PQ lookup table
at::Tensor build_pq_lut_cuda(const at::Tensor& q_rot, const at::Tensor& cb) {
  TORCH_CHECK(q_rot.is_cuda() && cb.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(q_rot.dtype() == at::kFloat && cb.dtype() == at::kFloat, "Must be float32");
  
  const at::cuda::CUDAGuard device_guard(q_rot.device());
  
  auto m = q_rot.size(0);
  auto dsub = q_rot.size(1);
  TORCH_CHECK(cb.size(0) == m && cb.size(1) == 256 && cb.size(2) == dsub,
              "Codebook shape mismatch");
  
  auto lut = at::empty({m, 256}, q_rot.options());
  
  dim3 blocks(m);
  dim3 threads(256);
  
  build_pq_lut_kernel<<<blocks, threads>>>(
    q_rot.data_ptr<float>(),
    cb.data_ptr<float>(),
    lut.data_ptr<float>(),
    m, dsub
  );
  
  return lut;
}

// ADC scan with lookup table
at::Tensor adc_scan_cuda(const at::Tensor& codes, const at::Tensor& lut) {
  TORCH_CHECK(codes.dtype() == at::kByte && lut.dtype() == at::kFloat, 
              "codes must be uint8, lut must be float32");
  TORCH_CHECK(codes.is_cuda() && lut.is_cuda(), "Tensors must be on CUDA");
  
  const at::cuda::CUDAGuard device_guard(codes.device());
  
  auto K = codes.size(0);
  auto M = codes.size(1);
  TORCH_CHECK(lut.size(0) == M && lut.size(1) == 256, "LUT shape mismatch");
  
  auto out = at::empty({K}, lut.options());
  
  const int threads = 256;
  const int blocks = (K + threads - 1) / threads;
  
  // Specialize for common M values for better performance
  if (M == 64) {
    adc_scan_kernel<64><<<blocks, threads>>>(
      codes.data_ptr<uint8_t>(),
      lut.data_ptr<float>(),
      out.data_ptr<float>(),
      K
    );
  } else if (M == 32) {
    adc_scan_kernel<32><<<blocks, threads>>>(
      codes.data_ptr<uint8_t>(),
      lut.data_ptr<float>(),
      out.data_ptr<float>(),
      K
    );
  } else if (M == 128) {
    adc_scan_kernel<128><<<blocks, threads>>>(
      codes.data_ptr<uint8_t>(),
      lut.data_ptr<float>(),
      out.data_ptr<float>(),
      K
    );
  } else {
    // Generic kernel for other M values
    adc_scan_generic_kernel<<<blocks, threads>>>(
      codes.data_ptr<uint8_t>(),
      lut.data_ptr<float>(),
      out.data_ptr<float>(),
      K, M
    );
  }
  
  return out;
}

// IVF list gathering kernel
__global__ void gather_ivf_codes_kernel(const uint8_t* __restrict__ codes,     // [N, M]
                                        const int64_t* __restrict__ ids,       // [N]
                                        const int* __restrict__ list_ids,      // [nprobe]
                                        const int* __restrict__ list_offsets,  // [nlists+1]
                                        uint8_t* __restrict__ out_codes,       // [K', M]
                                        int64_t* __restrict__ out_ids,         // [K']
                                        int nprobe, int M) {
  // Each block handles one list
  int list_idx = blockIdx.x;
  if (list_idx >= nprobe) return;
  
  int list_id = list_ids[list_idx];
  int start = list_offsets[list_id];
  int end = list_offsets[list_id + 1];
  int list_size = end - start;
  
  // Each thread copies one vector
  int tid = threadIdx.x;
  while (tid < list_size) {
    int src_idx = start + tid;
    int dst_idx = blockIdx.y * blockDim.x + tid;  // Output position
    
    // Copy codes
    for (int j = 0; j < M; j++) {
      out_codes[dst_idx * M + j] = codes[src_idx * M + j];
    }
    
    // Copy ID
    out_ids[dst_idx] = ids[src_idx];
    
    tid += blockDim.x;
  }
}

// Gather codes from IVF lists
std::tuple<at::Tensor, at::Tensor> gather_ivf_codes_cuda(
    const at::Tensor& codes,
    const at::Tensor& ids,
    const at::Tensor& list_ids,
    const at::Tensor& list_offsets) {
  
  TORCH_CHECK(codes.is_cuda() && ids.is_cuda(), "Tensors must be on CUDA");
  const at::cuda::CUDAGuard device_guard(codes.device());
  
  auto nprobe = list_ids.size(0);
  auto M = codes.size(1);
  
  // Calculate total codes to gather
  auto list_ids_cpu = list_ids.cpu();
  auto list_offsets_cpu = list_offsets.cpu();
  int total_codes = 0;
  for (int i = 0; i < nprobe; i++) {
    int list_id = list_ids_cpu[i].item<int>();
    int start = list_offsets_cpu[list_id].item<int>();
    int end = list_offsets_cpu[list_id + 1].item<int>();
    total_codes += (end - start);
  }
  
  auto out_codes = at::empty({total_codes, M}, codes.options());
  auto out_ids = at::empty({total_codes}, ids.options());
  
  // Launch gathering kernel
  dim3 blocks(nprobe);
  dim3 threads(256);
  
  gather_ivf_codes_kernel<<<blocks, threads>>>(
    codes.data_ptr<uint8_t>(),
    ids.data_ptr<int64_t>(),
    list_ids.data_ptr<int>(),
    list_offsets.data_ptr<int>(),
    out_codes.data_ptr<uint8_t>(),
    out_ids.data_ptr<int64_t>(),
    nprobe, M
  );
  
  return std::make_tuple(out_codes, out_ids);
}