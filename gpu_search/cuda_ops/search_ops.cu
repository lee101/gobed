// search_ops.cu - Custom CUDA operations for GPU search using LibTorch
// Implements IVF + OPQ + PQ + ADC + tiny re-rank architecture
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// INT8 dot product using __dp4a intrinsic for 4x speedup
// __dp4a requires compute capability 6.1+
template<int D>
__device__ __forceinline__ int dot_dp4a(const int8_t* __restrict__ a,
                                        const int8_t* __restrict__ b) {
    int acc = 0;
    #pragma unroll
    for (int i = 0; i < D; i += 4) {
        #if __CUDA_ARCH__ >= 610
        // Use fast __dp4a instruction for compute capability 6.1+
        int pa = *reinterpret_cast<const int*>(a + i);
        int pb = *reinterpret_cast<const int*>(b + i);
        acc = __dp4a(pa, pb, acc);
        #else
        // Fallback for older GPUs
        for (int j = 0; j < 4; j++) {
            acc += static_cast<int>(a[i + j]) * static_cast<int>(b[i + j]);
        }
        #endif
    }
    return acc;
}

// Kernel for computing similarity scores between query and database vectors
__global__ void i8dot512_scores_kernel(
    const int8_t* __restrict__ query,     // [512] query vector
    const int8_t* __restrict__ database,  // [N, 512] database vectors
    float* __restrict__ scores,           // [N] output scores
    int N                                 // number of database vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        const int8_t* db_vec = database + idx * 512;
        int dot_product = dot_dp4a<512>(query, db_vec);
        scores[idx] = static_cast<float>(dot_product);
    }
}

// Build Product Quantization lookup table
__global__ void build_pq_lut_kernel(
    const int8_t* __restrict__ query,     // [512] query vector
    const int8_t* __restrict__ codebooks, // [M, K, D/M] PQ codebooks
    float* __restrict__ lut,              // [M, K] lookup table
    int M,                                // number of subquantizers
    int K,                                // codebook size (256 for 8-bit)
    int D                                 // vector dimension (512)
) {
    int m = blockIdx.x;  // subquantizer index
    int k = threadIdx.x; // codeword index
    
    if (m < M && k < K) {
        int subvec_dim = D / M;
        const int8_t* query_sub = query + m * subvec_dim;
        const int8_t* codeword = codebooks + (m * K + k) * subvec_dim;
        
        int dot_product = 0;
        for (int i = 0; i < subvec_dim; i += 4) {
            #if __CUDA_ARCH__ >= 610
            // Use fast __dp4a instruction for compute capability 6.1+
            int qa = *reinterpret_cast<const int*>(query_sub + i);
            int ca = *reinterpret_cast<const int*>(codeword + i);
            dot_product = __dp4a(qa, ca, dot_product);
            #else
            // Fallback for older GPUs
            for (int j = 0; j < 4; j++) {
                dot_product += static_cast<int>(query_sub[i + j]) * static_cast<int>(codeword[i + j]);
            }
            #endif
        }
        
        lut[m * K + k] = static_cast<float>(dot_product);
    }
}

// Asymmetric Distance Computation scan using lookup table
__global__ void adc_scan_kernel(
    const float* __restrict__ lut,        // [M, K] lookup table
    const uint8_t* __restrict__ codes,    // [N, M] quantized codes
    float* __restrict__ scores,           // [N] output scores
    int N,                                // number of vectors
    int M,                                // number of subquantizers
    int K                                 // codebook size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float score = 0.0f;
        const uint8_t* vec_codes = codes + idx * M;
        
        #pragma unroll
        for (int m = 0; m < M; m++) {
            uint8_t code = vec_codes[m];
            score += lut[m * K + code];
        }
        
        scores[idx] = score;
    }
}

// Host functions for LibTorch integration
torch::Tensor i8dot512_scores_cuda(
    torch::Tensor query,
    torch::Tensor database
) {
    auto N = database.size(0);
    auto scores = torch::zeros({N}, torch::dtype(torch::kFloat32).device(query.device()));
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    i8dot512_scores_kernel<<<blocks, threads>>>(
        query.data_ptr<int8_t>(),
        database.data_ptr<int8_t>(),
        scores.data_ptr<float>(),
        N
    );
    
    return scores;
}

torch::Tensor build_pq_lut_cuda(
    torch::Tensor query,
    torch::Tensor codebooks
) {
    auto M = codebooks.size(0);
    auto K = codebooks.size(1);
    auto lut = torch::zeros({M, K}, torch::dtype(torch::kFloat32).device(query.device()));
    
    dim3 blocks(M);
    dim3 threads(K);
    
    build_pq_lut_kernel<<<blocks, threads>>>(
        query.data_ptr<int8_t>(),
        codebooks.data_ptr<int8_t>(),
        lut.data_ptr<float>(),
        M, K, 512
    );
    
    return lut;
}

torch::Tensor adc_scan_cuda(
    torch::Tensor lut,
    torch::Tensor codes
) {
    auto N = codes.size(0);
    auto M = codes.size(1);
    auto K = lut.size(1);
    auto scores = torch::zeros({N}, torch::dtype(torch::kFloat32).device(lut.device()));
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    adc_scan_kernel<<<blocks, threads>>>(
        lut.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        N, M, K
    );
    
    return scores;
}

// Register operations with PyTorch
TORCH_LIBRARY(gpu_search_ops, m) {
    m.def("i8dot512_scores", i8dot512_scores_cuda);
    m.def("build_pq_lut", build_pq_lut_cuda);
    m.def("adc_scan", adc_scan_cuda);
}