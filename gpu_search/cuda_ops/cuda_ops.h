#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    CUDA_OP_SUCCESS = 0,
    CUDA_OP_ERROR_INVALID_ARGS = 1,
    CUDA_OP_ERROR_CUDA_RUNTIME = 2,
    CUDA_OP_ERROR_MEMORY = 3,
    CUDA_OP_ERROR_DEVICE = 4
} cuda_op_result_t;

// Memory management
cuda_op_result_t cuda_malloc(void** ptr, size_t size);
cuda_op_result_t cuda_free(void* ptr);
cuda_op_result_t cuda_memcpy_h2d(void* dst, const void* src, size_t size);
cuda_op_result_t cuda_memcpy_d2h(void* dst, const void* src, size_t size);
cuda_op_result_t cuda_memset(void* ptr, int value, size_t size);

// Device management
cuda_op_result_t cuda_set_device(int device);
cuda_op_result_t cuda_get_device_count(int* count);
cuda_op_result_t cuda_synchronize();

// Core operations
cuda_op_result_t i8dot512_scores(
    const int8_t* q,        // query vector [512]
    const int8_t* db,       // database vectors [N, 512]
    int32_t* out,          // output scores [N]
    int64_t N              // number of database vectors
);

cuda_op_result_t i8dot512_batch(
    const int8_t* queries,  // batch of queries [B, 512]
    const int8_t* db,       // database vectors [N, 512]
    int32_t* out,          // output scores [B, N]
    int64_t B,             // batch size
    int64_t N              // number of database vectors
);

cuda_op_result_t build_pq_lut(
    const float* q_rot,     // rotated query [D]
    const float* cb,        // codebook [M, K, D/M]
    float* lut,            // output lookup table [M, K]
    int D,                 // dimension
    int M,                 // number of subspaces
    int K                  // codebook size
);

cuda_op_result_t adc_scan(
    const uint8_t* codes,   // PQ codes [N, M]
    const float* lut,       // lookup table [M, K]
    float* out,            // output distances [N]
    int64_t N,             // number of vectors
    int M,                 // number of subspaces
    int K                  // codebook size
);

cuda_op_result_t gather_ivf_codes(
    const uint8_t* codes,      // input codes [total_size, M]
    const int64_t* ids,        // vector IDs [total_size]
    const int32_t* list_ids,   // list IDs for each vector [total_size]
    const int64_t* list_offsets, // offsets for each list [num_lists+1]
    uint8_t* out_codes,        // output codes [total_size, M]
    int64_t* out_ids,          // output IDs [total_size]
    int64_t total_size,        // total number of vectors
    int M,                     // code dimension
    int num_lists              // number of IVF lists
);

// Capability and version checking
cuda_op_result_t check_cuda_capabilities(
    int* compute_major,
    int* compute_minor,
    int* cuda_version
);

cuda_op_result_t test_version_compatibility(
    const int8_t* db,       // test database [1024, 512]
    const int8_t* query,    // test query [512]
    int32_t* result,       // output result [1024]
    float* benchmark_time  // benchmark time in ms
);

#ifdef __cplusplus
}
#endif