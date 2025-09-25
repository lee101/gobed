// cuda_memory.cu - LibTorch-free CUDA memory management
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_ops.h"

// Memory management functions
extern "C" cuda_op_result_t cuda_malloc(void** ptr, size_t size) {
    if (!ptr || size == 0) {
        return CUDA_OP_ERROR_INVALID_ARGS;
    }

    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_MEMORY;
    }

    return CUDA_OP_SUCCESS;
}

extern "C" cuda_op_result_t cuda_free(void* ptr) {
    if (!ptr) {
        return CUDA_OP_ERROR_INVALID_ARGS;
    }

    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_MEMORY;
    }

    return CUDA_OP_SUCCESS;
}

extern "C" cuda_op_result_t cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        return CUDA_OP_ERROR_INVALID_ARGS;
    }

    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_MEMORY;
    }

    return CUDA_OP_SUCCESS;
}

extern "C" cuda_op_result_t cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        return CUDA_OP_ERROR_INVALID_ARGS;
    }

    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_MEMORY;
    }

    return CUDA_OP_SUCCESS;
}

extern "C" cuda_op_result_t cuda_memset(void* ptr, int value, size_t size) {
    if (!ptr || size == 0) {
        return CUDA_OP_ERROR_INVALID_ARGS;
    }

    cudaError_t err = cudaMemset(ptr, value, size);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_MEMORY;
    }

    return CUDA_OP_SUCCESS;
}

// Device management functions
extern "C" cuda_op_result_t cuda_set_device(int device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_DEVICE;
    }

    return CUDA_OP_SUCCESS;
}

extern "C" cuda_op_result_t cuda_get_device_count(int* count) {
    if (!count) {
        return CUDA_OP_ERROR_INVALID_ARGS;
    }

    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_DEVICE;
    }

    return CUDA_OP_SUCCESS;
}

extern "C" cuda_op_result_t cuda_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return CUDA_OP_ERROR_CUDA_RUNTIME;
    }

    return CUDA_OP_SUCCESS;
}