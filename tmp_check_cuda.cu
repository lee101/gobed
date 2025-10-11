#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("deviceCount=%d\n", deviceCount);
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    void* ptr = NULL;
    err = cudaMalloc(&ptr, 1024);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaMalloc success\n");
    cudaFree(ptr);
    return 0;
}
