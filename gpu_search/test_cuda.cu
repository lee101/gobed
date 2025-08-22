#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    printf("CUDA kernel running on thread %d\n", threadIdx.x);
}

int main() {
    printf("Testing CUDA...\n");
    test_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    printf("✅ CUDA works!\n");
    return 0;
}
