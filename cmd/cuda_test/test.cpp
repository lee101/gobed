#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "=== LibTorch CUDA Test ===" << std::endl;
    
    // Check LibTorch version
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    
    // Check CUDA runtime
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA runtime device count: " << deviceCount << std::endl;
    if (error != cudaSuccess) {
        std::cout << "CUDA runtime error: " << cudaGetErrorString(error) << std::endl;
    }
    
    // Check LibTorch CUDA
    std::cout << "LibTorch CUDA available: " << torch::cuda::is_available() << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "LibTorch CUDA device count: " << torch::cuda::device_count() << std::endl;
        
        // Test GPU tensor creation
        try {
            torch::Tensor x = torch::randn({3, 3}, torch::TensorOptions().device(torch::kCUDA));
            std::cout << "GPU tensor created successfully!" << std::endl;
            std::cout << "Tensor device: " << x.device() << std::endl;
            std::cout << "Tensor data:\n" << x << std::endl;
            
            // Test GPU computation
            torch::Tensor y = torch::randn({3, 3}, torch::TensorOptions().device(torch::kCUDA));
            torch::Tensor z = torch::mm(x, y);
            std::cout << "GPU matrix multiplication successful!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "GPU tensor error: " << e.what() << std::endl;
        }
    } else {
        std::cout << "LibTorch CUDA NOT available - investigating..." << std::endl;
        
        // More detailed CUDA investigation
        std::cout << "\nDetailed CUDA Investigation:" << std::endl;
        
        // Check CUDA driver
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        std::cout << "CUDA driver version: " << driverVersion << std::endl;
        
        // Check CUDA runtime version  
        int runtimeVersion = 0;
        cudaRuntimeGetVersion(&runtimeVersion);
        std::cout << "CUDA runtime version: " << runtimeVersion << std::endl;
        
        // Device properties
        if (deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "GPU 0: " << prop.name << std::endl;
            std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        }
    }
    
    return 0;
}