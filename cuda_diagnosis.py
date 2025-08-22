#!/usr/bin/env python3
"""
Comprehensive CUDA + LibTorch Diagnosis Script
Deep analysis of what's wrong with our current setup
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and return output"""
    print(f"\n🔍 {description}")
    print(f"   Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   ✅ Success:")
            for line in result.stdout.strip().split('\n')[:10]:  # Limit output
                print(f"      {line}")
            if len(result.stdout.strip().split('\n')) > 10:
                print(f"      ... ({len(result.stdout.strip().split('\n')) - 10} more lines)")
        else:
            print(f"   ❌ Failed (code {result.returncode}):")
            for line in result.stderr.strip().split('\n')[:5]:
                if line.strip():
                    print(f"      {line}")
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return "", str(e), -1

def check_file_exists(path, description):
    """Check if file exists and show info"""
    print(f"\n📁 {description}")
    print(f"   Path: {path}")
    
    if os.path.exists(path):
        stat = os.stat(path)
        print(f"   ✅ Exists - Size: {stat.st_size} bytes")
        if os.path.isfile(path):
            # Try to get file info
            run_command(f"file {path}", f"File type of {path}")
            if path.endswith('.so'):
                run_command(f"ldd {path} | head -10", f"Dependencies of {path}")
        return True
    else:
        print(f"   ❌ Does not exist")
        return False

def main():
    print("🚀 CUDA + LibTorch Deep Diagnosis")
    print("=" * 60)
    
    # 1. System CUDA check
    print("\n" + "=" * 60)
    print("1. SYSTEM CUDA ANALYSIS")
    print("=" * 60)
    
    run_command("nvidia-smi", "NVIDIA Driver Status")
    run_command("nvcc --version", "NVCC Compiler Version")
    run_command("cat /proc/driver/nvidia/version", "NVIDIA Driver Version")
    
    # Check CUDA installations
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.0", 
        "/usr/local/cuda-12.2",
        "/opt/cuda"
    ]
    
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            print(f"\n   ✅ Found CUDA installation: {cuda_path}")
            run_command(f"ls -la {cuda_path}/lib64/ | grep libcudart", f"CUDA Runtime in {cuda_path}")
        
    # 2. Current LibTorch Analysis  
    print("\n" + "=" * 60)
    print("2. CURRENT LIBTORCH ANALYSIS")
    print("=" * 60)
    
    libtorch_path = "/home/lee/code/gobed/libtorch"
    
    if check_file_exists(libtorch_path, "LibTorch Directory"):
        run_command(f"ls -la {libtorch_path}/lib/ | grep torch", "LibTorch Libraries")
        run_command(f"ls -la {libtorch_path}/lib/ | grep cuda", "CUDA Libraries in LibTorch")
        
        # Check specific files
        torch_lib = f"{libtorch_path}/lib/libtorch.so"
        torch_cuda_lib = f"{libtorch_path}/lib/libtorch_cuda.so"
        
        if check_file_exists(torch_lib, "Main LibTorch Library"):
            run_command(f"strings {torch_lib} | grep -i cuda | head -5", "CUDA strings in libtorch.so")
            run_command(f"objdump -p {torch_lib} | grep NEEDED | grep cuda", "CUDA dependencies in libtorch.so")
        
        if check_file_exists(torch_cuda_lib, "LibTorch CUDA Library"):
            run_command(f"ldd {torch_cuda_lib} | grep cuda", "CUDA dependencies of libtorch_cuda.so")
            run_command(f"nm -D {torch_cuda_lib} | grep cuda | head -5", "CUDA symbols in libtorch_cuda.so")
    
    # 3. Build Info Analysis
    print("\n" + "=" * 60)
    print("3. LIBTORCH BUILD ANALYSIS")
    print("=" * 60)
    
    build_info = f"{libtorch_path}/build-version"
    build_hash = f"{libtorch_path}/build-hash"
    
    check_file_exists(build_info, "Build Version File")
    check_file_exists(build_hash, "Build Hash File")
    
    # Try to determine LibTorch build type
    run_command(f"strings {libtorch_path}/lib/libtorch.so | grep -E '(CUDA|cpu|gpu)' | head -10", "Build type indicators")
    
    # 4. Test Basic CUDA
    print("\n" + "=" * 60)
    print("4. BASIC CUDA TEST")
    print("=" * 60)
    
    # Create simple CUDA test
    cuda_test_code = '''
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU 0: " << prop.name << std::endl;
        std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
        
        // Test memory allocation
        float *d_data;
        cudaError_t malloc_error = cudaMalloc(&d_data, 1000 * sizeof(float));
        std::cout << "Memory allocation: " << cudaGetErrorString(malloc_error) << std::endl;
        if (malloc_error == cudaSuccess) {
            cudaFree(d_data);
            std::cout << "CUDA basic test: PASSED" << std::endl;
        }
    }
    
    return 0;
}
'''
    
    with open('/tmp/cuda_test.cpp', 'w') as f:
        f.write(cuda_test_code)
    
    # Compile and run basic CUDA test
    run_command("g++ -I/usr/local/cuda-12.0/include -L/usr/local/cuda-12.0/targets/x86_64-linux/lib -lcudart /tmp/cuda_test.cpp -o /tmp/cuda_test", "Compile Basic CUDA Test")
    run_command("/tmp/cuda_test", "Run Basic CUDA Test")
    
    # 5. LibTorch CUDA Test
    print("\n" + "=" * 60)
    print("5. LIBTORCH CUDA TEST")
    print("=" * 60)
    
    torch_test_code = '''
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA Available: " << torch::cuda::is_available() << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA Device Count: " << torch::cuda::device_count() << std::endl;
        
        try {
            torch::Tensor x = torch::randn({2, 3}).cuda();
            std::cout << "GPU Tensor Test: SUCCESS" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU Tensor Test: FAILED - " << e.what() << std::endl;
        }
    } else {
        std::cout << "LibTorch CUDA not available - investigating..." << std::endl;
    }
    
    return 0;
}
'''
    
    with open('/tmp/torch_test.cpp', 'w') as f:
        f.write(torch_test_code)
    
    # Compile LibTorch test
    libtorch_includes = f"-I{libtorch_path}/include -I{libtorch_path}/include/torch/csrc/api/include"
    libtorch_libs = f"-L{libtorch_path}/lib -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda"
    cuda_libs = "-L/usr/local/cuda-12.0/targets/x86_64-linux/lib -lcudart"
    
    compile_cmd = f"g++ -std=c++17 {libtorch_includes} {libtorch_libs} {cuda_libs} /tmp/torch_test.cpp -o /tmp/torch_test"
    
    run_command(compile_cmd, "Compile LibTorch CUDA Test")
    
    if os.path.exists('/tmp/torch_test'):
        run_command(f"LD_LIBRARY_PATH={libtorch_path}/lib:/usr/local/cuda-12.0/targets/x86_64-linux/lib /tmp/torch_test", "Run LibTorch CUDA Test")
    
    # 6. Environment Analysis
    print("\n" + "=" * 60)
    print("6. ENVIRONMENT ANALYSIS")
    print("=" * 60)
    
    env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"   {var}: {value}")
    
    # 7. Recommendations
    print("\n" + "=" * 60)
    print("7. DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
Based on the analysis above, likely issues are:
1. LibTorch was built without proper CUDA support
2. CUDA version mismatch between LibTorch and system CUDA
3. Missing or incorrect library linking
4. Environment variables not set correctly

Next steps:
1. Download official CUDA-enabled LibTorch
2. Verify CUDA version compatibility  
3. Test minimal torch::cuda::is_available()
4. Rebuild our integration with working LibTorch
""")

if __name__ == "__main__":
    main()