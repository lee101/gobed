cmake_minimum_required(VERSION 3.18)

# Allow setting CUDA compiler from environment
if(DEFINED ENV{CUDAHOSTCXX})
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}")
endif()

project(gobed_ann_ops LANGUAGES CXX CUDA)

# Find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch")
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA version detection and configuration
message(STATUS "========================================")
message(STATUS "CUDA Configuration:")
message(STATUS "  CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "  CUDA Toolkit Root: ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")

# Set architecture if passed from command line, otherwise use defaults
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
    # Auto-detect based on CUDA version (excluding 5.0 which doesn't support __dp4a)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
        set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86;89;90")
        message(STATUS "  CUDA 12.x detected: Including Hopper support (6.1+)")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86;89")
        message(STATUS "  CUDA 11.8+ detected: Including Ada Lovelace support (6.1+)")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86")
        message(STATUS "  CUDA 11.x detected: Up to Ampere support (6.1+)")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80")
        message(STATUS "  Older CUDA detected: Conservative architecture set (6.1+)")
    endif()
endif()

message(STATUS "  Target Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "========================================")

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Version-specific optimizations
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    message(STATUS "Enabled CUDA 12.x extended lambda support")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "Enabled CUDA 11.x+ relaxed constexpr")
endif()

# Handle potential version file if it exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version_check.cu")
    # Create a simple version check file
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/version_check.cu" 
"#include <cuda_runtime.h>
#include <iostream>

extern \"C\" {
    int get_cuda_runtime_version() {
        int version;
        cudaRuntimeGetVersion(&version);
        return version;
    }
    
    int get_compute_capability(int device_id = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        return prop.major * 10 + prop.minor;
    }
}
")
endif()

# Create shared library
set(CUDA_SOURCES
    i8_dot512.cu 
    pq_ops.cu
    search_ops.cu
    version_check.cu
)

set(CPP_SOURCES
    ops_cuda.cpp
)

add_library(gobed_ann_ops SHARED ${CPP_SOURCES} ${CUDA_SOURCES})

# Include directories
target_include_directories(gobed_ann_ops PRIVATE ${TORCH_INCLUDE_DIRS})

# Link libraries
target_link_libraries(gobed_ann_ops "${TORCH_LIBRARIES}")

# Set target properties
set_target_properties(gobed_ann_ops PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS gobed_ann_ops DESTINATION lib)

# Print summary
message(STATUS "")
message(STATUS "Build Configuration Summary:")
message(STATUS "  Library: gobed_ann_ops")
message(STATUS "  Type: SHARED")
message(STATUS "  C++ Standard: 17")
message(STATUS "  CUDA Standard: 17")
message(STATUS "  Optimization: Release")
message(STATUS "")