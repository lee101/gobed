cmake_minimum_required(VERSION 3.18)

# Set CUDA host compiler to GCC 12
set(CMAKE_CUDA_HOST_COMPILER "g++-12")
# Allow unsupported compiler flag (not needed with GCC 12)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(gobed_ann_ops LANGUAGES CXX CUDA)

# Find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch")
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA settings with version compatibility
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Runtime Version: ${CUDA_VERSION_STRING}")

# Auto-detect GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Common architectures for different generations
    # 60,61: Pascal (GTX 10 series)  
    # 70,75: Volta/Turing (RTX 20 series)
    # 80,86: Ampere (RTX 30 series) 
    # 89,90: Ada Lovelace/Hopper (RTX 40 series)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
    message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Version-specific flags
if(CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    message(STATUS "CUDA 12.x: Enabled extended lambda support")
endif()

if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "CUDA 11.x+: Enabled relaxed constexpr")
endif()

# Common optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")  # For better debugging

# Create shared library
add_library(gobed_ann_ops SHARED 
    ops_cuda.cpp 
    i8_dot512.cu 
    pq_ops.cu
    search_ops.cu
    version_check.cu
)

# Include directories
target_include_directories(gobed_ann_ops PRIVATE ${TORCH_INCLUDE_DIRS})

# Link libraries
target_link_libraries(gobed_ann_ops "${TORCH_LIBRARIES}")

# Set properties
set_property(TARGET gobed_ann_ops PROPERTY CXX_STANDARD 17)
set_property(TARGET gobed_ann_ops PROPERTY CUDA_ARCHITECTURES 86)  # RTX 3080

# Install
install(TARGETS gobed_ann_ops DESTINATION lib)