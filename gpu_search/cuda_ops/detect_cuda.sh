#!/bin/bash

# Robust CUDA detection script that prioritizes newer versions
# Returns CUDA path and version information

detect_cuda() {
    local cuda_paths=()
    local cuda_versions=()
    
    # Check standard locations for CUDA installations
    for dir in /usr/local/cuda* /opt/cuda* /usr/lib/cuda*; do
        if [ -d "$dir" ] && [ -f "$dir/bin/nvcc" ]; then
            cuda_paths+=("$dir")
            # Extract version from nvcc
            version=$("$dir/bin/nvcc" --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
            if [ -n "$version" ]; then
                cuda_versions+=("$version")
            fi
        fi
    done
    
    # Sort by version (newest first)
    if [ ${#cuda_paths[@]} -gt 0 ]; then
        # Create array of indices sorted by version
        local sorted_indices=($(
            for i in "${!cuda_versions[@]}"; do
                echo "$i ${cuda_versions[$i]}"
            done | sort -t. -k1,1nr -k2,2nr | awk '{print $1}'
        ))
        
        # Return the newest version path
        if [ ${#sorted_indices[@]} -gt 0 ]; then
            local best_idx=${sorted_indices[0]}
            echo "${cuda_paths[$best_idx]}"
            return 0
        fi
    fi
    
    # Fallback to environment variable
    if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
        echo "$CUDA_HOME"
        return 0
    fi
    
    # Check if nvcc is in PATH
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(which nvcc)
        local cuda_dir=$(dirname $(dirname "$nvcc_path"))
        if [ -d "$cuda_dir" ]; then
            echo "$cuda_dir"
            return 0
        fi
    fi
    
    return 1
}

get_cuda_version() {
    local cuda_path=$1
    if [ -f "$cuda_path/bin/nvcc" ]; then
        "$cuda_path/bin/nvcc" --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'
    fi
}

get_cuda_arch_flags() {
    local cuda_version=$1
    local arch_flags=""
    
    # Version-specific architecture support (6.1+ for __dp4a support)
    case "$cuda_version" in
        11.*)
            # CUDA 11.x supports up to compute capability 8.6 (RTX 30 series)
            arch_flags="61;70;75;80;86"
            ;;
        12.[0-2])
            # CUDA 12.0-12.2 adds support for Hopper (9.0)
            arch_flags="61;70;75;80;86;89;90"
            ;;
        12.[3-9]|12.1[0-9])
            # CUDA 12.3+ adds Ada Lovelace optimizations
            arch_flags="61;70;75;80;86;89;90"
            ;;
        *)
            # Conservative default (6.1+ required)
            arch_flags="61;70;75;80"
            ;;
    esac
    
    echo "$arch_flags"
}

get_compatible_gcc() {
    local cuda_version=$1
    local gcc_compiler=""
    
    # CUDA version to GCC compatibility
    case "$cuda_version" in
        11.[0-3])
            # CUDA 11.0-11.3: GCC <= 10
            for gcc in gcc-10 gcc-9 gcc-8 gcc; do
                if command -v $gcc &> /dev/null; then
                    gcc_compiler=$gcc
                    break
                fi
            done
            ;;
        11.[4-7])
            # CUDA 11.4-11.7: GCC <= 11
            for gcc in gcc-11 gcc-10 gcc-9 gcc; do
                if command -v $gcc &> /dev/null; then
                    gcc_compiler=$gcc
                    break
                fi
            done
            ;;
        11.8|12.[0-2])
            # CUDA 11.8, 12.0-12.2: GCC <= 12
            for gcc in gcc-12 gcc-11 gcc-10 gcc; do
                if command -v $gcc &> /dev/null; then
                    gcc_compiler=$gcc
                    break
                fi
            done
            ;;
        12.[3-9]|12.1[0-9])
            # CUDA 12.3+: GCC <= 13
            for gcc in gcc-13 gcc-12 gcc-11 gcc; do
                if command -v $gcc &> /dev/null; then
                    gcc_compiler=$gcc
                    break
                fi
            done
            ;;
        *)
            # Default to newest available
            for gcc in gcc-13 gcc-12 gcc-11 gcc-10 gcc; do
                if command -v $gcc &> /dev/null; then
                    gcc_compiler=$gcc
                    break
                fi
            done
            ;;
    esac
    
    echo "$gcc_compiler"
}

# Main execution
if [ "$1" == "--info" ]; then
    CUDA_PATH=$(detect_cuda)
    if [ $? -eq 0 ]; then
        CUDA_VERSION=$(get_cuda_version "$CUDA_PATH")
        ARCH_FLAGS=$(get_cuda_arch_flags "$CUDA_VERSION")
        GCC_COMPILER=$(get_compatible_gcc "$CUDA_VERSION")
        
        echo "CUDA_PATH=$CUDA_PATH"
        echo "CUDA_VERSION=$CUDA_VERSION"
        echo "CUDA_ARCHITECTURES=$ARCH_FLAGS"
        echo "COMPATIBLE_GCC=$GCC_COMPILER"
        echo ""
        echo "Detected CUDA $CUDA_VERSION at $CUDA_PATH"
        echo "Using GCC: $GCC_COMPILER"
        echo "Target architectures: $ARCH_FLAGS"
    else
        echo "ERROR: No CUDA installation found"
        exit 1
    fi
elif [ "$1" == "--path" ]; then
    detect_cuda
elif [ "$1" == "--version" ]; then
    CUDA_PATH=$(detect_cuda)
    if [ $? -eq 0 ]; then
        get_cuda_version "$CUDA_PATH"
    fi
elif [ "$1" == "--gcc" ]; then
    CUDA_PATH=$(detect_cuda)
    if [ $? -eq 0 ]; then
        CUDA_VERSION=$(get_cuda_version "$CUDA_PATH")
        get_compatible_gcc "$CUDA_VERSION"
    fi
else
    echo "Usage: $0 [--info|--path|--version|--gcc]"
    echo "  --info    : Show all CUDA information"
    echo "  --path    : Return CUDA installation path"
    echo "  --version : Return CUDA version"
    echo "  --gcc     : Return compatible GCC version"
fi