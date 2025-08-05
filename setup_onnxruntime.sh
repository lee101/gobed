#!/bin/bash

# Script to download and install the correct ONNX Runtime library
# for gobed project

set -e

ONNX_VERSION="1.22.0"
OS="linux"
ARCH="x64"

# Determine if GPU or CPU version is needed
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, downloading GPU version..."
    GPU_SUFFIX="-gpu"
else
    echo "No NVIDIA GPU detected, downloading CPU version..."
    GPU_SUFFIX=""
fi

# Create download directory
mkdir -p /tmp/onnxruntime_download
cd /tmp/onnxruntime_download

# Download the appropriate ONNX Runtime package
PACKAGE_NAME="onnxruntime-${OS}-${ARCH}${GPU_SUFFIX}-${ONNX_VERSION}"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${PACKAGE_NAME}.tgz"

echo "Downloading ${PACKAGE_NAME}..."
wget -O "${PACKAGE_NAME}.tgz" "${DOWNLOAD_URL}"

# Extract the package
echo "Extracting package..."
tar -xzf "${PACKAGE_NAME}.tgz"

# Find the library file
LIB_FILE=$(find "${PACKAGE_NAME}" -name "libonnxruntime.so*" | head -1)

if [ -z "$LIB_FILE" ]; then
    echo "Error: Could not find libonnxruntime.so in the package"
    exit 1
fi

echo "Found library: ${LIB_FILE}"

# Install to system location
INSTALL_DIR="/usr/local/lib"
if [ ! -w "${INSTALL_DIR}" ]; then
    echo "Installing library to ${INSTALL_DIR} (requires sudo)..."
    sudo mkdir -p "${INSTALL_DIR}"
    sudo cp "${LIB_FILE}" "${INSTALL_DIR}/"
    sudo chmod 755 "${INSTALL_DIR}/$(basename ${LIB_FILE})"
    
    # Create symlink without version suffix if needed
    BASENAME=$(basename "${LIB_FILE}")
    if [[ "${BASENAME}" == *".so."* ]]; then
        SYMLINK_NAME="${BASENAME%%.so.*}.so"
        sudo ln -sf "${BASENAME}" "${INSTALL_DIR}/${SYMLINK_NAME}"
        echo "Created symlink: ${INSTALL_DIR}/${SYMLINK_NAME} -> ${BASENAME}"
    fi
else
    echo "Installing library to ${INSTALL_DIR}..."
    cp "${LIB_FILE}" "${INSTALL_DIR}/"
    chmod 755 "${INSTALL_DIR}/$(basename ${LIB_FILE})"
fi

# If GPU version, also copy CUDA provider library
if [ -n "$GPU_SUFFIX" ]; then
    CUDA_LIB=$(find "${PACKAGE_NAME}" -name "libonnxruntime_providers_cuda.so*" | head -1)
    if [ -n "$CUDA_LIB" ]; then
        echo "Installing CUDA provider library..."
        if [ ! -w "${INSTALL_DIR}" ]; then
            sudo cp "${CUDA_LIB}" "${INSTALL_DIR}/"
            sudo chmod 755 "${INSTALL_DIR}/$(basename ${CUDA_LIB})"
        else
            cp "${CUDA_LIB}" "${INSTALL_DIR}/"
            chmod 755 "${INSTALL_DIR}/$(basename ${CUDA_LIB})"
        fi
    fi
fi

# Update ldconfig
if command -v ldconfig &> /dev/null; then
    echo "Updating library cache..."
    if [ ! -w "${INSTALL_DIR}" ]; then
        sudo ldconfig
    else
        ldconfig
    fi
fi

echo "Installation complete!"
echo "Library installed to: ${INSTALL_DIR}/$(basename ${LIB_FILE})"

# Update the library path in main.go
cd - > /dev/null
MAIN_GO_PATH="$(pwd)/main.go"
if [ -f "${MAIN_GO_PATH}" ]; then
    echo "Updating library path in main.go..."
    sed -i "s|onnxruntime.SetSharedLibraryPath.*|onnxruntime.SetSharedLibraryPath(\"${INSTALL_DIR}/$(basename ${LIB_FILE})\")|" "${MAIN_GO_PATH}"
    echo "Updated main.go with correct library path"
fi

# Clean up
rm -rf /tmp/onnxruntime_download

echo ""
echo "Setup complete! You can now run your Go application."
echo "Library path: ${INSTALL_DIR}/$(basename ${LIB_FILE})"

# Check CUDA setup if GPU version
if [ -n "$GPU_SUFFIX" ]; then
    echo ""
    echo "GPU version installed. Checking CUDA setup..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
    else
        echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    fi
    
    echo ""
    echo "Make sure the following CUDA libraries are available:"
    echo "- libcudnn.so"
    echo "- libcublas.so"
    echo "- libcurand.so"
    echo "- libcufft.so"
fi
