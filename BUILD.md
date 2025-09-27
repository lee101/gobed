# GoBeD Build System

This document describes the comprehensive build system for GoBeD, including multi-platform builds, GPU acceleration, and CI/CD pipeline.

## Quick Start

```bash
# Run full CI pipeline locally
./scripts/ci_test.sh

# Or run individual components:
./scripts/detect_gpu.sh      # Detect GPU and set up environment
./scripts/run_all_tests.sh   # Run comprehensive test suite
./scripts/validate_build.sh  # Validate multi-platform builds
./scripts/performance_test.sh # Run performance benchmarks
```

## Build Matrix

The build system supports the following target platforms:

### CPU-Only Builds
- **linux-amd64-cpu** - Linux x86_64
- **linux-arm64-cpu** - Linux ARM64
- **darwin-amd64-cpu** - macOS Intel
- **darwin-arm64-cpu** - macOS Apple Silicon
- **windows-amd64-cpu** - Windows x86_64

### GPU-Accelerated Builds
- **linux-amd64-gpu** - Linux x86_64 with CUDA support

## GPU Detection and Setup

The build system automatically detects GPU capabilities:

```bash
# Run GPU detection
./scripts/detect_gpu.sh

# Source the generated environment
source gpu_env.sh

# Now build with GPU support
make gpu-build
```

### CUDA Compatibility

| CUDA Version | Supported GCC | Auto-Selected |
|--------------|---------------|---------------|
| 12.3+        | GCC 13        | gcc-13        |
| 12.0-12.2    | GCC 12        | gcc-12        |
| 11.x         | GCC 11        | gcc-11        |
| Fallback     | System GCC    | gcc           |

## Build Commands

### Local Development

```bash
# CPU build
make build

# GPU build (auto-detects CUDA)
make gpu-build

# Build all variants
make build-all

# Clean builds
make clean
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) provides:

1. **Multi-platform build matrix** - Builds for all supported platforms
2. **GPU acceleration testing** - Tests CUDA builds on Linux
3. **Comprehensive testing** - Unit tests, integration tests, benchmarks
4. **Code quality checks** - Linting, security scanning, formatting
5. **Performance validation** - Benchmark regression testing
6. **Release automation** - Automatic binary releases on tags

### Manual Build Validation

```bash
# Validate all build targets
./scripts/validate_build.sh

# This creates binaries in /tmp/gobed-builds/ and release archives
```

## Testing

### Test Data

The `testdata/` directory contains small test files for CI:
- `sample1.txt` - Machine learning text
- `sample2.txt` - Deep learning text
- `sample3.txt` - NLP text
- `sample4.txt` - Computer vision text
- `sample5.txt` - Vector database text

### Test Suite

```bash
# Run all tests
./scripts/run_all_tests.sh

# This includes:
# - Go unit tests (CPU + GPU)
# - Build validation
# - Binary functionality tests
# - Code quality checks
# - Security scans
# - Integration tests
```

### Performance Testing

```bash
# Run performance benchmarks
./scripts/performance_test.sh

# Tests:
# - Go benchmark suite
# - Build performance
# - Search performance (CPU vs GPU)
# - Memory usage analysis
# - Stress testing
```

## GitHub Actions Workflow

### Triggers
- Push to `main` or `develop`
- Pull requests to `main`
- Release publications
- Manual dispatch

### Jobs

1. **build-matrix** - Multi-platform builds with GPU variants
2. **test** - Comprehensive testing on Ubuntu
3. **quality** - Code quality and security checks
4. **gpu-test** - GPU-specific testing (mock/self-hosted)
5. **benchmark** - Performance benchmarking
6. **release** - Release asset creation

### Artifacts

Each build produces:
- Binary executables for target platform
- Test coverage reports
- Benchmark results
- Security scan results
- Release archives (on tags)

## Local Development Setup

1. **Install dependencies:**
   ```bash
   # Go 1.21+
   # CUDA Toolkit 12.x (for GPU builds)
   # GCC (version matching CUDA requirements)
   ```

2. **Set up environment:**
   ```bash
   git clone <repository>
   cd gobed
   ./scripts/detect_gpu.sh
   source gpu_env.sh
   ```

3. **Run tests:**
   ```bash
   ./scripts/run_all_tests.sh
   ```

4. **Build all variants:**
   ```bash
   ./scripts/validate_build.sh
   ```

## Continuous Integration

### Pull Request Workflow
1. All tests must pass
2. Code quality checks must pass
3. Build validation for all platforms
4. Performance regression checks

### Release Workflow
1. Tag creation triggers release build
2. All platforms built and tested
3. Release archives created automatically
4. GitHub release published with assets

## Troubleshooting

### Common Issues

**CUDA not detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA installation
ls /usr/local/cuda*/

# Re-run detection
./scripts/detect_gpu.sh
```

**Build failures:**
```bash
# Check Go version
go version

# Clean and rebuild
make clean
./scripts/validate_build.sh
```

**Test failures:**
```bash
# Run specific test categories
go test -v ./...                    # Unit tests
go test -tags gpu -v ./...          # GPU tests
go test -tags integration -v ./...  # Integration tests
```

### Debug Mode

Enable verbose output:
```bash
export DEBUG=true
export DEV=true
./scripts/run_all_tests.sh
```

## Contributing

When adding new features:

1. Update test data if needed (`testdata/`)
2. Add appropriate tests (unit, integration, performance)
3. Update build matrix if new platforms needed
4. Test with `./scripts/ci_test.sh`
5. Ensure all CI checks pass

The build system is designed to be:
- **Comprehensive** - Tests everything that could break
- **Fast** - Parallelized builds and cached dependencies
- **Reliable** - Extensive validation and error handling
- **Portable** - Works across all supported platforms