# Build & Test Setup

## Quick Commands

```bash
# Build
make build          # Build main binary
make build-all      # Build all binaries

# Test
make test           # Run all tests
make test-short     # Run short tests only
make test-race      # Test with race detector

# Code Quality
make fmt            # Format code
make vet            # Run go vet
make lint           # Run linters (requires golangci-lint)
make security       # Security scan

# Benchmarks
make bench          # Run benchmarks
make bench-cpu      # CPU profiling
make bench-mem      # Memory profiling

# Docker
make docker-build   # Build Docker image
make docker-run     # Run container
```

## Release Process

### Simple Version Bumping

```bash
# Bump version (creates and pushes git tag)
./scripts/bump-version.sh patch  # v1.0.0 -> v1.0.1
./scripts/bump-version.sh minor  # v1.0.0 -> v1.1.0
./scripts/bump-version.sh major  # v1.0.0 -> v2.0.0
```

When you push a tag starting with `v`, GitHub Actions will automatically:
- Build binaries for Linux, macOS, Windows (amd64/arm64)
- Create a GitHub release with changelog
- Build and push Docker images
- Publish the Go module

## GitHub Actions

All workflows run automatically on push/PR:

- **ci.yml**: Main build and test pipeline
- **quality.yml**: Code quality checks
- **benchmark.yml**: Performance benchmarks
- **release.yml**: Automated releases on tags

## Local Testing

Run these before pushing:

```bash
# Quick check
make fmt         # Format code
make test-short  # Quick tests
make vet         # Basic linting

# Full check
make build       # Ensure it builds
make test        # Run all tests
make lint        # Full linting (if golangci-lint installed)
```

## Known Issues

Some test files have build issues that need fixing:
- Multiple main functions in benchmark directories
- Some imports need updating

These don't affect the main library functionality.