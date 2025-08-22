# GitHub Actions CI/CD Configuration

This directory contains the CI/CD pipeline configuration for the gobed project.

## Workflows

### 1. **CI Workflow** (`ci.yml`)
**Trigger:** Push to main/develop, Pull requests
**Purpose:** Main build and test pipeline

- ✅ Multi-OS build (Ubuntu, macOS)
- ✅ Multi-Go version testing (1.20, 1.21)
- ✅ Unit tests with race detection
- ✅ Code coverage reporting to Codecov
- ✅ Security scanning with gosec
- ✅ Vulnerability checking with govulncheck
- ✅ Integration tests (when model files available)
- ✅ Docker build verification

### 2. **Code Quality** (`quality.yml`)
**Trigger:** Pull requests
**Purpose:** Comprehensive code quality checks

- 📝 Format checking (gofmt, goimports)
- 🔍 Static analysis (staticcheck, ineffassign)
- 📖 Spell checking (misspell)
- 🔄 Cyclomatic complexity analysis
- 👥 Duplicate code detection
- 📊 Generates quality summary report

### 3. **Benchmarks** (`benchmark.yml`)
**Trigger:** Push to main, Daily schedule
**Purpose:** Performance tracking and regression detection

- 🚀 CPU performance benchmarks
- 💾 Memory usage benchmarks
- 🔍 Search performance testing
- ⚡ Parallel processing benchmarks
- 📈 Historical performance tracking
- ⚠️ Regression alerts (150% threshold)

### 4. **Release** (`release.yml`)
**Trigger:** Version tags (v*), Manual dispatch
**Purpose:** Automated release process

- 📦 Multi-platform binary builds (Linux, macOS, Windows)
- 🐳 Docker image creation and push
- 📝 Automatic changelog generation
- 🏷️ GitHub release creation
- 📚 Go module publishing to pkg.go.dev

## Additional Configuration

### Dependabot (`dependabot.yml`)
- 🔄 Weekly Go module updates
- 🔄 Weekly GitHub Actions updates
- 🔒 Daily security updates
- 🤖 Automatic PR creation for updates

### golangci-lint (`.golangci.yml`)
Comprehensive linting configuration with:
- 25+ enabled linters
- Custom rules for code quality
- Performance-focused checks
- Security scanning integration

### Docker (`Dockerfile`)
Multi-stage build for minimal production images:
- Alpine-based final image
- Non-root user execution
- Optimized binary size
- Support for multiple architectures

### Makefile
Developer-friendly build automation:
```bash
make build       # Build main binary
make test        # Run tests
make bench       # Run benchmarks
make lint        # Run linters
make docker-build # Build Docker image
make ci          # Run full CI locally
```

## Status Badges

Add these to your README.md:

```markdown
![CI](https://github.com/lee101/gobed/workflows/CI/badge.svg)
![Code Quality](https://github.com/lee101/gobed/workflows/Code%20Quality/badge.svg)
![Benchmarks](https://github.com/lee101/gobed/workflows/Benchmarks/badge.svg)
[![codecov](https://codecov.io/gh/lee101/gobed/branch/main/graph/badge.svg)](https://codecov.io/gh/lee101/gobed)
```

## Local CI Testing

Run CI checks locally before pushing:

```bash
# Quick CI check
make ci

# Full CI check (includes race detection and coverage)
make ci-full

# Individual checks
make fmt         # Format code
make lint        # Run linters
make test-race   # Test with race detector
make security    # Security scan
```

## Secrets Required

For full functionality, configure these GitHub secrets:

- `CODECOV_TOKEN` - For coverage reporting (optional, public repos work without it)
- `DOCKER_USERNAME` - Docker Hub username (optional)
- `DOCKER_PASSWORD` - Docker Hub password (optional)

The `GITHUB_TOKEN` is automatically provided by GitHub Actions.

## Performance Tracking

Benchmarks run daily and on pushes to main. Results are:
- Stored as artifacts
- Tracked for regressions
- Available in the Actions tab under "Benchmarks"

## Security

Security scanning includes:
- gosec static security analysis
- govulncheck for known vulnerabilities
- Dependabot for dependency updates
- SARIF reports uploaded to GitHub Security tab

## Maintenance

### Updating Go Version
Update `GO_VERSION` in:
- `.github/workflows/ci.yml`
- `.github/workflows/quality.yml`
- `.github/workflows/benchmark.yml`
- `.github/workflows/release.yml`
- `Dockerfile`

### Adding New Checks
1. Add to appropriate workflow file
2. Update Makefile target
3. Document in this README

## Troubleshooting

### Build Failures
- Check Go version compatibility
- Verify dependencies with `go mod tidy`
- Check for duplicate main functions

### Test Failures
- Run `make test-short` for quick validation
- Check race conditions with `make test-race`
- Verify model files are present for integration tests

### Benchmark Issues
- Ensure consistent environment (use Docker)
- Check for background processes affecting results
- Review benchmark history in GitHub Actions