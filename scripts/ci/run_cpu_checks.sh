#!/usr/bin/env bash
#
# Lightweight CI entrypoint for CPU-only environments (GitHub hosted runners).
# Focuses on core packages that compile without CUDA or custom dependencies.

set -euo pipefail

echo "🔧 Running gobed CPU checks"
echo "Go version: $(go version)"

PACKAGES_UNDER_TEST=(
  "github.com/lee101/gobed"
  "github.com/lee101/gobed/ann/simd"
)

for pkg in "${PACKAGES_UNDER_TEST[@]}"; do
  echo "🧪 go test ${pkg}"
  go test -v "${pkg}"
done

echo "🔍 go vet github.com/lee101/gobed"
go vet github.com/lee101/gobed

echo "✅ CPU checks completed"
