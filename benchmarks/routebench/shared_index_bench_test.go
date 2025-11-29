package routebench

import (
	"math/rand"
	"testing"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func BenchmarkSharedMemorySearch100K(b *testing.B) {
	const (
		numVectors = 100000
		topK       = 50
	)

	config := gobed.SharedMemoryConfig{
		BasePath:    b.TempDir(),
		MaxVectors:  numVectors,
		CreateIfNew: true,
	}

	index, err := gobed.NewSharedMemoryIndex(config)
	if err != nil {
		b.Fatalf("NewSharedMemoryIndex failed: %v", err)
	}
	defer index.Close()

	var vec simd.Vec512
	for i := 0; i < numVectors; i++ {
		for j := range vec {
			vec[j] = int8(rand.Intn(255) - 128)
		}
		if err := index.AddVector(&vec, 1.0, i); err != nil {
			b.Fatalf("AddVector %d failed: %v", i, err)
		}
	}

	// Query vector
	var query simd.Vec512
	for i := range query {
		query[i] = int8(rand.Intn(255) - 128)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		results := index.SearchTopK(&query, topK)
		if len(results) > numVectors {
			b.Fatalf("unexpected results len: %d", len(results))
		}
	}
}
