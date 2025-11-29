package routebench

import (
	"fmt"
	"sync"
	"testing"

	"github.com/lee101/gobed"
)

var (
	loadBaseModelOnce sync.Once
	baseModel         *gobed.EmbeddingModel
	baseModelErr      error
)

func getBaseModel(b *testing.B) *gobed.EmbeddingModel {
	loadBaseModelOnce.Do(func() {
		baseModel, baseModelErr = gobed.LoadModel()
	})
	if baseModelErr != nil {
		b.Fatalf("LoadModel failed: %v", baseModelErr)
	}
	return baseModel
}

// BenchmarkEmbedInt8 measures the baseline EmbedInt8 latency/allocations.
func BenchmarkEmbedInt8(b *testing.B) {
	model := getBaseModel(b)

	queries := make([]string, b.N)
	for i := range queries {
		queries[i] = fmt.Sprintf("anime girl with cyberpunk neon lights %d", i)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := model.EmbedInt8(queries[i]); err != nil {
			b.Fatalf("EmbedInt8 failed: %v", err)
		}
	}
}
