package gobed

import (
	"fmt"
	"testing"
)

func benchmarkAppendEngine(b *testing.B) *SearchEngine {
	b.Helper()
	model := mustLoadModelForBenchmark(b)
	config := DefaultSearchConfig()
	config.AutoMode = false
	config.EnableGPU = false
	config.MaxExactSearchSize = 100000000
	return NewSearchEngineWithConfig(model, config)
}

func BenchmarkPrepareAppendBatch(b *testing.B) {
	for _, size := range []int{1, 4, 8, 16} {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			engine := benchmarkAppendEngine(b)
			ids := make([]int, size)
			texts := make([]string, size)
			for i := range ids {
				ids[i] = i
				texts[i] = fmt.Sprintf("generated asset class %d", i)
			}
			if _, err := engine.PrepareAppendBatchWithIDs(ids, texts); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := engine.PrepareAppendBatchWithIDs(ids, texts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAppendBatch(b *testing.B) {
	for _, size := range []int{1, 4, 8, 16} {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			engine := benchmarkAppendEngine(b)
			texts := make([]string, size)
			for i := range texts {
				texts[i] = fmt.Sprintf("generated asset class %d", i)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				ids := make([]int, size)
				for i := range ids {
					ids[i] = n*size + i
				}
				if err := engine.AppendBatchWithIDs(ids, texts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
