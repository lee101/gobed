//go:build legacy && cgo && cuda && gpu && cagra
// +build legacy,cgo,cuda,gpu,cagra

package cagra_gpu_bench

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

const maxDatasetSize = 100000
const queryBatchSize = 256

var (
	loadModelOnce sync.Once
	benchModel    *gobed.Int8EmbeddingModel512
	modelErr      error

	datasetOnce   sync.Once
	datasetVecs   []simd.Vec512
	datasetScales []float32
	datasetErr    error

	chdirOnce sync.Once
	chdirErr  error
)

func getModel(b *testing.B) *gobed.Int8EmbeddingModel512 {
	loadModelOnce.Do(func() {
		benchModel, modelErr = gobed.LoadInt8Model512()
	})
	if modelErr != nil {
		b.Fatalf("LoadInt8Model512 failed: %v", modelErr)
	}
	return benchModel
}

func ensureDataset(b *testing.B, size int) {
	if size > maxDatasetSize {
		b.Fatalf("dataset size %d exceeds supported maximum %d", size, maxDatasetSize)
	}
	datasetOnce.Do(func() {
		ensureRepoRoot()
		model := getModel(b)
		datasetVecs = make([]simd.Vec512, maxDatasetSize)
		datasetScales = make([]float32, maxDatasetSize)

		tokenBuf := make([]int16, 0, 4)
		for i := 0; i < maxDatasetSize; i++ {
			tokenBuf = tokensForDoc(tokenBuf[:0], i)
			floatVec, err := model.EmbedTokens(tokenBuf)
			if err != nil {
				datasetErr = fmt.Errorf("EmbedTokens doc %d failed: %w", i, err)
				return
			}
			int8Vec, scale := quantizeToInt8(floatVec)
			copy(datasetVecs[i][:], int8Vec)
			datasetScales[i] = scale
		}
	})

	if datasetErr != nil {
		b.Fatalf("dataset generation failed: %v", datasetErr)
	}
	if chdirErr != nil {
		b.Fatalf("failed to change working directory: %v", chdirErr)
	}
}

func ensureRepoRoot() {
	chdirOnce.Do(func() {
		_, file, _, ok := runtime.Caller(0)
		if !ok {
			chdirErr = fmt.Errorf("runtime.Caller lookup failed")
			return
		}
		root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
		if err := os.Chdir(root); err != nil {
			chdirErr = err
		}
	})
}

func tokensForDoc(buf []int16, docID int) []int16 {
	vocab := gobed.Int8VocabSize
	base := int16(docID % vocab)
	buf = append(buf, base)

	if docID >= vocab {
		buf = append(buf, int16((docID*31)%vocab))
	}
	if docID >= 2*vocab {
		buf = append(buf, int16((docID*17)%vocab))
	}
	if docID >= 3*vocab {
		buf = append(buf, int16((docID*7)%vocab))
	}
	return buf
}

func quantizeToInt8(vec []float32) ([]int8, float32) {
	maxAbs := float32(0)
	for _, v := range vec {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}
	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1.0
	}

	out := make([]int8, len(vec))
	inv := 1 / scale
	for i, v := range vec {
		q := int(math.Round(float64(v * inv)))
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		out[i] = int8(q)
	}
	return out, scale
}

func BenchmarkCAGRASearch(b *testing.B) {
	sizes := []int{1024, 5000, 10000, 20000, 50000, 100000}
	const topK = 10

	ensureDataset(b, sizes[len(sizes)-1])

	queries := datasetVecs[:queryBatchSize]

	for _, size := range sizes {
		size := size
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			config := gobed.DefaultCAGRAConfig()
			config.MaxVectors = size
			config.CachePath = ""

			index, err := gobed.NewCAGRAIndex(config)
			if err != nil {
				b.Fatalf("NewCAGRAIndex: %v", err)
			}
			defer index.Close()

			if err := index.BuildIndex(datasetVecs[:size], datasetScales[:size]); err != nil {
				b.Fatalf("BuildIndex(%d): %v", size, err)
			}

			b.ResetTimer()
			lastID := -1
			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				results, err := index.Search(q, 1.0, topK)
				if err != nil {
					b.Fatalf("Search: %v", err)
				}
				if len(results) > 0 {
					lastID = results[0].ID
				}
			}
			b.StopTimer()

			if lastID < 0 {
				b.Fatalf("unexpected result id: %d", lastID)
			}

			elapsed := b.Elapsed()
			totalQueries := float64(b.N)
			if elapsed > 0 {
				qps := totalQueries / elapsed.Seconds()
				b.ReportMetric(qps, "qps")
				avgLatency := elapsed / time.Duration(b.N)
				b.ReportMetric(float64(avgLatency.Microseconds()), "us_per_query")
			}
			b.ReportMetric(float64(size), "dataset_size")

			getModel(b) // ensure model stays referenced for the lifetime of the benchmark
		})
	}
}
