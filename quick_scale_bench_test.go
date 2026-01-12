package gobed

import (
	"fmt"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func generateQuickBenchVectors(n int) ([]simd.Vec512, []float32, []int) {
	vectors := make([]simd.Vec512, n)
	scales := make([]float32, n)
	ids := make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8((i*j + j) % 256 - 128)
		}
		scales[i] = 1.0
		ids[i] = i
	}
	return vectors, scales, ids
}

func TestQuickScaleBenchmark(t *testing.T) {
	scales := []struct {
		name  string
		count int
	}{
		{"10K", 10000},
		{"50K", 50000},
		{"100K", 100000},
		{"500K", 500000},
		{"1M", 1000000},
	}

	fmt.Printf("\n=== Quick Scale Benchmark (Flat Index) ===\n")
	fmt.Printf("%-8s %-12s %-15s %-12s %-10s %-10s\n",
		"Scale", "Alloc(ms)", "Vec/sec", "Search(us)", "QPS", "Mem(MB)")

	for _, s := range scales {
		t.Run(s.name, func(t *testing.T) {
			runtime.GC()
			var memBefore runtime.MemStats
			runtime.ReadMemStats(&memBefore)

			allocStart := time.Now()
			vectors, scales, _ := generateQuickBenchVectors(s.count)
			allocTime := time.Since(allocStart)

			runtime.GC()
			var memAfter runtime.MemStats
			runtime.ReadMemStats(&memAfter)
			memMB := int64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024

			allocVecPerSec := float64(s.count) / allocTime.Seconds()

			queries := make([]simd.Vec512, 100)
			for i := range queries {
				for j := 0; j < 512; j++ {
					queries[i][j] = int8((i*17 + j*31) % 256 - 128)
				}
			}

			latencies := make([]time.Duration, 100)
			for i := 0; i < 100; i++ {
				start := time.Now()

				query := &queries[i%len(queries)]
				maxScore := int32(-1 << 30)
				bestIdx := -1

				limit := s.count
				if limit > 50000 {
					limit = 50000
				}

				for j := 0; j < limit; j++ {
					score := int32(0)
					for k := 0; k < 512; k++ {
						score += int32(query[k]) * int32(vectors[j][k])
					}
					if score > maxScore {
						maxScore = score
						bestIdx = j
					}
				}
				_ = bestIdx
				latencies[i] = time.Since(start)
			}

			var total time.Duration
			for _, l := range latencies {
				total += l
			}
			avgUs := float64(total.Microseconds()) / 100.0
			qps := 100.0 / total.Seconds()

			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

			fmt.Printf("%-8s %-12.0f %-15.0f %-12.0f %-10.0f %-10d\n",
				s.name, float64(allocTime.Milliseconds()), allocVecPerSec,
				avgUs, qps, memMB)

			_ = scales
		})
	}
}

func TestEmbeddingGenerationBenchmark(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}

	testTexts := []string{
		"anime character with blue hair",
		"fantasy dragon warrior",
		"cyberpunk city at night",
		"cute cat playing with yarn",
		"mysterious wizard casting spells",
	}

	fmt.Printf("\n=== Embedding Generation Benchmark ===\n")

	for _, text := range testTexts {
		t.Run(text[:20], func(t *testing.T) {
			latencies := make([]time.Duration, 1000)

			for i := 0; i < 1000; i++ {
				start := time.Now()
				_, err := model.EmbedInt8(text)
				if err != nil {
					t.Fatal(err)
				}
				latencies[i] = time.Since(start)
			}

			var total time.Duration
			for _, l := range latencies {
				total += l
			}

			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

			fmt.Printf("%-25s avg=%v p50=%v p99=%v (%.0f/sec)\n",
				text[:20],
				total/1000,
				latencies[500],
				latencies[990],
				1000.0/total.Seconds())
		})
	}
}

func TestMemoryAllocationProfile(t *testing.T) {
	sizes := []int{100000, 500000, 1000000, 2000000}

	fmt.Printf("\n=== Memory Allocation Profile ===\n")
	fmt.Printf("%-10s %-12s %-15s %-12s\n",
		"Vectors", "Time(ms)", "MB/sec", "Total(MB)")

	for _, size := range sizes {
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)

		start := time.Now()
		vectors := make([]simd.Vec512, size)
		scales := make([]float32, size)

		for i := 0; i < size; i++ {
			for j := 0; j < 512; j++ {
				vectors[i][j] = int8(i % 256 - 128)
			}
			scales[i] = 1.0
		}
		elapsed := time.Since(start)

		runtime.GC()
		var after runtime.MemStats
		runtime.ReadMemStats(&after)

		allocMB := float64(after.Alloc-before.Alloc) / 1024 / 1024
		mbPerSec := allocMB / elapsed.Seconds()

		fmt.Printf("%-10d %-12.0f %-15.0f %-12.0f\n",
			size, float64(elapsed.Milliseconds()), mbPerSec, allocMB)

		_ = vectors
		_ = scales
	}
}

func BenchmarkDotProduct(b *testing.B) {
	v1 := simd.Vec512{}
	v2 := simd.Vec512{}

	for i := 0; i < 512; i++ {
		v1[i] = int8(i % 256 - 128)
		v2[i] = int8((i * 3) % 256 - 128)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		score := int32(0)
		for k := 0; k < 512; k++ {
			score += int32(v1[k]) * int32(v2[k])
		}
		_ = score
	}
}
