package simd

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// Performance regression test - ensures SIMD is significantly faster than generic
func TestPerformanceRegression(t *testing.T) {
	// Skip in short mode
	if testing.Short() {
		t.Skip("Skipping performance regression test in short mode")
	}

	const iterations = 100000

	// Create test vectors
	a := &Vec512{}
	b := &Vec512{}

	rand.Seed(42)
	for i := 0; i < 512; i++ {
		a[i] = int8(rand.Intn(256) - 128)
		b[i] = int8(rand.Intn(256) - 128)
	}

	// Warm up
	for i := 0; i < 1000; i++ {
		_ = dot512_generic(a, b)
		_ = dot512_i8_avx2(a, b)
	}

	// Benchmark generic implementation
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = dot512_generic(a, b)
	}
	genericTime := time.Since(start)

	// Benchmark AVX2 implementation
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_ = dot512_i8_avx2(a, b)
	}
	avx2Time := time.Since(start)

	// Calculate speedup
	speedup := float64(genericTime) / float64(avx2Time)

	t.Logf("Generic time: %v", genericTime)
	t.Logf("AVX2 time: %v", avx2Time)
	t.Logf("Speedup: %.2fx", speedup)

	// Verify significant speedup (at least 3x)
	if speedup < 3.0 {
		t.Errorf("Performance regression: AVX2 speedup %.2fx is less than expected minimum 3x", speedup)
	}

	// Verify results are still correct
	genericResult := dot512_generic(a, b)
	avx2Result := dot512_i8_avx2(a, b)

	if genericResult != avx2Result {
		t.Errorf("Correctness regression: results don't match - generic: %d, avx2: %d",
			genericResult, avx2Result)
	}
}

// Latency test - measures single operation latency
func TestLatencyMeasurement(t *testing.T) {
	const warmupRuns = 10000
	const measureRuns = 100000

	a := &Vec512{}
	b := &Vec512{}

	// Fill with test data
	for i := 0; i < 512; i++ {
		a[i] = int8(i%256 - 128)
		b[i] = int8((i*3)%256 - 128)
	}

	implementations := map[string]func(*Vec512, *Vec512) int32{
		"Generic":  dot512_generic,
		"AVX2":     dot512_i8_avx2,
		"AVX2_Alt": dot512_i8_avx2_alt,
		"VNNI":     dot512_i8_vnni,
		"Dispatch": Dot512,
	}

	for name, impl := range implementations {
		// Warmup
		for i := 0; i < warmupRuns; i++ {
			_ = impl(a, b)
		}

		// Measure
		start := time.Now()
		for i := 0; i < measureRuns; i++ {
			_ = impl(a, b)
		}
		elapsed := time.Since(start)

		avgLatency := elapsed.Nanoseconds() / int64(measureRuns)
		t.Logf("%s: Average latency %d ns/op", name, avgLatency)
	}
}

// Throughput test - measures operations per second
func TestThroughputMeasurement(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping throughput test in short mode")
	}

	const testDurationSeconds = 2

	a := &Vec512{}
	b := &Vec512{}

	// Fill with test data
	for i := 0; i < 512; i++ {
		a[i] = int8(i%256 - 128)
		b[i] = int8((i*7)%256 - 128)
	}

	implementations := map[string]func(*Vec512, *Vec512) int32{
		"Generic":  dot512_generic,
		"AVX2":     dot512_i8_avx2,
		"Dispatch": Dot512,
	}

	for name, impl := range implementations {
		// Warmup
		deadline := time.Now().Add(100 * time.Millisecond)
		for time.Now().Before(deadline) {
			_ = impl(a, b)
		}

		// Measure throughput
		count := 0
		deadline = time.Now().Add(testDurationSeconds * time.Second)

		start := time.Now()
		for time.Now().Before(deadline) {
			_ = impl(a, b)
			count++
		}
		elapsed := time.Since(start)

		throughput := float64(count) / elapsed.Seconds()
		t.Logf("%s: Throughput %.0f ops/sec", name, throughput)
	}
}

// Memory access pattern test
func TestMemoryAccessPatterns(t *testing.T) {
	// Test sequential access pattern
	t.Run("Sequential", func(t *testing.T) {
		vectors := make([][2]Vec512, 1000)

		// Initialize vectors
		for i := 0; i < len(vectors); i++ {
			for j := 0; j < 512; j++ {
				vectors[i][0][j] = int8(rand.Intn(256) - 128)
				vectors[i][1][j] = int8(rand.Intn(256) - 128)
			}
		}

		start := time.Now()
		sum := int64(0)
		for i := 0; i < len(vectors); i++ {
			sum += int64(Dot512(&vectors[i][0], &vectors[i][1]))
		}
		elapsed := time.Since(start)

		t.Logf("Sequential access: %v for %d vectors (sum: %d)", elapsed, len(vectors), sum)
	})

	// Test random access pattern
	t.Run("Random", func(t *testing.T) {
		vectors := make([][2]Vec512, 1000)
		indices := make([]int, 1000)

		// Initialize vectors and random indices
		for i := 0; i < len(vectors); i++ {
			for j := 0; j < 512; j++ {
				vectors[i][0][j] = int8(rand.Intn(256) - 128)
				vectors[i][1][j] = int8(rand.Intn(256) - 128)
			}
			indices[i] = rand.Intn(len(vectors))
		}

		start := time.Now()
		sum := int64(0)
		for i := 0; i < len(indices); i++ {
			idx := indices[i]
			sum += int64(Dot512(&vectors[idx][0], &vectors[idx][1]))
		}
		elapsed := time.Since(start)

		t.Logf("Random access: %v for %d vectors (sum: %d)", elapsed, len(indices), sum)
	})
}

// Cache behavior test
func TestCacheBehavior(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping cache behavior test in short mode")
	}

	// Test different dataset sizes to observe cache effects
	sizes := []int{
		100,     // L1 cache
		10000,   // L2 cache
		100000,  // L3 cache
		1000000, // Main memory
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
			vectors := make([][2]Vec512, size)

			// Initialize vectors
			for i := 0; i < len(vectors); i++ {
				for j := 0; j < 512; j++ {
					vectors[i][0][j] = int8(rand.Intn(256) - 128)
					vectors[i][1][j] = int8(rand.Intn(256) - 128)
				}
			}

			// Multiple passes to measure cache effects
			const passes = 3

			for pass := 0; pass < passes; pass++ {
				start := time.Now()
				sum := int64(0)

				for i := 0; i < len(vectors); i++ {
					sum += int64(Dot512(&vectors[i][0], &vectors[i][1]))
				}

				elapsed := time.Since(start)
				avgNsPerOp := elapsed.Nanoseconds() / int64(len(vectors))

				t.Logf("Pass %d: %v (%d ns/op) for %d vectors",
					pass+1, elapsed, avgNsPerOp, len(vectors))
			}
		})
	}
}

// Concurrent access test
func TestConcurrentAccess(t *testing.T) {
	const numGoroutines = 8
	const opsPerGoroutine = 10000

	vectors := make([][2]Vec512, numGoroutines)

	// Initialize vectors
	for i := 0; i < len(vectors); i++ {
		for j := 0; j < 512; j++ {
			vectors[i][0][j] = int8(rand.Intn(256) - 128)
			vectors[i][1][j] = int8(rand.Intn(256) - 128)
		}
	}

	// Channel to collect results
	results := make(chan int64, numGoroutines)

	start := time.Now()

	// Launch goroutines
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			sum := int64(0)
			a := &vectors[id][0]
			b := &vectors[id][1]

			for j := 0; j < opsPerGoroutine; j++ {
				sum += int64(Dot512(a, b))
			}

			results <- sum
		}(i)
	}

	// Collect results
	totalSum := int64(0)
	for i := 0; i < numGoroutines; i++ {
		totalSum += <-results
	}

	elapsed := time.Since(start)
	totalOps := numGoroutines * opsPerGoroutine

	t.Logf("Concurrent test: %v for %d operations across %d goroutines (sum: %d)",
		elapsed, totalOps, numGoroutines, totalSum)
	t.Logf("Average: %d ns/op", elapsed.Nanoseconds()/int64(totalOps))
}
