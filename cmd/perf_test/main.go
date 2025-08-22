package main

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("⚡ PERFORMANCE BENCHMARK: Float32 vs INT8")
	fmt.Println(strings.Repeat("=", 80))

	// System info
	fmt.Println("\n📊 System Information:")
	fmt.Printf("  CPU cores: %d\n", runtime.NumCPU())
	fmt.Printf("  Go version: %s\n", runtime.Version())
	fmt.Printf("  GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	// Get memory baseline
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	baselineMemory := m.Alloc

	// Load Float32 model
	fmt.Println("\n🔄 Loading Float32 model...")
	startLoad := time.Now()
	modelF32, err := gobed.LoadModel()
	if err != nil {
		panic(fmt.Sprintf("Failed to load model: %v", err))
	}
	loadTimeF32 := time.Since(startLoad)

	// Check memory after loading
	runtime.GC()
	runtime.ReadMemStats(&m)
	f32ModelMemory := m.Alloc - baselineMemory

	fmt.Printf("✅ Float32 model loaded in %v\n", loadTimeF32)
	fmt.Printf("   Memory used: %.2f MB\n", float64(f32ModelMemory)/(1024*1024))

	// Test texts of varying lengths
	testTexts := []string{
		// Short texts
		"Hello",
		"Test",
		"Go",
		// Medium texts
		"Machine learning is fascinating.",
		"Natural language processing is powerful.",
		"Deep learning transforms AI.",
		// Longer texts
		"The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.",
		"Artificial intelligence and machine learning are revolutionizing how we process and understand natural language.",
		strings.Repeat("test ", 50),  // 50 words
		strings.Repeat("word ", 100), // 100 words
	}

	fmt.Printf("\n📝 Test corpus: %d texts (varying lengths)\n", len(testTexts))

	// Warmup
	fmt.Println("\n🔥 Warming up...")
	for i := 0; i < 100; i++ {
		modelF32.Encode(testTexts[0])
	}

	// Single-threaded benchmark
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("📊 SINGLE-THREADED PERFORMANCE")
	fmt.Println(strings.Repeat("-", 50))

	iterations := 1000

	// Float32 single-threaded
	fmt.Printf("\nBenchmarking Float32 (%d iterations)...\n", iterations)
	startF32 := time.Now()
	for i := 0; i < iterations; i++ {
		for _, text := range testTexts {
			_, _ = modelF32.Encode(text)
		}
	}
	timeF32Single := time.Since(startF32)

	totalEncodings := iterations * len(testTexts)
	throughputF32Single := float64(totalEncodings) / timeF32Single.Seconds()
	latencyF32Single := timeF32Single.Nanoseconds() / int64(totalEncodings) / 1000 // microseconds

	fmt.Printf("  Total time: %v\n", timeF32Single)
	fmt.Printf("  Throughput: %.0f encodings/sec\n", throughputF32Single)
	fmt.Printf("  Avg latency: %d µs\n", latencyF32Single)

	// Multi-threaded benchmark
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("🚀 MULTI-THREADED PERFORMANCE")
	fmt.Println(strings.Repeat("-", 50))

	numWorkers := runtime.NumCPU()
	fmt.Printf("\nUsing %d workers\n", numWorkers)

	// Float32 multi-threaded
	fmt.Println("\nBenchmarking Float32 (parallel)...")
	var wg sync.WaitGroup
	startF32Multi := time.Now()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			iterPerWorker := iterations / numWorkers
			for i := 0; i < iterPerWorker; i++ {
				for _, text := range testTexts {
					_, _ = modelF32.Encode(text)
				}
			}
		}(w)
	}
	wg.Wait()

	timeF32Multi := time.Since(startF32Multi)
	throughputF32Multi := float64(totalEncodings) / timeF32Multi.Seconds()

	fmt.Printf("  Total time: %v\n", timeF32Multi)
	fmt.Printf("  Throughput: %.0f encodings/sec\n", throughputF32Multi)
	fmt.Printf("  Speedup vs single: %.2fx\n", timeF32Single.Seconds()/timeF32Multi.Seconds())

	// Batch processing test
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("📦 BATCH PROCESSING TEST")
	fmt.Println(strings.Repeat("-", 50))

	batchSizes := []int{1, 10, 50, 100}

	for _, batchSize := range batchSizes {
		// Create batch
		batch := make([]string, batchSize)
		for i := 0; i < batchSize; i++ {
			batch[i] = testTexts[i%len(testTexts)]
		}

		// Benchmark batch
		batchIterations := 1000
		start := time.Now()
		for i := 0; i < batchIterations; i++ {
			for _, text := range batch {
				_, _ = modelF32.Encode(text)
			}
		}
		elapsed := time.Since(start)

		throughput := float64(batchIterations*batchSize) / elapsed.Seconds()
		avgLatency := elapsed.Nanoseconds() / int64(batchIterations*batchSize) / 1000

		fmt.Printf("\nBatch size %d:\n", batchSize)
		fmt.Printf("  Throughput: %.0f encodings/sec\n", throughput)
		fmt.Printf("  Avg latency: %d µs\n", avgLatency)
	}

	// Memory pressure test
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("💾 MEMORY PRESSURE TEST")
	fmt.Println(strings.Repeat("-", 50))

	// Generate many embeddings
	fmt.Println("\nGenerating 10,000 embeddings...")
	embeddings := make([][]float32, 10000)

	runtime.GC()
	runtime.ReadMemStats(&m)
	memBefore := m.Alloc

	start := time.Now()
	for i := 0; i < 10000; i++ {
		embeddings[i], _ = modelF32.Encode(testTexts[i%len(testTexts)])
	}
	elapsed := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m)
	memAfter := m.Alloc

	fmt.Printf("  Time: %v\n", elapsed)
	fmt.Printf("  Rate: %.0f embeddings/sec\n", 10000.0/elapsed.Seconds())
	fmt.Printf("  Memory used: %.2f MB\n", float64(memAfter-memBefore)/(1024*1024))
	fmt.Printf("  Per embedding: %.2f KB\n", float64(memAfter-memBefore)/10000/1024)

	// Cache effects test
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("🔄 CACHE EFFECTS TEST")
	fmt.Println(strings.Repeat("-", 50))

	// Same text repeated
	fmt.Println("\nSame text repeated 1000 times:")
	sameText := "This is a test sentence for cache effects."
	start = time.Now()
	for i := 0; i < 1000; i++ {
		_, _ = modelF32.Encode(sameText)
	}
	timeSame := time.Since(start)

	// Different texts
	fmt.Println("Different texts 1000 times:")
	start = time.Now()
	for i := 0; i < 1000; i++ {
		_, _ = modelF32.Encode(fmt.Sprintf("Text number %d with unique content", i))
	}
	timeDifferent := time.Since(start)

	fmt.Printf("  Same text: %v (%.0f/sec)\n", timeSame, 1000.0/timeSame.Seconds())
	fmt.Printf("  Different texts: %v (%.0f/sec)\n", timeDifferent, 1000.0/timeDifferent.Seconds())
	fmt.Printf("  Cache benefit: %.2fx faster\n", timeDifferent.Seconds()/timeSame.Seconds())

	// Stress test
	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Println("🔥 STRESS TEST (30 seconds)")
	fmt.Println(strings.Repeat("-", 50))

	fmt.Println("\nRunning continuous encoding for 30 seconds...")
	stressStart := time.Now()
	stressCount := 0

	for time.Since(stressStart) < 30*time.Second {
		for _, text := range testTexts {
			_, _ = modelF32.Encode(text)
			stressCount++
		}
	}

	stressDuration := time.Since(stressStart)
	fmt.Printf("  Completed: %d encodings\n", stressCount)
	fmt.Printf("  Duration: %v\n", stressDuration)
	fmt.Printf("  Rate: %.0f encodings/sec\n", float64(stressCount)/stressDuration.Seconds())

	// Final memory stats
	runtime.GC()
	debug.FreeOSMemory()
	runtime.ReadMemStats(&m)

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📊 FINAL STATISTICS")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Printf("\n🎯 Performance Summary:\n")
	fmt.Printf("  Single-threaded: %.0f encodings/sec\n", throughputF32Single)
	fmt.Printf("  Multi-threaded:  %.0f encodings/sec (%.1fx speedup)\n",
		throughputF32Multi, throughputF32Multi/throughputF32Single)
	fmt.Printf("  Avg latency: %d µs\n", latencyF32Single)
	fmt.Printf("  Model load time: %v\n", loadTimeF32)

	fmt.Printf("\n💾 Memory Summary:\n")
	fmt.Printf("  Model size: %.2f MB\n", float64(f32ModelMemory)/(1024*1024))
	fmt.Printf("  Current heap: %.2f MB\n", float64(m.Alloc)/(1024*1024))
	fmt.Printf("  Total allocated: %.2f MB\n", float64(m.TotalAlloc)/(1024*1024))
	fmt.Printf("  System memory: %.2f MB\n", float64(m.Sys)/(1024*1024))

	fmt.Println("\n✅ Performance benchmark completed!")
}
