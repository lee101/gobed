package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L. -lcuda_ultra_fast -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt
// #include <stdlib.h>
// extern void* ultra_gpu_create(int max_docs, int dim);
// extern void ultra_gpu_destroy(void* handle);
// extern int ultra_gpu_add_batch_async(void* handle, const signed char* embeddings, int num_vectors);
// extern int ultra_gpu_search(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"

type BenchmarkResult struct {
	Name               string
	TotalTime          time.Duration
	DocumentsPerSecond float64
	QueriesPerSecond   float64
	MemoryUsedMB       float64
	GPUMemoryMB        float64
	Latency50          time.Duration
	Latency95          time.Duration
	Latency99          time.Duration
}

type UltraBenchmark struct {
	gpuHandle unsafe.Pointer
	results   []BenchmarkResult
	mu        sync.Mutex
}

func NewUltraBenchmark() *UltraBenchmark {
	return &UltraBenchmark{
		gpuHandle: C.ultra_gpu_create(C.int(10000000), C.int(512)),
		results:   make([]BenchmarkResult, 0),
	}
}

func (b *UltraBenchmark) RunAll() {
	fmt.Println("🚀 Ultra Performance Benchmark Suite")
	fmt.Println("=====================================")
	fmt.Printf("CPU: %d cores | Go version: %s\n", runtime.NumCPU(), runtime.Version())
	fmt.Println()

	// Run benchmarks
	b.benchmarkIndexing()
	b.benchmarkSearch()
	b.benchmarkBatchProcessing()
	b.benchmarkMemoryEfficiency()
	b.benchmarkConcurrency()
	b.benchmarkScalability()

	// Print summary
	b.printSummary()
}

func (b *UltraBenchmark) benchmarkIndexing() {
	fmt.Println("📊 Benchmark: Indexing Performance")
	fmt.Println("-----------------------------------")

	sizes := []int{1000, 10000, 100000, 1000000}

	for _, size := range sizes {
		// Generate random embeddings
		embeddings := make([]int8, size*512)
		for i := range embeddings {
			embeddings[i] = int8(rand.Intn(256) - 128)
		}

		// Measure indexing time
		start := time.Now()

		batchSize := 2048
		for offset := 0; offset < size; offset += batchSize {
			end := offset + batchSize
			if end > size {
				end = size
			}

			ret := C.ultra_gpu_add_batch_async(
				b.gpuHandle,
				(*C.schar)(unsafe.Pointer(&embeddings[offset*512])),
				C.int(end-offset),
			)

			if ret < 0 {
				log.Printf("Failed to add batch")
			}
		}

		elapsed := time.Since(start)
		docsPerSec := float64(size) / elapsed.Seconds()

		fmt.Printf("  %7d docs: %8.2f ms | %10.0f docs/sec | %6.2f GB/s\n",
			size, float64(elapsed.Milliseconds()), docsPerSec,
			float64(size*512)/1024/1024/1024/elapsed.Seconds())

		b.results = append(b.results, BenchmarkResult{
			Name:               fmt.Sprintf("Index_%d", size),
			TotalTime:          elapsed,
			DocumentsPerSecond: docsPerSec,
		})
	}
	fmt.Println()
}

func (b *UltraBenchmark) benchmarkSearch() {
	fmt.Println("📊 Benchmark: Search Performance")
	fmt.Println("--------------------------------")

	// Add test documents first
	numDocs := 100000
	embeddings := make([]int8, numDocs*512)
	for i := range embeddings {
		embeddings[i] = int8(rand.Intn(256) - 128)
	}

	C.ultra_gpu_add_batch_async(
		b.gpuHandle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		C.int(numDocs),
	)

	// Benchmark different k values
	kValues := []int{1, 10, 50, 100}
	numQueries := 1000

	for _, k := range kValues {
		query := make([]int8, 512)
		for i := range query {
			query[i] = int8(rand.Intn(256) - 128)
		}

		indices := make([]int32, k)
		scores := make([]float32, k)

		latencies := make([]time.Duration, numQueries)

		start := time.Now()
		for i := 0; i < numQueries; i++ {
			queryStart := time.Now()

			C.ultra_gpu_search(
				b.gpuHandle,
				(*C.schar)(unsafe.Pointer(&query[0])),
				C.int(k),
				(*C.int)(unsafe.Pointer(&indices[0])),
				(*C.float)(unsafe.Pointer(&scores[0])),
			)

			latencies[i] = time.Since(queryStart)
		}
		elapsed := time.Since(start)

		// Calculate percentiles
		p50 := calculatePercentile(latencies, 50)
		p95 := calculatePercentile(latencies, 95)
		p99 := calculatePercentile(latencies, 99)

		qps := float64(numQueries) / elapsed.Seconds()

		fmt.Printf("  k=%3d: QPS: %8.0f | P50: %6.2fμs | P95: %6.2fμs | P99: %6.2fμs\n",
			k, qps,
			float64(p50.Microseconds()),
			float64(p95.Microseconds()),
			float64(p99.Microseconds()))

		b.results = append(b.results, BenchmarkResult{
			Name:             fmt.Sprintf("Search_k%d", k),
			TotalTime:        elapsed,
			QueriesPerSecond: qps,
			Latency50:        p50,
			Latency95:        p95,
			Latency99:        p99,
		})
	}
	fmt.Println()
}

func (b *UltraBenchmark) benchmarkBatchProcessing() {
	fmt.Println("📊 Benchmark: Batch Processing")
	fmt.Println("------------------------------")

	batchSizes := []int{1, 10, 100, 1000}
	numDocs := 100000

	// Add documents
	embeddings := make([]int8, numDocs*512)
	for i := range embeddings {
		embeddings[i] = int8(rand.Intn(256) - 128)
	}

	C.ultra_gpu_add_batch_async(
		b.gpuHandle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		C.int(numDocs),
	)

	for _, batchSize := range batchSizes {
		queries := make([]int8, batchSize*512)
		for i := range queries {
			queries[i] = int8(rand.Intn(256) - 128)
		}

		start := time.Now()

		// Process batch
		for i := 0; i < batchSize; i++ {
			query := queries[i*512 : (i+1)*512]
			indices := make([]int32, 10)
			scores := make([]float32, 10)

			C.ultra_gpu_search(
				b.gpuHandle,
				(*C.schar)(unsafe.Pointer(&query[0])),
				C.int(10),
				(*C.int)(unsafe.Pointer(&indices[0])),
				(*C.float)(unsafe.Pointer(&scores[0])),
			)
		}

		elapsed := time.Since(start)
		throughput := float64(batchSize) / elapsed.Seconds()

		fmt.Printf("  Batch %4d: %8.2f ms | %8.0f queries/sec\n",
			batchSize, float64(elapsed.Milliseconds()), throughput)
	}
	fmt.Println()
}

func (b *UltraBenchmark) benchmarkMemoryEfficiency() {
	fmt.Println("📊 Benchmark: Memory Efficiency")
	fmt.Println("-------------------------------")

	var m runtime.MemStats

	// Baseline memory
	runtime.GC()
	runtime.ReadMemStats(&m)
	baselineMemory := m.Alloc

	// Add documents progressively
	docCounts := []int{10000, 50000, 100000, 500000}

	for _, count := range docCounts {
		embeddings := make([]int8, count*512)
		for i := range embeddings {
			embeddings[i] = int8(rand.Intn(256) - 128)
		}

		C.ultra_gpu_add_batch_async(
			b.gpuHandle,
			(*C.schar)(unsafe.Pointer(&embeddings[0])),
			C.int(count),
		)

		runtime.GC()
		runtime.ReadMemStats(&m)

		memUsed := (m.Alloc - baselineMemory) / 1024 / 1024
		memPerDoc := float64(m.Alloc-baselineMemory) / float64(count)

		fmt.Printf("  %7d docs: %6d MB total | %6.2f bytes/doc\n",
			count, memUsed, memPerDoc)
	}
	fmt.Println()
}

func (b *UltraBenchmark) benchmarkConcurrency() {
	fmt.Println("📊 Benchmark: Concurrent Operations")
	fmt.Println("-----------------------------------")

	// Add base documents
	numDocs := 100000
	embeddings := make([]int8, numDocs*512)
	for i := range embeddings {
		embeddings[i] = int8(rand.Intn(256) - 128)
	}

	C.ultra_gpu_add_batch_async(
		b.gpuHandle,
		(*C.schar)(unsafe.Pointer(&embeddings[0])),
		C.int(numDocs),
	)

	workerCounts := []int{1, 2, 4, 8, 16, 32}

	for _, workers := range workerCounts {
		queriesPerWorker := 100
		totalQueries := workers * queriesPerWorker

		var wg sync.WaitGroup
		var totalOps atomic.Int64

		start := time.Now()

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				query := make([]int8, 512)
				for i := range query {
					query[i] = int8(rand.Intn(256) - 128)
				}

				indices := make([]int32, 10)
				scores := make([]float32, 10)

				for i := 0; i < queriesPerWorker; i++ {
					C.ultra_gpu_search(
						b.gpuHandle,
						(*C.schar)(unsafe.Pointer(&query[0])),
						C.int(10),
						(*C.int)(unsafe.Pointer(&indices[0])),
						(*C.float)(unsafe.Pointer(&scores[0])),
					)
					totalOps.Add(1)
				}
			}()
		}

		wg.Wait()
		elapsed := time.Since(start)

		qps := float64(totalQueries) / elapsed.Seconds()
		fmt.Printf("  %2d workers: %8.0f QPS | %6.2f ms/query\n",
			workers, qps, float64(elapsed.Milliseconds())/float64(totalQueries))
	}
	fmt.Println()
}

func (b *UltraBenchmark) benchmarkScalability() {
	fmt.Println("📊 Benchmark: Scalability")
	fmt.Println("------------------------")

	docSizes := []int{10000, 100000, 500000, 1000000}

	for _, size := range docSizes {
		// Create new index for each test
		handle := C.ultra_gpu_create(C.int(size+1000), C.int(512))

		// Add documents
		embeddings := make([]int8, size*512)
		for i := range embeddings {
			embeddings[i] = int8(rand.Intn(256) - 128)
		}

		indexStart := time.Now()
		C.ultra_gpu_add_batch_async(
			handle,
			(*C.schar)(unsafe.Pointer(&embeddings[0])),
			C.int(size),
		)
		indexTime := time.Since(indexStart)

		// Search benchmark
		query := make([]int8, 512)
		for i := range query {
			query[i] = int8(rand.Intn(256) - 128)
		}

		indices := make([]int32, 10)
		scores := make([]float32, 10)

		searchTimes := make([]time.Duration, 100)
		for i := 0; i < 100; i++ {
			start := time.Now()
			C.ultra_gpu_search(
				handle,
				(*C.schar)(unsafe.Pointer(&query[0])),
				C.int(10),
				(*C.int)(unsafe.Pointer(&indices[0])),
				(*C.float)(unsafe.Pointer(&scores[0])),
			)
			searchTimes[i] = time.Since(start)
		}

		avgSearchTime := calculateAverage(searchTimes)

		fmt.Printf("  %7d docs: Index: %6.2fs | Search: %6.2fμs\n",
			size, indexTime.Seconds(), float64(avgSearchTime.Microseconds()))

		C.ultra_gpu_destroy(handle)
	}
	fmt.Println()
}

func (b *UltraBenchmark) printSummary() {
	fmt.Println("📈 Performance Summary")
	fmt.Println("=====================")

	if len(b.results) > 0 {
		var totalDPS, totalQPS float64
		var count int

		for _, r := range b.results {
			if r.DocumentsPerSecond > 0 {
				totalDPS += r.DocumentsPerSecond
				count++
			}
			if r.QueriesPerSecond > 0 {
				totalQPS += r.QueriesPerSecond
			}
		}

		if count > 0 {
			fmt.Printf("  Average Indexing Speed: %.0f docs/sec\n", totalDPS/float64(count))
		}
		if totalQPS > 0 {
			fmt.Printf("  Average Query Speed: %.0f QPS\n", totalQPS/float64(len(b.results)))
		}
	}

	// GPU memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("  Peak Memory Usage: %.2f MB\n", float64(m.Sys)/1024/1024)
}

func (b *UltraBenchmark) Cleanup() {
	C.ultra_gpu_destroy(b.gpuHandle)
}

func calculatePercentile(latencies []time.Duration, percentile float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	index := int(float64(len(latencies)) * percentile / 100)
	if index >= len(latencies) {
		index = len(latencies) - 1
	}

	return latencies[index]
}

func calculateAverage(times []time.Duration) time.Duration {
	if len(times) == 0 {
		return 0
	}

	var total time.Duration
	for _, t := range times {
		total += t
	}

	return total / time.Duration(len(times))
}

func main() {
	// CPU profiling
	if os.Getenv("CPUPROFILE") != "" {
		f, err := os.Create("cpu.prof")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	benchmark := NewUltraBenchmark()
	defer benchmark.Cleanup()

	benchmark.RunAll()
}