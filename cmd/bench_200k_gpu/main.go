package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	numDocs := 200000
	dim := 384 // Standard embedding dimension
	k := 100   // Top-k results

	fmt.Printf("🚀 GPU Maximum Performance Benchmark - RTX 3090\n")
	fmt.Printf("================================================\n")
	fmt.Printf("Documents: %d\n", numDocs)
	fmt.Printf("Dimension: %d\n", dim)
	fmt.Printf("Top-K: %d\n\n", k)

	// Check if we have cached embeddings
	embeddingsFile := "embeddings_200k.bin"
	scalesFile := "scales_200k.bin"

	var vectors []int8
	var scales []float32

	if fileExists(embeddingsFile) && fileExists(scalesFile) {
		fmt.Println("Loading cached embeddings...")
		vectors = loadInt8Array(embeddingsFile)
		scales = loadFloat32Array(scalesFile)
	} else {
		fmt.Println("Generating synthetic int8 embeddings...")
		vectors, scales = generateInt8Embeddings(numDocs, dim)

		// Cache for next run
		saveInt8Array(embeddingsFile, vectors)
		saveFloat32Array(scalesFile, scales)
	}

	// Initialize GPU index with maximum performance settings
	config := gobed.DefaultIndexConfig()
	config.MaxBatchSize = 10000
	config.NList = 1024  // IVF clusters
	config.NProbe = 32   // Clusters to search

	fmt.Println("\n📊 Initializing GPU Index...")
	index, err := gobed.NewCUDAMaxPerfIndex(dim, numDocs*2,
		gobed.WithConfig(config))
	if err != nil {
		log.Fatalf("Failed to create GPU index: %v", err)
	}
	defer index.Close()

	// Add vectors to GPU
	fmt.Println("🔄 Adding vectors to GPU...")
	addStart := time.Now()
	err = index.AddInt8Vectors(vectors, scales)
	if err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	addTime := time.Since(addStart)
	fmt.Printf("✅ Added %d vectors in %.2fms (%.0f vectors/sec)\n",
		numDocs, float64(addTime.Microseconds())/1000.0,
		float64(numDocs)/addTime.Seconds())

	// Build IVF index if enabled
	if config.NList > 0 {
		fmt.Printf("\n🏗️ Building IVF index with %d clusters...\n", config.NList)
		ivfStart := time.Now()

		// Use subset for training
		trainingSize := min(100000, numDocs)
		trainingVectors := vectors[:trainingSize*dim]

		err = index.BuildIVF(trainingVectors)
		if err != nil {
			fmt.Printf("⚠️ IVF build failed (continuing with flat index): %v\n", err)
		} else {
			ivfTime := time.Since(ivfStart)
			fmt.Printf("✅ IVF index built in %.2fs\n", ivfTime.Seconds())
		}
	}

	// Generate test queries
	fmt.Println("\n🔍 Generating test queries...")
	numQueries := 1000
	queries, _ := generateInt8Embeddings(numQueries, dim)

	// Warmup phase - critical for GPU performance
	fmt.Println("\n🔥 Warming up GPU...")
	for i := 0; i < 10; i++ {
		warmupQueries := queries[:100*dim]
		_, _, _ = index.SearchBatch(warmupQueries, 10)
	}

	// Benchmark different batch sizes
	fmt.Println("\n⚡ PERFORMANCE BENCHMARKS")
	fmt.Println("=" * 50)

	batchSizes := []int{1, 10, 100, 500, 1000}

	for _, batchSize := range batchSizes {
		batchQueries := queries[:batchSize*dim]

		// Multiple runs for consistency
		var times []float64
		runs := 10

		for run := 0; run < runs; run++ {
			start := time.Now()
			_, _, err := index.SearchBatch(batchQueries, k)
			elapsed := time.Since(start)

			if err != nil {
				log.Printf("Search failed: %v", err)
				continue
			}

			times = append(times, float64(elapsed.Microseconds())/1000.0)
		}

		// Calculate statistics
		avgTime := average(times)
		minTime := minimum(times)
		maxTime := maximum(times)
		qps := float64(batchSize) * 1000.0 / avgTime

		fmt.Printf("\nBatch Size: %d\n", batchSize)
		fmt.Printf("  Average: %.3fms (%.0f QPS)\n", avgTime, qps)
		fmt.Printf("  Min:     %.3fms\n", minTime)
		fmt.Printf("  Max:     %.3fms\n", maxTime)
		fmt.Printf("  Per Query: %.3fms\n", avgTime/float64(batchSize))
	}

	// Test latency for single queries
	fmt.Println("\n🎯 SINGLE QUERY LATENCY TEST")
	fmt.Println("=" * 50)

	singleQuery := queries[:dim]
	var singleTimes []float64

	for i := 0; i < 100; i++ {
		start := time.Now()
		_, _, err := index.SearchBatch(singleQuery, k)
		elapsed := time.Since(start)

		if err != nil {
			continue
		}

		singleTimes = append(singleTimes, float64(elapsed.Microseconds())/1000.0)
	}

	fmt.Printf("Average latency: %.3fms\n", average(singleTimes))
	fmt.Printf("P50 latency:     %.3fms\n", percentile(singleTimes, 0.5))
	fmt.Printf("P95 latency:     %.3fms\n", percentile(singleTimes, 0.95))
	fmt.Printf("P99 latency:     %.3fms\n", percentile(singleTimes, 0.99))

	// Get final statistics
	stats := index.GetStats()
	fmt.Println("\n📊 FINAL STATISTICS")
	fmt.Println("=" * 50)
	for key, value := range stats {
		fmt.Printf("%s: %v\n", key, value)
	}

	// Target performance check
	fmt.Println("\n🎯 TARGET PERFORMANCE")
	fmt.Println("=" * 50)
	targetMs := 5.0
	if average(singleTimes) < targetMs {
		fmt.Printf("✅ ACHIEVED: Single query < %.1fms (%.3fms)\n",
			targetMs, average(singleTimes))
	} else {
		fmt.Printf("❌ NOT MET: Single query > %.1fms (%.3fms)\n",
			targetMs, average(singleTimes))
		fmt.Println("   Consider: Reducing nprobe, using smaller k, or enabling IVF")
	}
}

func generateInt8Embeddings(n, dim int) ([]int8, []float32) {
	rand.Seed(42)
	vectors := make([]int8, n*dim)
	scales := make([]float32, n)

	for i := 0; i < n; i++ {
		// Generate random float embeddings first
		floatVec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			floatVec[j] = rand.Float32()*2 - 1 // Range [-1, 1]
		}

		// Normalize
		var norm float32
		for _, v := range floatVec {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for j := range floatVec {
				floatVec[j] /= norm
			}
		}

		// Quantize to int8
		minVal, maxVal := floatVec[0], floatVec[0]
		for _, v := range floatVec {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}

		scale := (maxVal - minVal) / 255.0
		if scale == 0 {
			scale = 1.0
		}
		scales[i] = scale

		for j, v := range floatVec {
			quantized := int8((v-minVal)/scale - 128)
			vectors[i*dim+j] = quantized
		}
	}

	return vectors, scales
}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

func saveInt8Array(filename string, data []int8) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	bytes := make([]byte, len(data))
	for i, v := range data {
		bytes[i] = byte(v)
	}
	_, err = file.Write(bytes)
	return err
}

func loadInt8Array(filename string) []int8 {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil
	}

	result := make([]int8, len(data))
	for i, b := range data {
		result[i] = int8(b)
	}
	return result
}

func saveFloat32Array(filename string, data []float32) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, v := range data {
		err := binary.Write(file, binary.LittleEndian, v)
		if err != nil {
			return err
		}
	}
	return nil
}

func loadFloat32Array(filename string) []float32 {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil
	}

	numFloats := len(data) / 4
	result := make([]float32, numFloats)

	for i := 0; i < numFloats; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		result[i] = math.Float32frombits(bits)
	}
	return result
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func minimum(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}
	return min
}

func maximum(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Simple percentile calculation (not exact but good enough)
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Basic sort
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	index := int(p * float64(len(sorted)-1))
	return sorted[index]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}