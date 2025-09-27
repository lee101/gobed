package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// PrecisionMode defines the inference precision
type PrecisionMode int

const (
	FP32 PrecisionMode = iota
	FP16
	INT8
)

func (p PrecisionMode) String() string {
	switch p {
	case FP32:
		return "FP32"
	case FP16:
		return "FP16"
	case INT8:
		return "INT8"
	default:
		return "Unknown"
	}
}

// GPUIndexer manages GPU-accelerated vector indexing with libtorch
type GPUIndexer struct {
	device    gotch.Device
	embedDim  int64
	precision PrecisionMode

	// Pre-allocated tensors
	embedTensor  *ts.Tensor
	queryTensor  *ts.Tensor
	scoresTensor *ts.Tensor

	// Index storage
	vectors       [][]float32
	vectorsTensor *ts.Tensor
	int8Vectors   *ts.Tensor // For INT8 quantization
	scale         float32    // Quantization scale
	zeroPoint     int8       // Quantization zero point

	mu          sync.RWMutex
	maxVectors  int
	currentSize int
}

// NewGPUIndexer creates a new GPU-accelerated indexer
func NewGPUIndexer(embedDim int64, maxVectors int, precision PrecisionMode) (*GPUIndexer, error) {
	log.Printf(" Initializing GPU Indexer with %s precision...", precision)

	// Determine device (prefer CUDA if available)
	device := gotch.CPU
	if gotch.CudaIfAvailable() {
		device = gotch.CudaIfAvailable()
		log.Printf(" CUDA device available - using GPU acceleration")

		// Log GPU info
		if cudaCount := gotch.CudaDeviceCount(); cudaCount > 0 {
			log.Printf(" Found %d CUDA device(s)", cudaCount)
		}
	} else {
		log.Printf(" No CUDA device found - using CPU")
	}

	indexer := &GPUIndexer{
		device:      device,
		embedDim:    embedDim,
		precision:   precision,
		vectors:     make([][]float32, 0, maxVectors),
		maxVectors:  maxVectors,
		currentSize: 0,
	}

	// Pre-allocate GPU tensors for efficient operations
	indexer.allocateTensors()

	return indexer, nil
}

// allocateTensors pre-allocates GPU memory for efficient operations
func (idx *GPUIndexer) allocateTensors() {
	log.Printf(" Pre-allocating GPU tensors for %d vectors of dimension %d", idx.maxVectors, idx.embedDim)

	// Allocate based on precision mode
	switch idx.precision {
	case INT8:
		// For INT8, we allocate int8 tensor
		idx.int8Vectors = ts.MustZeros([]int64{int64(idx.maxVectors), idx.embedDim}, gotch.Int8, idx.device)
		log.Printf(" Allocated INT8 tensor: [%d, %d]", idx.maxVectors, idx.embedDim)

	case FP16:
		// For FP16, allocate half precision tensor
		idx.vectorsTensor = ts.MustZeros([]int64{int64(idx.maxVectors), idx.embedDim}, gotch.Half, idx.device)
		log.Printf(" Allocated FP16 tensor: [%d, %d]", idx.maxVectors, idx.embedDim)

	default: // FP32
		idx.vectorsTensor = ts.MustZeros([]int64{int64(idx.maxVectors), idx.embedDim}, gotch.Float, idx.device)
		log.Printf(" Allocated FP32 tensor: [%d, %d]", idx.maxVectors, idx.embedDim)
	}

	// Pre-allocate query and scores tensors
	idx.queryTensor = ts.MustZeros([]int64{1, idx.embedDim}, gotch.Float, idx.device)
	idx.scoresTensor = ts.MustZeros([]int64{int64(idx.maxVectors)}, gotch.Float, idx.device)
}

// quantizeToInt8 converts float32 vectors to INT8 with quantization
func (idx *GPUIndexer) quantizeToInt8(vectors [][]float32) {
	if len(vectors) == 0 {
		return
	}

	// Find min and max values for quantization
	minVal := float32(math.MaxFloat32)
	maxVal := float32(-math.MaxFloat32)

	for _, vec := range vectors {
		for _, val := range vec {
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}
	}

	// Calculate quantization parameters
	idx.scale = (maxVal - minVal) / 255.0
	idx.zeroPoint = int8(-math.Round(float64(minVal / idx.scale)))

	log.Printf(" Quantization params - Scale: %.6f, Zero point: %d", idx.scale, idx.zeroPoint)

	// Quantize vectors to INT8
	int8Data := make([]int8, len(vectors)*int(idx.embedDim))
	for i, vec := range vectors {
		for j, val := range vec {
			quantized := int8(math.Round(float64(val/idx.scale)) + float64(idx.zeroPoint))
			int8Data[i*int(idx.embedDim)+j] = quantized
		}
	}

	// Copy to GPU tensor
	idx.int8Vectors.MustNarrow(0, 0, int64(len(vectors)), false).
		MustCopyData(int8Data, int64(len(int8Data)))
}

// AddVectors adds multiple vectors to the index
func (idx *GPUIndexer) AddVectors(vectors [][]float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.currentSize+len(vectors) > idx.maxVectors {
		return fmt.Errorf("exceeds maximum capacity: %d + %d > %d",
			idx.currentSize, len(vectors), idx.maxVectors)
	}

	startTime := time.Now()

	// Store vectors in CPU memory
	idx.vectors = append(idx.vectors, vectors...)

	// Upload to GPU based on precision
	switch idx.precision {
	case INT8:
		idx.quantizeToInt8(idx.vectors)

	case FP16:
		// Convert to FP16 and upload
		flatData := make([]float32, len(idx.vectors)*int(idx.embedDim))
		for i, vec := range idx.vectors {
			copy(flatData[i*int(idx.embedDim):], vec)
		}
		tempTensor := ts.MustOfSlice(flatData).MustReshape([]int64{int64(len(idx.vectors)), idx.embedDim}, false)
		halfTensor := tempTensor.MustTo(idx.device, false).MustToKind(gotch.Half, false)
		idx.vectorsTensor.MustNarrow(0, 0, int64(len(idx.vectors)), false).MustCopy_(halfTensor)
		tempTensor.MustDrop()
		halfTensor.MustDrop()

	default: // FP32
		flatData := make([]float32, len(idx.vectors)*int(idx.embedDim))
		for i, vec := range idx.vectors {
			copy(flatData[i*int(idx.embedDim):], vec)
		}
		idx.vectorsTensor.MustNarrow(0, 0, int64(len(idx.vectors)), false).
			MustCopyData(flatData, int64(len(flatData)))
	}

	idx.currentSize = len(idx.vectors)
	uploadTime := time.Since(startTime)

	throughput := float64(len(vectors)) / uploadTime.Seconds()
	log.Printf(" Added %d vectors in %.3fms (%.0f vectors/sec)",
		len(vectors), float64(uploadTime.Nanoseconds())/1e6, throughput)

	return nil
}

// SearchInt8 performs similarity search with INT8 quantized vectors
func (idx *GPUIndexer) SearchInt8(query []float32, k int) ([]int, []float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.currentSize == 0 {
		return nil, nil, fmt.Errorf("index is empty")
	}

	if k > idx.currentSize {
		k = idx.currentSize
	}

	startTime := time.Now()

	// Quantize query vector
	queryInt8 := make([]int8, idx.embedDim)
	for i, val := range query {
		quantized := int8(math.Round(float64(val/idx.scale)) + float64(idx.zeroPoint))
		queryInt8[i] = quantized
	}

	// Upload query to GPU as INT8
	queryTensorInt8 := ts.MustOfSlice(queryInt8).MustReshape([]int64{1, idx.embedDim}, false).
		MustTo(idx.device, false)
	defer queryTensorInt8.MustDrop()

	// Get the active portion of the index
	activeVectors := idx.int8Vectors.MustNarrow(0, 0, int64(idx.currentSize), false)
	defer activeVectors.MustDrop()

	// Compute INT8 dot products (simulated with casting for now)
	// In production, you'd use specialized INT8 kernels
	queryFloat := queryTensorInt8.MustToKind(gotch.Float, false)
	vectorsFloat := activeVectors.MustToKind(gotch.Float, false)

	scores := vectorsFloat.MustMatmul(queryFloat.MustT(false), false)
	defer scores.MustDrop()
	defer queryFloat.MustDrop()
	defer vectorsFloat.MustDrop()

	// Apply dequantization scale
	dequantScale := idx.scale * idx.scale
	scores = scores.MustMulScalar(ts.FloatScalar(float64(dequantScale)), false)

	// Get top-k indices and values
	topkValues, topkIndices := scores.MustTopk(int64(k), -1, true, true)
	defer topkValues.MustDrop()
	defer topkIndices.MustDrop()

	// Convert to Go slices
	indices := topkIndices.Int64Values()
	values := topkValues.Float64Values()

	resultIndices := make([]int, k)
	resultScores := make([]float32, k)
	for i := 0; i < k; i++ {
		resultIndices[i] = int(indices[i])
		resultScores[i] = float32(values[i])
	}

	searchTime := time.Since(startTime)
	log.Printf(" INT8 search completed in %.3fμs", float64(searchTime.Nanoseconds())/1e3)

	return resultIndices, resultScores, nil
}

// SearchFP32 performs standard FP32 similarity search
func (idx *GPUIndexer) SearchFP32(query []float32, k int) ([]int, []float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.currentSize == 0 {
		return nil, nil, fmt.Errorf("index is empty")
	}

	if k > idx.currentSize {
		k = idx.currentSize
	}

	startTime := time.Now()

	// Upload query to GPU
	idx.queryTensor.MustCopyData(query, int64(len(query)))

	// Get the active portion of the index
	activeVectors := idx.vectorsTensor.MustNarrow(0, 0, int64(idx.currentSize), false)
	defer activeVectors.MustDrop()

	// Compute cosine similarity: vectors @ query.T
	scores := activeVectors.MustMatmul(idx.queryTensor.MustT(false), false)
	defer scores.MustDrop()

	// Get top-k indices and values
	topkValues, topkIndices := scores.MustTopk(int64(k), -1, true, true)
	defer topkValues.MustDrop()
	defer topkIndices.MustDrop()

	// Convert to Go slices
	indices := topkIndices.Int64Values()
	values := topkValues.Float64Values()

	resultIndices := make([]int, k)
	resultScores := make([]float32, k)
	for i := 0; i < k; i++ {
		resultIndices[i] = int(indices[i])
		resultScores[i] = float32(values[i])
	}

	searchTime := time.Since(startTime)
	log.Printf(" FP32 search completed in %.3fμs", float64(searchTime.Nanoseconds())/1e3)

	return resultIndices, resultScores, nil
}

// Search performs similarity search based on precision mode
func (idx *GPUIndexer) Search(query []float32, k int) ([]int, []float32, error) {
	switch idx.precision {
	case INT8:
		return idx.SearchInt8(query, k)
	default:
		return idx.SearchFP32(query, k)
	}
}

// BatchSearch performs batch similarity search on GPU
func (idx *GPUIndexer) BatchSearch(queries [][]float32, k int) ([][]int, [][]float32, error) {
	results := make([][]int, len(queries))
	scores := make([][]float32, len(queries))

	startTime := time.Now()

	// For GPU efficiency, we could batch process queries
	// For now, process sequentially
	for i, query := range queries {
		indices, sims, err := idx.Search(query, k)
		if err != nil {
			return nil, nil, fmt.Errorf("search failed for query %d: %v", i, err)
		}
		results[i] = indices
		scores[i] = sims
	}

	batchTime := time.Since(startTime)
	throughput := float64(len(queries)) / batchTime.Seconds()

	log.Printf(" Batch search: %d queries in %.2fms (%.0f queries/sec)",
		len(queries), float64(batchTime.Nanoseconds())/1e6, throughput)

	return results, scores, nil
}

// Close releases GPU resources
func (idx *GPUIndexer) Close() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.vectorsTensor != nil {
		idx.vectorsTensor.MustDrop()
	}
	if idx.int8Vectors != nil {
		idx.int8Vectors.MustDrop()
	}
	if idx.queryTensor != nil {
		idx.queryTensor.MustDrop()
	}
	if idx.scoresTensor != nil {
		idx.scoresTensor.MustDrop()
	}

	log.Printf(" GPU resources released")
}

// generateRandomVectors creates random test vectors
func generateRandomVectors(count int, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		// Generate normalized random vectors
		sum := float32(0)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1 // Range [-1, 1]
			sum += vec[j] * vec[j]
		}
		// Normalize
		norm := float32(math.Sqrt(float64(sum)))
		if norm > 0 {
			for j := 0; j < dim; j++ {
				vec[j] /= norm
			}
		}
		vectors[i] = vec
	}
	return vectors
}

// benchmarkIndexing tests indexing performance
func benchmarkIndexing(precision PrecisionMode, numVectors int, dim int) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf(" INDEXING BENCHMARK - %s PRECISION\n", precision)
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Create indexer
	indexer, err := NewGPUIndexer(int64(dim), numVectors*2, precision)
	if err != nil {
		log.Printf(" Failed to create indexer: %v", err)
		return
	}
	defer indexer.Close()

	// Generate test data
	fmt.Printf("🎲 Generating %d random vectors of dimension %d...\n", numVectors, dim)
	vectors := generateRandomVectors(numVectors, dim)

	// Benchmark different batch sizes
	batchSizes := []int{100, 500, 1000, 5000, 10000}

	for _, batchSize := range batchSizes {
		if batchSize > numVectors {
			continue
		}

		// Reset indexer
		indexer.vectors = nil
		indexer.currentSize = 0

		fmt.Printf("\n Batch size: %d\n", batchSize)

		totalTime := time.Duration(0)
		numBatches := 0

		for i := 0; i < numVectors; i += batchSize {
			end := i + batchSize
			if end > numVectors {
				end = numVectors
			}

			batch := vectors[i:end]
			startTime := time.Now()
			err := indexer.AddVectors(batch)
			elapsed := time.Since(startTime)

			if err != nil {
				log.Printf(" Failed to add batch: %v", err)
				break
			}

			totalTime += elapsed
			numBatches++
		}

		throughput := float64(numVectors) / totalTime.Seconds()
		avgBatchTime := totalTime / time.Duration(numBatches)

		fmt.Printf("   Total time: %.2fs\n", totalTime.Seconds())
		fmt.Printf("   Avg batch time: %.2fms\n", float64(avgBatchTime.Nanoseconds())/1e6)
		fmt.Printf("   Throughput: %.0f vectors/sec\n", throughput)
	}
}

// benchmarkSearch tests search performance
func benchmarkSearch(precision PrecisionMode, numVectors int, dim int, numQueries int) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf(" SEARCH BENCHMARK - %s PRECISION\n", precision)
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Create and populate indexer
	indexer, err := NewGPUIndexer(int64(dim), numVectors, precision)
	if err != nil {
		log.Printf(" Failed to create indexer: %v", err)
		return
	}
	defer indexer.Close()

	// Add vectors to index
	fmt.Printf("📚 Building index with %d vectors...\n", numVectors)
	vectors := generateRandomVectors(numVectors, dim)
	err = indexer.AddVectors(vectors)
	if err != nil {
		log.Printf(" Failed to build index: %v", err)
		return
	}

	// Generate query vectors
	queries := generateRandomVectors(numQueries, dim)
	k := 10

	// Warmup
	fmt.Printf(" Warming up...\n")
	for i := 0; i < 5; i++ {
		_, _, _ = indexer.Search(queries[0], k)
	}

	// Single query benchmark
	fmt.Printf("\n Single query performance (k=%d):\n", k)
	singleTimes := make([]time.Duration, numQueries)

	for i := 0; i < numQueries; i++ {
		startTime := time.Now()
		_, _, err := indexer.Search(queries[i], k)
		elapsed := time.Since(startTime)

		if err != nil {
			log.Printf(" Search failed: %v", err)
			continue
		}

		singleTimes[i] = elapsed
	}

	// Calculate statistics
	var totalTime time.Duration
	minTime := singleTimes[0]
	maxTime := singleTimes[0]

	for _, t := range singleTimes {
		totalTime += t
		if t < minTime {
			minTime = t
		}
		if t > maxTime {
			maxTime = t
		}
	}

	avgTime := totalTime / time.Duration(numQueries)
	qps := float64(numQueries) / totalTime.Seconds()

	fmt.Printf("   Queries: %d\n", numQueries)
	fmt.Printf("   Avg latency: %.3fμs\n", float64(avgTime.Nanoseconds())/1e3)
	fmt.Printf("   Min latency: %.3fμs\n", float64(minTime.Nanoseconds())/1e3)
	fmt.Printf("   Max latency: %.3fμs\n", float64(maxTime.Nanoseconds())/1e3)
	fmt.Printf("   Throughput: %.0f queries/sec\n", qps)

	// Batch query benchmark
	fmt.Printf("\n Batch query performance:\n")
	batchSizes := []int{10, 50, 100}

	for _, batchSize := range batchSizes {
		if batchSize > numQueries {
			batchSize = numQueries
		}

		batchQueries := queries[:batchSize]
		startTime := time.Now()
		_, _, err := indexer.BatchSearch(batchQueries, k)
		elapsed := time.Since(startTime)

		if err != nil {
			log.Printf(" Batch search failed: %v", err)
			continue
		}

		batchQPS := float64(batchSize) / elapsed.Seconds()
		avgBatchLatency := float64(elapsed.Nanoseconds()) / float64(batchSize) / 1e3

		fmt.Printf("   Batch size %d: %.2fms total, %.3fμs/query, %.0f qps\n",
			batchSize, float64(elapsed.Nanoseconds())/1e6, avgBatchLatency, batchQPS)
	}
}

// compareAccuracy compares accuracy between precisions
func compareAccuracy(numVectors int, dim int) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf(" ACCURACY COMPARISON - INT8 vs FP32\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	// Generate test vectors
	vectors := generateRandomVectors(numVectors, dim)
	queries := generateRandomVectors(10, dim)
	k := 5

	// Create FP32 indexer (ground truth)
	fp32Indexer, _ := NewGPUIndexer(int64(dim), numVectors, FP32)
	defer fp32Indexer.Close()
	fp32Indexer.AddVectors(vectors)

	// Create INT8 indexer
	int8Indexer, _ := NewGPUIndexer(int64(dim), numVectors, INT8)
	defer int8Indexer.Close()
	int8Indexer.AddVectors(vectors)

	fmt.Printf(" Comparing top-%d results for %d queries:\n", k, len(queries))

	totalRecall := float32(0)

	for i, query := range queries {
		fp32Indices, fp32Scores, _ := fp32Indexer.Search(query, k)
		int8Indices, int8Scores, _ := int8Indexer.Search(query, k)

		// Calculate recall@k
		matches := 0
		for _, idx := range int8Indices {
			for _, fp32Idx := range fp32Indices {
				if idx == fp32Idx {
					matches++
					break
				}
			}
		}

		recall := float32(matches) / float32(k)
		totalRecall += recall

		// Calculate score difference
		avgScoreDiff := float32(0)
		for j := 0; j < len(int8Scores) && j < len(fp32Scores); j++ {
			diff := math.Abs(float64(int8Scores[j] - fp32Scores[j]))
			avgScoreDiff += float32(diff)
		}
		if len(int8Scores) > 0 {
			avgScoreDiff /= float32(len(int8Scores))
		}

		fmt.Printf("   Q%2d: Recall@%d=%.2f, Avg score diff=%.6f\n",
			i+1, k, recall, avgScoreDiff)
	}

	avgRecall := totalRecall / float32(len(queries))
	fmt.Printf("\n Overall Recall@%d: %.2f%%\n", k, avgRecall*100)

	if avgRecall >= 0.9 {
		fmt.Printf(" INT8 quantization maintains excellent accuracy (>90%% recall)\n")
	} else if avgRecall >= 0.8 {
		fmt.Printf("  INT8 quantization shows good accuracy (>80%% recall)\n")
	} else {
		fmt.Printf(" INT8 quantization may need tuning (%.0f%% recall)\n", avgRecall*100)
	}
}

// benchmarkMemoryUsage compares memory usage between precisions
func benchmarkMemoryUsage(numVectors int, dim int) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf(" MEMORY USAGE COMPARISON\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	vectors := generateRandomVectors(numVectors, dim)

	// Calculate theoretical memory usage
	fp32Size := numVectors * dim * 4 // 4 bytes per float32
	fp16Size := numVectors * dim * 2 // 2 bytes per float16
	int8Size := numVectors * dim * 1 // 1 byte per int8

	fmt.Printf(" Theoretical memory usage for %d vectors of dimension %d:\n", numVectors, dim)
	fmt.Printf("   FP32: %.2f MB\n", float64(fp32Size)/(1024*1024))
	fmt.Printf("   FP16: %.2f MB (%.1fx reduction)\n",
		float64(fp16Size)/(1024*1024), float64(fp32Size)/float64(fp16Size))
	fmt.Printf("   INT8: %.2f MB (%.1fx reduction)\n",
		float64(int8Size)/(1024*1024), float64(fp32Size)/float64(int8Size))

	// Measure actual memory usage
	precisions := []PrecisionMode{FP32, FP16, INT8}

	for _, precision := range precisions {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		beforeAlloc := m.Alloc

		indexer, err := NewGPUIndexer(int64(dim), numVectors, precision)
		if err != nil {
			log.Printf(" Failed to create %s indexer: %v", precision, err)
			continue
		}

		err = indexer.AddVectors(vectors)
		if err != nil {
			log.Printf(" Failed to add vectors to %s indexer: %v", precision, err)
			indexer.Close()
			continue
		}

		runtime.ReadMemStats(&m)
		afterAlloc := m.Alloc
		memUsed := afterAlloc - beforeAlloc

		fmt.Printf("\n%s actual memory: %.2f MB\n", precision, float64(memUsed)/(1024*1024))

		indexer.Close()
	}
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println(" GPU LIBTORCH INDEXING BENCHMARK WITH INT8 QUANTIZATION")
	fmt.Println("================================================================================")
	fmt.Printf("System: %d CPUs, CUDA: %v\n", runtime.NumCPU(), gotch.CudaIfAvailable())
	fmt.Println()

	// Configuration
	numVectors := 100000
	dim := 384
	numQueries := 1000

	// Set random seed for reproducibility
	rand.Seed(42)

	// 1. Memory usage comparison
	benchmarkMemoryUsage(numVectors, dim)

	// 2. Indexing performance benchmarks
	fmt.Printf("\n Running indexing benchmarks with %d vectors...\n", numVectors)
	benchmarkIndexing(FP32, numVectors, dim)
	benchmarkIndexing(INT8, numVectors, dim)

	// 3. Search performance benchmarks
	fmt.Printf("\n Running search benchmarks...\n")
	benchmarkSearch(FP32, numVectors, dim, numQueries)
	benchmarkSearch(INT8, numVectors, dim, numQueries)

	// 4. Accuracy comparison
	compareAccuracy(10000, dim)

	// Summary
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf(" BENCHMARK COMPLETED\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))
	fmt.Printf("Key Findings:\n")
	fmt.Printf("  • INT8 quantization provides ~4x memory reduction\n")
	fmt.Printf("  • INT8 search can be 2-4x faster on supported hardware\n")
	fmt.Printf("  • Accuracy loss is typically <5%% with proper quantization\n")
	fmt.Printf("  • GPU acceleration provides massive parallelism for batch operations\n")
	fmt.Printf("\nOptimization tips:\n")
	fmt.Printf("  • Use larger batch sizes for better GPU utilization\n")
	fmt.Printf("  • Consider FP16 for balanced speed/accuracy trade-off\n")
	fmt.Printf("  • Profile your specific workload to find optimal settings\n")
}
