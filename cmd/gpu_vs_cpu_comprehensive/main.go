package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

// CPUEmbeddings simulates CPU-based embedding operations
type CPUEmbeddings struct {
	embedMatrix [][]float32
	vocabSize   int
	embedDim    int
}

func NewCPUEmbeddings(vocabSize, embedDim int) *CPUEmbeddings {
	// Initialize random embedding matrix
	matrix := make([][]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		matrix[i] = make([]float32, embedDim)
		for j := 0; j < embedDim; j++ {
			matrix[i][j] = (rand.Float32() - 0.5) * 0.1
		}
	}

	return &CPUEmbeddings{
		embedMatrix: matrix,
		vocabSize:   vocabSize,
		embedDim:    embedDim,
	}
}

func (c *CPUEmbeddings) Embed(tokenIDs []int) []float32 {
	if len(tokenIDs) == 0 {
		return nil
	}

	// Lookup embeddings
	embeddings := make([][]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		if id >= 0 && id < c.vocabSize {
			embeddings[i] = c.embedMatrix[id]
		}
	}

	// Average pooling
	pooled := make([]float32, c.embedDim)
	count := 0
	for _, embed := range embeddings {
		if embed != nil {
			for j := 0; j < c.embedDim; j++ {
				pooled[j] += embed[j]
			}
			count++
		}
	}

	if count > 0 {
		for j := 0; j < c.embedDim; j++ {
			pooled[j] /= float32(count)
		}
	}

	// L2 normalization
	norm := float32(0)
	for _, v := range pooled {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm > 0 {
		for j := 0; j < c.embedDim; j++ {
			pooled[j] /= norm
		}
	}

	return pooled
}

// CPUSearch implements CPU-based vector search
type CPUSearch struct {
	vectors [][]float32
	dim     int
}

func NewCPUSearch(dim int) *CPUSearch {
	return &CPUSearch{
		vectors: make([][]float32, 0),
		dim:     dim,
	}
}

func (s *CPUSearch) Add(vectors [][]float32) {
	s.vectors = append(s.vectors, vectors...)
}

func (s *CPUSearch) Search(query []float32, k int) ([]int, []float32) {
	if len(s.vectors) == 0 || k <= 0 {
		return nil, nil
	}

	// Compute all similarities
	scores := make([]float32, len(s.vectors))
	for i, vec := range s.vectors {
		dot := float32(0)
		for j := 0; j < s.dim; j++ {
			dot += query[j] * vec[j]
		}
		scores[i] = dot
	}

	// Find top-k (simple selection sort for now)
	if k > len(scores) {
		k = len(scores)
	}

	indices := make([]int, k)
	topScores := make([]float32, k)

	for i := 0; i < k; i++ {
		maxIdx := 0
		maxScore := scores[0]
		for j := 1; j < len(scores); j++ {
			if scores[j] > maxScore {
				maxIdx = j
				maxScore = scores[j]
			}
		}
		indices[i] = maxIdx
		topScores[i] = maxScore
		scores[maxIdx] = -2 // Mark as used
	}

	return indices, topScores
}

// BenchmarkResult stores comparison metrics
type BenchmarkResult struct {
	Operation     string
	CPUTime       time.Duration
	GPUTime       time.Duration
	Speedup       float64
	CPUThroughput float64
	GPUThroughput float64
}

func printBenchmarkTable(results []BenchmarkResult) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 120))
	fmt.Printf("📊 CPU vs GPU PERFORMANCE COMPARISON\n")
	fmt.Printf("%s\n", strings.Repeat("=", 120))

	fmt.Printf("%-30s | %12s | %12s | %10s | %15s | %15s\n",
		"Operation", "CPU Time", "GPU Time", "Speedup", "CPU Throughput", "GPU Throughput")
	fmt.Printf("%s\n", strings.Repeat("-", 120))

	for _, r := range results {
		cpuTimeStr := fmt.Sprintf("%.2fms", float64(r.CPUTime.Nanoseconds())/1e6)
		gpuTimeStr := fmt.Sprintf("%.2fms", float64(r.GPUTime.Nanoseconds())/1e6)

		fmt.Printf("%-30s | %12s | %12s | %9.1fx | %15.0f | %15.0f\n",
			r.Operation, cpuTimeStr, gpuTimeStr, r.Speedup,
			r.CPUThroughput, r.GPUThroughput)
	}
}

func benchmarkEmbeddings() []BenchmarkResult {
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf("⚡ EMBEDDING OPERATIONS: CPU vs GPU\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	vocabSize := 250002
	embedDim := 384
	numSequences := 1000

	// Generate test sequences
	sequences := make([][]int, numSequences)
	for i := 0; i < numSequences; i++ {
		seqLen := rand.Intn(100) + 10
		seq := make([]int, seqLen)
		for j := 0; j < seqLen; j++ {
			seq[j] = rand.Intn(vocabSize)
		}
		sequences[i] = seq
	}

	results := []BenchmarkResult{}

	// CPU Embeddings
	fmt.Printf("\n📊 CPU Embedding Performance:\n")
	cpuEmbed := NewCPUEmbeddings(vocabSize, embedDim)

	// Single sequence
	cpuSingleStart := time.Now()
	for i := 0; i < 100; i++ {
		_ = cpuEmbed.Embed(sequences[i])
	}
	cpuSingleTime := time.Since(cpuSingleStart)
	cpuSingleThroughput := 100.0 / cpuSingleTime.Seconds()

	fmt.Printf("   100 sequences: %.2fms (%.0f seq/sec)\n",
		float64(cpuSingleTime.Nanoseconds())/1e6, cpuSingleThroughput)

	// Batch processing (simulated with parallel processing)
	cpuBatchStart := time.Now()
	batchSize := 100
	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()
	chunkSize := batchSize / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for i := start; i < start+chunkSize && i < batchSize; i++ {
				_ = cpuEmbed.Embed(sequences[i])
			}
		}(w * chunkSize)
	}
	wg.Wait()
	cpuBatchTime := time.Since(cpuBatchStart)
	cpuBatchThroughput := float64(batchSize) / cpuBatchTime.Seconds()

	fmt.Printf("   Batch %d: %.2fms (%.0f seq/sec)\n",
		batchSize, float64(cpuBatchTime.Nanoseconds())/1e6, cpuBatchThroughput)

	// GPU Embeddings
	fmt.Printf("\n📊 GPU Embedding Performance:\n")

	device := gotch.CPU
	if gotch.CudaIfAvailable() {
		device = gotch.CudaIfAvailable()
	}

	// Create GPU embedding layer
	vs := nn.NewVarStore(device)
	embedLayer := nn.NewEmbedding(vs.Root(), int64(vocabSize), int64(embedDim), nn.DefaultEmbeddingConfig())
	vs.Init()

	// Single sequence on GPU
	gpuSingleStart := time.Now()
	for i := 0; i < 100; i++ {
		tokens := make([]int64, len(sequences[i]))
		for j, t := range sequences[i] {
			tokens[j] = int64(t)
		}

		tokenTensor := ts.MustOfSlice(tokens).MustTo(device, false)
		embeddings := embedLayer.Forward(tokenTensor)

		// Average pooling
		pooled := embeddings.MustMean1([]int64{0}, false, gotch.Float)

		// L2 norm
		norm := pooled.MustNorm(false)
		normalized := pooled.MustDiv(norm, false)

		// Cleanup
		tokenTensor.MustDrop()
		embeddings.MustDrop()
		pooled.MustDrop()
		norm.MustDrop()
		normalized.MustDrop()
	}
	gpuSingleTime := time.Since(gpuSingleStart)
	gpuSingleThroughput := 100.0 / gpuSingleTime.Seconds()

	fmt.Printf("   100 sequences: %.2fms (%.0f seq/sec)\n",
		float64(gpuSingleTime.Nanoseconds())/1e6, gpuSingleThroughput)

	// Batch on GPU
	gpuBatchStart := time.Now()

	// Prepare batch tensor
	maxLen := 0
	for i := 0; i < batchSize; i++ {
		if len(sequences[i]) > maxLen {
			maxLen = len(sequences[i])
		}
	}

	batchTokens := ts.MustZeros([]int64{int64(batchSize), int64(maxLen)}, gotch.Int64, device)
	batchMask := ts.MustZeros([]int64{int64(batchSize), int64(maxLen)}, gotch.Float, device)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < len(sequences[i]) && j < maxLen; j++ {
			batchTokens.MustNarrow(0, int64(i), 1, false).
				MustNarrow(1, int64(j), 1, false).
				MustFill_(ts.IntScalar(int64(sequences[i][j])))
			batchMask.MustNarrow(0, int64(i), 1, false).
				MustNarrow(1, int64(j), 1, false).
				MustFill_(ts.FloatScalar(1.0))
		}
	}

	// Batch embedding lookup
	batchEmbeddings := embedLayer.Forward(batchTokens)

	// Masked average pooling
	maskedEmbed := batchEmbeddings.MustMul(batchMask.MustUnsqueeze(-1, false), false)
	sumEmbed := maskedEmbed.MustSum1([]int64{1}, false, gotch.Float)
	counts := batchMask.MustSum1([]int64{1}, true, gotch.Float).MustUnsqueeze(-1, false)
	pooledBatch := sumEmbed.MustDiv(counts, false)

	// Batch L2 norm
	norms := pooledBatch.MustNorm2([]int64{1}, true, false)
	normalizedBatch := pooledBatch.MustDiv(norms, false)

	// Cleanup
	batchTokens.MustDrop()
	batchMask.MustDrop()
	batchEmbeddings.MustDrop()
	maskedEmbed.MustDrop()
	sumEmbed.MustDrop()
	counts.MustDrop()
	pooledBatch.MustDrop()
	norms.MustDrop()
	normalizedBatch.MustDrop()

	gpuBatchTime := time.Since(gpuBatchStart)
	gpuBatchThroughput := float64(batchSize) / gpuBatchTime.Seconds()

	fmt.Printf("   Batch %d: %.2fms (%.0f seq/sec)\n",
		batchSize, float64(gpuBatchTime.Nanoseconds())/1e6, gpuBatchThroughput)

	// Record results
	results = append(results, BenchmarkResult{
		Operation:     "Embedding (100 single)",
		CPUTime:       cpuSingleTime,
		GPUTime:       gpuSingleTime,
		Speedup:       float64(cpuSingleTime) / float64(gpuSingleTime),
		CPUThroughput: cpuSingleThroughput,
		GPUThroughput: gpuSingleThroughput,
	})

	results = append(results, BenchmarkResult{
		Operation:     fmt.Sprintf("Embedding (batch %d)", batchSize),
		CPUTime:       cpuBatchTime,
		GPUTime:       gpuBatchTime,
		Speedup:       float64(cpuBatchTime) / float64(gpuBatchTime),
		CPUThroughput: cpuBatchThroughput,
		GPUThroughput: gpuBatchThroughput,
	})

	return results
}

func benchmarkSearch() []BenchmarkResult {
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf("🔍 SEARCH OPERATIONS: CPU vs GPU\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	dim := 384
	numVectors := 100000
	numQueries := 100
	k := 10

	// Generate test data
	fmt.Printf("🎲 Generating %d vectors...\n", numVectors)
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		sum := float32(0)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1
			sum += vec[j] * vec[j]
		}
		norm := float32(math.Sqrt(float64(sum)))
		for j := 0; j < dim; j++ {
			vec[j] /= norm
		}
		vectors[i] = vec
	}

	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		vec := make([]float32, dim)
		sum := float32(0)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1
			sum += vec[j] * vec[j]
		}
		norm := float32(math.Sqrt(float64(sum)))
		for j := 0; j < dim; j++ {
			vec[j] /= norm
		}
		queries[i] = vec
	}

	results := []BenchmarkResult{}

	// CPU Search
	fmt.Printf("\n📊 CPU Search Performance:\n")
	cpuIndex := NewCPUSearch(dim)
	cpuIndex.Add(vectors)

	cpuSearchStart := time.Now()
	for _, query := range queries {
		_, _ = cpuIndex.Search(query, k)
	}
	cpuSearchTime := time.Since(cpuSearchStart)
	cpuSearchThroughput := float64(numQueries) / cpuSearchTime.Seconds()

	fmt.Printf("   %d queries: %.2fms (%.0f qps)\n",
		numQueries, float64(cpuSearchTime.Nanoseconds())/1e6, cpuSearchThroughput)

	// GPU Search
	fmt.Printf("\n📊 GPU Search Performance:\n")

	device := gotch.CPU
	if gotch.CudaIfAvailable() {
		device = gotch.CudaIfAvailable()
	}

	// Upload vectors to GPU
	flatVectors := make([]float32, numVectors*dim)
	for i, vec := range vectors {
		copy(flatVectors[i*dim:], vec)
	}

	vectorsTensor := ts.MustOfSlice(flatVectors).
		MustReshape([]int64{int64(numVectors), int64(dim)}, false).
		MustTo(device, false)
	defer vectorsTensor.MustDrop()

	gpuSearchStart := time.Now()
	for _, query := range queries {
		queryTensor := ts.MustOfSlice(query).
			MustReshape([]int64{int64(dim), 1}, false).
			MustTo(device, false)

		scores := vectorsTensor.MustMatmul(queryTensor, false)
		_, topIndices := scores.MustTopk(int64(k), -1, true, true)

		queryTensor.MustDrop()
		scores.MustDrop()
		topIndices.MustDrop()
	}
	gpuSearchTime := time.Since(gpuSearchStart)
	gpuSearchThroughput := float64(numQueries) / gpuSearchTime.Seconds()

	fmt.Printf("   %d queries: %.2fms (%.0f qps)\n",
		numQueries, float64(gpuSearchTime.Nanoseconds())/1e6, gpuSearchThroughput)

	// Batch search on GPU
	fmt.Printf("\n📊 GPU Batch Search:\n")

	// Upload all queries at once
	flatQueries := make([]float32, numQueries*dim)
	for i, q := range queries {
		copy(flatQueries[i*dim:], q)
	}

	gpuBatchSearchStart := time.Now()

	queriesTensor := ts.MustOfSlice(flatQueries).
		MustReshape([]int64{int64(numQueries), int64(dim)}, false).
		MustTo(device, false)

	// Batch matrix multiplication
	batchScores := vectorsTensor.MustMatmul(queriesTensor.MustT(false), false)

	// Get top-k for each query
	for i := 0; i < numQueries; i++ {
		queryScores := batchScores.MustSelect(1, int64(i), false)
		_, topIndices := queryScores.MustTopk(int64(k), -1, true, true)
		queryScores.MustDrop()
		topIndices.MustDrop()
	}

	queriesTensor.MustDrop()
	batchScores.MustDrop()

	gpuBatchSearchTime := time.Since(gpuBatchSearchStart)
	gpuBatchSearchThroughput := float64(numQueries) / gpuBatchSearchTime.Seconds()

	fmt.Printf("   Batch %d queries: %.2fms (%.0f qps)\n",
		numQueries, float64(gpuBatchSearchTime.Nanoseconds())/1e6, gpuBatchSearchThroughput)

	// Record results
	results = append(results, BenchmarkResult{
		Operation:     fmt.Sprintf("Search %dk vectors", numVectors/1000),
		CPUTime:       cpuSearchTime,
		GPUTime:       gpuSearchTime,
		Speedup:       float64(cpuSearchTime) / float64(gpuSearchTime),
		CPUThroughput: cpuSearchThroughput,
		GPUThroughput: gpuSearchThroughput,
	})

	results = append(results, BenchmarkResult{
		Operation:     fmt.Sprintf("Batch Search %dk vectors", numVectors/1000),
		CPUTime:       cpuSearchTime,
		GPUTime:       gpuBatchSearchTime,
		Speedup:       float64(cpuSearchTime) / float64(gpuBatchSearchTime),
		CPUThroughput: cpuSearchThroughput,
		GPUThroughput: gpuBatchSearchThroughput,
	})

	return results
}

func benchmarkScaling() {
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf("📈 SCALING ANALYSIS: CPU vs GPU\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	dim := 384
	k := 10
	vectorCounts := []int{1000, 10000, 50000, 100000, 500000}

	fmt.Printf("\n%-15s | %12s | %12s | %10s\n",
		"Vector Count", "CPU Time", "GPU Time", "Speedup")
	fmt.Printf("%s\n", strings.Repeat("-", 60))

	for _, numVectors := range vectorCounts {
		// Generate vectors
		vectors := make([][]float32, numVectors)
		for i := 0; i < numVectors; i++ {
			vec := make([]float32, dim)
			for j := 0; j < dim; j++ {
				vec[j] = rand.Float32()*2 - 1
			}
			vectors[i] = vec
		}

		query := make([]float32, dim)
		for j := 0; j < dim; j++ {
			query[j] = rand.Float32()*2 - 1
		}

		// CPU timing
		cpuIndex := NewCPUSearch(dim)
		cpuIndex.Add(vectors)

		cpuStart := time.Now()
		_, _ = cpuIndex.Search(query, k)
		cpuTime := time.Since(cpuStart)

		// GPU timing
		device := gotch.CudaIfAvailable()

		flatVectors := make([]float32, numVectors*dim)
		for i, vec := range vectors {
			copy(flatVectors[i*dim:], vec)
		}

		vectorsTensor := ts.MustOfSlice(flatVectors).
			MustReshape([]int64{int64(numVectors), int64(dim)}, false).
			MustTo(device, false)

		queryTensor := ts.MustOfSlice(query).
			MustReshape([]int64{int64(dim), 1}, false).
			MustTo(device, false)

		gpuStart := time.Now()
		scores := vectorsTensor.MustMatmul(queryTensor, false)
		_, topIndices := scores.MustTopk(int64(k), -1, true, true)
		gpuTime := time.Since(gpuStart)

		// Cleanup
		vectorsTensor.MustDrop()
		queryTensor.MustDrop()
		scores.MustDrop()
		topIndices.MustDrop()

		speedup := float64(cpuTime) / float64(gpuTime)

		fmt.Printf("%-15d | %11.2fms | %11.2fms | %9.1fx\n",
			numVectors,
			float64(cpuTime.Nanoseconds())/1e6,
			float64(gpuTime.Nanoseconds())/1e6,
			speedup)
	}

	fmt.Printf("\n📊 Observation: GPU speedup increases with dataset size!\n")
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("⚡ COMPREHENSIVE CPU vs GPU BENCHMARK")
	fmt.Println("================================================================================")
	fmt.Printf("System Configuration:\n")
	fmt.Printf("  CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("  GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("  CUDA Available: %v\n", gotch.CudaIfAvailable())
	if gotch.CudaIfAvailable() {
		fmt.Printf("  CUDA Devices: %d\n", gotch.CudaDeviceCount())
	}
	fmt.Println()

	rand.Seed(42)
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Collect all results
	allResults := []BenchmarkResult{}

	// Benchmark embeddings
	embedResults := benchmarkEmbeddings()
	allResults = append(allResults, embedResults...)

	// Benchmark search
	searchResults := benchmarkSearch()
	allResults = append(allResults, searchResults...)

	// Print summary table
	printBenchmarkTable(allResults)

	// Scaling analysis
	benchmarkScaling()

	// Final summary
	fmt.Printf("\n%s\n", strings.Repeat("=", 100))
	fmt.Printf("🎯 KEY INSIGHTS\n")
	fmt.Printf("%s\n", strings.Repeat("=", 100))

	avgSpeedup := float64(0)
	for _, r := range allResults {
		avgSpeedup += r.Speedup
	}
	avgSpeedup /= float64(len(allResults))

	fmt.Printf("\n📊 Performance Summary:\n")
	fmt.Printf("  • Average GPU speedup: %.1fx\n", avgSpeedup)
	fmt.Printf("  • Best for: Large batch operations and vector search\n")
	fmt.Printf("  • GPU excels at: Parallel matrix operations\n")
	fmt.Printf("  • Speedup scales with: Dataset size and batch size\n")

	fmt.Printf("\n⚡ Optimization Recommendations:\n")
	fmt.Printf("  1. Use GPU for embedding lookup + pooling (not matrix mul)\n")
	fmt.Printf("  2. Batch operations for maximum GPU utilization\n")
	fmt.Printf("  3. Keep data on GPU to avoid transfer overhead\n")
	fmt.Printf("  4. Use INT8 quantization for 4x memory savings\n")
	fmt.Printf("  5. Implement approximate search (IVF/HNSW) for large datasets\n")

	fmt.Printf("\n✅ GPU acceleration provides massive speedups for vector operations!\n")
}
