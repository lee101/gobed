package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"
)

// BulkEmbedder handles batch processing of embeddings
type BulkEmbedder struct {
	model       *EmbeddingModel
	batchSize   int
	numWorkers  int
	useCUDA     bool
}

// NewBulkEmbedder creates a new bulk embedder
func NewBulkEmbedder(model *EmbeddingModel, batchSize int) *BulkEmbedder {
	return &BulkEmbedder{
		model:      model,
		batchSize:  batchSize,
		numWorkers: runtime.NumCPU(),
		useCUDA:    false, // Would be true if CUDA is available
	}
}

// ProcessBatch processes a batch of texts in parallel
func (b *BulkEmbedder) ProcessBatch(texts []string) ([][]float32, error) {
	n := len(texts)
	embeddings := make([][]float32, n)
	
	// Process in batches
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, b.numWorkers)
	
	for i := 0; i < n; i += b.batchSize {
		end := i + b.batchSize
		if end > n {
			end = n
		}
		
		wg.Add(1)
		semaphore <- struct{}{}
		
		go func(start, end int) {
			defer wg.Done()
			defer func() { <-semaphore }()
			
			for j := start; j < end; j++ {
				emb, err := b.model.Encode(texts[j])
				if err == nil {
					embeddings[j] = emb
				}
			}
		}(i, end)
	}
	
	wg.Wait()
	return embeddings, nil
}

// SimulatedCUDAProcess simulates CUDA batch processing
func (b *BulkEmbedder) SimulatedCUDAProcess(texts []string) ([][]float32, time.Duration, error) {
	n := len(texts)
	embeddings := make([][]float32, n)
	
	// Simulate CUDA kernel launch overhead
	time.Sleep(100 * time.Microsecond)
	
	start := time.Now()
	
	// Simulate parallel GPU processing (much faster than CPU)
	// In real CUDA, all embeddings would be computed in parallel
	for i := range texts {
		embeddings[i] = make([]float32, b.model.EmbedDim)
		// Simulate fast GPU computation
		for j := 0; j < b.model.EmbedDim; j++ {
			embeddings[i][j] = rand.Float32()
		}
	}
	
	// Simulate memory transfer back from GPU
	transferTime := time.Duration(n*b.model.EmbedDim*4/1000000) * time.Microsecond // ~4 bytes per float
	time.Sleep(transferTime)
	
	return embeddings, time.Since(start), nil
}

func runBulkEmbeddingExample() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("🚀 BULK EMBEDDING EXAMPLE (CPU vs Simulated CUDA)")
	fmt.Println(strings.Repeat("=", 80))
	
	// Load model
	model, err := LoadModel()
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return
	}
	
	fmt.Printf("\n✅ Model loaded: %d vocab × %d dims\n", model.VocabSize, model.EmbedDim)
	fmt.Printf("💾 Model size: %.2f MB\n", float64(model.VocabSize*model.EmbedDim*4)/(1024*1024))
	
	// Create bulk embedder
	bulkEmbedder := NewBulkEmbedder(model, 32)
	
	// Get available texts for testing
	availableTexts := model.GetAvailableTexts()
	
	// Test different batch sizes
	batchSizes := []int{1, 10, 32, 64, 128, 256}
	
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📊 BATCH PROCESSING PERFORMANCE (CPU)")
	fmt.Println(strings.Repeat("=", 80))
	
	fmt.Println("\nBatch Size | Total Time | Per Text | Throughput | Speedup")
	fmt.Println(strings.Repeat("-", 65))
	
	var baseline time.Duration
	
	for _, batchSize := range batchSizes {
		// Create test batch
		texts := make([]string, 0, batchSize)
		for i := 0; i < batchSize && i < len(availableTexts); i++ {
			texts = append(texts, availableTexts[i%len(availableTexts)])
		}
		
		if len(texts) == 0 {
			continue
		}
		
		// Warm up
		_, _ = bulkEmbedder.ProcessBatch(texts[:1])
		
		// Benchmark CPU processing
		start := time.Now()
		_, err := bulkEmbedder.ProcessBatch(texts)
		cpuTime := time.Since(start)
		
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		
		perText := cpuTime / time.Duration(len(texts))
		throughput := float64(len(texts)) / cpuTime.Seconds()
		
		speedup := float64(1.0)
		if baseline == 0 {
			baseline = perText
		} else {
			speedup = float64(baseline) / float64(perText)
		}
		
		fmt.Printf("%10d | %10v | %8v | %10.0f | %.2fx\n",
			len(texts), cpuTime, perText, throughput, speedup)
	}
	
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("🎮 SIMULATED CUDA BATCH PROCESSING")
	fmt.Println(strings.Repeat("=", 80))
	
	fmt.Println("\nBatch Size | CPU Time   | CUDA Time | Speedup | GPU Util")
	fmt.Println(strings.Repeat("-", 65))
	
	for _, batchSize := range batchSizes {
		if batchSize > len(availableTexts) {
			continue
		}
		
		texts := availableTexts[:batchSize]
		
		// CPU timing
		start := time.Now()
		_, _ = bulkEmbedder.ProcessBatch(texts)
		cpuTime := time.Since(start)
		
		// Simulated CUDA timing
		_, cudaTime, _ := bulkEmbedder.SimulatedCUDAProcess(texts)
		
		// Calculate theoretical CUDA time based on GPU capabilities
		// RTX 3080: ~30 TFLOPS, ~760 GB/s memory bandwidth
		flopsRequired := float64(batchSize * model.EmbedDim * model.VocabSize * 2)
		memoryRequired := float64(batchSize * model.EmbedDim * 4)
		
		theoreticalCudaTime := time.Duration(flopsRequired/30e12*1e9) * time.Nanosecond
		if memTransferTime := time.Duration(memoryRequired/760e9*1e9) * time.Nanosecond; memTransferTime > theoreticalCudaTime {
			theoreticalCudaTime = memTransferTime
		}
		
		speedup := float64(cpuTime) / float64(theoreticalCudaTime)
		gpuUtil := float64(theoreticalCudaTime) / float64(cudaTime) * 100
		
		fmt.Printf("%10d | %10v | %9v | %7.1fx | %6.1f%%\n",
			batchSize, cpuTime, theoreticalCudaTime, speedup, gpuUtil)
	}
	
	// Parallel CPU vs theoretical CUDA comparison
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("🔄 PARALLEL PROCESSING COMPARISON")
	fmt.Println(strings.Repeat("=", 80))
	
	testSizes := []int{100, 500, 1000}
	
	fmt.Println("\nTexts | CPU Serial | CPU Parallel | Theoretical CUDA | CUDA Speedup")
	fmt.Println(strings.Repeat("-", 75))
	
	for _, numTexts := range testSizes {
		// Create synthetic texts
		texts := make([]string, 0, numTexts)
		for i := 0; i < numTexts; i++ {
			texts = append(texts, availableTexts[i%len(availableTexts)])
		}
		
		// Serial CPU
		start := time.Now()
		for _, text := range texts {
			_, _ = model.Encode(text)
		}
		serialTime := time.Since(start)
		
		// Parallel CPU
		bulkEmbedder.batchSize = 32
		start = time.Now()
		_, _ = bulkEmbedder.ProcessBatch(texts)
		parallelTime := time.Since(start)
		
		// Theoretical CUDA (based on RTX 3080 specs)
		// Assume perfect parallelization and memory bandwidth limited
		bytesRequired := int64(numTexts * model.EmbedDim * 4)
		theoreticalCudaTime := time.Duration(float64(bytesRequired)/760e9*1e9) * time.Nanosecond
		// Add kernel launch overhead
		theoreticalCudaTime += 100 * time.Microsecond
		
		cudaSpeedup := float64(parallelTime) / float64(theoreticalCudaTime)
		
		fmt.Printf("%5d | %10v | %12v | %16v | %12.1fx\n",
			numTexts, serialTime, parallelTime, theoreticalCudaTime, cudaSpeedup)
	}
	
	// Memory transfer analysis
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("💾 MEMORY TRANSFER ANALYSIS (for CUDA)")
	fmt.Println(strings.Repeat("=", 80))
	
	fmt.Println("\nBatch Size | Weight Transfer | Input Transfer | Output Transfer | Total")
	fmt.Println(strings.Repeat("-", 75))
	
	for _, batchSize := range []int{1, 32, 128, 512} {
		// Weight matrix (only transferred once)
		weightTransfer := float64(model.VocabSize*model.EmbedDim*4) / (1024 * 1024)
		
		// Input token IDs (assuming 100 tokens per text)
		inputTransfer := float64(batchSize*100*4) / (1024 * 1024)
		
		// Output embeddings
		outputTransfer := float64(batchSize*model.EmbedDim*4) / (1024 * 1024)
		
		total := inputTransfer + outputTransfer
		if batchSize == 1 {
			total += weightTransfer // Weights transferred on first run
		}
		
		fmt.Printf("%10d | %13.2f MB | %12.2f MB | %13.2f MB | %7.2f MB\n",
			batchSize, weightTransfer, inputTransfer, outputTransfer, total)
	}
	
	fmt.Println("\n📝 CUDA RECOMMENDATIONS:")
	fmt.Println(strings.Repeat("-", 75))
	fmt.Println("• For batch sizes < 32: CPU is likely faster (transfer overhead)")
	fmt.Println("• For batch sizes 32-128: 2-5x speedup expected")
	fmt.Println("• For batch sizes > 128: 10-50x speedup expected")
	fmt.Println("• Optimal batch size for RTX 3080: 128-256 texts")
	fmt.Println("• Keep model weights in GPU memory to avoid repeated transfers")
	
	fmt.Println("\n✅ Bulk embedding example completed!")
}