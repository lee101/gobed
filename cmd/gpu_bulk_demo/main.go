// +build gpu

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Printf(" GPU Bulk Indexing Demo\n")
	fmt.Printf("=========================\n\n")

	// Check GPU availability
	if !gobed.IsCUDAAvailable() {
		log.Fatal(" CUDA not available. GPU bulk indexing requires CUDA.")
	}

	// Display system info
	gpuInfo := gobed.GetGPUMemoryInfo()
	fmt.Printf("🖥  System Information:\n")
	fmt.Printf("   GPU VRAM: %d MB total, %d MB available\n", gpuInfo.TotalVRAM, gpuInfo.AvailableVRAM)
	fmt.Printf("   CPU Cores: %d\n", runtime.NumCPU())
	fmt.Printf("   Memory Utilization: %.1f%%\n\n", gpuInfo.Utilization)

	// Interactive mode selection
	mode := selectMode()
	
	switch mode {
	case "bulk":
		runBulkIndexingDemo()
	case "progressive":
		runProgressiveIndexingDemo()
	case "streaming":
		runStreamingDemo()
	case "benchmark":
		runBenchmarkDemo()
	default:
		fmt.Println("Invalid mode selected")
	}
}

func selectMode() string {
	fmt.Printf(" Select Demo Mode:\n")
	fmt.Printf("   1. bulk       - Bulk indexing demonstration\n")
	fmt.Printf("   2. progressive - Progressive indexing with live updates\n") 
	fmt.Printf("   3. streaming  - Streaming indexing from data source\n")
	fmt.Printf("   4. benchmark  - Performance benchmarking\n")
	fmt.Printf("\nEnter mode (1-4 or name): ")

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	input := strings.TrimSpace(scanner.Text())

	switch input {
	case "1", "bulk":
		return "bulk"
	case "2", "progressive":
		return "progressive"
	case "3", "streaming":
		return "streaming"
	case "4", "benchmark":
		return "benchmark"
	default:
		return "bulk"
	}
}

func runBulkIndexingDemo() {
	fmt.Printf(" Bulk Indexing Demo\n")
	fmt.Printf("=====================\n\n")

	// Get dataset size from user
	numVectors := getUserInput("Enter number of vectors to index", 50000)
	
	// Create indexer
	config := gobed.DefaultBulkIndexConfig()
	config.MaxMemoryMB = int(gobed.GetAvailableVRAM() * 0.8) // Use 80% of VRAM
	
	fmt.Printf(" Creating GPU bulk indexer with config:\n")
	fmt.Printf("   Clusters: %d, Search probes: %d\n", config.NList, config.NProbe)
	fmt.Printf("   Vector dim: %d, Max memory: %d MB\n", config.VectorDim, config.MaxMemoryMB)
	fmt.Printf("   Progressive mode: %v\n\n", config.ProgressiveMode)

	indexer, err := gobed.NewGPUIVFBulkIndexer(config)
	if err != nil {
		log.Fatalf(" Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Setup phase
	fmt.Printf("🔄 Setup Phase:\n")
	
	// Load embeddings
	fmt.Printf("   Loading quantized embeddings...")
	embeddings, scales := generateSampleEmbeddings(config.VocabSize, config.EmbedDim)
	start := time.Now()
	if err := indexer.LoadEmbeddings(embeddings, scales); err != nil {
		log.Fatalf(" Failed to load embeddings: %v", err)
	}
	fmt.Printf("  Done (%v)\n", time.Since(start))

	// Train k-means
	fmt.Printf("   Training GPU k-means...")
	trainingData := generateSampleTrainingData(5000, config.VectorDim)
	start = time.Now()
	if err := indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales)); err != nil {
		log.Fatalf(" Failed to train k-means: %v", err)
	}
	fmt.Printf("  Done (%v)\n", time.Since(start))

	// Generate token sequences
	fmt.Printf("   Generating %d token sequences...", numVectors)
	start = time.Now()
	tokenSequences := generateSampleTokenSequences(numVectors, config.VectorDim)
	fmt.Printf("  Done (%v)\n\n", time.Since(start))

	// Bulk indexing phase
	fmt.Printf(" Bulk Indexing Phase:\n")
	
	progressCallback := func(progress float64) {
		fmt.Printf("   Progress: %.1f%% complete\n", progress*100)
	}

	indexingStart := time.Now()
	indexed, err := indexer.BulkIndexTokenSequences(tokenSequences, progressCallback)
	indexingTime := time.Since(indexingStart)
	
	if err != nil {
		log.Fatalf(" Bulk indexing failed: %v", err)
	}

	// Results
	fmt.Printf("\n Results:\n")
	fmt.Printf("   Vectors indexed: %d\n", indexed)
	fmt.Printf("   Total time: %v\n", indexingTime)
	fmt.Printf("   Throughput: %.0f vectors/sec\n", float64(indexed)/indexingTime.Seconds())
	
	stats := indexer.GetStats()
	fmt.Printf("   Peak throughput: %.0f vectors/sec\n", stats.PeakThroughput)
	fmt.Printf("   Average latency: %.1f ms\n", stats.GetAvgLatencyMs())
	
	memUsage := indexer.GetMemoryUsage()
	fmt.Printf("   GPU memory used: %d MB\n", memUsage/1024/1024)

	fmt.Printf("\n Bulk indexing demo completed successfully!\n")
}

func runProgressiveIndexingDemo() {
	fmt.Printf(" Progressive Indexing Demo\n")
	fmt.Printf("============================\n\n")

	numBatches := getUserInput("Enter number of batches to process", 20)
	batchSize := getUserInput("Enter batch size", 1000)
	
	config := gobed.DefaultBulkIndexConfig()
	config.ProgressiveMode = true
	config.OnlineUpdates = true
	
	indexer, err := gobed.NewGPUIVFBulkIndexer(config)
	if err != nil {
		log.Fatalf(" Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Setup
	fmt.Printf(" Setting up indexer...\n")
	embeddings, scales := generateSampleEmbeddings(config.VocabSize, config.EmbedDim)
	indexer.LoadEmbeddings(embeddings, scales)
	
	trainingData := generateSampleTrainingData(2000, config.VectorDim)
	indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales))

	// Progressive indexing with live updates
	fmt.Printf("\n⏳ Progressive Indexing (Ctrl+C to stop):\n")
	
	// Setup signal handling
	ctx, cancel := context.WithCancel(context.Background())
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Printf("\n🛑 Received interrupt signal, stopping...\n")
		cancel()
	}()

	totalProcessed := 0
	startTime := time.Now()
	
	for batch := 0; batch < numBatches; batch++ {
		select {
		case <-ctx.Done():
			break
		default:
		}
		
		// Generate batch
		batchTokens := generateSampleTokenSequences(batchSize, config.VectorDim)
		
		// Process batch with progress tracking
		batchStart := time.Now()
		indexed, err := indexer.BulkIndexTokenSequences(batchTokens, nil)
		batchTime := time.Since(batchStart)
		
		if err != nil {
			fmt.Printf("    Batch %d failed: %v\n", batch+1, err)
			continue
		}
		
		totalProcessed += indexed
		overallTime := time.Since(startTime)
		throughput := float64(totalProcessed) / overallTime.Seconds()
		
		// Live progress display
		fmt.Printf("   Batch %d/%d: %d vectors, %.1fms, %.0f vec/sec overall\n",
			batch+1, numBatches, indexed, batchTime.Seconds()*1000, throughput)
		
		// Show memory usage every 5 batches
		if (batch+1) % 5 == 0 {
			memUsage := indexer.GetMemoryUsage()
			fmt.Printf("    Memory usage: %d MB, Progress: %.1f%%\n",
				memUsage/1024/1024, float64(batch+1)/float64(numBatches)*100)
		}
		
		// Brief pause to demonstrate progressive nature
		time.Sleep(100 * time.Millisecond)
	}
	
	totalTime := time.Since(startTime)
	fmt.Printf("\n Progressive Indexing Summary:\n")
	fmt.Printf("   Total vectors: %d\n", totalProcessed)
	fmt.Printf("   Total time: %v\n", totalTime)
	fmt.Printf("   Overall throughput: %.0f vectors/sec\n", float64(totalProcessed)/totalTime.Seconds())
	
	fmt.Printf("\n Progressive indexing demo completed!\n")
}

func runStreamingDemo() {
	fmt.Printf("🌊 Streaming Indexing Demo\n")
	fmt.Printf("==========================\n\n")

	totalVectors := getUserInput("Enter total vectors to stream", 10000)
	streamBatchSize := getUserInput("Enter stream batch size", 500)
	
	config := gobed.DefaultBulkIndexConfig()
	config.ProgressiveMode = true
	
	indexer, err := gobed.NewGPUIVFBulkIndexer(config)
	if err != nil {
		log.Fatalf(" Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Setup
	fmt.Printf(" Setting up streaming indexer...\n")
	embeddings, scales := generateSampleEmbeddings(config.VocabSize, config.EmbedDim)
	indexer.LoadEmbeddings(embeddings, scales)
	
	trainingData := generateSampleTrainingData(2000, config.VectorDim)
	indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales))

	// Create streaming channels
	tokenChan := make(chan []int, streamBatchSize)
	progressChan := make(chan gobed.IndexProgress, 100)

	// Start streaming indexer
	fmt.Printf("\n🌊 Starting streaming indexer...\n")
	go func() {
		if err := indexer.StreamingIndex(tokenChan, progressChan); err != nil {
			log.Printf(" Streaming indexer error: %v", err)
		}
	}()

	// Data producer goroutine
	go func() {
		defer close(tokenChan)
		
		fmt.Printf("📤 Streaming %d vectors in batches of %d...\n", totalVectors, streamBatchSize)
		
		for i := 0; i < totalVectors; i++ {
			tokens := generateSingleTokenSequence(config.VectorDim)
			
			select {
			case tokenChan <- tokens:
				// Successfully sent
			case <-time.After(5 * time.Second):
				fmt.Printf("  Timeout sending tokens at vector %d\n", i)
				return
			}
			
			// Small delay to simulate real-time streaming
			if i%100 == 0 {
				time.Sleep(10 * time.Millisecond)
			}
		}
		
		fmt.Printf("📤 Finished streaming all vectors\n")
	}()

	// Progress monitoring
	fmt.Printf(" Monitoring progress (updates every second):\n")
	lastUpdateTime := time.Now()
	
	for progress := range progressChan {
		now := time.Now()
		
		// Update display every second
		if now.Sub(lastUpdateTime) >= time.Second {
			fmt.Printf("   %d/%d (%.1f%%) - %.0f vec/sec - ETA: %v\n",
				progress.Current, progress.Total, progress.Percentage,
				progress.DocsPerSec, progress.TimeLeft)
			lastUpdateTime = now
		}
	}

	fmt.Printf("\n Final Statistics:\n")
	stats := indexer.GetStats()
	fmt.Printf("   Vectors processed: %d\n", stats.TotalVectors)
	fmt.Printf("   Batches processed: %d\n", stats.TotalBatches)
	fmt.Printf("   Peak throughput: %.0f vec/sec\n", stats.PeakThroughput)
	fmt.Printf("   Average latency: %.1f ms\n", stats.GetAvgLatencyMs())
	
	memUsage := indexer.GetMemoryUsage()
	fmt.Printf("   Final memory usage: %d MB\n", memUsage/1024/1024)

	fmt.Printf("\n Streaming demo completed successfully!\n")
}

func runBenchmarkDemo() {
	fmt.Printf("🏁 Benchmark Demo\n")
	fmt.Printf("=================\n\n")

	fmt.Printf(" Running comprehensive benchmarks...\n\n")

	// Small benchmark
	runQuickBenchmark("Small Dataset", 10000, 1000)
	
	// Medium benchmark  
	runQuickBenchmark("Medium Dataset", 50000, 2000)
	
	// Large benchmark (if user confirms)
	fmt.Printf("\nRun large dataset benchmark (500K vectors)? This may take several minutes. (y/n): ")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	if strings.ToLower(strings.TrimSpace(scanner.Text())) == "y" {
		runQuickBenchmark("Large Dataset", 500000, 5000)
	}

	fmt.Printf("\n Benchmark demo completed!\n")
}

func runQuickBenchmark(name string, numVectors, batchSize int) {
	fmt.Printf(" %s Benchmark:\n", name)
	fmt.Printf("   Vectors: %d, Batch size: %d\n", numVectors, batchSize)

	config := gobed.DefaultBulkIndexConfig()
	indexer, err := gobed.NewGPUIVFBulkIndexer(config)
	if err != nil {
		fmt.Printf("    Failed to create indexer: %v\n\n", err)
		return
	}
	defer indexer.Close()

	// Quick setup
	embeddings, scales := generateSampleEmbeddings(config.VocabSize, config.EmbedDim)
	indexer.LoadEmbeddings(embeddings, scales)
	
	trainingData := generateSampleTrainingData(2000, config.VectorDim)
	indexer.TrainKMeans(trainingData.vectors, trainingData.scales, len(trainingData.scales))

	// Generate data
	tokenSequences := generateSampleTokenSequences(numVectors, config.VectorDim)

	// Benchmark
	start := time.Now()
	indexed, err := indexer.BulkIndexTokenSequences(tokenSequences, nil)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("    Benchmark failed: %v\n\n", err)
		return
	}

	throughput := float64(indexed) / elapsed.Seconds()
	memUsage := indexer.GetMemoryUsage()

	fmt.Printf("    Results: %.0f vec/sec, %v total, %d MB memory\n\n",
		throughput, elapsed, memUsage/1024/1024)
}

// Helper functions

func getUserInput(prompt string, defaultValue int) int {
	fmt.Printf("%s [%d]: ", prompt, defaultValue)
	
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	input := strings.TrimSpace(scanner.Text())
	
	if input == "" {
		return defaultValue
	}
	
	value, err := strconv.Atoi(input)
	if err != nil {
		fmt.Printf("Invalid input, using default: %d\n", defaultValue)
		return defaultValue
	}
	
	return value
}

func generateSampleEmbeddings(vocabSize, embedDim int) ([]int8, []float32) {
	embeddings := make([]int8, vocabSize*embedDim)
	scales := make([]float32, vocabSize)
	
	// Generate realistic quantized embeddings
	for i := 0; i < vocabSize; i++ {
		scales[i] = 0.5 + (float32(i%100) / 200.0) // Varied scales
		for j := 0; j < embedDim; j++ {
			// Generate normal-ish distribution around 0
			embeddings[i*embedDim+j] = int8((i*7+j*11)%256 - 128)
		}
	}
	
	return embeddings, scales
}

func generateSampleTrainingData(numVectors, vectorDim int) TrainingData {
	vectors := make([]int8, numVectors*vectorDim)
	scales := make([]float32, numVectors)
	
	for i := 0; i < numVectors; i++ {
		scales[i] = 0.8 + (float32(i%50) / 100.0)
		for j := 0; j < vectorDim; j++ {
			vectors[i*vectorDim+j] = int8((i*13+j*17)%256 - 128)
		}
	}
	
	return TrainingData{vectors: vectors, scales: scales}
}

func generateSampleTokenSequences(numSequences, maxSeqLen int) [][]int {
	sequences := make([][]int, numSequences)
	
	for i := 0; i < numSequences; i++ {
		// Variable sequence lengths
		seqLen := 10 + (i*7)%int(maxSeqLen-10)
		if seqLen > maxSeqLen {
			seqLen = maxSeqLen
		}
		
		sequence := make([]int, seqLen)
		for j := 0; j < seqLen; j++ {
			// Generate realistic token IDs
			sequence[j] = (i*23 + j*29) % 30522 // BERT vocab size
		}
		sequences[i] = sequence
	}
	
	return sequences
}

func generateSingleTokenSequence(maxLen int) []int {
	seqLen := 10 + (int(time.Now().UnixNano())%int(maxLen-10))
	sequence := make([]int, seqLen)
	
	base := int(time.Now().UnixNano()) % 30522
	for j := 0; j < seqLen; j++ {
		sequence[j] = (base + j*31) % 30522
	}
	
	return sequence
}

type TrainingData struct {
	vectors []int8
	scales  []float32
}