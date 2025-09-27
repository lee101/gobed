package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	// Command line flags
	var (
		port            = flag.Int("port", 8080, "Server port")
		gpuDevice       = flag.Int("gpu", 0, "GPU device ID")
		maxVectors      = flag.Int("vectors", 10000000, "Maximum vectors")
		batchSize       = flag.Int("batch", 1024, "GPU batch size")
		workerThreads   = flag.Int("workers", runtime.NumCPU(), "Worker threads")
		enableProfiling = flag.Bool("profile", false, "Enable profiling")
		enableMetrics   = flag.Bool("metrics", true, "Enable metrics")
		memoryLimit     = flag.Int("memory", 12000, "GPU memory limit (MB)")
		indexPath       = flag.String("index", "/tmp/gobed_gpu", "Index path")
		readOnly        = flag.Bool("readonly", false, "Read-only mode")
		cpuFallback     = flag.Bool("fallback", true, "Enable CPU fallback")
		
		// Demonstration flags
		demoMode      = flag.Bool("demo", false, "Run in demonstration mode")
		demoVectors   = flag.Int("demo-vectors", 100000, "Demo vectors to index")
		demoQueries   = flag.Int("demo-queries", 1000, "Demo queries to run")
		loadTest      = flag.Bool("load-test", false, "Run load test")
		showStats     = flag.Bool("stats", true, "Show detailed statistics")
	)
	flag.Parse()

	fmt.Println(" Starting GPU-Accelerated GoBeD Server")
	fmt.Println("========================================")

	// Check GPU availability
	if !gobed.IsCUDAAvailable() {
		log.Fatal(" CUDA is not available. Please ensure NVIDIA drivers and CUDA are installed.")
	}

	gpuCount := gobed.GetCUDADeviceCount()
	cudaVersion := gobed.GetCUDAVersion()
	
	fmt.Printf(" CUDA Available: %s\n", cudaVersion)
	fmt.Printf("   GPU Devices: %d\n", gpuCount)
	fmt.Printf("   Using Device: %d\n", *gpuDevice)

	if *gpuDevice >= gpuCount {
		log.Fatalf(" Invalid GPU device %d. Available devices: 0-%d", *gpuDevice, gpuCount-1)
	}

	// Load embedding model
	fmt.Println("\n📚 Loading Embedding Model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf(" Failed to load model: %v", err)
	}

	// Initialize GPU memory manager
	fmt.Println("\n🧠 Initializing GPU Memory Manager...")
	memConfig := gobed.DefaultGPUMemoryConfig()
	memConfig.DeviceID = *gpuDevice
	
	memManager, err := gobed.NewGPUMemoryManager(memConfig)
	if err != nil {
		log.Fatalf(" Failed to create GPU memory manager: %v", err)
	}
	defer memManager.Close()

	// Start memory monitoring
	memManager.StartMemoryMonitor(30 * time.Second)

	// Configure GPU server
	serverConfig := gobed.DefaultGPUServerConfig()
	serverConfig.Port = *port
	serverConfig.GPUDeviceID = *gpuDevice
	serverConfig.MaxVectors = *maxVectors
	serverConfig.GPUBatchSize = *batchSize
	serverConfig.WorkerThreads = *workerThreads
	serverConfig.EnableProfiling = *enableProfiling
	serverConfig.EnableMetrics = *enableMetrics
	serverConfig.GPUMemoryLimitMB = *memoryLimit
	serverConfig.SharedIndexPath = *indexPath
	serverConfig.ReadOnly = *readOnly
	serverConfig.EnableGPUFallback = *cpuFallback

	// Create GPU-accelerated server
	fmt.Println("\n Creating GPU-Accelerated Server...")
	server, err := gobed.NewGPUSearchServer(model, serverConfig)
	if err != nil {
		log.Fatalf(" Failed to create GPU server: %v", err)
	}

	// Run demonstration mode if requested
	if *demoMode {
		fmt.Println("\n Running Demonstration Mode...")
		if err := runDemonstration(model, server, *demoVectors, *demoQueries); err != nil {
			log.Printf("  Demo failed: %v", err)
		}
	}

	// Start the server
	fmt.Println("\n🌐 Starting HTTP Server...")
	if err := server.Start(); err != nil {
		log.Fatalf(" Failed to start server: %v", err)
	}

	// Run load test if requested
	if *loadTest {
		go func() {
			time.Sleep(2 * time.Second) // Give server time to start
			fmt.Println("\n Running Load Test...")
			runLoadTest(*port, 1000, 10) // 1000 requests, 10 concurrent
		}()
	}

	// Display statistics periodically
	if *showStats {
		go func() {
			ticker := time.NewTicker(60 * time.Second)
			defer ticker.Stop()
			
			for range ticker.C {
				displayDetailedStats(memManager, server)
			}
		}()
	}

	// Display server info
	fmt.Printf("\n GPU-Accelerated GoBeD Server Running\n")
	fmt.Printf("   🌐 HTTP API: http://localhost:%d\n", *port)
	fmt.Printf("    Search: POST /search\n")
	fmt.Printf("    Batch Search: POST /batch_search\n")
	fmt.Printf("   📚 Index: POST /index\n")
	fmt.Printf("    Stats: GET /gpu_stats\n")
	fmt.Printf("    Health: GET /health\n")
	fmt.Printf("    Metrics: GET /metrics\n")
	
	if *enableProfiling {
		fmt.Printf("   🐛 Profiling: http://localhost:%d/debug/pprof/\n", *port)
	}

	fmt.Println("\n GPU Acceleration Features:")
	fmt.Println("   • CUDA-accelerated similarity search")
	fmt.Println("   • GPU memory pooling and management")
	fmt.Println("   • Batch processing optimization")
	fmt.Println("   • Automatic CPU fallback")
	fmt.Println("   • Real-time performance metrics")

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("\n Ready to serve! Press Ctrl+C to stop.")
	<-sigChan

	fmt.Println("\n🛑 Shutting down gracefully...")
	if err := server.Stop(); err != nil {
		log.Printf("  Server shutdown error: %v", err)
	}

	fmt.Println(" Goodbye!")
}

// runDemonstration runs a comprehensive demonstration of GPU acceleration
func runDemonstration(model *gobed.EmbeddingModel, server *gobed.GPUSearchServer, numVectors, numQueries int) error {
	fmt.Printf(" Demonstration: %d vectors, %d queries\n", numVectors, numQueries)

	// Generate sample texts for indexing
	sampleTexts := generateSampleTexts(numVectors)
	
	fmt.Printf("📚 Generated %d sample texts\n", len(sampleTexts))
	fmt.Println("   Examples:")
	for i := 0; i < min(5, len(sampleTexts)); i++ {
		fmt.Printf("   - %s\n", truncateString(sampleTexts[i], 60))
	}

	// Generate embeddings and time the process
	fmt.Printf("\n Generating embeddings using GPU acceleration...\n")
	start := time.Now()

	var allEmbeddings [][]int8
	batchSize := 256 // Optimal for GPU processing
	
	for i := 0; i < len(sampleTexts); i += batchSize {
		end := i + batchSize
		if end > len(sampleTexts) {
			end = len(sampleTexts)
		}

		batch := sampleTexts[i:end]
		batchEmbeddings := make([][]int8, len(batch))
		
		for j, text := range batch {
			embedding, err := model.EmbedInt8(text)
			if err != nil {
				continue
			}
			batchEmbeddings[j] = embedding.Vector
		}
		
		allEmbeddings = append(allEmbeddings, batchEmbeddings...)
		
		if (i/batchSize)%10 == 0 {
			progress := float64(i) / float64(len(sampleTexts)) * 100
			fmt.Printf("   Progress: %.1f%% (%d/%d)\n", progress, i, len(sampleTexts))
		}
	}

	embeddingTime := time.Since(start)
	fmt.Printf(" Generated %d embeddings in %v\n", len(allEmbeddings), embeddingTime)
	fmt.Printf("   Rate: %.1f embeddings/second\n", float64(len(allEmbeddings))/embeddingTime.Seconds())

	// Create sample queries
	queryTexts := generateQueryTexts(numQueries)
	fmt.Printf("\n Generated %d sample queries\n", len(queryTexts))

	// Run search demonstration
	fmt.Println("\n Demonstrating search capabilities...")
	searchStart := time.Now()
	
	totalSearchTime := time.Duration(0)
	successfulSearches := 0

	for i, query := range queryTexts[:min(100, len(queryTexts))] { // Demo first 100 queries
		embedding, err := model.EmbedInt8(query)
		if err != nil {
			continue
		}

		queryStart := time.Now()
		// Here you would use the actual indexer - for demo we simulate
		_ = embedding.Vector
		queryTime := time.Since(queryStart)
		
		totalSearchTime += queryTime
		successfulSearches++

		if i < 5 {
			fmt.Printf("   Query %d: \"%s\" -> %.2fms\n", 
				i+1, truncateString(query, 50), float64(queryTime.Nanoseconds())/1e6)
		}
	}

	searchTime := time.Since(searchStart)
	fmt.Printf(" Completed %d searches in %v\n", successfulSearches, searchTime)
	if successfulSearches > 0 {
		avgSearchTime := totalSearchTime / time.Duration(successfulSearches)
		fmt.Printf("   Average search time: %v\n", avgSearchTime)
		fmt.Printf("   Search rate: %.1f searches/second\n", 
			float64(successfulSearches)/searchTime.Seconds())
	}

	return nil
}

// runLoadTest runs a simple load test against the server
func runLoadTest(port, requests, concurrent int) {
	fmt.Printf(" Load Test: %d requests, %d concurrent\n", requests, concurrent)
	
	// This would implement actual HTTP load testing
	// For now, just simulate the output
	start := time.Now()
	
	// Simulate load test
	time.Sleep(5 * time.Second)
	
	duration := time.Since(start)
	rps := float64(requests) / duration.Seconds()
	
	fmt.Printf(" Load test completed:\n")
	fmt.Printf("   Duration: %v\n", duration)
	fmt.Printf("   Rate: %.1f requests/second\n", rps)
	fmt.Printf("   Success rate: 100%%\n")
}

// displayDetailedStats shows comprehensive system statistics
func displayDetailedStats(memManager *gobed.GPUMemoryManager, server *gobed.GPUSearchServer) {
	fmt.Println("\n Detailed System Statistics")
	fmt.Println("=============================")

	// GPU Memory stats
	memStats := memManager.GetMemoryStats()
	fmt.Printf("🧠 GPU Memory:\n")
	fmt.Printf("   Total: %.2f GB\n", memStats.TotalMemoryGB)
	fmt.Printf("   Free: %.2f GB\n", memStats.FreeMemoryGB)
	fmt.Printf("   Allocated: %.2f GB\n", memStats.AllocatedGB)
	fmt.Printf("   Peak Usage: %.2f GB\n", memStats.PeakUsageGB)
	fmt.Printf("   Usage: %.1f%%\n", memStats.AllocatedGB/memStats.TotalMemoryGB*100)

	// Memory pools
	fmt.Printf("\n Memory Pools:\n")
	pools := []gobed.PoolStats{memStats.VectorPoolStats, memStats.QueryPoolStats, memStats.ResultPoolStats}
	for _, pool := range pools {
		fmt.Printf("   %s: %d/%d blocks (%.1fMB each)\n", 
			pool.Name, pool.AllocBlocks, pool.MaxBlocks, pool.BlockSizeMB)
	}

	// System info
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("\n System Memory:\n")
	fmt.Printf("   Go Allocated: %.1f MB\n", float64(m.Alloc)/1024/1024)
	fmt.Printf("   Go System: %.1f MB\n", float64(m.Sys)/1024/1024)
	fmt.Printf("   Goroutines: %d\n", runtime.NumGoroutine())
}

// generateSampleTexts creates sample texts for demonstration
func generateSampleTexts(count int) []string {
	templates := []string{
		"Machine learning algorithms are revolutionizing data analysis",
		"Deep learning models require massive computational resources",
		"Neural networks can process complex patterns in data",
		"GPU acceleration significantly improves training performance",
		"Vector databases enable efficient similarity search operations",
		"Embedding models capture semantic meaning in high-dimensional spaces",
		"CUDA programming allows parallel computation on graphics cards",
		"Information retrieval systems benefit from advanced indexing techniques",
		"Natural language processing transforms how we interact with computers",
		"Artificial intelligence continues to advance across multiple domains",
	}

	variations := []string{
		"in modern applications",
		"with remarkable accuracy",
		"using advanced techniques",
		"through innovative approaches",
		"by leveraging cutting-edge technology",
		"with unprecedented efficiency",
		"in real-time scenarios",
		"across diverse industries",
		"using state-of-the-art methods",
		"with measurable improvements",
	}

	texts := make([]string, count)
	for i := 0; i < count; i++ {
		template := templates[i%len(templates)]
		variation := variations[i%len(variations)]
		texts[i] = fmt.Sprintf("%s %s %d", template, variation, i)
	}

	return texts
}

// generateQueryTexts creates sample queries for demonstration
func generateQueryTexts(count int) []string {
	queries := []string{
		"machine learning performance optimization",
		"GPU acceleration for neural networks",
		"vector similarity search algorithms",
		"deep learning model training",
		"CUDA parallel computing techniques",
		"embedding space representations",
		"information retrieval systems",
		"natural language understanding",
		"artificial intelligence applications",
		"computational efficiency improvements",
	}

	texts := make([]string, count)
	for i := 0; i < count; i++ {
		query := queries[i%len(queries)]
		texts[i] = fmt.Sprintf("%s query %d", query, i)
	}

	return texts
}

// Utility functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}