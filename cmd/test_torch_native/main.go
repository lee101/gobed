package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/lee101/gobed/gpu"
)

// Simple embedder that creates deterministic embeddings
type TestEmbedder struct {
	dim int
}

func NewTestEmbedder(dim int) *TestEmbedder {
	return &TestEmbedder{dim: dim}
}

func (e *TestEmbedder) Encode(text string) ([]float32, error) {
	// Create deterministic embedding based on text content
	hash := int64(0)
	for _, c := range text {
		hash = hash*31 + int64(c)
	}
	
	rand.Seed(hash)
	embedding := make([]float32, e.dim)
	for i := range embedding {
		embedding[i] = (rand.Float32() - 0.5) * 2 // Values between -1 and 1
	}
	
	return embedding, nil
}

func main() {
	fmt.Println("🔥 LibTorch Native Integration Test")
	fmt.Println("===================================")

	// Check LibTorch info
	version, cudaAvailable, deviceCount := gpu.GetTorchInfo()
	fmt.Printf("LibTorch version: %s\n", version)
	fmt.Printf("CUDA available: %v\n", cudaAvailable)
	fmt.Printf("Device count: %d\n", deviceCount)

	if !cudaAvailable {
		log.Fatal("❌ CUDA not available")
	}

	if deviceCount == 0 {
		log.Fatal("❌ No CUDA devices found")
	}

	// Test 1: Basic indexer functionality
	fmt.Println("\n🧪 Test 1: Basic Indexer")
	testBasicIndexer()

	// Test 2: Pipeline functionality  
	fmt.Println("\n🧪 Test 2: Pipeline")
	testPipeline()

	// Test 3: Performance benchmark
	fmt.Println("\n🧪 Test 3: Performance Benchmark")
	benchmarkPerformance()

	fmt.Println("\n✅ All tests completed successfully!")
}

func testBasicIndexer() {
	config := gpu.DefaultTorchNativeConfig()
	config.VectorDim = 256
	config.IVFClusters = 128
	config.NumSubquantizers = 32
	config.ProbeLists = 16
	config.RerankK = 200

	indexer, err := gpu.NewTorchNativeIndexer(config)
	if err != nil {
		log.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Generate training data
	fmt.Print("  Generating training data... ")
	start := time.Now()
	numTraining := 2000
	trainingVectors := make([][]int8, numTraining)
	for i := 0; i < numTraining; i++ {
		vec := make([]int8, config.VectorDim)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		trainingVectors[i] = vec
	}
	fmt.Printf("✅ (%v)\n", time.Since(start))

	// Train indexer
	fmt.Print("  Training indexer... ")
	start = time.Now()
	err = indexer.TrainIndex(trainingVectors)
	if err != nil {
		log.Fatalf("Failed to train indexer: %v", err)
	}
	fmt.Printf("✅ (%v)\n", time.Since(start))

	// Generate and add vectors
	fmt.Print("  Adding vectors to index... ")
	start = time.Now()
	numVectors := 50000
	vectors := make([][]int8, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]int8, config.VectorDim)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		vectors[i] = vec
	}

	err = indexer.AddVectors(vectors)
	if err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	indexTime := time.Since(start)
	fmt.Printf("✅ (%v)\n", indexTime)
	fmt.Printf("    Indexing rate: %.0f vectors/sec\n", float64(numVectors)/indexTime.Seconds())

	// Test search
	fmt.Print("  Testing search... ")
	query := vectors[42] // Use a known vector
	k := 10

	start = time.Now()
	ids, scores, err := indexer.Search(query, k)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}
	searchTime := time.Since(start)
	fmt.Printf("✅ (%v)\n", searchTime)

	fmt.Printf("    Found %d results\n", len(ids))
	fmt.Printf("    Search latency: %.2f ms\n", searchTime.Seconds()*1000)

	// Print top results
	for i := 0; i < min(5, len(ids)); i++ {
		fmt.Printf("    Result %d: ID=%d, Score=%.3f\n", i+1, ids[i], scores[i])
	}

	// Get and print stats
	stats, err := indexer.GetStats()
	if err != nil {
		log.Fatalf("Failed to get stats: %v", err)
	}

	fmt.Println("  📊 Index Statistics:")
	fmt.Printf("    Vectors: %d\n", stats.NumVectors)
	fmt.Printf("    Vector dim: %d\n", stats.VectorDim)
	fmt.Printf("    IVF clusters: %d\n", stats.IVFClusters)
	fmt.Printf("    PQ subquantizers: %d\n", stats.PQSubquantizers)
	fmt.Printf("    GPU memory: %.1f MB\n", stats.GPUMemoryMB)
	fmt.Printf("    Trained: %v\n", stats.IsTrained)
	fmt.Printf("    Built: %v\n", stats.IndexBuilt)
}

func testPipeline() {
	config := gpu.DefaultTorchNativeConfig()
	config.VectorDim = 256
	config.IVFClusters = 64
	config.NumSubquantizers = 32

	embedder := NewTestEmbedder(config.VectorDim)

	pipeline, err := gpu.NewTorchNativePipeline(config, embedder)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}
	defer pipeline.Close()

	// Training texts
	trainingTexts := []string{
		"machine learning algorithms and techniques",
		"deep neural network architectures",
		"computer vision and image processing",
		"natural language processing systems", 
		"artificial intelligence applications",
		"data science and analytics methods",
		"statistical modeling approaches",
		"optimization and search algorithms",
		"reinforcement learning frameworks",
		"distributed computing systems",
		"database management and queries",
		"software engineering practices",
		"web development technologies",
		"mobile application development",
		"cloud computing platforms",
	}

	fmt.Print("  Training pipeline... ")
	start := time.Now()
	err = pipeline.TrainPipeline(trainingTexts)
	if err != nil {
		log.Fatalf("Failed to train pipeline: %v", err)
	}
	fmt.Printf("✅ (%v)\n", time.Since(start))

	// Index texts
	indexTexts := []string{
		"advanced machine learning techniques for data analysis",
		"deep convolutional neural networks for image recognition", 
		"transformer models for natural language understanding",
		"computer vision algorithms for object detection",
		"reinforcement learning for autonomous systems",
		"statistical methods for predictive modeling",
		"distributed systems for large-scale computing",
		"database optimization for high-performance queries",
		"web frameworks for scalable application development",
		"mobile development for cross-platform applications",
		"cloud infrastructure for modern software deployment",
		"artificial intelligence in healthcare diagnostics",
		"machine learning operations and model deployment",
		"data engineering pipelines for real-time processing",
		"cybersecurity applications of machine learning",
		"recommendation systems using collaborative filtering",
		"time series analysis with deep learning models",
		"graph neural networks for social network analysis",
		"federated learning for privacy-preserving AI",
		"automated machine learning and hyperparameter tuning",
	}

	fmt.Print("  Indexing texts... ")
	start = time.Now()
	err = pipeline.IndexTexts(indexTexts)
	if err != nil {
		log.Fatalf("Failed to index texts: %v", err)
	}
	indexTime := time.Since(start)
	fmt.Printf("✅ (%v)\n", indexTime)
	fmt.Printf("    Indexing rate: %.1f texts/sec\n", float64(len(indexTexts))/indexTime.Seconds())

	// Test searches
	queries := []string{
		"neural networks for computer vision",
		"machine learning for healthcare",
		"distributed computing systems",
		"web application development",
		"artificial intelligence algorithms",
	}

	fmt.Println("  🔍 Search Results:")
	for i, query := range queries {
		start = time.Now()
		results, err := pipeline.Search(query, 3)
		if err != nil {
			log.Fatalf("Failed to search: %v", err)
		}
		searchTime := time.Since(start)

		fmt.Printf("    Query %d: '%s' (%.2f ms)\n", i+1, query, searchTime.Seconds()*1000)
		for j, result := range results {
			fmt.Printf("      %d. Score=%.3f: %s\n", j+1, result.Score, result.Text)
		}
	}

	// Get pipeline stats
	stats, err := pipeline.GetPipelineStats()
	if err != nil {
		log.Fatalf("Failed to get pipeline stats: %v", err)
	}

	fmt.Println("  📊 Pipeline Statistics:")
	fmt.Printf("    Texts: %d\n", stats.NumTexts)
	fmt.Printf("    Embeddings: %d\n", stats.NumEmbeddings)
	fmt.Printf("    GPU device: %s\n", stats.GPUDevice)
	fmt.Printf("    GPU memory: %.1f MB\n", stats.GPUMemoryMB)
}

func benchmarkPerformance() {
	config := gpu.DefaultTorchNativeConfig()
	config.VectorDim = 512 // Full dimension
	config.IVFClusters = 1024
	config.NumSubquantizers = 64

	indexer, err := gpu.NewTorchNativeIndexer(config)
	if err != nil {
		log.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()

	// Quick training
	fmt.Print("  Setting up benchmark... ")
	start := time.Now()
	
	numTraining := 5000
	trainingVectors := make([][]int8, numTraining)
	for i := 0; i < numTraining; i++ {
		vec := make([]int8, config.VectorDim)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		trainingVectors[i] = vec
	}

	err = indexer.TrainIndex(trainingVectors)
	if err != nil {
		log.Fatalf("Failed to train: %v", err)
	}

	// Add many vectors for realistic performance test
	numVectors := 100000
	vectors := make([][]int8, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]int8, config.VectorDim)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		vectors[i] = vec
	}

	err = indexer.AddVectors(vectors)
	if err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}

	setupTime := time.Since(start)
	fmt.Printf("✅ (%v)\n", setupTime)
	fmt.Printf("    Index size: %d vectors\n", numVectors)

	// Benchmark search performance
	fmt.Println("  🚀 Search Performance:")
	
	numQueries := 1000
	queries := make([][]int8, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = vectors[rand.Intn(numVectors)] // Use indexed vectors as queries
	}

	// Single query benchmark
	start = time.Now()
	for i := 0; i < numQueries; i++ {
		_, _, err := indexer.Search(queries[i], 10)
		if err != nil {
			log.Fatalf("Search failed: %v", err)
		}
	}
	totalTime := time.Since(start)

	avgLatency := totalTime.Seconds() * 1000 / float64(numQueries)
	qps := float64(numQueries) / totalTime.Seconds()

	fmt.Printf("    Queries: %d\n", numQueries)
	fmt.Printf("    Total time: %v\n", totalTime)
	fmt.Printf("    Average latency: %.2f ms\n", avgLatency)
	fmt.Printf("    Queries per second: %.0f\n", qps)

	// Memory usage
	stats, err := indexer.GetStats()
	if err != nil {
		log.Fatalf("Failed to get stats: %v", err)
	}

	fmt.Printf("    GPU memory usage: %.1f MB\n", stats.GPUMemoryMB)

	// Accuracy test (using exact match)
	fmt.Print("  🎯 Testing accuracy... ")
	exactQuery := vectors[1000]
	ids, scores, err := indexer.Search(exactQuery, 1)
	if err != nil {
		log.Fatalf("Accuracy test failed: %v", err)
	}

	if len(ids) > 0 && ids[0] == 1000 {
		fmt.Printf("✅ Exact match found (ID=%d, Score=%.3f)\n", ids[0], scores[0])
	} else {
		fmt.Printf("⚠️  Approximate match (ID=%d, Score=%.3f)\n", ids[0], scores[0])
	}

	fmt.Println("  📈 Performance Summary:")
	fmt.Printf("    Index: %d vectors, %d dimensions\n", numVectors, config.VectorDim)
	fmt.Printf("    Search: %.2f ms latency, %.0f QPS\n", avgLatency, qps)
	fmt.Printf("    Memory: %.1f MB GPU\n", stats.GPUMemoryMB)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}