package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/search"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	fmt.Println("=== Gobed ANN Search Demo ===\n")

	// Load the embedding model
	fmt.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✓ Model loaded successfully\n")

	// Demo 1: Small-scale exact search (SIMD-Flat)
	demoSmallScale(model)

	// Demo 2: Large-scale approximate search (IVF-HNSW-PQ)
	demoLargeScale()

	// Demo 3: Real text search with gobed embeddings
	demoTextSearch(model)
}

func demoSmallScale(model *gobed.EmbeddingModel) {
	fmt.Println("=== Demo 1: Small-scale Exact Search (SIMD-Flat) ===")

	// Create sample documents
	documents := []gobed.Document{
		{ID: 1, Text: "Machine learning is a subset of artificial intelligence"},
		{ID: 2, Text: "Deep learning uses neural networks with multiple layers"},
		{ID: 3, Text: "Natural language processing helps computers understand human language"},
		{ID: 4, Text: "Computer vision enables machines to interpret visual information"},
		{ID: 5, Text: "Reinforcement learning trains agents through rewards and penalties"},
		{ID: 6, Text: "Transfer learning reuses pre-trained models for new tasks"},
		{ID: 7, Text: "Gradient descent optimizes neural network weights"},
		{ID: 8, Text: "Transformers revolutionized NLP with attention mechanisms"},
		{ID: 9, Text: "BERT and GPT are popular transformer-based models"},
		{ID: 10, Text: "Convolutional neural networks excel at image recognition"},
	}

	// Create index with flat configuration
	config := gobed.VectorIndexConfig{
		MaxFlatSize: 1000, // Force flat index
		UseParallel: true,
	}

	index := gobed.NewVectorIndex(model, config)

	// Add documents
	fmt.Printf("Indexing %d documents...\n", len(documents))
	start := time.Now()
	err := index.AddDocuments(documents)
	if err != nil {
		log.Fatalf("Failed to add documents: %v", err)
	}
	fmt.Printf("✓ Indexed in %v\n\n", time.Since(start))

	// Perform searches
	queries := []string{
		"neural network architectures",
		"language understanding systems",
		"visual perception in AI",
	}

	for _, query := range queries {
		fmt.Printf("Query: '%s'\n", query)

		start := time.Now()
		results, err := index.Search(query, 3)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		latency := time.Since(start)

		fmt.Printf("Results (latency: %v):\n", latency)
		for i, r := range results {
			fmt.Printf("  %d. Doc #%d (similarity: %.3f)\n", i+1, r.ID, r.Similarity)
		}
		fmt.Println()
	}

	// Print stats
	stats := index.Stats()
	fmt.Printf("Index Stats: Type=%s, Size=%d, Memory=%.2f MB\n\n",
		stats.IndexType, stats.Size, float64(stats.MemoryUsage)/1024/1024)
}

func demoLargeScale() {
	fmt.Println("=== Demo 2: Large-scale Approximate Search (IVF-HNSW-PQ) ===")

	// Generate synthetic data
	numVectors := 100000
	fmt.Printf("Generating %d synthetic vectors...\n", numVectors)

	vectors := make([]simd.Vec512, numVectors)
	scales := make([]float32, numVectors)
	ids := make([]int, numVectors)

	for i := 0; i < numVectors; i++ {
		for j := 0; j < 512; j++ {
			vectors[i][j] = int8(rand.Intn(256) - 128)
		}
		scales[i] = 1.0
		ids[i] = i
	}

	// Create engine with IVF-HNSW-PQ configuration
	config := search.Config{
		MaxFlatSize: 10000,
		NList:       1024, // 1024 clusters
		NProbe:      8,    // Search 8 clusters
		M:           64,   // 64 subquantizers
		NBits:       8,    // 8 bits per code
		HNSWEnabled: true, // Use HNSW for routing
		HNSWM:       16,
		HNSWEfC:     200,
		RerankSize:  128, // Rerank top 128
		UseParallel: true,
	}

	engine := search.NewEngine(config)

	// Train the index
	fmt.Println("Training index...")
	trainStart := time.Now()
	trainSize := 10000
	err := engine.Train(vectors[:trainSize], scales[:trainSize])
	if err != nil {
		log.Fatalf("Failed to train: %v", err)
	}
	fmt.Printf("✓ Training completed in %v\n", time.Since(trainStart))

	// Add vectors in batches
	fmt.Println("Adding vectors to index...")
	addStart := time.Now()
	batchSize := 10000
	for i := 0; i < numVectors; i += batchSize {
		end := min(i+batchSize, numVectors)
		err := engine.AddBatch(vectors[i:end], scales[i:end], ids[i:end])
		if err != nil {
			log.Fatalf("Failed to add batch: %v", err)
		}
		fmt.Printf("  Added %d/%d vectors\r", end, numVectors)
	}
	fmt.Printf("\n✓ Indexing completed in %v\n\n", time.Since(addStart))

	// Perform benchmark searches
	numQueries := 100
	k := 10

	fmt.Printf("Running %d searches (k=%d)...\n", numQueries, k)
	searchStart := time.Now()

	for i := 0; i < numQueries; i++ {
		query := &vectors[rand.Intn(numVectors)]
		results, err := engine.Search(query, k)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		_ = results // Process results
	}

	searchTime := time.Since(searchStart)
	avgLatency := searchTime / time.Duration(numQueries)
	qps := float64(numQueries) / searchTime.Seconds()

	fmt.Printf("✓ Search benchmark completed\n")
	fmt.Printf("  Average latency: %v\n", avgLatency)
	fmt.Printf("  Throughput: %.1f QPS\n", qps)

	// Print stats
	stats := engine.Stats()
	fmt.Printf("\nIndex Stats:\n")
	fmt.Printf("  Type: %s\n", stats.IndexType)
	fmt.Printf("  Size: %d vectors\n", stats.Size)
	fmt.Printf("  Memory: %.2f MB\n", float64(stats.MemoryUsage)/1024/1024)
	fmt.Printf("  IVF Lists: %d\n", stats.NLists)
	fmt.Printf("  PQ Enabled: %v (M=%d, bits=%d)\n", stats.PQEnabled, stats.PQM, stats.PQBits)
	fmt.Printf("  HNSW Enabled: %v\n\n", stats.HNSWEnabled)
}

func demoTextSearch(model *gobed.EmbeddingModel) {
	fmt.Println("=== Demo 3: Real Text Search with Semantic Understanding ===")

	// Create a corpus of technical documents
	corpus := []gobed.Document{
		{ID: 1, Text: "Python is a high-level programming language known for its simplicity"},
		{ID: 2, Text: "JavaScript powers interactive web applications and runs in browsers"},
		{ID: 3, Text: "Go is a statically typed language designed for system programming"},
		{ID: 4, Text: "Rust provides memory safety without garbage collection"},
		{ID: 5, Text: "Docker containers package applications with their dependencies"},
		{ID: 6, Text: "Kubernetes orchestrates containerized applications at scale"},
		{ID: 7, Text: "PostgreSQL is a powerful open-source relational database"},
		{ID: 8, Text: "MongoDB stores data in flexible JSON-like documents"},
		{ID: 9, Text: "Redis provides in-memory data structure storage"},
		{ID: 10, Text: "Elasticsearch enables full-text search and analytics"},
		{ID: 11, Text: "React builds user interfaces with component-based architecture"},
		{ID: 12, Text: "Vue.js is a progressive framework for building web interfaces"},
		{ID: 13, Text: "TensorFlow facilitates machine learning model development"},
		{ID: 14, Text: "PyTorch provides dynamic computational graphs for deep learning"},
		{ID: 15, Text: "GraphQL offers flexible API queries compared to REST"},
		{ID: 16, Text: "WebAssembly runs high-performance code in web browsers"},
		{ID: 17, Text: "Blockchain technology enables decentralized ledgers"},
		{ID: 18, Text: "Microservices architecture breaks applications into small services"},
		{ID: 19, Text: "CI/CD pipelines automate software deployment processes"},
		{ID: 20, Text: "Cloud computing provides on-demand computing resources"},
	}

	// Create index optimized for this size
	config := gobed.VectorIndexConfig{
		MaxFlatSize: 100,   // Use IVF for demonstration
		NList:       4,     // Small number of clusters
		NProbe:      2,     // Search half the clusters
		UseHNSW:     false, // Not needed for this size
		RerankSize:  10,
	}

	index := gobed.NewVectorIndex(model, config)

	// Train on the corpus
	texts := make([]string, len(corpus))
	for i, doc := range corpus {
		texts[i] = doc.Text
	}

	fmt.Printf("Training index on %d documents...\n", len(corpus))
	err := index.Train(texts)
	if err != nil {
		log.Printf("Warning: Training failed: %v", err)
	}

	// Add documents
	fmt.Println("Indexing documents...")
	err = index.AddDocuments(corpus)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}
	fmt.Printf("✓ Indexed %d documents\n\n", len(corpus))

	// Semantic search queries
	queries := []struct {
		query string
		desc  string
	}{
		{"database for web applications", "Finding database technologies"},
		{"container orchestration platform", "Looking for Kubernetes-like tools"},
		{"frontend JavaScript framework", "Searching for UI frameworks"},
		{"compiled systems programming", "Finding low-level languages"},
		{"machine learning frameworks", "Looking for ML tools"},
	}

	for _, q := range queries {
		fmt.Printf("Query: '%s'\n", q.query)
		fmt.Printf("(%s)\n", q.desc)

		start := time.Now()
		results, err := index.Search(q.query, 3)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		latency := time.Since(start)

		fmt.Printf("Top 3 results (latency: %v):\n", latency)
		for i, r := range results {
			doc := corpus[r.ID-1] // IDs are 1-based
			fmt.Printf("  %d. [%.3f] %s\n", i+1, r.Similarity, doc.Text)
		}
		fmt.Println()
	}

	// Final stats
	stats := index.Stats()
	fmt.Printf("Final Index Stats:\n")
	fmt.Printf("  Documents: %d\n", stats.Size)
	fmt.Printf("  Index Type: %s\n", stats.IndexType)
	fmt.Printf("  Memory Usage: %.2f MB\n", float64(stats.MemoryUsage)/1024/1024)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
