package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	// Load the embedding model
	fmt.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✓ Model loaded\n")

	// Example 1: Basic usage with automatic configuration
	basicExample(model)

	// Example 2: Advanced search with options
	advancedExample(model)

	// Example 3: Large-scale example
	largeScaleExample(model)

	// Example 4: Finding similar documents
	similarityExample(model)
}

func basicExample(model *gobed.EmbeddingModel) {
	fmt.Println("=== Example 1: Basic Search ===")

	// Create a search engine with automatic configuration
	engine := gobed.NewSearchEngine(model)

	// Index some documents
	texts := []string{
		"Python is a high-level programming language",
		"Machine learning is a subset of artificial intelligence",
		"Neural networks are inspired by biological neurons",
		"Go is a statically typed compiled language",
		"Docker containers package applications",
		"Kubernetes orchestrates containerized applications",
		"React is a JavaScript library for building UIs",
		"PostgreSQL is a powerful relational database",
	}

	fmt.Printf("Indexing %d documents...\n", len(texts))
	ids, err := engine.IndexBatch(texts)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}
	fmt.Printf("✓ Indexed with IDs: %v\n\n", ids)

	// Perform searches
	queries := []string{
		"database systems",
		"AI and deep learning",
		"container technology",
		"programming languages",
	}

	for _, query := range queries {
		fmt.Printf("Query: '%s'\n", query)
		results, err := engine.Search(query, 3)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		for i, result := range results {
			fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Similarity, result.Text)
		}
		fmt.Println()
	}

	// Show statistics
	stats := engine.Stats()
	fmt.Printf("Engine stats: %d documents, Type: %s, Memory: %.2f MB\n\n",
		stats.NumDocuments, stats.IndexType, stats.MemoryUsageMB)
}

func advancedExample(model *gobed.EmbeddingModel) {
	fmt.Println("=== Example 2: Advanced Search with Options ===")

	// Create engine with custom configuration
	config := gobed.SearchConfig{
		AutoMode:           false,
		MaxExactSearchSize: 100,
		NumClusters:        4,
		SearchClusters:     2,
		UseCompression:     false,
	}
	engine := gobed.NewSearchEngineWithConfig(model, config)

	// Index documents with custom IDs
	documents := map[int]string{
		101: "The quick brown fox jumps over the lazy dog",
		102: "Machine learning algorithms can predict patterns",
		103: "Natural language processing understands text",
		104: "Computer vision processes visual information",
		105: "Deep learning uses multiple neural network layers",
	}

	for id, text := range documents {
		err := engine.IndexWithID(id, text)
		if err != nil {
			log.Printf("Failed to index doc %d: %v", id, err)
		}
	}

	// Search with options
	opts := gobed.SearchOptions{
		TopK:          3,
		MinSimilarity: 0.5, // Only return results with >50% similarity
	}

	query := "artificial intelligence and neural networks"
	fmt.Printf("Query: '%s' (min similarity: %.1f)\n", query, opts.MinSimilarity)

	results, err := engine.SearchWithOptions(query, opts)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	for i, result := range results {
		fmt.Printf("  %d. ID=%d [%.3f] %s\n", i+1, result.ID, result.Similarity, result.Text)
	}

	// Retrieve specific document
	if text, exists := engine.GetDocument(103); exists {
		fmt.Printf("\nDocument 103: %s\n", text)
	}
	fmt.Println()
}

func largeScaleExample(model *gobed.EmbeddingModel) {
	fmt.Println("=== Example 3: Large-Scale Search ===")

	engine := gobed.NewSearchEngine(model)

	// Simulate a larger corpus
	corpus := []string{
		// Programming languages
		"Python is known for its simplicity and readability",
		"JavaScript powers the modern web",
		"Go provides excellent concurrency support",
		"Rust ensures memory safety without garbage collection",
		"Java runs on billions of devices worldwide",
		"C++ offers high performance and control",
		"TypeScript adds static typing to JavaScript",
		"Swift is designed for iOS development",
		"Kotlin is the preferred language for Android",
		"Ruby emphasizes developer happiness",

		// Databases
		"MySQL is the world's most popular open source database",
		"PostgreSQL provides advanced SQL features",
		"MongoDB stores data in flexible JSON-like documents",
		"Redis is an in-memory data structure store",
		"Elasticsearch enables full-text search at scale",
		"Cassandra handles large amounts of data across many servers",
		"SQLite is a self-contained embedded database",
		"Neo4j is a graph database management system",
		"InfluxDB is optimized for time-series data",
		"DynamoDB is AWS's managed NoSQL database",

		// Cloud and DevOps
		"Docker containers ensure consistency across environments",
		"Kubernetes automates container deployment and scaling",
		"AWS provides comprehensive cloud computing services",
		"Azure is Microsoft's cloud computing platform",
		"Google Cloud Platform offers innovative AI services",
		"Terraform enables infrastructure as code",
		"Ansible automates IT infrastructure",
		"Jenkins facilitates continuous integration",
		"GitLab provides complete DevOps platform",
		"Prometheus monitors and alerts on metrics",

		// AI and ML
		"TensorFlow is Google's machine learning framework",
		"PyTorch provides dynamic computational graphs",
		"Scikit-learn offers simple machine learning tools",
		"Keras provides high-level neural network APIs",
		"OpenCV specializes in computer vision",
		"NLTK processes human language data",
		"Hugging Face democratizes NLP models",
		"JAX combines NumPy and automatic differentiation",
		"MLflow manages the machine learning lifecycle",
		"Apache Spark processes big data at scale",
	}

	// Index in batches for efficiency
	fmt.Printf("Indexing %d documents...\n", len(corpus))
	start := time.Now()

	batchSize := 10
	for i := 0; i < len(corpus); i += batchSize {
		end := min(i+batchSize, len(corpus))
		_, err := engine.IndexBatch(corpus[i:end])
		if err != nil {
			log.Printf("Failed to index batch: %v", err)
		}
	}

	indexTime := time.Since(start)
	fmt.Printf("✓ Indexed in %v\n\n", indexTime)

	// Optimize the index for better performance
	fmt.Println("Optimizing index...")
	start = time.Now()
	err := engine.Optimize()
	if err != nil {
		log.Printf("Optimization failed: %v", err)
	} else {
		fmt.Printf("✓ Optimized in %v\n\n", time.Since(start))
	}

	// Run benchmark queries
	queries := []string{
		"cloud native application deployment",
		"machine learning frameworks for Python",
		"NoSQL database solutions",
		"container orchestration platforms",
		"programming language for systems development",
	}

	totalLatency := time.Duration(0)
	for _, query := range queries {
		start := time.Now()
		results, err := engine.Search(query, 5)
		latency := time.Since(start)
		totalLatency += latency

		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("Query: '%s' (latency: %v)\n", query, latency)
		for i, result := range results[:min(3, len(results))] {
			fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Similarity, result.Text)
		}
		fmt.Println()
	}

	avgLatency := totalLatency / time.Duration(len(queries))
	fmt.Printf("Average search latency: %v\n", avgLatency)

	// Final statistics
	stats := engine.Stats()
	fmt.Printf("\nFinal stats: %d documents, Type: %s, Memory: %.2f MB\n",
		stats.NumDocuments, stats.IndexType, stats.MemoryUsageMB)
	fmt.Printf("Index details: %+v\n\n", stats.IndexDetails)
}

func similarityExample(model *gobed.EmbeddingModel) {
	fmt.Println("=== Example 4: Finding Similar Documents ===")

	engine := gobed.NewSearchEngine(model)

	// Index documents about different topics
	topics := map[int]string{
		1: "Machine learning uses statistical techniques to give computers the ability to learn",
		2: "Deep learning is part of machine learning based on artificial neural networks",
		3: "Natural language processing helps computers understand and generate human language",
		4: "Computer vision enables machines to interpret and understand visual information",
		5: "Reinforcement learning trains agents to make decisions through trial and error",
		6: "Supervised learning uses labeled data to train predictive models",
		7: "Unsupervised learning discovers hidden patterns in unlabeled data",
		8: "Transfer learning applies knowledge from one domain to another",
		9: "Federated learning trains models across decentralized data",
		10: "Active learning selects the most informative samples for labeling",
	}

	// Index all documents
	for id, text := range topics {
		err := engine.IndexWithID(id, text)
		if err != nil {
			log.Printf("Failed to index: %v", err)
		}
	}

	// Find documents similar to "Deep learning" (ID: 2)
	fmt.Println("Finding documents similar to: 'Deep learning is part of machine learning...'")
	similar, err := engine.FindSimilar(2, 3)
	if err != nil {
		log.Fatalf("Failed to find similar: %v", err)
	}

	fmt.Println("Most similar documents:")
	for i, result := range similar {
		fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Similarity, result.Text)
	}

	fmt.Println("\n✓ All examples completed!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}