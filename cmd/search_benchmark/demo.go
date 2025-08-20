package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("=== Gobed Semantic Search Demo ===\n")

	// Load model
	fmt.Println("Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Model loaded\n")

	// Create search engine
	engine := gobed.NewSearchEngine(model)

	// High-quality technical documents
	documents := []string{
		// Programming Languages
		"Python's simplicity and extensive libraries make it ideal for data science, machine learning, and rapid prototyping",
		"Go provides excellent concurrency through goroutines and channels, making it perfect for cloud-native applications",
		"Rust guarantees memory safety without garbage collection through its innovative ownership system",
		"JavaScript powers both client and server applications, dominating web development with frameworks like React and Node.js",
		"TypeScript adds static typing to JavaScript, improving code maintainability and developer productivity",
		
		// AI & Machine Learning
		"Deep learning neural networks with multiple hidden layers can learn complex patterns in data",
		"Transformer models like GPT and BERT have revolutionized natural language processing",
		"Reinforcement learning trains agents to make decisions by maximizing cumulative rewards",
		"Computer vision uses convolutional neural networks to understand and interpret images",
		"Gradient descent optimizes neural network weights by minimizing the loss function",
		
		// Databases
		"PostgreSQL provides ACID compliance, complex queries, and extensibility for relational data",
		"MongoDB offers flexible document storage with horizontal scaling for unstructured data",
		"Redis serves as an in-memory cache and message broker with sub-millisecond latency",
		"Elasticsearch enables full-text search and real-time analytics on large datasets",
		"Vector databases like Pinecone and Weaviate specialize in similarity search for embeddings",
		
		// Cloud & DevOps
		"Kubernetes orchestrates containers with automatic scaling, self-healing, and load balancing",
		"Docker containers ensure consistent application deployment across different environments",
		"Terraform enables infrastructure as code for reproducible cloud deployments",
		"GitHub Actions automates CI/CD pipelines directly from your repository",
		"Microservices architecture improves scalability by breaking applications into independent services",
		
		// Security
		"Zero-trust security assumes no implicit trust and verifies every transaction",
		"Encryption protects data in transit and at rest using cryptographic algorithms",
		"OAuth 2.0 provides secure authorization for third-party application access",
		"SQL injection attacks exploit vulnerabilities in database queries to access sensitive data",
		"Multi-factor authentication adds security layers beyond just passwords",
	}

	// Index documents
	fmt.Printf("Indexing %d documents...\n", len(documents))
	start := time.Now()
	ids, err := engine.IndexBatch(documents)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✓ Indexed in %v\n\n", time.Since(start))

	// Demonstrate semantic understanding with various queries
	queries := []struct {
		query       string
		explanation string
	}{
		{
			"How to handle concurrent tasks efficiently?",
			"Should find Go's concurrency features",
		},
		{
			"Best database for JSON documents with scaling",
			"Should identify MongoDB",
		},
		{
			"Modern AI for understanding text",
			"Should find transformer/NLP content",
		},
		{
			"Container deployment and management",
			"Should find Kubernetes/Docker",
		},
		{
			"Fast key-value storage for caching",
			"Should identify Redis",
		},
		{
			"Type safety in web development",
			"Should find TypeScript",
		},
		{
			"Memory safe systems programming",
			"Should identify Rust",
		},
		{
			"Search engine for logs and metrics",
			"Should find Elasticsearch",
		},
	}

	fmt.Println("=== SEMANTIC SEARCH RESULTS ===\n")
	
	for i, q := range queries {
		fmt.Printf("Query %d: \"%s\"\n", i+1, q.query)
		fmt.Printf("Expected: %s\n", q.explanation)
		
		start := time.Now()
		results, err := engine.Search(q.query, 3)
		latency := time.Since(start)
		
		if err != nil {
			log.Printf("Error: %v", err)
			continue
		}
		
		fmt.Printf("Search time: %v\n", latency)
		fmt.Println("Results:")
		for j, r := range results {
			fmt.Printf("  %d. [%.3f] %s\n", j+1, r.Similarity, r.Text)
		}
		fmt.Println()
	}

	// Show statistics
	stats := engine.Stats()
	fmt.Println("=== PERFORMANCE METRICS ===")
	fmt.Printf("Documents indexed: %d\n", stats.NumDocuments)
	fmt.Printf("Index type: %s\n", stats.IndexType)
	fmt.Printf("Memory usage: %.2f MB\n", stats.MemoryUsageMB)
	
	// Document retrieval check
	fmt.Println("\n=== DOCUMENT RETRIEVAL ===")
	if text, exists := engine.GetDocument(ids[0]); exists {
		fmt.Printf("Document ID %d: %s\n", ids[0], text[:50]+"...")
	}
	
	fmt.Println("\n✓ Demo completed successfully!")
}