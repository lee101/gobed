//go:build legacy

package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

// Sample texts from various domains - we'll generate variations of these
var baseTexts = []string{
	// Technology & Programming
	"Python is a high-level, interpreted programming language known for its simplicity and readability",
	"JavaScript is the programming language of the web, enabling interactive and dynamic websites",
	"Go is a statically typed, compiled language designed for simplicity and efficient concurrency",
	"Machine learning algorithms can automatically learn and improve from experience without being explicitly programmed",
	"Docker containers package applications with their dependencies for consistent deployment across environments",
	"Kubernetes orchestrates containerized applications, handling scaling, load balancing, and self-healing",
	"React is a JavaScript library for building user interfaces with a component-based architecture",
	"PostgreSQL is a powerful, open-source relational database management system with advanced features",
	"Redis is an in-memory data structure store used as a database, cache, and message broker",
	"Git is a distributed version control system for tracking changes in source code during development",

	// Science & Research
	"Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways",
	"CRISPR gene editing technology allows precise modifications to DNA sequences in living organisms",
	"Climate change refers to long-term shifts in global temperatures and weather patterns",
	"Artificial neural networks are computing systems inspired by biological neural networks in animal brains",
	"The theory of relativity describes the relationship between space, time, matter, and energy",
	"Photosynthesis is the process by which plants convert light energy into chemical energy",
	"Black holes are regions of spacetime where gravity is so strong that nothing can escape",
	"DNA stores genetic information using four nucleotide bases: adenine, guanine, cytosine, and thymine",
	"Evolution by natural selection explains the diversity of life through inherited variations",
	"The human brain contains approximately 86 billion neurons connected by trillions of synapses",

	// Business & Finance
	"Blockchain technology provides a decentralized, immutable ledger for recording transactions",
	"Cryptocurrency is a digital or virtual currency secured by cryptography and distributed networks",
	"Stock markets facilitate the buying and selling of company shares and other securities",
	"Supply chain management coordinates the flow of goods from suppliers to customers",
	"Artificial intelligence is transforming business operations through automation and data analysis",
	"Cloud computing delivers computing services over the internet on a pay-as-you-go basis",
	"Venture capital provides funding to startups with high growth potential in exchange for equity",
	"Digital marketing uses online channels to promote products and reach target audiences",
	"E-commerce enables buying and selling goods and services over the internet",
	"Data analytics helps organizations make informed decisions by analyzing large datasets",

	// Health & Medicine
	"Vaccines stimulate the immune system to develop immunity against specific diseases",
	"Antibiotics are medications that destroy or slow down the growth of bacteria",
	"MRI scanning uses magnetic fields and radio waves to create detailed images of internal organs",
	"Stem cells have the unique ability to develop into many different cell types in the body",
	"Telemedicine enables remote diagnosis and treatment of patients through telecommunications",
	"The cardiovascular system circulates blood throughout the body via the heart and blood vessels",
	"Mental health encompasses emotional, psychological, and social well-being throughout life",
	"Cancer occurs when abnormal cells divide uncontrollably and invade other tissues",
	"The immune system protects the body against disease-causing microorganisms and foreign substances",
	"Nutrition is the process of providing the body with food necessary for health and growth",

	// Education & Learning
	"Online learning platforms provide educational content accessible from anywhere with internet",
	"Critical thinking involves analyzing information objectively to form reasoned judgments",
	"STEM education focuses on science, technology, engineering, and mathematics subjects",
	"Cognitive psychology studies mental processes including perception, memory, and problem-solving",
	"Language acquisition is the process by which humans learn to understand and use language",
	"Educational technology integrates digital tools to enhance teaching and learning experiences",
	"Collaborative learning encourages students to work together to solve problems and complete tasks",
	"Assessment methods evaluate student learning, skill acquisition, and academic achievement",
	"Lifelong learning is the continuous pursuit of knowledge for personal or professional development",
	"Pedagogy is the art and science of teaching and instructional methods",
}

// generateCorpus creates a large corpus by generating variations
func generateCorpus(targetSize int) []string {
	corpus := make([]string, 0, targetSize)

	// Add original texts
	corpus = append(corpus, baseTexts...)

	// Generate variations
	variations := []string{
		"Understanding", "Exploring", "Introduction to", "Advanced", "Modern",
		"Practical", "Theoretical", "Applied", "Fundamental", "Essential",
		"Comprehensive guide to", "The science of", "The art of", "Mastering",
		"Deep dive into", "Overview of", "Principles of", "The future of",
	}

	domains := []string{
		"in production", "for beginners", "for experts", "in practice",
		"at scale", "in the cloud", "for enterprise", "for startups",
		"in 2024", "best practices", "case studies", "real-world applications",
		"industry trends", "research perspectives", "technical details",
	}

	// Generate combinations
	for len(corpus) < targetSize {
		base := baseTexts[rand.Intn(len(baseTexts))]

		// Sometimes add prefix
		if rand.Float32() < 0.5 {
			prefix := variations[rand.Intn(len(variations))]
			base = prefix + " " + strings.ToLower(base)
		}

		// Sometimes add suffix
		if rand.Float32() < 0.5 {
			suffix := domains[rand.Intn(len(domains))]
			base = base + " " + suffix
		}

		// Sometimes combine two topics
		if rand.Float32() < 0.3 && len(corpus) < targetSize-1 {
			other := baseTexts[rand.Intn(len(baseTexts))]
			combined := base + " and " + strings.ToLower(other)
			corpus = append(corpus, combined)
		}

		corpus = append(corpus, base)
	}

	return corpus[:targetSize]
}

// BenchmarkQueries are realistic search queries
var benchmarkQueries = []struct {
	query       string
	description string
}{
	{"machine learning algorithms for image recognition", "ML/Computer Vision"},
	{"database performance optimization techniques", "Database/Performance"},
	{"cloud native application development", "Cloud/Development"},
	{"quantum computing applications in cryptography", "Quantum/Security"},
	{"sustainable energy solutions and climate change", "Environment/Energy"},
	{"neural network architectures for NLP", "AI/NLP"},
	{"microservices vs monolithic architecture", "Architecture/Design"},
	{"blockchain technology in supply chain", "Blockchain/Business"},
	{"gene therapy and CRISPR applications", "Biotech/Medicine"},
	{"containerization and orchestration platforms", "DevOps/Infrastructure"},
	{"real-time data processing at scale", "BigData/Streaming"},
	{"mobile app development frameworks", "Mobile/Development"},
	{"cybersecurity threat detection systems", "Security/AI"},
	{"distributed systems consensus algorithms", "Distributed/Algorithms"},
	{"web3 and decentralized applications", "Blockchain/Web"},
}

func main() {
	fmt.Println("=== Gobed Search Performance Benchmark ===")
	fmt.Println("Testing with 100,000 real text embeddings\n")

	// Load the embedding model
	fmt.Println("Loading embedding model...")
	startTime := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ Model loaded in %v\n\n", time.Since(startTime))

	// Generate corpus
	corpusSize := 100000
	fmt.Printf("Generating corpus of %d texts...\n", corpusSize)
	startTime = time.Now()
	corpus := generateCorpus(corpusSize)
	fmt.Printf("✓ Corpus generated in %v\n\n", time.Since(startTime))

	// Create search engine
	fmt.Println("Creating search engine with automatic configuration...")
	engine := gobed.NewSearchEngine(model)

	// Benchmark indexing
	fmt.Println("=== INDEXING BENCHMARK ===")
	fmt.Printf("Indexing %d documents...\n", corpusSize)

	batchSize := 1000
	startTime = time.Now()
	indexingStart := time.Now()

	for i := 0; i < corpusSize; i += batchSize {
		end := min(i+batchSize, corpusSize)
		batch := corpus[i:end]

		batchStart := time.Now()
		_, err := engine.IndexBatch(batch)
		if err != nil {
			log.Printf("Failed to index batch: %v", err)
			continue
		}
		batchTime := time.Since(batchStart)

		// Progress update every 10k
		if (i+batchSize)%10000 == 0 {
			elapsed := time.Since(indexingStart)
			docsPerSec := float64(i+batchSize) / elapsed.Seconds()
			fmt.Printf("  Indexed %d/%d documents (%.0f docs/sec, batch: %v)\n",
				min(i+batchSize, corpusSize), corpusSize, docsPerSec, batchTime)
		}
	}

	totalIndexTime := time.Since(indexingStart)
	indexingThroughput := float64(corpusSize) / totalIndexTime.Seconds()

	fmt.Printf("\n✓ Indexing completed in %v\n", totalIndexTime)
	fmt.Printf("  Throughput: %.0f documents/second\n", indexingThroughput)
	fmt.Printf("  Average: %.2f ms/document\n\n", float64(totalIndexTime.Milliseconds())/float64(corpusSize))

	// Get and print statistics
	stats := engine.Stats()
	fmt.Println("=== INDEX STATISTICS ===")
	fmt.Printf("Documents:     %d\n", stats.NumDocuments)
	fmt.Printf("Index Type:    %s\n", stats.IndexType)
	fmt.Printf("Memory Usage:  %.2f MB\n", stats.MemoryUsageMB)
	fmt.Printf("Index Details: %+v\n\n", stats.IndexDetails)

	// Optimize index
	fmt.Println("Optimizing index for better search performance...")
	startTime = time.Now()
	err = engine.Optimize()
	if err != nil {
		log.Printf("Optimization failed: %v", err)
	} else {
		fmt.Printf("✓ Optimization completed in %v\n\n", time.Since(startTime))
	}

	// Benchmark search
	fmt.Println("=== SEARCH BENCHMARK ===")
	fmt.Printf("Running %d search queries...\n\n", len(benchmarkQueries))

	totalSearchTime := time.Duration(0)
	k := 5 // Top 5 results

	for i, bq := range benchmarkQueries {
		fmt.Printf("Query %d: \"%s\"\n", i+1, bq.query)
		fmt.Printf("Category: %s\n", bq.description)

		// Measure search time
		searchStart := time.Now()
		results, err := engine.Search(bq.query, k)
		searchTime := time.Since(searchStart)
		totalSearchTime += searchTime

		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("Search time: %v\n", searchTime)
		fmt.Println("Top 5 results:")
		for j, result := range results {
			// Truncate long texts for display
			text := result.Text
			if len(text) > 80 {
				text = text[:77] + "..."
			}
			fmt.Printf("  %d. [%.3f] %s\n", j+1, result.Similarity, text)
		}
		fmt.Println()
	}

	// Calculate search performance metrics
	avgSearchTime := totalSearchTime / time.Duration(len(benchmarkQueries))
	searchQPS := float64(len(benchmarkQueries)) / totalSearchTime.Seconds()

	fmt.Println("=== PERFORMANCE SUMMARY ===")
	fmt.Printf("Dataset size:           %d documents\n", corpusSize)
	fmt.Printf("Index type:             %s\n", stats.IndexType)
	fmt.Printf("Memory usage:           %.2f MB\n", stats.MemoryUsageMB)
	fmt.Println("\nIndexing Performance:")
	fmt.Printf("  Total time:           %v\n", totalIndexTime)
	fmt.Printf("  Throughput:           %.0f docs/sec\n", indexingThroughput)
	fmt.Printf("  Latency:              %.2f ms/doc\n", float64(totalIndexTime.Milliseconds())/float64(corpusSize))
	fmt.Println("\nSearch Performance:")
	fmt.Printf("  Average latency:      %v\n", avgSearchTime)
	fmt.Printf("  Throughput:           %.1f queries/sec\n", searchQPS)
	fmt.Printf("  Total queries:        %d\n", len(benchmarkQueries))

	// Interactive search demo
	fmt.Println("\n=== INTERACTIVE SEARCH ===")
	fmt.Println("Enter your search queries (or 'quit' to exit):")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nQuery> ")
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "quit" || query == "exit" || query == "q" {
			break
		}

		if query == "" {
			continue
		}

		// Search
		searchStart := time.Now()
		results, err := engine.Search(query, 10)
		searchTime := time.Since(searchStart)

		if err != nil {
			fmt.Printf("Search error: %v\n", err)
			continue
		}

		fmt.Printf("\nResults for '%s' (found in %v):\n", query, searchTime)
		for i, result := range results {
			// Show more of the text in interactive mode
			text := result.Text
			if len(text) > 100 {
				text = text[:97] + "..."
			}
			fmt.Printf("%2d. [%.3f] %s\n", i+1, result.Similarity, text)
		}
	}

	fmt.Println("\n✓ Benchmark completed!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
