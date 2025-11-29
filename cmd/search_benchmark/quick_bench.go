//go:build legacy

package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

// Realistic texts from different domains
var sampleTexts = []string{
	// Programming & Tech (20 texts)
	"Python programming language is known for its simplicity and extensive libraries for data science",
	"JavaScript enables dynamic web applications with client-side scripting and modern frameworks",
	"Go language provides excellent concurrency support through goroutines and channels",
	"Machine learning models can automatically learn patterns from data without explicit programming",
	"Docker containers package applications with dependencies for consistent deployment",
	"Kubernetes orchestrates containerized applications at scale with automated management",
	"React framework builds interactive user interfaces with component-based architecture",
	"PostgreSQL database offers advanced SQL features and excellent performance",
	"Redis provides in-memory data storage for caching and real-time applications",
	"Git version control tracks code changes and enables collaborative development",
	"REST APIs enable communication between different software systems using HTTP",
	"Microservices architecture breaks applications into small, independent services",
	"Cloud computing provides on-demand computing resources over the internet",
	"DevOps practices combine development and operations for faster delivery",
	"Cybersecurity protects systems and data from digital attacks and breaches",
	"Blockchain technology creates immutable, distributed ledgers for transactions",
	"Artificial intelligence simulates human intelligence in machines and systems",
	"Data engineering builds pipelines for collecting and processing large datasets",
	"Web development creates websites and web applications using various technologies",
	"Mobile app development builds applications for smartphones and tablets",

	// Science & Research (20 texts)
	"Quantum computing uses quantum mechanics principles for exponentially faster calculations",
	"CRISPR gene editing allows precise modifications to DNA sequences in organisms",
	"Climate change causes global temperature increases and weather pattern shifts",
	"Neural networks are computing systems inspired by biological brain structures",
	"Black holes are spacetime regions with gravitational pull so strong nothing escapes",
	"DNA stores genetic information using four nucleotide bases in double helix structure",
	"Evolution explains biodiversity through natural selection and genetic variation",
	"Photosynthesis converts light energy into chemical energy in plants and algae",
	"Stem cells can differentiate into various specialized cell types in the body",
	"Particle physics studies fundamental particles and forces in the universe",
	"Astronomy explores celestial objects, space, and the physical universe",
	"Biotechnology uses living organisms to develop products and technologies",
	"Renewable energy comes from naturally replenishing sources like solar and wind",
	"Nanotechnology manipulates matter at atomic and molecular scales",
	"Ecology studies relationships between organisms and their environment",
	"Neuroscience investigates the nervous system and brain functions",
	"Genetics examines heredity and variation in living organisms",
	"Chemistry studies matter properties, composition, and transformations",
	"Physics explores matter, energy, and fundamental forces of nature",
	"Biology is the scientific study of life and living organisms",

	// Business & Finance (20 texts)
	"Stock markets facilitate buying and selling of company shares and securities",
	"Cryptocurrency uses cryptography for secure digital currency transactions",
	"Supply chain management coordinates flow of goods from suppliers to customers",
	"Digital marketing promotes products through online channels and platforms",
	"E-commerce enables buying and selling products over the internet",
	"Venture capital funds startups with high growth potential for equity",
	"Data analytics helps organizations make informed decisions from data insights",
	"Project management organizes resources and tasks to achieve specific goals",
	"Business intelligence uses data analysis to support business decisions",
	"Financial planning manages money to achieve personal and business goals",
	"Investment banking provides financial services to corporations and governments",
	"Risk management identifies and mitigates potential business threats",
	"Customer relationship management systems track customer interactions and data",
	"Agile methodology emphasizes iterative development and continuous improvement",
	"Market research gathers information about target markets and customers",
	"Entrepreneurship involves starting and running new business ventures",
	"Corporate strategy defines long-term goals and competitive positioning",
	"Human resources manages employee relations, hiring, and development",
	"Accounting tracks financial transactions and prepares financial statements",
	"Economics studies production, distribution, and consumption of resources",
}

func generateVariations(base []string, targetSize int) []string {
	result := make([]string, 0, targetSize)
	result = append(result, base...)

	prefixes := []string{
		"Introduction to", "Advanced", "Modern", "Essential", "Practical",
		"Understanding", "Mastering", "Comprehensive guide to", "The future of",
	}

	suffixes := []string{
		"best practices", "in production", "at scale", "for beginners",
		"explained simply", "deep dive", "case study", "tutorial",
	}

	for len(result) < targetSize {
		text := base[rand.Intn(len(base))]

		switch rand.Intn(4) {
		case 0: // Add prefix
			text = prefixes[rand.Intn(len(prefixes))] + " " + strings.ToLower(text)
		case 1: // Add suffix
			text = text + " - " + suffixes[rand.Intn(len(suffixes))]
		case 2: // Combine two topics
			other := base[rand.Intn(len(base))]
			text = text + " and " + strings.ToLower(other[:len(other)/2])
		}

		result = append(result, text)
	}

	return result[:targetSize]
}

func main() {
	fmt.Println("=== Gobed Quick Search Benchmark ===")
	fmt.Println("Testing with 10,000 documents\n")

	// Load model
	fmt.Println("Loading model...")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ Model loaded in %v\n\n", time.Since(start))

	// Generate corpus
	corpusSize := 10000
	fmt.Printf("Generating %d texts...\n", corpusSize)
	corpus := generateVariations(sampleTexts, corpusSize)
	fmt.Printf("✓ Corpus ready\n\n")

	// Create search engine
	engine := gobed.NewSearchEngine(model)

	// Benchmark indexing
	fmt.Println("=== INDEXING ===")
	batchSize := 500
	indexStart := time.Now()

	for i := 0; i < corpusSize; i += batchSize {
		end := min(i+batchSize, corpusSize)
		_, err := engine.IndexBatch(corpus[i:end])
		if err != nil {
			log.Printf("Index error: %v", err)
		}

		if (i+batchSize)%2000 == 0 {
			fmt.Printf("  Indexed %d/%d documents\n", min(i+batchSize, corpusSize), corpusSize)
		}
	}

	indexTime := time.Since(indexStart)
	fmt.Printf("\n✓ Indexed %d documents in %v\n", corpusSize, indexTime)
	fmt.Printf("  Throughput: %.0f docs/sec\n\n", float64(corpusSize)/indexTime.Seconds())

	// Show stats
	stats := engine.Stats()
	fmt.Println("=== INDEX STATS ===")
	fmt.Printf("Type: %s\n", stats.IndexType)
	fmt.Printf("Memory: %.2f MB\n\n", stats.MemoryUsageMB)

	// Test queries
	queries := []string{
		"machine learning and neural networks",
		"web development frameworks",
		"database performance optimization",
		"quantum computing applications",
		"financial investment strategies",
		"cloud native applications",
		"gene editing technology",
		"mobile app development",
		"cryptocurrency and blockchain",
		"data science pipelines",
	}

	fmt.Println("=== SEARCH BENCHMARK ===")
	totalTime := time.Duration(0)

	for i, query := range queries {
		searchStart := time.Now()
		results, err := engine.Search(query, 5)
		searchTime := time.Since(searchStart)
		totalTime += searchTime

		if err != nil {
			log.Printf("Search error: %v", err)
			continue
		}

		fmt.Printf("\n%d. Query: '%s' (%v)\n", i+1, query, searchTime)
		for j, r := range results[:min(3, len(results))] {
			text := r.Text
			if len(text) > 70 {
				text = text[:67] + "..."
			}
			fmt.Printf("   %d. [%.3f] %s\n", j+1, r.Similarity, text)
		}
	}

	avgLatency := totalTime / time.Duration(len(queries))
	fmt.Printf("\n=== PERFORMANCE SUMMARY ===\n")
	fmt.Printf("Dataset: %d documents\n", corpusSize)
	fmt.Printf("Index time: %v (%.0f docs/sec)\n", indexTime, float64(corpusSize)/indexTime.Seconds())
	fmt.Printf("Avg search latency: %v\n", avgLatency)
	fmt.Printf("Search throughput: %.1f queries/sec\n", float64(len(queries))/totalTime.Seconds())
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
