package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("Gobed Basic Search Example")
	fmt.Println("=========================")

	// Create search engine with auto configuration
	// This will automatically detect GPU and optimize settings
	engine, err := gobed.NewAutoSearchEngine()
	if err != nil {
		log.Fatalf("Failed to create search engine: %v", err)
	}
	defer engine.Close()

	// Show configuration
	stats := engine.Stats()
	fmt.Printf("Engine initialized with GPU: %v, Int8: %v\n\n",
		stats["gpu_enabled"], stats["int8_enabled"])

	// Add some sample documents
	documents := map[string]string{
		"ml1":   "machine learning algorithms and neural networks for data science",
		"ml2":   "deep learning models using convolutional neural networks",
		"ml3":   "artificial intelligence and natural language processing",
		"web1":  "web development with JavaScript and React frameworks",
		"web2":  "backend development using Go and microservices architecture",
		"web3":  "database design and SQL query optimization techniques",
		"sci1":  "quantum computing and physics research applications",
		"sci2":  "biomedical research and computational biology methods",
		"sci3":  "climate science and environmental data analysis",
		"biz1":  "business strategy and market analysis frameworks",
		"biz2":  "financial modeling and investment portfolio management",
		"biz3":  "project management and agile development methodologies",
	}

	fmt.Printf("Adding %d documents...\n", len(documents))
	if err := engine.AddDocuments(documents); err != nil {
		log.Fatalf("Failed to add documents: %v", err)
	}

	// Perform several example searches
	queries := []string{
		"artificial intelligence and machine learning",
		"web development programming",
		"scientific research computing",
		"business and finance",
	}

	for _, query := range queries {
		fmt.Printf("\n🔍 Searching for: \"%s\"\n", query)
		fmt.Println(string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─'))

		results, metadata, err := engine.SearchWithMetadata(query, 3)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("Found %d results in %dms\n",
			len(results), metadata["query_time_ms"])

		for i, result := range results {
			fmt.Printf("%d. [%.3f] %s\n   %s\n",
				i+1, result.Similarity, result.ID, result.Content)
		}
	}

	// Demonstrate similarity computation
	fmt.Printf("\n💡 Text Similarity Examples\n")
	fmt.Println(string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─'))

	pairs := [][]string{
		{"machine learning", "artificial intelligence"},
		{"web development", "software engineering"},
		{"quantum physics", "cooking recipes"},
		{"database design", "data management"},
	}

	for _, pair := range pairs {
		sim, err := engine.Similarity(pair[0], pair[1])
		if err != nil {
			log.Printf("Similarity computation failed: %v", err)
			continue
		}
		fmt.Printf("'%s' ↔ '%s': %.3f\n", pair[0], pair[1], sim)
	}

	// Final statistics
	finalStats := engine.Stats()
	fmt.Printf("\n📊 Final Statistics\n")
	fmt.Println(string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─') + string('─'))
	fmt.Printf("Documents indexed: %v\n", finalStats["documents_count"])
	fmt.Printf("GPU acceleration: %v\n", finalStats["gpu_enabled"])
	fmt.Printf("Int8 quantization: %v\n", finalStats["int8_enabled"])
	fmt.Printf("Index type: %v\n", finalStats["index_type"])
	fmt.Printf("Batch size: %v\n", finalStats["batch_size"])
}