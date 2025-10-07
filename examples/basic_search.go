package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("Gobed Basic Search Example")
	fmt.Println("=========================")

	// Load the embedding model first
	fmt.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine with auto configuration
	// This will automatically detect GPU and optimize settings
	engine := gobed.NewSearchEngine(model)

	// Show configuration
	stats := engine.Stats()
	fmt.Printf("Engine initialized: %d documents, Type: %s\n\n",
		stats.NumDocuments, stats.IndexType)

	// Add some sample documents using correct API
	documentTexts := []string{
		"machine learning algorithms and neural networks for data science",
		"deep learning models using convolutional neural networks",
		"artificial intelligence and natural language processing",
		"web development with JavaScript and React frameworks",
		"backend development using Go and microservices architecture",
		"database design and SQL query optimization techniques",
		"quantum computing and physics research applications",
		"biomedical research and computational biology methods",
		"climate science and environmental data analysis",
		"business strategy and market analysis frameworks",
		"financial modeling and investment portfolio management",
		"project management and agile development methodologies",
	}

	fmt.Printf("Indexing %d documents...\n", len(documentTexts))
	ids, err := engine.IndexBatch(documentTexts)
	if err != nil {
		log.Fatalf("Failed to index documents: %v", err)
	}
	fmt.Printf("✓ Indexed with IDs: %v\n", ids)

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

		results, err := engine.Search(query, 3)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("Found %d results\n", len(results))

		for i, result := range results {
			fmt.Printf("%d. [%.3f] %s\n   %s\n",
				i+1, result.Similarity, result.ID, result.Text)
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
		emb1, err1 := model.Encode(pair[0])
		emb2, err2 := model.Encode(pair[1])

		if err1 != nil || err2 != nil {
			log.Printf("Failed to encode similarity pair: %v, %v", err1, err2)
			continue
		}

		similarity := gobed.CosineSimilarity(emb1, emb2)
		fmt.Printf("'%s' ↔ '%s': %.3f\n", pair[0], pair[1], similarity)
	}

	// Final statistics
	finalStats := engine.Stats()
	fmt.Printf("\n📊 Final Statistics\n")
	fmt.Println("----------------------------------------")
	fmt.Printf("Documents indexed: %d\n", finalStats.NumDocuments)
	fmt.Printf("Index type: %s\n", finalStats.IndexType)
	fmt.Printf("Memory usage: %.2f MB\n", finalStats.MemoryUsageMB)
	fmt.Printf("Index details: %+v\n", finalStats.IndexDetails)
}