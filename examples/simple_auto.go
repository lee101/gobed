package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("Gobed Auto GPU Detection Example")
	fmt.Println("===============================")

	// Load the embedding model
	fmt.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine with automatic GPU detection and optimization
	fmt.Println("Creating search engine with auto-optimization...")
	engine := gobed.NewAutoSearchEngine(model)

	// Add some sample documents
	documents := []struct {
		id      string
		content string
	}{
		{"tech1", "machine learning algorithms and artificial intelligence"},
		{"tech2", "web development with JavaScript and React"},
		{"tech3", "database design and SQL optimization"},
		{"sci1", "quantum computing and physics research"},
		{"sci2", "biomedical research and genomics"},
		{"biz1", "business strategy and market analysis"},
		{"biz2", "financial modeling and investment"},
	}

	fmt.Printf("Adding %d documents to search index...\n", len(documents))
	for _, doc := range documents {
		// Index the document content directly (SearchEngine handles encoding)
		_, err := engine.Index(doc.content)
		if err != nil {
			log.Printf("Failed to index document %s: %v", doc.id, err)
			continue
		}
	}

	// Perform searches
	queries := []string{
		"artificial intelligence and machine learning",
		"web programming and development",
		"scientific research and computing",
		"business and finance analysis",
	}

	for _, query := range queries {
		fmt.Printf("\n🔍 Search: \"%s\"\n", query)
		fmt.Println("----------------------------------------")

		// Search for similar documents (SearchEngine handles query encoding)
		results, err := engine.Search(query, 3)
		if err != nil {
			log.Printf("Failed to search: %v", err)
			continue
		}

		if len(results) == 0 {
			fmt.Println("No results found")
			continue
		}

		for i, result := range results {
			fmt.Printf("%d. [%.3f] ID:%d\n   %s\n",
				i+1, result.Similarity, result.ID, result.Text)
		}
	}

	// Test similarity computation
	fmt.Printf("\n💡 Text Similarity Examples\n")
	fmt.Println("----------------------------------------")

	similarityPairs := [][]string{
		{"machine learning", "artificial intelligence"},
		{"web development", "software programming"},
		{"quantum physics", "cooking recipes"},
		{"business strategy", "market analysis"},
	}

	for _, pair := range similarityPairs {
		emb1, err1 := model.Encode(pair[0])
		emb2, err2 := model.Encode(pair[1])

		if err1 != nil || err2 != nil {
			log.Printf("Failed to encode similarity pair: %v, %v", err1, err2)
			continue
		}

		similarity := gobed.CosineSimilarity(emb1, emb2)
		fmt.Printf("'%s' ↔ '%s': %.3f\n", pair[0], pair[1], similarity)
	}

	fmt.Printf("\n✅ Example completed successfully!\n")
	fmt.Printf("📊 Total documents indexed: %d\n", len(documents))
}