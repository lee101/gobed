//go:build legacy

package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	// Generate 200k synthetic documents
	numDocs := 200000
	fmt.Printf("Generating %d documents...\n", numDocs)
	documents := generateDocuments(numDocs)

	// Initialize model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine
	searchEngine := gobed.NewSearchEngine(model)

	fmt.Printf("Indexing %d documents...\n", numDocs)

	// Index documents
	indexStart := time.Now()
	ids := make([]int, numDocs)
	for i := range ids {
		ids[i] = i
	}

	err = searchEngine.IndexBatchWithIDs(ids, documents)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}

	indexTime := time.Since(indexStart)
	fmt.Printf("Indexed %d documents in %.2fs (%.0f docs/sec)\n",
		numDocs, indexTime.Seconds(), float64(numDocs)/indexTime.Seconds())

	// Test queries
	testQueries := []string{
		"fantasy adventure",
		"science fiction space",
		"romantic comedy",
		"horror thriller",
		"mystery detective",
		"action hero",
		"magical wizard",
		"cyberpunk future",
		"historical drama",
		"superhero powers",
	}

	// Warmup
	fmt.Println("\nWarming up...")
	for i := 0; i < 3; i++ {
		_, _ = searchEngine.Search(testQueries[0], 10)
	}

	// Benchmark searches
	fmt.Println("\nBenchmarking searches...")
	totalTime := time.Duration(0)
	minTime := time.Hour
	maxTime := time.Duration(0)

	for _, query := range testQueries {
		start := time.Now()
		results, err := searchEngine.Search(query, 100)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("Search failed for '%s': %v", query, err)
			continue
		}

		fmt.Printf("Query: %-20s Time: %6.2fms Results: %d\n",
			query, float64(elapsed.Microseconds())/1000.0, len(results))

		totalTime += elapsed
		if elapsed < minTime {
			minTime = elapsed
		}
		if elapsed > maxTime {
			maxTime = elapsed
		}
	}

	avgTime := totalTime / time.Duration(len(testQueries))
	fmt.Printf("\n=== Search Performance ===\n")
	fmt.Printf("Average: %.2fms\n", float64(avgTime.Microseconds())/1000.0)
	fmt.Printf("Min:     %.2fms\n", float64(minTime.Microseconds())/1000.0)
	fmt.Printf("Max:     %.2fms\n", float64(maxTime.Microseconds())/1000.0)
	fmt.Printf("QPS:     %.0f\n", 1000.0/(float64(avgTime.Microseconds())/1000.0))

	// Test with different batch sizes
	fmt.Println("\n=== Testing Different Result Sizes ===")
	for _, k := range []int{10, 50, 100, 200} {
		start := time.Now()
		_, err := searchEngine.Search("fantasy adventure", k)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("Failed for k=%d: %v", k, err)
			continue
		}

		fmt.Printf("Top-%d: %.2fms\n", k, float64(elapsed.Microseconds())/1000.0)
	}
}

func generateDocuments(n int) []string {
	rand.Seed(42)

	categories := []string{
		"fantasy", "science fiction", "romance", "horror", "mystery",
		"action", "adventure", "comedy", "drama", "thriller",
		"magical", "cyberpunk", "steampunk", "historical", "superhero",
		"detective", "space", "wizard", "vampire", "zombie",
	}

	adjectives := []string{
		"epic", "dark", "light", "mysterious", "ancient",
		"modern", "futuristic", "classic", "legendary", "mythical",
		"powerful", "weak", "strong", "fast", "slow",
		"intelligent", "brave", "cowardly", "noble", "evil",
	}

	nouns := []string{
		"hero", "villain", "warrior", "mage", "knight",
		"princess", "prince", "dragon", "monster", "creature",
		"spaceship", "robot", "alien", "detective", "scientist",
		"explorer", "pirate", "ninja", "samurai", "wizard",
	}

	docs := make([]string, n)
	for i := 0; i < n; i++ {
		// Generate a semi-realistic document
		cat := categories[rand.Intn(len(categories))]
		adj1 := adjectives[rand.Intn(len(adjectives))]
		adj2 := adjectives[rand.Intn(len(adjectives))]
		noun1 := nouns[rand.Intn(len(nouns))]
		noun2 := nouns[rand.Intn(len(nouns))]

		docs[i] = fmt.Sprintf("A %s %s story about a %s %s who encounters a %s %s in the world of %s",
			adj1, cat, adj2, noun1, adjectives[rand.Intn(len(adjectives))], noun2, cat)
	}

	return docs
}
