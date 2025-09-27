package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("=== Gobed Search Performance Test ===")
	fmt.Println("Testing with real netwrck character data simulation")

	// Load model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine
	searchEngine := gobed.NewGPUSearchEngine(model)

	// Simulate netwrck query patterns
	queries := []string{
		"popular characters adventure fantasy",
		"fantasy",
		"concept character design",
		"anime action heroes",
		"romantic comedy slice of life",
		"horror mystery thriller",
		"science fiction cyberpunk",
		"magical girl transformation",
		"isekai overpowered protagonist",
		"dragon quest medieval fantasy",
	}

	// Test embedding speed first
	fmt.Println("\n=== Embedding Performance ===")
	for _, query := range queries[:3] {
		start := time.Now()
		_, err := model.Encode(query)
		elapsed := time.Since(start)
		if err != nil {
			log.Printf("Embedding failed: %v", err)
			continue
		}
		fmt.Printf("Embed '%s': %.2fms\n", query, float64(elapsed.Microseconds())/1000.0)
	}

	// Generate and index test documents
	fmt.Println("\n=== Indexing Test Data ===")
	numDocs := 200000
	fmt.Printf("Generating %d synthetic documents...\n", numDocs)

	// Simulate character descriptions
	templates := []string{
		"A %s character with %s personality who enjoys %s and %s",
		"An anime %s from the world of %s specializing in %s magic",
		"A mysterious %s who travels through %s seeking %s",
		"The legendary %s of %s, known for their %s abilities",
		"A powerful %s warrior defending the realm of %s",
	}

	categories := []string{"fantasy", "sci-fi", "romance", "action", "comedy", "horror", "mystery", "adventure"}
	traits := []string{"brave", "cunning", "wise", "powerful", "mysterious", "cheerful", "dark", "heroic"}
	activities := []string{"battle", "exploration", "romance", "investigation", "training", "questing", "crafting", "diplomacy"}

	documents := make([]string, numDocs)
	ids := make([]int, numDocs)

	for i := 0; i < numDocs; i++ {
		template := templates[i%len(templates)]
		cat := categories[i%len(categories)]
		trait := traits[(i/10)%len(traits)]
		act1 := activities[(i/100)%len(activities)]
		act2 := activities[(i/50)%len(activities)]

		documents[i] = fmt.Sprintf(template, cat, trait, act1, act2)
		ids[i] = i
	}

	// Index in batches
	batchSize := 10000
	indexStart := time.Now()

	for i := 0; i < len(documents); i += batchSize {
		end := i + batchSize
		if end > len(documents) {
			end = len(documents)
		}

		err := searchEngine.IndexBatchWithIDs(ids[i:end], documents[i:end])
		if err != nil {
			log.Printf("Failed to index batch %d-%d: %v", i, end, err)
		}

		if (i/batchSize+1)%10 == 0 {
			fmt.Printf("Indexed %d/%d documents\n", end, len(documents))
		}
	}

	indexTime := time.Since(indexStart)
	fmt.Printf("Total indexing time: %.2fs (%.0f docs/sec)\n",
		indexTime.Seconds(), float64(numDocs)/indexTime.Seconds())

	// Warmup
	fmt.Println("\n=== Warming Up ===")
	for i := 0; i < 5; i++ {
		_, _ = searchEngine.Search(queries[0], 100)
	}

	// Benchmark searches
	fmt.Println("\n=== Search Performance Test ===")
	fmt.Println("Query                                     Time(ms)  Results")
	fmt.Println("----------------------------------------  --------  -------")

	totalTime := time.Duration(0)
	minTime := time.Hour
	maxTime := time.Duration(0)
	successCount := 0

	for _, query := range queries {
		start := time.Now()
		results, err := searchEngine.Search(query, 130)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("Search failed for '%s': %v", query, err)
			continue
		}

		fmt.Printf("%-40s  %8.2f  %7d\n",
			query, float64(elapsed.Microseconds())/1000.0, len(results))

		totalTime += elapsed
		successCount++

		if elapsed < minTime {
			minTime = elapsed
		}
		if elapsed > maxTime {
			maxTime = elapsed
		}
	}

	if successCount > 0 {
		avgTime := totalTime / time.Duration(successCount)
		fmt.Println("\n=== Performance Summary ===")
		fmt.Printf("Average: %.2fms\n", float64(avgTime.Microseconds())/1000.0)
		fmt.Printf("Min:     %.2fms\n", float64(minTime.Microseconds())/1000.0)
		fmt.Printf("Max:     %.2fms\n", float64(maxTime.Microseconds())/1000.0)
		fmt.Printf("QPS:     %.0f queries/sec\n", 1000.0/(float64(avgTime.Microseconds())/1000.0))

		// Compare to netwrck's current performance
		fmt.Println("\n=== Comparison to Current Netwrck Performance ===")
		currentLatency := 4500.0 // 4.5 seconds as reported
		improvement := currentLatency / (float64(avgTime.Microseconds())/1000.0)
		fmt.Printf("Current netwrck latency: %.0fms\n", currentLatency)
		fmt.Printf("This benchmark latency:  %.2fms\n", float64(avgTime.Microseconds())/1000.0)
		fmt.Printf("Improvement factor:      %.1fx faster\n", improvement)
	}
}