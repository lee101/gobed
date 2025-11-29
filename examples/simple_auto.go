//go:build legacy

package main

import (
	"fmt"
	"log"
	"sync"

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
	engine := gobed.NewSearchEngine(model) // Use NewSearchEngine which includes auto-optimization

	// Prepare sample documents for batch indexing (ultra-fast parallel processing)
	documentTexts := []string{
		"machine learning algorithms and artificial intelligence",
		"web development with JavaScript and React",
		"database design and SQL optimization",
		"quantum computing and physics research",
		"biomedical research and genomics",
		"business strategy and market analysis",
		"financial modeling and investment",
	}

	fmt.Printf("Ultra-fast batch indexing %d documents...\n", len(documentTexts))
	// Use IndexBatch for maximum parallel throughput instead of sequential indexing
	ids, err := engine.IndexBatch(documentTexts)
	if err != nil {
		log.Fatalf("Failed to batch index documents: %v", err)
	}
	fmt.Printf("✅ Indexed %d documents with IDs: %v\n", len(ids), ids)

	// Ultra-fast parallel searches (no for loops, maximum parallelism)
	queries := []string{
		"artificial intelligence and machine learning",
		"web programming and development",
		"scientific research and computing",
		"business and finance analysis",
	}

	fmt.Printf("\n🚀 Ultra-fast parallel searching %d queries...\n", len(queries))

	// Parallel search processing with channels (no sequential for loops)
	type searchResult struct {
		query   string
		results []gobed.SearchResult
		err     error
	}

	resultChan := make(chan searchResult, len(queries))

	// Launch all searches in parallel for maximum throughput
	for _, query := range queries {
		go func(q string) {
			results, err := engine.Search(q, 3)
			resultChan <- searchResult{query: q, results: results, err: err}
		}(query)
	}

	// Collect all results as they complete
	for i := 0; i < len(queries); i++ {
		result := <-resultChan

		fmt.Printf("\n🔍 Search: \"%s\"\n", result.query)
		fmt.Println("----------------------------------------")

		if result.err != nil {
			log.Printf("Failed to search: %v", result.err)
			continue
		}

		if len(result.results) == 0 {
			fmt.Println("No results found")
			continue
		}

		for j, searchRes := range result.results {
			fmt.Printf("%d. [%.3f] ID:%d\n   %s\n",
				j+1, searchRes.Similarity, searchRes.ID, searchRes.Text)
		}
	}

	// Ultra-fast parallel similarity computation (no for loops)
	fmt.Printf("\n💡 Ultra-Fast Parallel Similarity Examples\n")
	fmt.Println("--------------------------------------------")

	similarityPairs := [][]string{
		{"machine learning", "artificial intelligence"},
		{"web development", "software programming"},
		{"quantum physics", "cooking recipes"},
		{"business strategy", "market analysis"},
	}

	// Parallel similarity computation for maximum speed
	type similarityResult struct {
		pair       []string
		similarity float32
		err        error
	}

	simResultChan := make(chan similarityResult, len(similarityPairs))

	// Process all similarity pairs in parallel
	for _, pair := range similarityPairs {
		go func(p []string) {
			// Parallel embedding generation within each pair
			var emb1, emb2 []float32
			var err1, err2 error

			// Use sync.WaitGroup for ultra-fast parallel embedding
			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				emb1, err1 = model.Encode(p[0])
			}()

			go func() {
				defer wg.Done()
				emb2, err2 = model.Encode(p[1])
			}()

			wg.Wait()

			if err1 != nil || err2 != nil {
				simResultChan <- similarityResult{
					pair: p,
					err:  fmt.Errorf("encoding failed: %v, %v", err1, err2),
				}
				return
			}

			similarity := gobed.CosineSimilarity(emb1, emb2)
			simResultChan <- similarityResult{
				pair:       p,
				similarity: similarity,
			}
		}(pair)
	}

	// Collect all similarity results
	for i := 0; i < len(similarityPairs); i++ {
		result := <-simResultChan

		if result.err != nil {
			log.Printf("Failed to compute similarity: %v", result.err)
			continue
		}

		fmt.Printf("'%s' ↔ '%s': %.3f\n",
			result.pair[0], result.pair[1], result.similarity)
	}

	fmt.Printf("\n✅ Ultra-fast parallel example completed successfully!\n")
	fmt.Printf("📊 Total documents indexed: %d\n", len(documentTexts))
}
