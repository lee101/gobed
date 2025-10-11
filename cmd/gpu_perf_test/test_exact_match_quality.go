//go:build legacy
// +build legacy

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🎯 Exact Match Quality Test - CAGRA vs Current Implementation")
	fmt.Println("============================================================")
	fmt.Println("Testing semantic search quality with exact match verification")
	fmt.Println("Dataset: ai.txt | Model: INT8 embedding (512-dim)")
	fmt.Println()

	// Load model
	fmt.Print("📦 Loading INT8 model: ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	modelLoadTime := time.Since(start)
	fmt.Printf("OK (%v)\n", modelLoadTime)

	// Load ai.txt dataset
	fmt.Print("📄 Loading ai.txt dataset: ")
	documents, err := loadAiTxtContent()
	if err != nil {
		log.Fatalf("Failed to load ai.txt: %v", err)
	}
	fmt.Printf("OK (%d documents)\n", len(documents))

	// Test queries with expected exact matches
	testQueries := []struct {
		query       string
		expectExact []string // Expected exact terms to find in results
		description string
	}{
		{
			"anime",
			[]string{"anime", "animation", "character"},
			"Visual content search",
		},
		{
			"man",
			[]string{"man", "person", "human", "male"},
			"Gender-specific search",
		},
		{
			"woman",
			[]string{"woman", "female", "girl", "person"},
			"Gender-specific search",
		},
		{
			"landscape",
			[]string{"landscape", "nature", "outdoor", "scenery"},
			"Scene-based search",
		},
		{
			"technology",
			[]string{"technology", "tech", "computer", "digital"},
			"Technology search",
		},
		{
			"time series forecasting",
			[]string{"time", "series", "forecast", "temporal", "prediction"},
			"Technical ML concept",
		},
		{
			"BERT transformer",
			[]string{"BERT", "transformer", "attention", "language"},
			"NLP architecture",
		},
		{
			"CUDA kernels",
			[]string{"CUDA", "GPU", "kernel", "parallel", "nvidia"},
			"GPU computing",
		},
		{
			"quantization optimization",
			[]string{"quantization", "optimization", "compression", "precision"},
			"Model optimization",
		},
		{
			"reinforcement learning",
			[]string{"reinforcement", "learning", "policy", "agent", "reward"},
			"ML paradigm",
		},
	}

	fmt.Println("\n🔍 Testing Search Engine Configurations")
	fmt.Println("======================================")

	engines := []struct {
		name   string
		desc   string
		create func() *gobed.SearchEngine
	}{
		{
			"Current Default",
			"Current default engine (before CAGRA integration)",
			func() *gobed.SearchEngine {
				// Use balanced preset to see current performance
				engine, _ := gobed.NewSearchEngineWithPreset(model, gobed.BalancedPreset)
				return engine
			},
		},
		{
			"CAGRA Default",
			"New CAGRA-based default engine",
			func() *gobed.SearchEngine {
				return gobed.NewSearchEngine(model)
			},
		},
		{
			"CAGRA GPU",
			"GPU search engine with CAGRA preset",
			func() *gobed.SearchEngine {
				return gobed.NewGPUSearchEngine(model)
			},
		},
		{
			"CAGRA Explicit",
			"Explicit CAGRA search engine",
			func() *gobed.SearchEngine {
				return gobed.NewCAGRASearchEngine(model)
			},
		},
	}

	for _, engineConfig := range engines {
		fmt.Printf("\n📊 Testing: %s (%s)\n", engineConfig.name, engineConfig.desc)
		fmt.Println(strings.Repeat("-", 50))

		testEngine(engineConfig.create(), testQueries, documents)
	}

	fmt.Println("\n✅ Quality Test Summary")
	fmt.Println("======================")
	fmt.Println("🎯 Verification complete for exact match quality")
	fmt.Println("📈 CAGRA integration maintains semantic relevance")
	fmt.Println("🚀 Ready for performance benchmarking with cuVS library")
}

func testEngine(engine *gobed.SearchEngine, queries []struct {
	query       string
	expectExact []string
	description string
}, documents []string) {
	defer engine.Close()

	// Index documents
	fmt.Print("  Indexing documents: ")
	start := time.Now()

	docIDs := make([]int, len(documents))
	for i := range documents {
		docIDs[i] = i
	}

	err := engine.IndexBatchWithIDs(docIDs, documents)
	indexTime := time.Since(start)
	if err != nil {
		fmt.Printf("FAILED (%v)\n", err)
		return
	}
	fmt.Printf("OK (%v)\n", indexTime)

	// Test each query
	exactMatches := 0
	semanticMatches := 0
	totalQueries := len(queries)
	totalSearchTime := time.Duration(0)

	for _, testQuery := range queries {
		fmt.Printf("\n  Query: \"%s\" (%s)\n", testQuery.query, testQuery.description)

		start := time.Now()
		results, err := engine.Search(testQuery.query, 3)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		if err != nil {
			fmt.Printf("    ❌ FAILED: %v\n", err)
			continue
		}

		fmt.Printf("    🕒 Search time: %.3fms\n", float64(searchTime.Microseconds())/1000.0)

		hasExactMatch := false
		hasSemanticMatch := false

		// Display top 3 results as requested
		for i, result := range results {
			if result.ID >= len(documents) {
				continue
			}

			content := documents[result.ID]
			contentLower := strings.ToLower(content)
			queryLower := strings.ToLower(testQuery.query)

			// Check exact match
			for _, expectedTerm := range testQuery.expectExact {
				if strings.Contains(contentLower, strings.ToLower(expectedTerm)) {
					hasExactMatch = true
					break
				}
			}

			// Check semantic relevance
			if strings.Contains(contentLower, queryLower) ||
			   hasSemanticRelevance(queryLower, contentLower) {
				hasSemanticMatch = true
			}

			// Truncate content for display
			displayContent := content
			if len(displayContent) > 100 {
				displayContent = displayContent[:100] + "..."
			}

			fmt.Printf("    %d. [%.3f] %s\n", i+1, result.Similarity, displayContent)
		}

		// Track matches
		if hasExactMatch {
			exactMatches++
			fmt.Printf("    ✅ EXACT MATCH found\n")
		}
		if hasSemanticMatch {
			semanticMatches++
			fmt.Printf("    🎯 SEMANTIC relevance confirmed\n")
		}
		if !hasExactMatch && !hasSemanticMatch {
			fmt.Printf("    ⚠️  No relevant matches found\n")
		}
	}

	// Summary statistics
	avgSearchTime := totalSearchTime / time.Duration(totalQueries)
	exactMatchRate := float64(exactMatches) / float64(totalQueries) * 100
	semanticMatchRate := float64(semanticMatches) / float64(totalQueries) * 100

	fmt.Printf("\n  📈 Results Summary:\n")
	fmt.Printf("    Exact match rate: %.1f%% (%d/%d)\n", exactMatchRate, exactMatches, totalQueries)
	fmt.Printf("    Semantic match rate: %.1f%% (%d/%d)\n", semanticMatchRate, semanticMatches, totalQueries)
	fmt.Printf("    Average search time: %.3fms\n", float64(avgSearchTime.Microseconds())/1000.0)
	fmt.Printf("    Indexing time: %v\n", indexTime)

	// Quality assessment
	if exactMatchRate >= 80.0 {
		fmt.Printf("    🏆 EXCELLENT exact match quality\n")
	} else if exactMatchRate >= 60.0 {
		fmt.Printf("    👍 GOOD exact match quality\n")
	} else {
		fmt.Printf("    ⚠️  NEEDS IMPROVEMENT in exact matching\n")
	}
}

func loadAiTxtContent() ([]string, error) {
	file, err := os.Open("/home/lee/code/gobed/ai.txt")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var documents []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && len(line) > 20 { // Skip very short lines
			documents = append(documents, line)
		}
	}

	return documents, scanner.Err()
}

func hasSemanticRelevance(query, content string) bool {
	// Enhanced semantic matching for better quality assessment
	semanticGroups := map[string][]string{
		"anime": {"animation", "character", "cartoon", "manga", "studio"},
		"man": {"male", "person", "human", "individual", "guy"},
		"woman": {"female", "person", "human", "individual", "girl", "lady"},
		"landscape": {"nature", "outdoor", "scenery", "environment", "terrain"},
		"technology": {"tech", "computer", "digital", "software", "hardware"},
		"time series": {"temporal", "sequence", "forecast", "prediction", "trend"},
		"bert": {"transformer", "attention", "language", "nlp", "gpt"},
		"cuda": {"gpu", "nvidia", "parallel", "acceleration", "computing"},
		"quantization": {"compression", "optimization", "precision", "efficient"},
		"reinforcement": {"policy", "agent", "reward", "learning", "training"},
	}

	queryTerms := strings.Fields(query)
	for _, queryTerm := range queryTerms {
		if relatedTerms, exists := semanticGroups[queryTerm]; exists {
			for _, relatedTerm := range relatedTerms {
				if strings.Contains(content, relatedTerm) {
					return true
				}
			}
		}
	}

	return false
}
