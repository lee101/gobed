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

// QualityBenchmarkResult stores comprehensive test results
type QualityBenchmarkResult struct {
	Query               string
	TopResults          []SearchResult
	ExactMatches        int
	SemanticMatches     int
	SearchTime          time.Duration
	Relevance           float64
	ExpectedTerms       []string
	FoundExpectedTerms  int
}

type SearchResult struct {
	ID         int
	Content    string
	Similarity float32
	Rank       int
}

func main() {
	fmt.Println("🎯 Current Search Quality Benchmark (ai.txt)")
	fmt.Println("=============================================")
	fmt.Println("Testing current search engine quality before CAGRA integration")
	fmt.Println("Focus: Exact matching, semantic relevance, and performance baseline")
	fmt.Println()

	// Load model
	fmt.Print("📦 Loading model: ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	modelLoadTime := time.Since(start)
	fmt.Printf("OK (%v)\n", modelLoadTime)

	// Load ai.txt content
	fmt.Print("📄 Loading ai.txt content: ")
	documents, err := loadAiTxtContent()
	if err != nil {
		log.Fatalf("Failed to load ai.txt: %v", err)
	}
	fmt.Printf("OK (%d documents)\n", len(documents))

	// Index documents with timing
	fmt.Print("🔍 Indexing documents: ")
	start = time.Now()
	engine, err := indexDocuments(model, documents)
	if err != nil {
		log.Fatalf("Failed to index documents: %v", err)
	}
	defer engine.Close()
	indexTime := time.Since(start)
	fmt.Printf("OK (%v)\n", indexTime)

	// Run comprehensive quality tests
	fmt.Println("\n🎯 Current Search Quality Tests")
	fmt.Println("==============================")

	// Test queries with expected results
	qualityTests := []struct {
		query         string
		expectedTerms []string
		description   string
	}{
		{
			query:         "time series forecasting",
			expectedTerms: []string{"time series", "forecasting", "RNNs", "LSTMs"},
			description:   "Time series analysis",
		},
		{
			query:         "BERT transformer",
			expectedTerms: []string{"BERT", "GPT", "transformer", "attention"},
			description:   "Transformer models",
		},
		{
			query:         "CUDA kernels",
			expectedTerms: []string{"CUDA", "GPU", "acceleration", "kernels"},
			description:   "GPU computing",
		},
		{
			query:         "quantization optimization",
			expectedTerms: []string{"quantization", "optimization", "compression", "precision"},
			description:   "Model optimization",
		},
		{
			query:         "reinforcement learning",
			expectedTerms: []string{"reinforcement", "policy", "gradient", "learning"},
			description:   "RL algorithms",
		},
		{
			query:         "computer vision CNN",
			expectedTerms: []string{"computer vision", "convolutional", "CNN", "image"},
			description:   "Computer vision",
		},
		{
			query:         "natural language processing",
			expectedTerms: []string{"natural language", "NLP", "transformer", "BERT"},
			description:   "NLP techniques",
		},
		{
			query:         "federated learning",
			expectedTerms: []string{"federated", "privacy", "distributed", "learning"},
			description:   "Distributed ML",
		},
	}

	var allResults []QualityBenchmarkResult
	totalSearchTime := time.Duration(0)

	for _, test := range qualityTests {
		fmt.Printf("\n📝 Testing: %s (\"%s\")\n", test.description, test.query)

		start := time.Now()
		results, err := engine.Search(test.query, 5)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		if err != nil {
			fmt.Printf("  ❌ Search failed: %v\n", err)
			continue
		}

		// Convert to our result format
		searchResults := make([]SearchResult, len(results))
		for i, r := range results {
			content := ""
			if r.ID < len(documents) {
				content = documents[r.ID]
			}
			searchResults[i] = SearchResult{
				ID:         r.ID,
				Content:    content,
				Similarity: r.Similarity,
				Rank:       i + 1,
			}
		}

		// Analyze quality
		result := analyzeQuality(test.query, test.expectedTerms, searchResults, searchTime)
		allResults = append(allResults, result)

		// Print top 3 results
		fmt.Printf("  Top 3 results:\n")
		for i, res := range searchResults {
			if i >= 3 {
				break
			}
			content := res.Content
			if len(content) > 80 {
				content = content[:80] + "..."
			}
			fmt.Printf("    %d. (%.3f) %s\n", i+1, res.Similarity, content)
		}

		fmt.Printf("  Search time: %v\n", result.SearchTime)
		fmt.Printf("  Expected terms found: %d/%d\n", result.FoundExpectedTerms, len(test.expectedTerms))
		fmt.Printf("  Relevance score: %.1f%%\n", result.Relevance*100)

		qualityEmoji := "✅"
		if result.Relevance < 0.7 {
			qualityEmoji = "⚠️"
		}
		if result.Relevance < 0.4 {
			qualityEmoji = "❌"
		}
		fmt.Printf("  Quality: %s %s\n", qualityEmoji, getQualityDescription(result.Relevance))
	}

	// Overall benchmark results
	fmt.Println("\n📊 Current Search Engine Benchmark Summary")
	fmt.Println("==========================================")
	printBenchmarkSummary(allResults, totalSearchTime, modelLoadTime, indexTime)

	// Performance analysis
	fmt.Println("\n⚡ Performance Baseline Analysis")
	fmt.Println("==============================")
	printPerformanceAnalysis(allResults, len(documents))

	// CAGRA integration recommendations
	fmt.Println("\n🚀 CAGRA Integration Readiness Assessment")
	fmt.Println("=========================================")
	printCAGRAReadiness(allResults)
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
		if line != "" {
			documents = append(documents, line)
		}
	}

	return documents, scanner.Err()
}

func indexDocuments(model *gobed.EmbeddingModel, documents []string) (*gobed.SearchEngine, error) {
	// Use the current default engine (CPU mode to avoid library issues)
	engine := gobed.NewSearchEngine(model)

	// Create document IDs
	docIDs := make([]int, len(documents))
	for i := range documents {
		docIDs[i] = i
	}

	err := engine.IndexBatchWithIDs(docIDs, documents)
	if err != nil {
		engine.Close()
		return nil, err
	}

	return engine, nil
}

func analyzeQuality(query string, expectedTerms []string, results []SearchResult, searchTime time.Duration) QualityBenchmarkResult {
	result := QualityBenchmarkResult{
		Query:      query,
		TopResults: results,
		SearchTime: searchTime,
		ExpectedTerms: expectedTerms,
	}

	if len(results) == 0 {
		return result
	}

	// Count expected terms found in top results
	foundTerms := 0
	for _, term := range expectedTerms {
		termLower := strings.ToLower(term)
		for _, res := range results {
			if strings.Contains(strings.ToLower(res.Content), termLower) {
				foundTerms++
				break
			}
		}
	}
	result.FoundExpectedTerms = foundTerms

	// Calculate relevance score
	termRelevance := float64(foundTerms) / float64(len(expectedTerms))

	// Check similarity scores
	avgSimilarity := 0.0
	for i, res := range results {
		weight := 1.0 / float64(i+1) // Weight by rank
		avgSimilarity += float64(res.Similarity) * weight
	}
	if len(results) > 0 {
		avgSimilarity /= 3.0 // Normalize by top 3
	}

	// Combined relevance score
	result.Relevance = (termRelevance * 0.7) + (avgSimilarity * 0.3)

	// Count exact and semantic matches
	queryLower := strings.ToLower(query)
	for _, res := range results {
		contentLower := strings.ToLower(res.Content)
		if strings.Contains(contentLower, queryLower) {
			result.ExactMatches++
		}
		if containsSemanticMatch(queryLower, contentLower) {
			result.SemanticMatches++
		}
	}

	return result
}

func containsSemanticMatch(query, content string) bool {
	// Simple semantic matching for ML terms
	semanticPairs := map[string][]string{
		"time series": {"forecasting", "rnn", "lstm", "temporal"},
		"bert":        {"transformer", "attention", "gpt", "language model"},
		"cuda":        {"gpu", "acceleration", "parallel", "kernels"},
		"quantization":{"compression", "precision", "optimization", "int8"},
		"reinforcement":{"policy", "gradient", "reward", "agent"},
		"computer vision":{"cnn", "convolutional", "image", "detection"},
		"federated":   {"distributed", "privacy", "decentralized"},
	}

	for key, related := range semanticPairs {
		if strings.Contains(query, key) {
			for _, term := range related {
				if strings.Contains(content, strings.ToLower(term)) {
					return true
				}
			}
		}
	}
	return false
}

func getQualityDescription(relevance float64) string {
	if relevance >= 0.8 {
		return "Excellent relevance"
	} else if relevance >= 0.6 {
		return "Good relevance"
	} else if relevance >= 0.4 {
		return "Fair relevance"
	}
	return "Poor relevance"
}

func printBenchmarkSummary(results []QualityBenchmarkResult, totalSearchTime, modelLoadTime, indexTime time.Duration) {
	if len(results) == 0 {
		return
	}

	avgRelevance := 0.0
	avgSearchTime := totalSearchTime / time.Duration(len(results))
	highQualityQueries := 0

	for _, r := range results {
		avgRelevance += r.Relevance
		if r.Relevance >= 0.7 {
			highQualityQueries++
		}
	}
	avgRelevance /= float64(len(results))

	fmt.Printf("Model Load Time: %v\n", modelLoadTime)
	fmt.Printf("Index Time: %v (%v per document)\n", indexTime, indexTime/time.Duration(len(results)))
	fmt.Printf("Average Search Time: %v\n", avgSearchTime)
	fmt.Printf("Total Queries: %d\n", len(results))
	fmt.Printf("High Quality Results: %d/%d (%.1f%%)\n",
		highQualityQueries, len(results), float64(highQualityQueries)/float64(len(results))*100)
	fmt.Printf("Average Relevance: %.1f%%\n", avgRelevance*100)

	// Overall score
	overallScore := avgRelevance * 100
	if overallScore >= 80 {
		fmt.Println("Current Quality: ✅ Excellent - Ready for CAGRA integration")
	} else if overallScore >= 60 {
		fmt.Println("Current Quality: ⚠️ Good - CAGRA integration viable")
	} else {
		fmt.Println("Current Quality: ❌ Needs improvement before CAGRA integration")
	}
}

func printPerformanceAnalysis(results []QualityBenchmarkResult, numDocs int) {
	if len(results) == 0 {
		return
	}

	totalSearchTime := time.Duration(0)
	for _, r := range results {
		totalSearchTime += r.SearchTime
	}

	avgSearchTime := totalSearchTime / time.Duration(len(results))
	qps := float64(len(results)) / totalSearchTime.Seconds()

	fmt.Printf("Dataset Size: %d documents\n", numDocs)
	fmt.Printf("Current Average Latency: %v\n", avgSearchTime)
	fmt.Printf("Current Queries Per Second: %.0f\n", qps)

	// Performance baseline for CAGRA comparison
	fmt.Printf("\nCurrent vs CAGRA Performance Targets:\n")

	cagraTargetLatency := 1 * time.Millisecond
	cagraTargetQPS := 100000.0

	latencyImprovement := float64(avgSearchTime) / float64(cagraTargetLatency)
	qpsImprovement := cagraTargetQPS / qps

	fmt.Printf("  Latency improvement potential: %.1fx (current: %v → target: <1ms)\n",
		latencyImprovement, avgSearchTime)
	fmt.Printf("  Throughput improvement potential: %.1fx (current: %.0f → target: 100K+ QPS)\n",
		qpsImprovement, qps)

	if avgSearchTime > 10*time.Millisecond {
		fmt.Printf("  Status: 🚀 Major speedup expected with CAGRA\n")
	} else {
		fmt.Printf("  Status: ⚡ Good speedup expected with CAGRA\n")
	}
}

func printCAGRAReadiness(results []QualityBenchmarkResult) {
	avgRelevance := 0.0
	for _, r := range results {
		avgRelevance += r.Relevance
	}
	avgRelevance /= float64(len(results))

	fmt.Printf("Current Quality Score: %.1f%%\n", avgRelevance*100)

	if avgRelevance >= 0.7 {
		fmt.Println("✅ Quality is sufficient for CAGRA integration")
		fmt.Println("\n🔧 Next Integration Steps:")
		fmt.Println("  1. Add CAGRA preset to search presets")
		fmt.Println("  2. Modify GPUSearchConfig to use CAGRA by default")
		fmt.Println("  3. Ensure CAGRA library is properly built and linked")
		fmt.Println("  4. Update indexing pipeline for CAGRA graph construction")
		fmt.Println("  5. Benchmark quality retention after CAGRA integration")

		fmt.Println("\n⚡ Expected CAGRA Benefits:")
		fmt.Println("  - 10-50x faster search latency")
		fmt.Println("  - 100K+ queries per second throughput")
		fmt.Println("  - Maintained semantic quality (current baseline)")
		fmt.Println("  - Better GPU memory utilization")
	} else {
		fmt.Println("⚠️ Quality may need monitoring during CAGRA integration")
		fmt.Println("\n🔧 Recommended Quality Safeguards:")
		fmt.Println("  1. Implement quality regression testing")
		fmt.Println("  2. A/B test CAGRA vs current implementation")
		fmt.Println("  3. Monitor relevance scores in production")
		fmt.Println("  4. Consider hybrid approach for quality-critical queries")
	}

	fmt.Println("\n📈 Immediate Optimization Opportunities:")
	fmt.Println("  - GPU batch embedding generation")
	fmt.Println("  - Faster indexing with CAGRA graph construction")
	fmt.Println("  - Memory-mapped data loading for large datasets")
	fmt.Println("  - Streaming indexing for real-time updates")
}
