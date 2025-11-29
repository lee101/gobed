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
	fmt.Println("🎯 Real Quality Benchmark for ai.txt")
	fmt.Println("====================================")
	fmt.Println("Testing search quality with real embeddings on actual content")
	fmt.Println("Focus: Exact matching, semantic relevance, and benchmark scores")
	fmt.Println()

	// Load model
	fmt.Print("📦 Loading real model: ")
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
	engine, indexTime, err := indexDocumentsWithTiming(model, documents)
	if err != nil {
		log.Fatalf("Failed to index documents: %v", err)
	}
	defer engine.Close()
	fmt.Printf("OK (%v)\n", indexTime)

	// Run comprehensive quality tests
	fmt.Println("\n🎯 Quality Benchmark Tests")
	fmt.Println("==========================")

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
	fmt.Println("\n📊 Benchmark Summary")
	fmt.Println("===================")
	printBenchmarkSummary(allResults, totalSearchTime, modelLoadTime, indexTime)

	// Performance analysis
	fmt.Println("\n⚡ Performance Analysis")
	fmt.Println("======================")
	printPerformanceAnalysis(allResults, len(documents))

	// Quality recommendations
	fmt.Println("\n🔧 Quality Optimization Recommendations")
	fmt.Println("=======================================")
	printQualityRecommendations(allResults)
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

func indexDocumentsWithTiming(model *gobed.EmbeddingModel, documents []string) (*gobed.SearchEngine, time.Duration, error) {
	engine := gobed.NewGPUSearchEngine(model)

	// Create document IDs
	docIDs := make([]int, len(documents))
	for i := range documents {
		docIDs[i] = i
	}

	start := time.Now()
	err := engine.IndexBatchWithIDs(docIDs, documents)
	indexTime := time.Since(start)

	if err != nil {
		engine.Close()
		return nil, 0, err
	}

	return engine, indexTime, nil
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
	avgSimilarity /= 3.0 // Normalize by top 3

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
		"time series": {"forecasting", "RNN", "LSTM", "temporal"},
		"BERT":        {"transformer", "attention", "GPT", "language model"},
		"CUDA":        {"GPU", "acceleration", "parallel", "kernels"},
		"quantization":{"compression", "precision", "optimization", "int8"},
		"reinforcement":{"policy", "gradient", "reward", "agent"},
		"computer vision":{"CNN", "convolutional", "image", "detection"},
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
	fmt.Printf("Index Time: %v\n", indexTime)
	fmt.Printf("Average Search Time: %v\n", avgSearchTime)
	fmt.Printf("Total Queries: %d\n", len(results))
	fmt.Printf("High Quality Results: %d/%d (%.1f%%)\n",
		highQualityQueries, len(results), float64(highQualityQueries)/float64(len(results))*100)
	fmt.Printf("Average Relevance: %.1f%%\n", avgRelevance*100)

	// Overall score
	overallScore := avgRelevance * 100
	if overallScore >= 80 {
		fmt.Println("Overall Quality: ✅ Excellent")
	} else if overallScore >= 60 {
		fmt.Println("Overall Quality: ⚠️ Good")
	} else {
		fmt.Println("Overall Quality: ❌ Needs Improvement")
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
	fmt.Printf("Average Latency: %v\n", avgSearchTime)
	fmt.Printf("Queries Per Second: %.0f\n", qps)

	// Performance targets
	fmt.Printf("Performance vs Targets:\n")
	if avgSearchTime < 5*time.Millisecond {
		fmt.Printf("  Latency: ✅ Sub-5ms (current: %v)\n", avgSearchTime)
	} else {
		fmt.Printf("  Latency: ⚠️ >5ms (current: %v, target: <5ms)\n", avgSearchTime)
	}

	if qps > 100 {
		fmt.Printf("  Throughput: ✅ >100 QPS (current: %.0f)\n", qps)
	} else {
		fmt.Printf("  Throughput: ⚠️ <100 QPS (current: %.0f, target: >100)\n", qps)
	}
}

func printQualityRecommendations(results []QualityBenchmarkResult) {
	lowQualityQueries := 0
	for _, r := range results {
		if r.Relevance < 0.6 {
			lowQualityQueries++
		}
	}

	if lowQualityQueries == 0 {
		fmt.Println("✅ All queries achieve good quality!")
		fmt.Println("📈 Recommendations:")
		fmt.Println("  - Ready for production deployment")
		fmt.Println("  - Consider integrating CAGRA for 10-50x speedup")
		fmt.Println("  - Monitor quality with larger datasets")
	} else {
		fmt.Printf("⚠️ %d queries have quality issues\n", lowQualityQueries)
		fmt.Println("🔧 Recommendations:")
		fmt.Println("  - Increase embedding dimensions")
		fmt.Println("  - Use better preprocessing/tokenization")
		fmt.Println("  - Consider fine-tuning on domain data")
		fmt.Println("  - Implement query expansion")
	}

	fmt.Println("\n🚀 Next Steps for CAGRA Integration:")
	fmt.Println("  1. Replace IVF with CAGRA as default")
	fmt.Println("  2. Optimize GPU indexing pipeline")
	fmt.Println("  3. Implement batch search for throughput")
	fmt.Println("  4. Add quality monitoring in production")
}
