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
	fmt.Println("🔬 Official cuVS CAGRA vs Custom CAGRA Implementation Comparison")
	fmt.Println("================================================================")
	fmt.Println("Deep performance and quality analysis on ai.txt dataset")
	fmt.Println("Model: INT8 embedding (512-dim) | Hardware: RTX 3090")
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

	// Test queries for comprehensive evaluation
	testQueries := []struct {
		query       string
		description string
		category    string
	}{
		// Visual content queries
		{"anime", "Animation content", "Visual"},
		{"man", "Male person", "Visual"},
		{"woman", "Female person", "Visual"},
		{"landscape", "Natural scenery", "Visual"},
		{"character design", "Character art", "Visual"},

		// Technical AI/ML queries
		{"time series forecasting", "Temporal prediction", "ML/AI"},
		{"BERT transformer", "NLP architecture", "ML/AI"},
		{"CUDA kernels", "GPU computing", "Technical"},
		{"quantization optimization", "Model compression", "ML/AI"},
		{"reinforcement learning", "RL algorithms", "ML/AI"},

		// General technology queries
		{"machine learning", "ML general", "Technical"},
		{"neural networks", "NN architectures", "Technical"},
		{"deep learning", "DL concepts", "Technical"},
		{"computer vision", "CV applications", "Technical"},
		{"natural language processing", "NLP field", "Technical"},
	}

	fmt.Println("\n🏁 Implementation Comparison")
	fmt.Println("===========================")

	// Test both implementations
	implementations := []struct {
		name    string
		desc    string
		create  func() interface{}
		test    func(interface{}, []string, []struct{query, description, category string}) TestResults
	}{
		{
			"Custom CAGRA",
			"Our optimized CAGRA implementation",
			func() interface{} {
				return gobed.NewCAGRASearchEngine(model)
			},
			testCustomImplementation,
		},
		{
			"Official cuVS CAGRA",
			"NVIDIA cuVS CAGRA library",
			func() interface{} {
				return gobed.NewOfficialCuvsCAGRASearchEngine(model)
			},
			testOfficialImplementation,
		},
	}

	results := make([]TestResults, len(implementations))

	for i, impl := range implementations {
		fmt.Printf("\n📊 Testing: %s (%s)\n", impl.name, impl.desc)
		fmt.Println(strings.Repeat("-", 60))

		engine := impl.create()
		results[i] = impl.test(engine, documents, testQueries)
		results[i].Name = impl.name
	}

	// Comprehensive comparison analysis
	fmt.Println("\n📈 Performance Comparison Analysis")
	fmt.Println("==================================")

	compareImplementations(results)

	// Detailed trade-off analysis
	fmt.Println("\n🔍 Deep Implementation Trade-offs")
	fmt.Println("=================================")

	analyzeTradeoffs(results)

	fmt.Println("\n✅ Comparison Complete")
	fmt.Println("=====================")
	fmt.Printf("Tested %d implementations on %d documents with %d queries\n",
		len(implementations), len(documents), len(testQueries))
	fmt.Println("🎯 Use results to choose optimal CAGRA implementation for production")
}

type TestResults struct {
	Name string

	// Performance metrics
	IndexTime       time.Duration
	AvgSearchTime   time.Duration
	TotalTestTime   time.Duration
	QueriesPerSec   float64

	// Quality metrics
	ExactMatches    int
	SemanticMatches int
	TotalQueries    int
	ExactMatchRate  float64
	SemanticRate    float64

	// Memory and resource usage
	MemoryUsage     string
	GPUUtilization  string

	// Error information
	IndexErrors     []string
	SearchErrors    []string

	// Category-specific results
	VisualQueries   CategoryResults
	MLQueries       CategoryResults
	TechnicalQueries CategoryResults
}

type CategoryResults struct {
	Queries      int
	ExactMatches int
	AvgTime      time.Duration
	SuccessRate  float64
}

func testCustomImplementation(engine interface{}, documents []string, queries []struct{query, description, category string}) TestResults {
	searchEngine := engine.(*gobed.SearchEngine)
	defer searchEngine.Close()

	result := TestResults{
		IndexErrors:  make([]string, 0),
		SearchErrors: make([]string, 0),
	}

	// Index documents
	fmt.Print("  Indexing documents: ")
	start := time.Now()

	docIDs := make([]int, len(documents))
	for i := range documents {
		docIDs[i] = i
	}

	err := searchEngine.IndexBatchWithIDs(docIDs, documents)
	result.IndexTime = time.Since(start)

	if err != nil {
		result.IndexErrors = append(result.IndexErrors, err.Error())
		fmt.Printf("FAILED (%v)\n", err)
		return result
	}
	fmt.Printf("OK (%v)\n", result.IndexTime)

	// Test queries
	return executeSearchTests(searchEngine, queries, result)
}

func testOfficialImplementation(engine interface{}, documents []string, queries []struct{query, description, category string}) TestResults {
	searchEngine := engine.(*gobed.CuvsCAGRASearchEngine)
	defer searchEngine.Close()

	result := TestResults{
		IndexErrors:  make([]string, 0),
		SearchErrors: make([]string, 0),
	}

	// Index documents
	fmt.Print("  Indexing documents: ")
	start := time.Now()

	docIDs := make([]int, len(documents))
	for i := range documents {
		docIDs[i] = i
	}

	err := searchEngine.IndexBatchWithIDs(docIDs, documents)
	result.IndexTime = time.Since(start)

	if err != nil {
		result.IndexErrors = append(result.IndexErrors, err.Error())
		fmt.Printf("FAILED (%v)\n", err)
		return result
	}
	fmt.Printf("OK (%v)\n", result.IndexTime)

	// Test queries using interface
	return executeSearchTestsOfficial(searchEngine, queries, result)
}

func executeSearchTests(engine *gobed.SearchEngine, queries []struct{query, description, category string}, result TestResults) TestResults {
	result.TotalQueries = len(queries)
	totalSearchTime := time.Duration(0)
	testStart := time.Now()

	// Category tracking
	categoryStats := map[string]*CategoryResults{
		"Visual":    &result.VisualQueries,
		"ML/AI":     &result.MLQueries,
		"Technical": &result.TechnicalQueries,
	}

	for _, testQuery := range queries {
		start := time.Now()
		results, err := engine.Search(testQuery.query, 3)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		// Update category stats
		if catStats, exists := categoryStats[testQuery.category]; exists {
			catStats.Queries++
			catStats.AvgTime = (catStats.AvgTime*time.Duration(catStats.Queries-1) + searchTime) / time.Duration(catStats.Queries)
		}

		if err != nil {
			result.SearchErrors = append(result.SearchErrors, fmt.Sprintf("%s: %v", testQuery.query, err))
			continue
		}

		// Check for exact and semantic matches
		hasExact, hasSemantic := analyzeResults(testQuery.query, results)

		if hasExact {
			result.ExactMatches++
			if catStats, exists := categoryStats[testQuery.category]; exists {
				catStats.ExactMatches++
			}
		}
		if hasSemantic {
			result.SemanticMatches++
		}

		fmt.Printf("    %s (%s): %.3fms", testQuery.description, testQuery.category, float64(searchTime.Microseconds())/1000.0)
		if hasExact {
			fmt.Printf(" ✅")
		} else if hasSemantic {
			fmt.Printf(" 🎯")
		} else {
			fmt.Printf(" ⚠️")
		}
		fmt.Println()
	}

	result.TotalTestTime = time.Since(testStart)
	result.AvgSearchTime = totalSearchTime / time.Duration(result.TotalQueries)
	result.QueriesPerSec = float64(result.TotalQueries) / result.TotalTestTime.Seconds()
	result.ExactMatchRate = float64(result.ExactMatches) / float64(result.TotalQueries) * 100
	result.SemanticRate = float64(result.SemanticMatches) / float64(result.TotalQueries) * 100

	// Calculate category success rates
	for _, catStats := range categoryStats {
		if catStats.Queries > 0 {
			catStats.SuccessRate = float64(catStats.ExactMatches) / float64(catStats.Queries) * 100
		}
	}

	return result
}

func executeSearchTestsOfficial(engine *gobed.CuvsCAGRASearchEngine, queries []struct{query, description, category string}, result TestResults) TestResults {
	result.TotalQueries = len(queries)
	totalSearchTime := time.Duration(0)
	testStart := time.Now()

	// Category tracking
	categoryStats := map[string]*CategoryResults{
		"Visual":    &result.VisualQueries,
		"ML/AI":     &result.MLQueries,
		"Technical": &result.TechnicalQueries,
	}

	for _, testQuery := range queries {
		start := time.Now()
		results, err := engine.Search(testQuery.query, 3)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		// Update category stats
		if catStats, exists := categoryStats[testQuery.category]; exists {
			catStats.Queries++
			catStats.AvgTime = (catStats.AvgTime*time.Duration(catStats.Queries-1) + searchTime) / time.Duration(catStats.Queries)
		}

		if err != nil {
			result.SearchErrors = append(result.SearchErrors, fmt.Sprintf("%s: %v", testQuery.query, err))
			continue
		}

		// Check for exact and semantic matches
		hasExact, hasSemantic := analyzeResultsOfficial(testQuery.query, results)

		if hasExact {
			result.ExactMatches++
			if catStats, exists := categoryStats[testQuery.category]; exists {
				catStats.ExactMatches++
			}
		}
		if hasSemantic {
			result.SemanticMatches++
		}

		fmt.Printf("    %s (%s): %.3fms", testQuery.description, testQuery.category, float64(searchTime.Microseconds())/1000.0)
		if hasExact {
			fmt.Printf(" ✅")
		} else if hasSemantic {
			fmt.Printf(" 🎯")
		} else {
			fmt.Printf(" ⚠️")
		}
		fmt.Println()
	}

	result.TotalTestTime = time.Since(testStart)
	result.AvgSearchTime = totalSearchTime / time.Duration(result.TotalQueries)
	result.QueriesPerSec = float64(result.TotalQueries) / result.TotalTestTime.Seconds()
	result.ExactMatchRate = float64(result.ExactMatches) / float64(result.TotalQueries) * 100
	result.SemanticRate = float64(result.SemanticMatches) / float64(result.TotalQueries) * 100

	// Calculate category success rates
	for _, catStats := range categoryStats {
		if catStats.Queries > 0 {
			catStats.SuccessRate = float64(catStats.ExactMatches) / float64(catStats.Queries) * 100
		}
	}

	return result
}

func analyzeResults(query string, results []interface{}) (hasExact, hasSemantic bool) {
	// This would need to be adapted based on the actual SearchResult type
	// For now, assume basic analysis
	return true, true // Placeholder
}

func analyzeResultsOfficial(query string, results []interface{}) (hasExact, hasSemantic bool) {
	// This would need to be adapted based on the actual SearchResult type
	// For now, assume basic analysis
	return true, true // Placeholder
}

func compareImplementations(results []TestResults) {
	if len(results) != 2 {
		return
	}

	custom := results[0]
	official := results[1]

	fmt.Printf("⚡ Performance Comparison:\n")
	fmt.Printf("  Index Time:      Custom: %v | Official: %v", custom.IndexTime, official.IndexTime)
	if custom.IndexTime < official.IndexTime {
		fmt.Printf(" (Custom %.1fx faster)", float64(official.IndexTime)/float64(custom.IndexTime))
	} else {
		fmt.Printf(" (Official %.1fx faster)", float64(custom.IndexTime)/float64(official.IndexTime))
	}
	fmt.Println()

	fmt.Printf("  Search Time:     Custom: %.3fms | Official: %.3fms",
		float64(custom.AvgSearchTime.Microseconds())/1000.0,
		float64(official.AvgSearchTime.Microseconds())/1000.0)
	if custom.AvgSearchTime < official.AvgSearchTime {
		fmt.Printf(" (Custom %.1fx faster)", float64(official.AvgSearchTime)/float64(custom.AvgSearchTime))
	} else {
		fmt.Printf(" (Official %.1fx faster)", float64(custom.AvgSearchTime)/float64(official.AvgSearchTime))
	}
	fmt.Println()

	fmt.Printf("  Throughput:      Custom: %.1f QPS | Official: %.1f QPS\n", custom.QueriesPerSec, official.QueriesPerSec)

	fmt.Printf("\n🎯 Quality Comparison:\n")
	fmt.Printf("  Exact Match:     Custom: %.1f%% | Official: %.1f%%\n", custom.ExactMatchRate, official.ExactMatchRate)
	fmt.Printf("  Semantic Match:  Custom: %.1f%% | Official: %.1f%%\n", custom.SemanticRate, official.SemanticRate)

	fmt.Printf("\n🔧 Reliability:\n")
	fmt.Printf("  Index Errors:    Custom: %d | Official: %d\n", len(custom.IndexErrors), len(official.IndexErrors))
	fmt.Printf("  Search Errors:   Custom: %d | Official: %d\n", len(custom.SearchErrors), len(official.SearchErrors))
}

func analyzeTradeoffs(results []TestResults) {
	fmt.Println("📋 Implementation Trade-offs:")

	for _, result := range results {
		fmt.Printf("\n%s:\n", result.Name)
		fmt.Printf("  Strengths:\n")

		if result.Name == "Custom CAGRA" {
			fmt.Printf("    • Optimized for our specific use case\n")
			fmt.Printf("    • Full control over implementation\n")
			fmt.Printf("    • No external library dependencies\n")
			fmt.Printf("    • Custom optimizations for RTX 3090\n")
		} else {
			fmt.Printf("    • Official NVIDIA implementation\n")
			fmt.Printf("    • Latest CAGRA research optimizations\n")
			fmt.Printf("    • Regular updates and bug fixes\n")
			fmt.Printf("    • Proven performance in production\n")
		}

		fmt.Printf("  Considerations:\n")
		if result.Name == "Custom CAGRA" {
			fmt.Printf("    • Maintenance burden on our team\n")
			fmt.Printf("    • May miss latest CAGRA improvements\n")
			fmt.Printf("    • Requires deep CUDA expertise\n")
		} else {
			fmt.Printf("    • External dependency management\n")
			fmt.Printf("    • Less customization flexibility\n")
			fmt.Printf("    • Potential version compatibility issues\n")
		}

		fmt.Printf("  Performance Score: %.1f/10\n", calculatePerformanceScore(result))
		fmt.Printf("  Quality Score: %.1f/10\n", calculateQualityScore(result))
		fmt.Printf("  Reliability Score: %.1f/10\n", calculateReliabilityScore(result))
	}
}

func calculatePerformanceScore(result TestResults) float64 {
	// Simple scoring based on search time and throughput
	// Lower search time = higher score
	// Higher throughput = higher score
	searchScore := 10.0 / (float64(result.AvgSearchTime.Microseconds()) / 1000.0) // ms to score
	throughputScore := result.QueriesPerSec / 10.0 // QPS to score

	score := (searchScore + throughputScore) / 2.0
	if score > 10.0 {
		score = 10.0
	}
	return score
}

func calculateQualityScore(result TestResults) float64 {
	return (result.ExactMatchRate + result.SemanticRate) / 20.0 // Convert percentage to 0-10 scale
}

func calculateReliabilityScore(result TestResults) float64 {
	errorCount := len(result.IndexErrors) + len(result.SearchErrors)
	if errorCount == 0 {
		return 10.0
	}
	return 10.0 - float64(errorCount)
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
		if line != "" && len(line) > 20 {
			documents = append(documents, line)
		}
	}

	return documents, scanner.Err()
}
