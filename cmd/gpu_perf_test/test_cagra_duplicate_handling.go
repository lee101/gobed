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
	fmt.Println("🔬 CAGRA Implementations Benchmark & Duplicate Handling Test")
	fmt.Println("============================================================")
	fmt.Println("Testing both Custom and Official CAGRA for quality and duplicates")
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

	// Create test dataset with intentional duplicates
	fmt.Println("\n🔍 Creating Test Dataset with Duplicates")
	fmt.Println("========================================")

	// Sample dataset sizes for testing
	testSizes := []int{100, 1000, 10000}

	for _, size := range testSizes {
		fmt.Printf("\n📊 Testing with %d documents (including duplicates)\n", size)

		// Create dataset with duplicates
		testDocs, duplicateInfo := createTestDatasetWithDuplicates(documents, size)

		fmt.Printf("  Dataset composition:\n")
		fmt.Printf("    Unique documents: %d\n", duplicateInfo.uniqueCount)
		fmt.Printf("    Duplicate documents: %d\n", duplicateInfo.duplicateCount)
		fmt.Printf("    Total documents: %d\n", len(testDocs))

		// Test both implementations
		implementations := []struct {
			name   string
			desc   string
			create func() interface{}
			isCustom bool
		}{
			{
				"Custom CAGRA",
				"Our optimized CAGRA implementation",
				func() interface{} {
					return gobed.NewCAGRASearchEngine(model)
				},
				true,
			},
			// Uncomment when cuVS is properly linked
			// {
			// 	"Official cuVS CAGRA",
			// 	"NVIDIA cuVS CAGRA library",
			// 	func() interface{} {
			// 		return gobed.NewOfficialCuvsCAGRASearchEngine(model)
			// 	},
			// 	false,
			// },
		}

		for _, impl := range implementations {
			fmt.Printf("\n  🚀 Testing: %s\n", impl.name)
			fmt.Println("  " + strings.Repeat("-", 40))

			if impl.isCustom {
				testCustomImplementation(impl.create().(*gobed.SearchEngine), testDocs, duplicateInfo)
			} else {
				// testOfficialImplementation(impl.create().(*gobed.CuvsCAGRASearchEngine), testDocs, duplicateInfo)
			}
		}
	}

	// Quality benchmark on real queries
	fmt.Println("\n📈 Quality Benchmark on Real Queries")
	fmt.Println("====================================")

	performQualityBenchmark(model, documents)

	fmt.Println("\n✅ Benchmark Complete")
	fmt.Println("====================")
}

type DuplicateInfo struct {
	uniqueCount    int
	duplicateCount int
	duplicateMap   map[string][]int // content -> list of indices
}

func createTestDatasetWithDuplicates(documents []string, targetSize int) ([]string, DuplicateInfo) {
	info := DuplicateInfo{
		duplicateMap: make(map[string][]int),
	}

	result := make([]string, 0, targetSize)

	// Use 70% unique documents, 30% duplicates
	uniqueTarget := int(float64(targetSize) * 0.7)
	duplicateTarget := targetSize - uniqueTarget

	// Add unique documents
	for i := 0; i < uniqueTarget && i < len(documents); i++ {
		result = append(result, documents[i])
		info.duplicateMap[documents[i]] = append(info.duplicateMap[documents[i]], len(result)-1)
	}
	info.uniqueCount = len(result)

	// Add duplicates
	for i := 0; i < duplicateTarget; i++ {
		// Pick a random document from the first part
		sourceIdx := i % uniqueTarget
		if sourceIdx < len(documents) {
			result = append(result, documents[sourceIdx])
			info.duplicateMap[documents[sourceIdx]] = append(info.duplicateMap[documents[sourceIdx]], len(result)-1)
		}
	}
	info.duplicateCount = len(result) - info.uniqueCount

	// Count actual duplicates
	actualDuplicates := 0
	for _, indices := range info.duplicateMap {
		if len(indices) > 1 {
			actualDuplicates += len(indices) - 1
		}
	}
	info.duplicateCount = actualDuplicates

	return result, info
}

func testCustomImplementation(engine *gobed.SearchEngine, documents []string, duplicateInfo DuplicateInfo) {
	defer engine.Close()

	// Index documents
	fmt.Print("    Indexing: ")
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

	// Test search performance
	testQueries := []string{
		"machine learning",
		"neural networks",
		"deep learning",
		"computer vision",
		"natural language processing",
	}

	fmt.Println("    Search performance:")
	totalSearchTime := time.Duration(0)
	duplicatesFound := 0

	for _, query := range testQueries {
		start := time.Now()
		results, err := engine.Search(query, 10)
		searchTime := time.Since(start)
		totalSearchTime += searchTime

		if err != nil {
			fmt.Printf("      %s: FAILED (%v)\n", query, err)
			continue
		}

		// Check for duplicates in results
		seenContent := make(map[string]bool)
		duplicatesInResult := 0
		for _, result := range results {
			if result.ID < len(documents) {
				content := documents[result.ID]
				if seenContent[content] {
					duplicatesInResult++
				}
				seenContent[content] = true
			}
		}

		duplicatesFound += duplicatesInResult
		fmt.Printf("      %s: %.3fms (duplicates in top-10: %d)\n",
			query, float64(searchTime.Microseconds())/1000.0, duplicatesInResult)
	}

	avgSearchTime := totalSearchTime / time.Duration(len(testQueries))
	fmt.Printf("    Average search time: %.3fms\n", float64(avgSearchTime.Microseconds())/1000.0)
	fmt.Printf("    Total duplicates in results: %d\n", duplicatesFound)

	// Test exact duplicate search
	fmt.Println("    Duplicate handling test:")
	testDuplicateHandling(engine, documents, duplicateInfo)
}

func testDuplicateHandling(engine *gobed.SearchEngine, documents []string, duplicateInfo DuplicateInfo) {
	// Find a document that has duplicates
	var testDoc string
	var duplicateIndices []int

	for doc, indices := range duplicateInfo.duplicateMap {
		if len(indices) > 1 {
			testDoc = doc
			duplicateIndices = indices
			break
		}
	}

	if testDoc == "" {
		fmt.Println("      No duplicates to test")
		return
	}

	// Search for the duplicate document
	results, err := engine.Search(testDoc, 10)
	if err != nil {
		fmt.Printf("      FAILED: %v\n", err)
		return
	}

	// Check if we find all duplicates
	foundIndices := make(map[int]bool)
	for _, result := range results {
		for _, dupIdx := range duplicateIndices {
			if result.ID == dupIdx {
				foundIndices[dupIdx] = true
			}
		}
	}

	fmt.Printf("      Found %d/%d duplicate instances\n", len(foundIndices), len(duplicateIndices))
	fmt.Printf("      Duplicate indices: %v\n", duplicateIndices)

	foundList := make([]int, 0, len(foundIndices))
	for idx := range foundIndices {
		foundList = append(foundList, idx)
	}
	fmt.Printf("      Found indices: %v\n", foundList)

	// Check similarity scores for duplicates
	fmt.Println("      Similarity scores for duplicates:")
	for i, result := range results {
		if i >= 5 { // Only show top 5
			break
		}
		isDuplicate := false
		for _, dupIdx := range duplicateIndices {
			if result.ID == dupIdx {
				isDuplicate = true
				break
			}
		}
		marker := ""
		if isDuplicate {
			marker = " [DUPLICATE]"
		}
		fmt.Printf("        %d. ID:%d Score:%.4f%s\n", i+1, result.ID, result.Similarity, marker)
	}
}

func performQualityBenchmark(model *gobed.EmbeddingModel, documents []string) {
	// Use a subset for quality testing
	testSize := 10000
	if len(documents) < testSize {
		testSize = len(documents)
	}

	testDocs := documents[:testSize]

	// Complex queries to test quality
	qualityQueries := []struct {
		query    string
		expected []string // Expected terms in good results
	}{
		{"anime character design", []string{"anime", "character", "design", "art"}},
		{"time series forecasting with LSTM", []string{"time", "series", "lstm", "rnn", "forecast"}},
		{"BERT transformer attention mechanism", []string{"bert", "transformer", "attention", "nlp"}},
		{"CUDA kernel optimization", []string{"cuda", "kernel", "gpu", "optimization"}},
		{"reinforcement learning policy gradient", []string{"reinforcement", "learning", "policy", "gradient"}},
		{"computer vision convolutional networks", []string{"vision", "convolutional", "cnn", "image"}},
		{"quantization for model compression", []string{"quantization", "compression", "optimization", "model"}},
		{"distributed training with data parallelism", []string{"distributed", "training", "parallel", "data"}},
		{"graph neural networks", []string{"graph", "neural", "network", "gnn"}},
		{"federated learning privacy", []string{"federated", "learning", "privacy", "distributed"}},
	}

	fmt.Println("\nTesting search quality on complex queries:")
	fmt.Println(strings.Repeat("-", 50))

	// Test with Custom CAGRA
	engine := gobed.NewCAGRASearchEngine(model)
	defer engine.Close()

	// Index documents
	fmt.Print("  Indexing test dataset: ")
	start := time.Now()
	docIDs := make([]int, len(testDocs))
	for i := range testDocs {
		docIDs[i] = i
	}

	err := engine.IndexBatchWithIDs(docIDs, testDocs)
	if err != nil {
		fmt.Printf("FAILED (%v)\n", err)
		return
	}
	fmt.Printf("OK (%v)\n", time.Since(start))

	totalQuality := 0.0
	for _, test := range qualityQueries {
		start := time.Now()
		results, err := engine.Search(test.query, 5)
		searchTime := time.Since(start)

		if err != nil {
			fmt.Printf("  %s: FAILED (%v)\n", test.query, err)
			continue
		}

		// Calculate quality score
		qualityScore := 0.0
		for _, result := range results {
			if result.ID < len(testDocs) {
				content := strings.ToLower(testDocs[result.ID])
				matchCount := 0
				for _, term := range test.expected {
					if strings.Contains(content, term) {
						matchCount++
					}
				}
				qualityScore += float64(matchCount) / float64(len(test.expected))
			}
		}
		qualityScore = qualityScore / float64(len(results)) * 100

		totalQuality += qualityScore
		fmt.Printf("  %s: %.3fms (Quality: %.1f%%)\n",
			test.query, float64(searchTime.Microseconds())/1000.0, qualityScore)
	}

	avgQuality := totalQuality / float64(len(qualityQueries))
	fmt.Printf("\nAverage quality score: %.1f%%\n", avgQuality)
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
