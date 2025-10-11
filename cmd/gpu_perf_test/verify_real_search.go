//go:build legacy

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
	fmt.Println("🔍 Real Search Verification Test")
	fmt.Println("================================")
	fmt.Println("This test verifies that search is actually working with real embeddings")
	fmt.Println("and finding semantically similar content with varying scores.")
	fmt.Println()

	// Load model
	fmt.Println("📦 Loading model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create search engine
	engine := gobed.NewGPUSearchEngine(model)
	defer engine.Close()

	// Test with small sample files first (easier to verify)
	fmt.Println("📄 Loading sample test documents...")
	docs := loadSampleDocuments()

	fmt.Printf("Loaded %d sample documents:\n", len(docs))
	for i, doc := range docs {
		preview := doc
		if len(preview) > 100 {
			preview = preview[:100] + "..."
		}
		fmt.Printf("  Doc %d: %s\n", i+1, preview)
	}
	fmt.Println()

	// Index the documents
	fmt.Println("🔍 Indexing documents...")
	ids := make([]int, len(docs))
	for i := range docs {
		ids[i] = i
	}

	err = engine.IndexBatchWithIDs(ids, docs)
	if err != nil {
		log.Fatalf("Failed to index: %v", err)
	}

	fmt.Println("✅ Documents indexed successfully")
	fmt.Println()

	// Test queries based on actual ai.txt content
	testCases := []struct {
		query    string
		expected string // What we expect to find
	}{
		{"state-of-the-art", "should find many docs with state-of-the-art phrases"},
		{"transfer learning", "should find transfer learning article"},
		{"AutoML neural architecture", "should find AutoML article"},
		{"support vector machines", "should find SVM article"},
		{"object detection", "should find object detection article"},
		{"computer vision", "should find computer vision article"},
		{"machine learning algorithms", "should find ML algorithms article"},
	}

	fmt.Println("🎯 Testing search with verification...")
	fmt.Println("=====================================")

	allScoresIdentical := true
	var firstScore float32 = -1

	for i, tc := range testCases {
		fmt.Printf("\n%d. Query: '%s'\n", i+1, tc.query)
		fmt.Printf("   Expected: %s\n", tc.expected)

		start := time.Now()
		results, err := engine.Search(tc.query, 3)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("   ❌ ERROR: %v\n", err)
			continue
		}

		fmt.Printf("   ⏱️  Search time: %.2fms\n", float64(elapsed.Microseconds())/1000.0)
		fmt.Printf("   📊 Results (%d found):\n", len(results))

		for j, result := range results {
			docPreview := docs[result.ID]
			if len(docPreview) > 80 {
				docPreview = docPreview[:80] + "..."
			}

			fmt.Printf("     %d. Score: %.6f - %s\n", j+1, result.Similarity, docPreview)

			// Check for identical scores (suspicious)
			if firstScore == -1 {
				firstScore = result.Similarity
			} else if result.Similarity != firstScore {
				allScoresIdentical = false
			}
		}

		// Verify the search makes sense
		if len(results) > 0 {
			topResult := docs[results[0].ID]
			if strings.Contains(strings.ToLower(topResult), strings.ToLower(tc.query)) ||
			   containsRelatedTerms(strings.ToLower(topResult), strings.ToLower(tc.query)) {
				fmt.Printf("   ✅ GOOD: Found semantically relevant document\n")
			} else {
				fmt.Printf("   ⚠️  WARNING: Top result may not be semantically relevant\n")
			}
		}
	}

	// Final verification
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("🔬 VERIFICATION RESULTS")
	fmt.Println(strings.Repeat("=", 50))

	if allScoresIdentical {
		fmt.Println("❌ CRITICAL ISSUE: All similarity scores are identical!")
		fmt.Printf("   All scores: %.6f\n", firstScore)
		fmt.Println("   This suggests embeddings are not working properly.")
		fmt.Println("   Expected: Diverse scores based on semantic similarity")
	} else {
		fmt.Println("✅ GOOD: Found diverse similarity scores")
		fmt.Println("   This suggests embeddings are working correctly")
	}

	fmt.Println("\n🏁 Verification Complete")
}

func loadSampleDocuments() []string {
	// Try to load actual content from ai.txt first
	if aiContent := loadFromAIFile(); len(aiContent) > 0 {
		return aiContent
	}

	// Load the sample files from testdata as fallback
	sampleFiles := []string{
		"testdata/sample1.txt",
		"testdata/sample2.txt",
		"testdata/sample3.txt",
		"testdata/sample4.txt",
		"testdata/sample5.txt",
	}

	var docs []string

	for _, filename := range sampleFiles {
		if content := readFileContent(filename); content != "" {
			docs = append(docs, content)
		}
	}

	// If sample files not found, create some test docs
	if len(docs) == 0 {
		docs = []string{
			"Machine learning is a method of data analysis that automates analytical model building using artificial intelligence.",
			"Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
			"Computer vision enables machines to interpret and understand visual information from the world around them.",
			"Natural language processing allows computers to understand, interpret and generate human language in a valuable way.",
			"Reinforcement learning is an area where agents learn to make decisions by performing actions in an environment.",
		}
	}

	return docs
}

func loadFromAIFile() []string {
	aiPaths := []string{"../../ai.txt", "../../../ai.txt", "ai.txt"}

	for _, path := range aiPaths {
		if file, err := os.Open(path); err == nil {
			defer file.Close()

			var docs []string
			scanner := bufio.NewScanner(file)
			count := 0

			for scanner.Scan() && count < 20 {
				line := strings.TrimSpace(scanner.Text())
				if line != "" {
					docs = append(docs, line)
					count++
				}
			}

			if len(docs) > 0 {
				return docs
			}
		}
	}

	return nil
}

func readFileContent(filename string) string {
	file, err := os.Open(filename)
	if err != nil {
		return ""
	}
	defer file.Close()

	var content strings.Builder
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		content.WriteString(scanner.Text())
		content.WriteString(" ")
	}

	return strings.TrimSpace(content.String())
}

func containsRelatedTerms(text, query string) bool {
	// Simple semantic relatedness check
	mlTerms := []string{"machine", "learning", "model", "data", "algorithm", "artificial", "intelligence"}
	dlTerms := []string{"deep", "neural", "network", "layer", "pattern"}
	cvTerms := []string{"computer", "vision", "visual", "image", "recognition"}
	nlpTerms := []string{"language", "processing", "text", "understand", "interpret"}

	queryLower := strings.ToLower(query)
	textLower := strings.ToLower(text)

	var relevantTerms []string
	if strings.Contains(queryLower, "machine") || strings.Contains(queryLower, "learning") {
		relevantTerms = mlTerms
	} else if strings.Contains(queryLower, "deep") || strings.Contains(queryLower, "neural") {
		relevantTerms = dlTerms
	} else if strings.Contains(queryLower, "computer") || strings.Contains(queryLower, "vision") {
		relevantTerms = cvTerms
	} else if strings.Contains(queryLower, "language") || strings.Contains(queryLower, "processing") {
		relevantTerms = nlpTerms
	}

	for _, term := range relevantTerms {
		if strings.Contains(textLower, term) {
			return true
		}
	}

	return false
}
