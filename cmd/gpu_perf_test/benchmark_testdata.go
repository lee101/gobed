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
	fmt.Println("🏋️  Testdata Benchmark - GPU Performance")
	fmt.Println("==========================================")

	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	engine := gobed.NewGPUSearchEngine(model)
	defer engine.Close()

	// Test with different sized datasets
	testFiles := []struct {
		name     string
		path     string
		maxLines int
	}{
		{"Small", "../../testdata/small_test.txt", 1000},
		{"Medium", "../../testdata/medium_test.txt", 10000},
		{"Large", "../../testdata/large_test.txt", 50000},
		{"AI Full", "../../testdata/ai.txt", 100000},
	}

	queries := []string{
		"machine learning",
		"neural networks",
		"deep learning",
		"artificial intelligence",
		"computer vision",
	}

	for _, testFile := range testFiles {
		fmt.Printf("\n📊 Testing %s Dataset (%s)\n", testFile.name, testFile.path)
		fmt.Println(strings.Repeat("=", 50))

		docs, err := loadFile(testFile.path, testFile.maxLines)
		if err != nil {
			fmt.Printf("❌ Failed to load %s: %v\n", testFile.name, err)
			continue
		}

		fmt.Printf("📄 Loaded %d documents\n", len(docs))

		// Index documents
		fmt.Print("🔄 Indexing... ")
		start := time.Now()
		ids := make([]int, len(docs))
		for i := range docs {
			ids[i] = i
		}

		err = engine.IndexBatchWithIDs(ids, docs)
		if err != nil {
			fmt.Printf("❌ Failed to index: %v\n", err)
			continue
		}
		indexTime := time.Since(start)
		fmt.Printf("✅ Done in %.2fs (%.1f docs/sec)\n",
			indexTime.Seconds(), float64(len(docs))/indexTime.Seconds())

		// Run search benchmarks
		fmt.Printf("\n%-20s %10s %8s %10s\n", "Query", "Time(ms)", "Results", "QPS")
		fmt.Println(strings.Repeat("-", 50))

		var totalTime time.Duration
		successCount := 0

		for _, query := range queries {
			start := time.Now()
			results, err := engine.Search(query, 10)
			elapsed := time.Since(start)

			if err != nil {
				fmt.Printf("%-20s %10s %8s %10s\n", query, "ERROR", "-", "-")
				continue
			}

			qps := 1000.0 / (float64(elapsed.Microseconds()) / 1000.0)
			fmt.Printf("%-20s %10.2f %8d %10.0f\n",
				query,
				float64(elapsed.Microseconds())/1000.0,
				len(results),
				qps)

			totalTime += elapsed
			successCount++

			// Show first result for verification
			if len(results) > 0 && successCount == 1 {
				preview := docs[results[0].ID]
				if len(preview) > 60 {
					preview = preview[:60] + "..."
				}
				fmt.Printf("   Top result: [%.3f] %s\n", results[0].Similarity, preview)
			}
		}

		if successCount > 0 {
			avgTime := totalTime / time.Duration(successCount)
			avgQPS := 1000.0 / (float64(avgTime.Microseconds()) / 1000.0)

			fmt.Println(strings.Repeat("-", 50))
			fmt.Printf("📈 Average: %.2fms (%.0f QPS)\n",
				float64(avgTime.Microseconds())/1000.0, avgQPS)

			// Performance analysis
			targetMs := 100.0
			actualMs := float64(avgTime.Microseconds()) / 1000.0
			if actualMs < targetMs {
				fmt.Printf("✅ %.1fx faster than %0.0fms target\n", targetMs/actualMs, targetMs)
			} else {
				fmt.Printf("⚠️  %.1fx slower than %0.0fms target\n", actualMs/targetMs, targetMs)
			}
		}
	}

	fmt.Println("\n🏁 Benchmark Complete!")
}

func loadFile(filePath string, maxLines int) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var docs []string
	scanner := bufio.NewScanner(file)
	count := 0

	for scanner.Scan() && count < maxLines {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
			count++
		}
	}

	return docs, scanner.Err()
}