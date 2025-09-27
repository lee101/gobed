package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 GPU Performance Test - Custom CUDA + Int8 Model")
	fmt.Println("==================================================")

	// Load the embedding model
	fmt.Println("📦 Loading EmbeddingModel...")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load EmbeddingModel: %v", err)
	}
	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded in %.2fms\n", float64(loadTime.Microseconds())/1000.0)

	// Create GPU search engine
	fmt.Println("🏗  Creating GPU search engine...")
	start = time.Now()
	engine := gobed.NewGPUSearchEngine(model)
	defer engine.Close()
	engineTime := time.Since(start)
	fmt.Printf("✅ GPU engine created in %.2fms\n", float64(engineTime.Microseconds())/1000.0)

	// Load and index ai.txt data
	fmt.Println("📁 Loading ai.txt data...")
	documents, err := loadAITextData()
	if err != nil {
		log.Fatalf("Failed to load ai.txt: %v", err)
	}
	fmt.Printf("📄 Loaded %d documents\n", len(documents))

	// Index documents
	fmt.Println("🔍 Indexing documents on GPU...")
	start = time.Now()
	ids := make([]int, len(documents))
	for i := range documents {
		ids[i] = i
	}

	err = engine.IndexBatchWithIDs(ids, documents)
	if err != nil {
		log.Fatalf("Failed to index documents: %v", err)
	}
	indexTime := time.Since(start)
	fmt.Printf("✅ Indexed %d documents in %.2fs (%.1f docs/sec)\n",
		len(documents), indexTime.Seconds(), float64(len(documents))/indexTime.Seconds())

	// Test queries focusing on AI/ML terms
	testQueries := []string{
		"machine learning",
		"neural networks",
		"deep learning",
		"transformer models",
		"reinforcement learning",
		"computer vision",
		"natural language processing",
		"CUDA optimization",
		"quantization techniques",
		"time series forecasting",
	}

	fmt.Println("\n🎯 Running GPU Search Performance Tests")
	fmt.Println("========================================")
	fmt.Printf("%-25s %10s %8s %12s\n", "Query", "Time(ms)", "Results", "QPS")
	fmt.Println(strings.Repeat("-", 60))

	var totalTime time.Duration
	var minTime = time.Hour
	var maxTime time.Duration
	var successCount int

	for _, query := range testQueries {
		start := time.Now()
		results, err := engine.Search(query, 10)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("%-25s %10s %8s %12s\n", query, "ERROR", "-", "-")
			log.Printf("Search failed for '%s': %v", query, err)
			continue
		}

		// Calculate QPS (queries per second)
		qps := 1000.0 / (float64(elapsed.Microseconds()) / 1000.0)

		fmt.Printf("%-25s %10.2f %8d %12.0f\n",
			query,
			float64(elapsed.Microseconds())/1000.0,
			len(results),
			qps)

		// Show top 3 results for first query as example
		if successCount == 0 && len(results) > 0 {
			fmt.Printf("\n📋 Sample results for '%s':\n", query)
			for i, result := range results {
				if i >= 3 { break }
				docText := documents[result.ID]
				if len(docText) > 80 {
					docText = docText[:80] + "..."
				}
				fmt.Printf("  %d. [Score: %.3f] %s\n", i+1, result.Similarity, docText)
			}
			fmt.Println()
		}

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
		avgQPS := 1000.0 / (float64(avgTime.Microseconds()) / 1000.0)

		fmt.Println(strings.Repeat("-", 60))
		fmt.Println("\n📊 Performance Summary")
		fmt.Println("======================")
		fmt.Printf("Average:     %.2fms (%.0f QPS)\n", float64(avgTime.Microseconds())/1000.0, avgQPS)
		fmt.Printf("Fastest:     %.2fms\n", float64(minTime.Microseconds())/1000.0)
		fmt.Printf("Slowest:     %.2fms\n", float64(maxTime.Microseconds())/1000.0)
		fmt.Printf("Successful:  %d/%d queries\n", successCount, len(testQueries))

		// Performance targets
		fmt.Println("\n🎯 Performance Analysis")
		fmt.Println("=======================")
		targetTime := 100.0 // Target <100ms
		avgTimeMs := float64(avgTime.Microseconds()) / 1000.0

		if avgTimeMs < targetTime {
			improvement := targetTime / avgTimeMs
			fmt.Printf("✅ EXCELLENT: %.1fx faster than 100ms target!\n", improvement)
		} else {
			slowdown := avgTimeMs / targetTime
			fmt.Printf("⚠️  NEEDS OPTIMIZATION: %.1fx slower than 100ms target\n", slowdown)
		}

		fmt.Printf("🏆 Target QPS (100ms): %.0f\n", 1000.0/targetTime)
		fmt.Printf("📈 Actual QPS:        %.0f\n", avgQPS)

		// Memory usage (if available)
		fmt.Println("\n💾 System Info")
		fmt.Println("===============")
		fmt.Printf("📄 Documents indexed: %d\n", len(documents))
		fmt.Printf("🧠 Model: EmbeddingModel (GPU-optimized)\n")
		fmt.Printf("⚡ GPU: Custom CUDA implementation\n")
	}

	fmt.Println("\n🏁 GPU Performance Test Complete!")
}

func loadAITextData() ([]string, error) {
	dataPath := filepath.Join("..", "..", "ai.txt")
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open ai.txt: %v", err)
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

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading ai.txt: %v", err)
	}

	return documents, nil
}