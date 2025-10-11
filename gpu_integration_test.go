package gobed

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"
)

// TestGPUEnvironment verifies GPU environment is properly configured
func TestGPUEnvironment(t *testing.T) {
	// Check environment variables
	useGPU := os.Getenv("USE_GPU")
	cudaEnabled := os.Getenv("CUDA_ENABLED")
	cudaPath := os.Getenv("CUDA_PATH")

	t.Logf("GPU Environment:")
	t.Logf("  USE_GPU: %s", useGPU)
	t.Logf("  CUDA_ENABLED: %s", cudaEnabled)
	t.Logf("  CUDA_PATH: %s", cudaPath)

	if useGPU != "true" {
		t.Skip("GPU not enabled (USE_GPU != true)")
	}

	if cudaPath == "" {
		t.Error("CUDA_PATH not set but GPU is enabled")
	}
}

// TestModelLoadingPerformance tests model loading with GPU
func TestModelLoadingPerformance(t *testing.T) {
	start := time.Now()
	model := loadModelOrSkip(t)
	loadTime := time.Since(start)

	t.Logf("Model loaded in %v", loadTime)

	// Basic validation
	if model == nil {
		t.Fatal("Model is nil")
	}

	t.Log("Model loaded successfully")
}

// TestSmallDatasetSearch tests search on small dataset
func TestSmallDatasetSearch(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Skipf("Model load failed: %v", err)
	}

	engine := NewSearchEngine(model)

	// Test documents
	docs := []string{
		"Machine learning transforms data into insights",
		"Deep learning mimics human neural networks",
		"Natural language processing understands text",
		"Computer vision analyzes images and videos",
		"Reinforcement learning optimizes decision making",
	}

	// Index documents
	for _, doc := range docs {
		_, err := engine.Index(doc)
		if err != nil {
			t.Fatalf("Failed to index document: %v", err)
		}
	}

	// Test queries
	queries := []string{
		"neural networks",
		"image processing",
		"machine learning",
	}

	for _, query := range queries {
		t.Run(query, func(t *testing.T) {
			start := time.Now()
			results, err := engine.Search(query, 3)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			searchTime := time.Since(start)

			t.Logf("Query: %q - Time: %v", query, searchTime)

			if len(results) == 0 {
				t.Errorf("No results for query: %q", query)
			}

			for i, r := range results {
				t.Logf("  [%d] Score: %.4f - %s", i+1, r.Similarity, r.Text)
			}

			// Check latency
			if searchTime > 5*time.Millisecond {
				t.Logf("Warning: Search took %v (target: <5ms)", searchTime)
			}
		})
	}
}

// TestBatchIndexing tests batch indexing performance
func TestBatchIndexing(t *testing.T) {
	model, err := LoadModel()
	if err != nil {
		t.Skipf("Model load failed: %v", err)
	}

	engine := NewSearchEngine(model)

	// Generate test documents
	numDocs := 1000
	docs := make([]string, numDocs)
	for i := 0; i < numDocs; i++ {
		docs[i] = fmt.Sprintf("Test document %d about AI and machine learning topics", i)
	}

	// Measure indexing performance
	start := time.Now()

	// Add in batches
	batchSize := 100
	for i := 0; i < len(docs); i += batchSize {
		end := i + batchSize
		if end > len(docs) {
			end = len(docs)
		}

		batch := docs[i:end]
		for _, doc := range batch {
			_, err := engine.Index(doc)
			if err != nil {
				t.Fatalf("Failed to index: %v", err)
			}
		}
	}

	indexTime := time.Since(start)
	docsPerSecond := float64(numDocs) / indexTime.Seconds()

	t.Logf("Indexed %d documents in %v", numDocs, indexTime)
	t.Logf("Throughput: %.0f docs/second", docsPerSecond)

	// Test search after indexing
	results, err := engine.Search("machine learning", 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	t.Logf("Found %d results for 'machine learning'", len(results))
}

// TestMemoryUsage monitors memory during operations
func TestMemoryUsage(t *testing.T) {
	var m runtime.MemStats

	// Baseline memory
	runtime.GC()
	runtime.ReadMemStats(&m)
	baselineMemory := m.Alloc

	model, err := LoadModel()
	if err != nil {
		t.Skipf("Model load failed: %v", err)
	}

	// Memory after model load
	runtime.ReadMemStats(&m)
	modelMemory := m.Alloc - baselineMemory

	engine := NewSearchEngine(model)

	// Add documents
	numDocs := 5000
	for i := 0; i < numDocs; i++ {
		doc := fmt.Sprintf("Document %d with content about various AI topics", i)
		_, err := engine.Index(doc)
		if err != nil {
			t.Fatalf("Failed to index: %v", err)
		}
	}

	// Memory after indexing
	runtime.ReadMemStats(&m)
	totalMemory := m.Alloc - baselineMemory

	t.Logf("Memory usage:")
	t.Logf("  Model: %.2f MB", float64(modelMemory)/1024/1024)
	t.Logf("  Total with %d docs: %.2f MB", numDocs, float64(totalMemory)/1024/1024)
	t.Logf("  Per document: %.2f KB", float64(totalMemory-modelMemory)/float64(numDocs)/1024)
}

// BenchmarkSearch measures search performance
func BenchmarkSearch(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Skipf("Model load failed: %v", err)
	}

	engine := NewSearchEngine(model)

	// Add test documents (keep under 10k to avoid training requirement)
	numDocs := 5000
	for i := 0; i < numDocs; i++ {
		_, err := engine.Index(fmt.Sprintf("Document %d about neural networks and deep learning", i))
		if err != nil {
			b.Fatalf("Failed to index: %v", err)
		}
	}

	queries := []string{
		"neural networks",
		"deep learning",
		"machine learning",
		"artificial intelligence",
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		_, err := engine.Search(query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "QPS")
}

// BenchmarkIndexing measures indexing performance
func BenchmarkIndexing(b *testing.B) {
	model, err := LoadModel()
	if err != nil {
		b.Skipf("Model load failed: %v", err)
	}

	// Prepare documents
	docs := make([]string, b.N)
	for i := 0; i < b.N; i++ {
		docs[i] = fmt.Sprintf("Benchmark document %d with AI content", i)
	}

	b.ResetTimer()

	engine := NewSearchEngine(model)
	for i := 0; i < b.N; i++ {
		_, err := engine.Index(docs[i])
		if err != nil {
			b.Fatalf("Failed to index: %v", err)
		}
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "docs/sec")
}
