//go:build legacy

package src

import (
	"context"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

type benchQuery struct {
	text     string
	relevant []string
}

// TestGPUSearchIndex tests the GPU search functionality
func TestGPUSearchIndex(t *testing.T) {
	// Skip if CUDA not available
	if os.Getenv("CUDA_VISIBLE_DEVICES") == "-1" {
		t.Skip("CUDA not available")
	}

	config := DefaultSearchConfig()
	config.ChunkSize = 128
	config.IVFClusters = 16
	config.MaxVectors = 10000

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Fatalf("Failed to create GPU index: %v", err)
	}
	defer index.Close()

	// Create test directory
	tempDir, err := ioutil.TempDir("", "gpu_search_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create test files
	testFiles := map[string]string{
		"code.go": `package main

import "fmt"

func calculateSimilarity(vec1, vec2 []float32) float32 {
    // Calculate cosine similarity between vectors
    var dotProduct, norm1, norm2 float32
    for i := range vec1 {
        dotProduct += vec1[i] * vec2[i]
        norm1 += vec1[i] * vec1[i]
        norm2 += vec2[i] * vec2[i]
    }
    return dotProduct / (sqrt(norm1) * sqrt(norm2))
}

func searchDatabase(query []float32, database [][]float32, k int) []int {
    // Search for similar vectors in database
    scores := make([]float32, len(database))
    for i, vec := range database {
        scores[i] = calculateSimilarity(query, vec)
    }
    return topK(scores, k)
}`,

		"readme.md": `# GPU-Accelerated Search System

This system provides fast similarity search using GPU acceleration.

## Features
- Hierarchical IVF indexing for large-scale search
- Int8 quantization for memory efficiency
- Batch processing for high throughput
- File chunking for handling large documents

## Performance
- 100,000+ queries per second on RTX 3090
- Sub-millisecond latency for small batches
- Linear scaling with multiple GPUs`,

		"data.txt": `The quick brown fox jumps over the lazy dog.
Machine learning models can learn patterns from data.
Deep neural networks have revolutionized AI.
GPU acceleration enables faster computation.
Vector similarity search is fundamental to many applications.
Embeddings capture semantic meaning in numerical form.`,
	}

	// Write test files
	for filename, content := range testFiles {
		filepath := filepath.Join(tempDir, filename)
		if err := ioutil.WriteFile(filepath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	// Test indexing
	ctx := context.Background()
	if err := index.IndexDirectory(ctx, tempDir, nil); err != nil {
		t.Fatalf("Failed to index directory: %v", err)
	}

	// Verify chunks were created
	if len(index.chunks) == 0 {
		t.Fatal("No chunks created")
	}

	// Verify file index
	if len(index.fileIndex) != len(testFiles) {
		t.Errorf("Expected %d files indexed, got %d", len(testFiles), len(index.fileIndex))
	}

	// Test search
	testCases := []struct {
		query         string
		expectedFiles []string
	}{
		{
			query:         "calculate similarity vectors",
			expectedFiles: []string{"code.go"},
		},
		{
			query:         "GPU acceleration performance",
			expectedFiles: []string{"readme.md"},
		},
		{
			query:         "quick brown fox",
			expectedFiles: []string{"data.txt"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.query, func(t *testing.T) {
			results, err := index.Search(tc.query, 3)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			if len(results) == 0 {
				t.Fatal("No results returned")
			}

			// Check if expected file is in top result
			topResult := results[0]
			found := false
			for _, expected := range tc.expectedFiles {
				if strings.Contains(topResult.FilePath, expected) {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("Expected one of %v in top result, got %s",
					tc.expectedFiles, topResult.FilePath)
			}

			// Verify scores are sorted
			for i := 1; i < len(results); i++ {
				if results[i].Score > results[i-1].Score {
					t.Error("Results not sorted by score")
				}
			}
		})
	}
}

// TestChunking tests file chunking logic
func TestChunking(t *testing.T) {
	config := DefaultSearchConfig()
	config.ChunkSize = 50
	config.ChunkOverlap = 10

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Create test file
	tempFile, err := ioutil.TempFile("", "chunk_test.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tempFile.Name())

	// Write test content
	lines := make([]string, 100)
	for i := range lines {
		lines[i] = fmt.Sprintf("Line %d: %s", i, strings.Repeat("word ", 10))
	}
	content := strings.Join(lines, "\n")
	if _, err := tempFile.WriteString(content); err != nil {
		t.Fatal(err)
	}
	tempFile.Close()

	// Test chunking
	chunks, err := index.chunkFile(tempFile.Name())
	if err != nil {
		t.Fatalf("Failed to chunk file: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("No chunks created")
	}

	// Verify chunk properties
	for i, chunk := range chunks {
		if chunk.FilePath != tempFile.Name() {
			t.Errorf("Chunk %d has wrong file path", i)
		}

		if chunk.LineEnd < chunk.LineStart {
			t.Errorf("Chunk %d has invalid line range: %d-%d",
				i, chunk.LineStart, chunk.LineEnd)
		}

		if len(chunk.Text) == 0 {
			t.Errorf("Chunk %d has empty text", i)
		}

		if chunk.ID == 0 {
			t.Errorf("Chunk %d has no ID", i)
		}
	}

	// Check for overlap
	if len(chunks) > 1 {
		for i := 0; i < len(chunks)-1; i++ {
			// There should be some overlap or small gap
			gap := chunks[i+1].LineStart - chunks[i].LineEnd
			if gap > 5 {
				t.Errorf("Large gap between chunks %d and %d: %d lines",
					i, i+1, gap)
			}
		}
	}
}

// TestBatchSearch tests batch search functionality
func TestBatchSearch(t *testing.T) {
	config := DefaultSearchConfig()
	config.MaxVectors = 10000

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Create test directory
	tempDir, err := ioutil.TempDir("", "batch_search_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create test files
	for i := 0; i < 10; i++ {
		filename := fmt.Sprintf("file%d.txt", i)
		content := fmt.Sprintf("Document %d about topic %d and subtopic %d",
			i, i%3, i%5)
		filepath := filepath.Join(tempDir, filename)
		if err := ioutil.WriteFile(filepath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	// Index
	ctx := context.Background()
	if err := index.IndexDirectory(ctx, tempDir, []string{".txt"}); err != nil {
		t.Fatalf("Failed to index: %v", err)
	}

	// Test batch search
	queries := []string{
		"document about topic 0",
		"document about topic 1",
		"document about topic 2",
		"subtopic 3",
		"subtopic 4",
	}

	results, err := index.BatchSearch(queries, 3)
	if err != nil {
		t.Fatalf("Batch search failed: %v", err)
	}

	if len(results) != len(queries) {
		t.Errorf("Expected %d result sets, got %d", len(queries), len(results))
	}

	for i, queryResults := range results {
		if len(queryResults) == 0 {
			t.Errorf("No results for query %d: %s", i, queries[i])
		}
	}
}

// TestSaveLoad tests index persistence
func TestSaveLoad(t *testing.T) {
	config := DefaultSearchConfig()
	config.MaxVectors = 1000

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Create test data
	tempDir, err := ioutil.TempDir("", "saveload_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create test files
	for i := 0; i < 5; i++ {
		filename := fmt.Sprintf("test%d.txt", i)
		content := fmt.Sprintf("Test content %d with some text", i)
		filepath := filepath.Join(tempDir, filename)
		if err := ioutil.WriteFile(filepath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	// Index
	ctx := context.Background()
	if err := index.IndexDirectory(ctx, tempDir, []string{".txt"}); err != nil {
		t.Fatalf("Failed to index: %v", err)
	}

	originalChunkCount := len(index.chunks)
	originalFileCount := len(index.fileIndex)

	// Save index
	indexPath := filepath.Join(tempDir, "test_index")
	if err := index.SaveIndex(indexPath); err != nil {
		t.Fatalf("Failed to save index: %v", err)
	}

	// Create new index and load
	newIndex, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Fatal(err)
	}
	defer newIndex.Close()

	if err := newIndex.LoadIndex(indexPath); err != nil {
		t.Fatalf("Failed to load index: %v", err)
	}

	// Verify loaded data
	if len(newIndex.chunks) != originalChunkCount {
		t.Errorf("Chunk count mismatch: expected %d, got %d",
			originalChunkCount, len(newIndex.chunks))
	}

	if len(newIndex.fileIndex) != originalFileCount {
		t.Errorf("File count mismatch: expected %d, got %d",
			originalFileCount, len(newIndex.fileIndex))
	}

	// Test search on loaded index
	results, err := newIndex.Search("test content", 3)
	if err != nil {
		t.Fatalf("Search on loaded index failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("No results from loaded index")
	}
}

// TestStats tests statistics collection
func TestStats(t *testing.T) {
	config := DefaultSearchConfig()

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Get initial stats
	stats := index.GetStats()

	// Verify required fields
	requiredFields := []string{
		"num_chunks",
		"num_files",
		"embedding_dim",
		"search_count",
		"memory_mb",
	}

	for _, field := range requiredFields {
		if _, ok := stats[field]; !ok {
			t.Errorf("Missing required stat field: %s", field)
		}
	}

	// Create and index test data
	tempDir, err := ioutil.TempDir("", "stats_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	testFile := filepath.Join(tempDir, "test.txt")
	if err := ioutil.WriteFile(testFile, []byte("test content"), 0644); err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	if err := index.IndexDirectory(ctx, tempDir, nil); err != nil {
		t.Fatal(err)
	}

	// Perform some searches
	for i := 0; i < 5; i++ {
		index.Search("test query", 5)
	}

	// Get updated stats
	newStats := index.GetStats()

	// Verify stats updated
	if newStats["num_chunks"].(int) == 0 {
		t.Error("No chunks in stats after indexing")
	}

	if newStats["search_count"].(int64) != 5 {
		t.Errorf("Expected 5 searches, got %d", newStats["search_count"])
	}
}

// BenchmarkIndexing benchmarks indexing performance
func BenchmarkIndexing(b *testing.B) {
	config := DefaultSearchConfig()
	config.MaxVectors = 100000

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		b.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Create test files
	tempDir, err := ioutil.TempDir("", "bench_index")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create corpus
	for i := 0; i < 100; i++ {
		filename := fmt.Sprintf("file%d.txt", i)
		content := fmt.Sprintf("File %d\n%s", i, strings.Repeat("test content line\n", 50))
		filepath := filepath.Join(tempDir, filename)
		if err := ioutil.WriteFile(filepath, []byte(content), 0644); err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		index.chunks = nil
		index.fileIndex = make(map[string][]uint64)
		index.chunkMap = make(map[uint64]*Chunk)

		ctx := context.Background()
		if err := index.IndexDirectory(ctx, tempDir, nil); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportMetric(float64(len(index.chunks)), "chunks")
	b.ReportMetric(float64(len(index.chunks))/b.Elapsed().Seconds(), "chunks/sec")
}

// BenchmarkSearch benchmarks search performance
func BenchmarkSearch(b *testing.B) {
	config := DefaultSearchConfig()
	config.MaxVectors = 100000

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		b.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Create and index test data
	tempDir, err := ioutil.TempDir("", "bench_search")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create corpus with unique topic tokens per document
	numDocs := 1000
	for i := 0; i < numDocs; i++ {
		filename := fmt.Sprintf("file%d.txt", i)
		topicTag := fmt.Sprintf("topic_%04d", i)
		content := fmt.Sprintf("Document %d about %s and various topics\n%sRelevant token: %s\n",
			i, topicTag, strings.Repeat("line with content\n", 20), topicTag)
		filepath := filepath.Join(tempDir, filename)
		if err := ioutil.WriteFile(filepath, []byte(content), 0644); err != nil {
			b.Fatal(err)
		}
	}

	ctx := context.Background()
	if err := index.IndexDirectory(ctx, tempDir, nil); err != nil {
		b.Fatal(err)
	}

	docIDs := []int{5, 42, 123, 256, 512, 789, 900, 999}
	benchQueries := make([]benchQuery, len(docIDs))
	for i, id := range docIDs {
		if id >= numDocs {
			b.Fatalf("docID %d exceeds corpus size %d", id, numDocs)
		}
		benchQueries[i] = benchQuery{
			text:     fmt.Sprintf("topic_%04d", id),
			relevant: []string{fmt.Sprintf("file%d.txt", id)},
		}
	}

	totalNDCG := 0.0

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		q := benchQueries[i%len(benchQueries)]
		results, err := index.Search(q.text, 10)
		if err != nil {
			b.Fatal(err)
		}
		totalNDCG += computeNDCGForBedResults(results, q.relevant, 10)
	}

	qps := float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(qps, "queries/sec")
	b.ReportMetric(b.Elapsed().Seconds()/float64(b.N)*1000, "ms/query")
	if b.N > 0 {
		b.ReportMetric(totalNDCG/float64(b.N), "ndcg@10")
	}

	stats := index.GetStats()
	if avg, ok := stats["avg_qps"]; ok {
		switch v := avg.(type) {
		case float32:
			b.ReportMetric(float64(v), "gpu_avg_qps")
		case float64:
			b.ReportMetric(v, "gpu_avg_qps")
		}
	}
	if peak, ok := stats["peak_qps"]; ok {
		switch v := peak.(type) {
		case float32:
			b.ReportMetric(float64(v), "gpu_peak_qps")
		case float64:
			b.ReportMetric(v, "gpu_peak_qps")
		}
	}
	if mem, ok := stats["memory_mb"]; ok {
		switch v := mem.(type) {
		case float32:
			b.ReportMetric(float64(v), "gpu_memory_mb")
		case float64:
			b.ReportMetric(v, "gpu_memory_mb")
		}
	}
}

// BenchmarkBatchSearch benchmarks batch search performance
func BenchmarkBatchSearch(b *testing.B) {
	config := DefaultSearchConfig()
	config.MaxVectors = 100000
	config.BatchSize = 1000

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		b.Skipf("GPU not available: %v", err)
	}
	defer index.Close()

	// Create and index test data
	tempDir, err := ioutil.TempDir("", "bench_batch")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	numDocs := 1000
	for i := 0; i < numDocs; i++ {
		filename := fmt.Sprintf("file%d.txt", i)
		topicTag := fmt.Sprintf("topic_%04d", i)
		content := fmt.Sprintf("Document %d about %s\n%sRelevant token: %s\n", i, topicTag, strings.Repeat("content\n", 20), topicTag)
		filepath := filepath.Join(tempDir, filename)
		if err := ioutil.WriteFile(filepath, []byte(content), 0644); err != nil {
			b.Fatal(err)
		}
	}

	ctx := context.Background()
	if err := index.IndexDirectory(ctx, tempDir, nil); err != nil {
		b.Fatal(err)
	}

	// Prepare batch queries
	batchSizes := []int{10, 100, 1000}

	docIDs := []int{5, 42, 123, 256, 512, 789, 900, 999}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			queries := make([]string, batchSize)
			queryMeta := make([]benchQuery, batchSize)
			for i := range queries {
				docID := docIDs[i%len(docIDs)]
				if docID >= numDocs {
					b.Fatalf("docID %d exceeds corpus size %d", docID, numDocs)
				}
				queryMeta[i] = benchQuery{
					text:     fmt.Sprintf("topic_%04d", docID),
					relevant: []string{fmt.Sprintf("file%d.txt", docID)},
				}
				queries[i] = queryMeta[i].text
			}

			totalNDCG := 0.0

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				results, err := index.BatchSearch(queries, 10)
				if err != nil {
					b.Fatal(err)
				}
				if len(results) != batchSize {
					b.Fatalf("expected %d batch results, got %d", batchSize, len(results))
				}
				for q := range results {
					totalNDCG += computeNDCGForBedResults(results[q], queryMeta[q].relevant, 10)
				}
			}

			totalQueries := b.N * batchSize
			qps := float64(totalQueries) / b.Elapsed().Seconds()
			b.ReportMetric(qps, "queries/sec")
			b.ReportMetric(float64(batchSize), "batch_size")
			if b.N > 0 && batchSize > 0 {
				b.ReportMetric(totalNDCG/float64(b.N*batchSize), "ndcg@10")
			}
		})
	}
}

func computeNDCGForBedResults(results []*GPUSearchResult, relevant []string, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}

	gradeMap := make(map[string]float64, len(relevant))
	for i, file := range relevant {
		gradeMap[file] = float64(len(relevant) - i)
	}

	actual := make([]float64, 0, min(k, len(results)))
	for i := 0; i < len(results) && i < k; i++ {
		base := filepath.Base(results[i].FilePath)
		actual = append(actual, gradeMap[base])
	}

	if len(actual) == 0 {
		return 0
	}

	ideal := make([]float64, 0, len(gradeMap))
	for _, grade := range gradeMap {
		if grade > 0 {
			ideal = append(ideal, grade)
		}
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(ideal)))
	if len(ideal) > len(actual) {
		ideal = ideal[:len(actual)]
	}

	return computeNDCGFromGrades(actual, ideal)
}

func computeNDCGFromGrades(actual, ideal []float64) float64 {
	if len(actual) == 0 || len(ideal) == 0 {
		return 0
	}

	dcg := discountedGainGrades(actual)
	idcg := discountedGainGrades(ideal)
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func discountedGainGrades(grades []float64) float64 {
	var sum float64
	for i, grade := range grades {
		if grade <= 0 {
			continue
		}
		sum += grade / math.Log2(float64(i)+2)
	}
	return sum
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
