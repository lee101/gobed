package gpu

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// MockEmbedder implements EmbedderInterface for testing
type MockEmbedder struct {
	dim int
}

func NewMockEmbedder(dim int) *MockEmbedder {
	return &MockEmbedder{dim: dim}
}

func (m *MockEmbedder) Encode(text string) ([]float32, error) {
	// Generate deterministic embeddings based on text hash
	hash := 0
	for _, c := range text {
		hash = hash*31 + int(c)
	}
	
	rand.Seed(int64(hash))
	embedding := make([]float32, m.dim)
	for i := range embedding {
		embedding[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}
	
	return embedding, nil
}

func TestTorchInfo(t *testing.T) {
	t.Log(" Testing LibTorch info...")
	
	version, cudaAvailable, deviceCount := GetTorchInfo()
	
	t.Logf("LibTorch version: %s", version)
	t.Logf("CUDA available: %v", cudaAvailable)
	t.Logf("Device count: %d", deviceCount)
	
	if !cudaAvailable {
		t.Skip("CUDA not available, skipping GPU tests")
	}
	
	if deviceCount == 0 {
		t.Fatal("No CUDA devices found")
	}
}

func TestTorchNativeIndexerCreation(t *testing.T) {
	t.Log(" Testing LibTorch native indexer creation...")
	
	config := DefaultTorchNativeConfig()
	config.VectorDim = 128 // Smaller for faster testing
	config.IVFClusters = 32
	config.NumSubquantizers = 16
	
	indexer, err := NewTorchNativeIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()
	
	// Check initial stats
	stats, err := indexer.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}
	
	t.Logf("Initial stats: %+v", stats)
	
	if stats.VectorDim != 128 {
		t.Errorf("Expected vector dim 128, got %d", stats.VectorDim)
	}
	
	if stats.IsTrained {
		t.Error("Indexer should not be trained initially")
	}
	
	if stats.IndexBuilt {
		t.Error("Index should not be built initially")
	}
}

func TestTorchNativeIndexerTraining(t *testing.T) {
	t.Log(" Testing LibTorch native indexer training...")
	
	config := DefaultTorchNativeConfig()
	config.VectorDim = 128
	config.IVFClusters = 16 // Small for testing
	config.NumSubquantizers = 16
	
	indexer, err := NewTorchNativeIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()
	
	// Generate training vectors
	numTraining := 1000
	trainingVectors := make([][]int8, numTraining)
	for i := 0; i < numTraining; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		trainingVectors[i] = vec
	}
	
	// Train the indexer
	start := time.Now()
	err = indexer.TrainIndex(trainingVectors)
	trainTime := time.Since(start)
	
	if err != nil {
		t.Fatalf("Failed to train indexer: %v", err)
	}
	
	t.Logf("Training completed in %v", trainTime)
	
	// Check stats after training
	stats, err := indexer.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats after training: %v", err)
	}
	
	if !stats.IsTrained {
		t.Error("Indexer should be trained")
	}
	
	t.Logf("Stats after training: %+v", stats)
}

func TestTorchNativeIndexerAddVectors(t *testing.T) {
	t.Log("📚 Testing LibTorch native indexer add vectors...")
	
	config := DefaultTorchNativeConfig()
	config.VectorDim = 128
	config.IVFClusters = 16
	config.NumSubquantizers = 16
	
	indexer, err := NewTorchNativeIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()
	
	// Generate training vectors
	numTraining := 500
	trainingVectors := make([][]int8, numTraining)
	for i := 0; i < numTraining; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		trainingVectors[i] = vec
	}
	
	// Train first
	err = indexer.TrainIndex(trainingVectors)
	if err != nil {
		t.Fatalf("Failed to train indexer: %v", err)
	}
	
	// Generate vectors to index
	numVectors := 2000
	vectors := make([][]int8, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		vectors[i] = vec
	}
	
	// Add vectors to index
	start := time.Now()
	err = indexer.AddVectors(vectors)
	addTime := time.Since(start)
	
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	
	t.Logf("Added %d vectors in %v", numVectors, addTime)
	
	// Check stats after adding vectors
	stats, err := indexer.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats after adding vectors: %v", err)
	}
	
	if !stats.IndexBuilt {
		t.Error("Index should be built after adding vectors")
	}
	
	if stats.NumVectors != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, stats.NumVectors)
	}
	
	t.Logf("Stats after adding vectors: %+v", stats)
}

func TestTorchNativeIndexerSearch(t *testing.T) {
	t.Log(" Testing LibTorch native indexer search...")
	
	config := DefaultTorchNativeConfig()
	config.VectorDim = 128
	config.IVFClusters = 16
	config.NumSubquantizers = 16
	config.ProbeLists = 4
	config.RerankK = 50
	
	indexer, err := NewTorchNativeIndexer(config)
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()
	
	// Generate and train
	numTraining := 500
	trainingVectors := make([][]int8, numTraining)
	for i := 0; i < numTraining; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		trainingVectors[i] = vec
	}
	
	err = indexer.TrainIndex(trainingVectors)
	if err != nil {
		t.Fatalf("Failed to train indexer: %v", err)
	}
	
	// Generate vectors to index (include some training vectors for exact matches)
	numVectors := 1000
	vectors := make([][]int8, numVectors)
	
	// First few vectors are from training set (for testing exact matches)
	for i := 0; i < min(100, numTraining); i++ {
		vectors[i] = make([]int8, len(trainingVectors[i]))
		copy(vectors[i], trainingVectors[i])
	}
	
	// Rest are random
	for i := 100; i < numVectors; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		vectors[i] = vec
	}
	
	err = indexer.AddVectors(vectors)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	
	// Test search with exact match (should find itself)
	query := vectors[0] // Use first vector as query
	k := 10
	
	start := time.Now()
	ids, scores, err := indexer.Search(query, k)
	searchTime := time.Since(start)
	
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}
	
	t.Logf("Search completed in %v", searchTime)
	t.Logf("Found %d results", len(ids))
	
	if len(ids) == 0 {
		t.Fatal("No search results returned")
	}
	
	if len(ids) != len(scores) {
		t.Fatalf("Mismatch between IDs (%d) and scores (%d)", len(ids), len(scores))
	}
	
	// Check that the first result is the query itself (ID 0)
	if ids[0] != 0 {
		t.Logf("Warning: Expected exact match (ID 0) as first result, got ID %d", ids[0])
	}
	
	// Print top results
	for i := 0; i < min(5, len(ids)); i++ {
		t.Logf("Result %d: ID=%d, Score=%.3f", i+1, ids[i], scores[i])
	}
	
	// Verify scores are in descending order
	for i := 1; i < len(scores); i++ {
		if scores[i] > scores[i-1] {
			t.Errorf("Scores not in descending order: scores[%d]=%.3f > scores[%d]=%.3f", 
				i, scores[i], i-1, scores[i-1])
		}
	}
}

func TestTorchNativePipeline(t *testing.T) {
	t.Log("🔄 Testing LibTorch native pipeline...")
	
	config := DefaultTorchNativeConfig()
	config.VectorDim = 128
	config.IVFClusters = 8  // Very small for testing
	config.NumSubquantizers = 16
	
	embedder := NewMockEmbedder(128)
	
	pipeline, err := NewTorchNativePipeline(config, embedder)
	if err != nil {
		t.Fatalf("Failed to create pipeline: %v", err)
	}
	defer pipeline.Close()
	
	// Training texts
	trainingTexts := []string{
		"machine learning algorithms",
		"neural network architectures", 
		"deep learning frameworks",
		"computer vision models",
		"natural language processing",
		"artificial intelligence systems",
		"data science techniques",
		"statistical modeling approaches",
		"optimization algorithms",
		"reinforcement learning methods",
	}
	
	// Train pipeline
	err = pipeline.TrainPipeline(trainingTexts)
	if err != nil {
		t.Fatalf("Failed to train pipeline: %v", err)
	}
	
	// Index texts
	indexTexts := []string{
		"advanced machine learning techniques for computer vision",
		"deep neural networks for image classification",
		"natural language understanding with transformers",
		"reinforcement learning for autonomous systems",
		"statistical methods in data analysis",
		"optimization techniques for machine learning",
		"computer vision algorithms for object detection",
		"neural network architectures for sequence modeling",
		"artificial intelligence in healthcare applications",
		"machine learning frameworks and libraries",
	}
	
	start := time.Now()
	err = pipeline.IndexTexts(indexTexts)
	indexTime := time.Since(start)
	
	if err != nil {
		t.Fatalf("Failed to index texts: %v", err)
	}
	
	t.Logf("Indexed %d texts in %v", len(indexTexts), indexTime)
	
	// Search
	query := "neural networks for computer vision"
	k := 5
	
	start = time.Now()
	results, err := pipeline.Search(query, k)
	searchTime := time.Since(start)
	
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}
	
	t.Logf("Search completed in %v", searchTime)
	t.Logf("Found %d results for query: '%s'", len(results), query)
	
	if len(results) == 0 {
		t.Fatal("No search results")
	}
	
	// Print results
	for i, result := range results {
		t.Logf("Result %d: Score=%.3f, Text='%s'", i+1, result.Score, result.Text)
	}
	
	// Get pipeline stats
	stats, err := pipeline.GetPipelineStats()
	if err != nil {
		t.Fatalf("Failed to get pipeline stats: %v", err)
	}
	
	t.Logf("Pipeline stats: %+v", stats)
	
	if stats.NumTexts != len(indexTexts) {
		t.Errorf("Expected %d texts, got %d", len(indexTexts), stats.NumTexts)
	}
}

func BenchmarkTorchNativeSearch(b *testing.B) {
	config := DefaultTorchNativeConfig()
	config.VectorDim = 128
	config.IVFClusters = 64
	config.NumSubquantizers = 16
	
	indexer, err := NewTorchNativeIndexer(config)
	if err != nil {
		b.Fatalf("Failed to create indexer: %v", err)
	}
	defer indexer.Close()
	
	// Quick training and indexing setup
	numTraining := 1000
	trainingVectors := make([][]int8, numTraining)
	for i := 0; i < numTraining; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		trainingVectors[i] = vec
	}
	
	err = indexer.TrainIndex(trainingVectors)
	if err != nil {
		b.Fatalf("Failed to train: %v", err)
	}
	
	numVectors := 10000
	vectors := make([][]int8, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]int8, 128)
		for j := range vec {
			vec[j] = int8(rand.Intn(256) - 128)
		}
		vectors[i] = vec
	}
	
	err = indexer.AddVectors(vectors)
	if err != nil {
		b.Fatalf("Failed to add vectors: %v", err)
	}
	
	// Benchmark search
	query := vectors[0]
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := indexer.Search(query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}