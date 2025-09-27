package src

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../ -lcudart -lcublas -lstdc++ -lm
#cgo LDFLAGS: -Wl,-rpath,/usr/local/cuda/lib64

#include <stdint.h>
#include <stdlib.h>

// C interface to CUDA functions
extern void* create_max_performance_index(int max_vectors, int dim, int max_batch_size);
extern void destroy_max_performance_index(void* handle);
extern int add_vectors_cuda(void* handle, const int8_t* vectors, const float* scales, int num_vectors);
extern int search_batch_cuda(void* handle, const int8_t* queries, int num_queries, int k,
                           int* result_indices, float* result_scores);
extern int build_ivf_cuda(void* handle, int nlist, int nprobe, const int8_t* training_data, int num_training);
extern void get_stats_cuda(void* handle, float* avg_qps, float* peak_qps, size_t* memory_bytes);
*/
import "C"

import (
	"context"
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/lee101/gobed"
)

// SearchConfig configures the GPU search system
type SearchConfig struct {
	Device       string
	EmbeddingDim int
	ChunkSize    int   // tokens per chunk
	ChunkOverlap int   // token overlap between chunks
	IVFClusters  int   // number of IVF clusters
	ProbeLists   int   // clusters to probe during search
	BatchSize    int   // batch size for processing
	MaxFileSize  int64 // max file size to index
	IndexPath    string
	UseInt8      bool
	NumWorkers   int
	MaxVectors   int // max vectors in index
}

// DefaultSearchConfig returns optimized default configuration
func DefaultSearchConfig() SearchConfig {
	return SearchConfig{
		Device:       "cuda:0",
		EmbeddingDim: 512,
		ChunkSize:    512,
		ChunkOverlap: 64,
		IVFClusters:  1024,
		ProbeLists:   32,
		BatchSize:    256,
		MaxFileSize:  10 * 1024 * 1024, // 10MB
		IndexPath:    ".bed_gpu_index",
		UseInt8:      true,
		NumWorkers:   runtime.NumCPU(),
		MaxVectors:   1000000, // 1M vectors max
	}
}

// Chunk represents a searchable chunk of text
type Chunk struct {
	ID          uint64
	FilePath    string
	StartOffset int
	EndOffset   int
	LineStart   int
	LineEnd     int
	Text        string
	Embedding   []int8 // Int8 quantized embedding
	Scale       float32
}

// GPUSearchIndex provides high-performance GPU-accelerated search
type GPUSearchIndex struct {
	config   SearchConfig
	embedder *gobed.EmbeddingModel

	// CUDA handle
	cudaHandle unsafe.Pointer

	// Index data
	chunks      []*Chunk
	fileIndex   map[string][]uint64 // file -> chunk IDs
	chunkMap    map[uint64]*Chunk   // ID -> chunk
	numVectors  int32

	// IVF index
	ivfTrained bool
	centroids  [][]int8

	// Stats
	indexTime   time.Duration
	searchCount int64
	totalQPS    float64

	mu sync.RWMutex
}

// NewGPUSearchIndex creates a new GPU-accelerated search index
func NewGPUSearchIndex(config SearchConfig) (*GPUSearchIndex, error) {
	// Load gobed model
	embedder, err := gobed.LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load gobed model: %w", err)
	}

	// Initialize CUDA
	handle := C.create_max_performance_index(
		C.int(config.MaxVectors),
		C.int(config.EmbeddingDim),
		C.int(config.BatchSize),
	)
	if handle == nil {
		return nil, fmt.Errorf("failed to initialize CUDA index")
	}

	log.Printf("🚀 Initialized GPU search index (dim=%d, max_vectors=%d)",
		config.EmbeddingDim, config.MaxVectors)

	return &GPUSearchIndex{
		config:     config,
		embedder:   embedder,
		cudaHandle: handle,
		chunks:     make([]*Chunk, 0),
		fileIndex:  make(map[string][]uint64),
		chunkMap:   make(map[uint64]*Chunk),
	}, nil
}

// Close releases GPU resources
func (idx *GPUSearchIndex) Close() {
	if idx.cudaHandle != nil {
		C.destroy_max_performance_index(idx.cudaHandle)
		idx.cudaHandle = nil
	}
}

// chunkFile splits a file into overlapping chunks
func (idx *GPUSearchIndex) chunkFile(filePath string) ([]*Chunk, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	text := string(content)
	lines := strings.Split(text, "\n")

	var chunks []*Chunk
	var currentChunk []string
	currentTokens := 0
	lineStart := 0

	for lineNum, line := range lines {
		// Simple token estimation
		tokens := len(strings.Fields(line))

		if currentTokens+tokens > idx.config.ChunkSize && len(currentChunk) > 0 {
			// Create chunk
			chunkText := strings.Join(currentChunk, "\n")

			// Generate chunk ID
			h := fnv.New64a()
			h.Write([]byte(filePath))
			h.Write([]byte{byte(lineStart >> 8), byte(lineStart)})
			chunkID := h.Sum64()

			chunk := &Chunk{
				ID:        chunkID,
				FilePath:  filePath,
				LineStart: lineStart,
				LineEnd:   lineNum - 1,
				Text:      chunkText,
			}

			// Calculate offsets
			chunk.StartOffset = 0
			for i := 0; i < lineStart && i < len(lines); i++ {
				chunk.StartOffset += len(lines[i]) + 1
			}
			chunk.EndOffset = chunk.StartOffset + len(chunkText)

			chunks = append(chunks, chunk)

			// Start new chunk with overlap
			overlapLines := idx.config.ChunkOverlap / 20 // Rough estimation
			if overlapLines > len(currentChunk)/2 {
				overlapLines = len(currentChunk) / 2
			}

			if overlapLines > 0 {
				currentChunk = currentChunk[len(currentChunk)-overlapLines:]
				currentTokens = 0
				for _, l := range currentChunk {
					currentTokens += len(strings.Fields(l))
				}
				lineStart = lineNum - overlapLines
			} else {
				currentChunk = []string{}
				currentTokens = 0
				lineStart = lineNum
			}
		}

		currentChunk = append(currentChunk, line)
		currentTokens += tokens
	}

	// Add final chunk
	if len(currentChunk) > 0 {
		chunkText := strings.Join(currentChunk, "\n")

		h := fnv.New64a()
		h.Write([]byte(filePath))
		h.Write([]byte{byte(lineStart >> 8), byte(lineStart)})
		chunkID := h.Sum64()

		chunk := &Chunk{
			ID:        chunkID,
			FilePath:  filePath,
			LineStart: lineStart,
			LineEnd:   len(lines) - 1,
			Text:      chunkText,
			EndOffset: len(text),
		}

		chunk.StartOffset = 0
		for i := 0; i < lineStart && i < len(lines); i++ {
			chunk.StartOffset += len(lines[i]) + 1
		}

		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

// embedChunks generates embeddings for chunks using gobed
func (idx *GPUSearchIndex) embedChunks(chunks []*Chunk) error {
	for _, chunk := range chunks {
		// Get float32 embedding from gobed
		embedding, err := idx.embedder.Encode(chunk.Text)
		if err != nil {
			log.Printf("Failed to embed chunk: %v", err)
			continue
		}

		// Quantize to int8 for GPU efficiency
		chunk.Embedding, chunk.Scale = quantizeToInt8(embedding)
	}
	return nil
}

// quantizeToInt8 converts float32 embeddings to int8
func quantizeToInt8(embedding []float32) ([]int8, float32) {
	// Find scale
	maxAbs := float32(0)
	for _, v := range embedding {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}

	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1.0
	}

	// Quantize
	quantized := make([]int8, len(embedding))
	for i, v := range embedding {
		q := int(v / scale)
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		quantized[i] = int8(q)
	}

	return quantized, scale
}

// IndexDirectory indexes all files in a directory
func (idx *GPUSearchIndex) IndexDirectory(ctx context.Context, dir string, extensions []string) error {
	startTime := time.Now()

	if extensions == nil {
		extensions = []string{".go", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".md", ".txt"}
	}

	log.Printf("🔍 Indexing directory: %s", dir)

	// Find all files
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}

		// Check size
		if info.Size() > idx.config.MaxFileSize {
			return nil
		}

		// Check extension
		for _, ext := range extensions {
			if strings.HasSuffix(strings.ToLower(path), ext) {
				files = append(files, path)
				break
			}
		}
		return nil
	})
	if err != nil {
		return err
	}

	log.Printf("📁 Found %d files to index", len(files))

	// Process files in parallel
	var wg sync.WaitGroup
	chunkChan := make(chan []*Chunk, idx.config.NumWorkers)

	// Start workers
	for i := 0; i < idx.config.NumWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for chunks := range chunkChan {
				if err := idx.embedChunks(chunks); err != nil {
					log.Printf("Embedding error: %v", err)
				}

				idx.mu.Lock()
				for _, chunk := range chunks {
					idx.chunks = append(idx.chunks, chunk)
					idx.chunkMap[chunk.ID] = chunk
					idx.fileIndex[chunk.FilePath] = append(idx.fileIndex[chunk.FilePath], chunk.ID)
				}
				idx.mu.Unlock()
			}
		}()
	}

	// Process files
	for _, file := range files {
		select {
		case <-ctx.Done():
			close(chunkChan)
			wg.Wait()
			return ctx.Err()
		default:
			chunks, err := idx.chunkFile(file)
			if err != nil {
				log.Printf("Failed to chunk %s: %v", file, err)
				continue
			}
			chunkChan <- chunks
		}
	}

	close(chunkChan)
	wg.Wait()

	// Upload to GPU
	if err := idx.uploadToGPU(); err != nil {
		return fmt.Errorf("failed to upload to GPU: %w", err)
	}

	// Train IVF if we have enough data
	if len(idx.chunks) > 100 {
		if err := idx.trainIVF(); err != nil {
			log.Printf("Warning: IVF training failed: %v", err)
		}
	}

	idx.indexTime = time.Since(startTime)
	atomic.StoreInt32(&idx.numVectors, int32(len(idx.chunks)))

	log.Printf("✅ Indexed %d chunks from %d files in %.2fs",
		len(idx.chunks), len(idx.fileIndex), idx.indexTime.Seconds())

	return nil
}

// uploadToGPU uploads embeddings to GPU memory
func (idx *GPUSearchIndex) uploadToGPU() error {
	if len(idx.chunks) == 0 {
		return nil
	}

	// Flatten embeddings
	dim := idx.config.EmbeddingDim
	vectors := make([]int8, len(idx.chunks)*dim)
	scales := make([]float32, len(idx.chunks))

	for i, chunk := range idx.chunks {
		copy(vectors[i*dim:(i+1)*dim], chunk.Embedding)
		scales[i] = chunk.Scale
	}

	// Upload to GPU
	result := C.add_vectors_cuda(
		idx.cudaHandle,
		(*C.int8_t)(unsafe.Pointer(&vectors[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int(len(idx.chunks)),
	)

	if result != 0 {
		return fmt.Errorf("CUDA upload failed with code %d", result)
	}

	return nil
}

// trainIVF trains the IVF index
func (idx *GPUSearchIndex) trainIVF() error {
	// Use subset for training
	numTraining := len(idx.chunks)
	if numTraining > 10000 {
		numTraining = 10000
	}

	dim := idx.config.EmbeddingDim
	trainingData := make([]int8, numTraining*dim)

	for i := 0; i < numTraining; i++ {
		copy(trainingData[i*dim:(i+1)*dim], idx.chunks[i].Embedding)
	}

	result := C.build_ivf_cuda(
		idx.cudaHandle,
		C.int(idx.config.IVFClusters),
		C.int(idx.config.ProbeLists),
		(*C.int8_t)(unsafe.Pointer(&trainingData[0])),
		C.int(numTraining),
	)

	if result == 0 {
		idx.ivfTrained = true
		log.Printf("✅ Trained IVF index with %d clusters", idx.config.IVFClusters)
	}

	return nil
}

// GPUSearchResult represents a search result from GPU search
type GPUSearchResult struct {
	Chunk    *Chunk
	Score    float32
	FilePath string
	Preview  string
	Lines    string
}

// Search performs similarity search
func (idx *GPUSearchIndex) Search(query string, k int) ([]*GPUSearchResult, error) {
	// Embed query
	embedding, err := idx.embedder.Encode(query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Quantize
	queryInt8, _ := quantizeToInt8(embedding)

	// Search on GPU
	indices := make([]int32, k)
	scores := make([]float32, k)

	result := C.search_batch_cuda(
		idx.cudaHandle,
		(*C.int8_t)(unsafe.Pointer(&queryInt8[0])),
		C.int(1), // Single query
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("CUDA search failed with code %d", result)
	}

	// Build results
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	results := make([]*GPUSearchResult, 0, k)
	for i := 0; i < k && indices[i] >= 0 && int(indices[i]) < len(idx.chunks); i++ {
		chunk := idx.chunks[indices[i]]

		preview := chunk.Text
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}

		results = append(results, &GPUSearchResult{
			Chunk:    chunk,
			Score:    scores[i],
			FilePath: chunk.FilePath,
			Preview:  preview,
			Lines:    fmt.Sprintf("%d-%d", chunk.LineStart+1, chunk.LineEnd+1),
		})
	}

	// Update stats
	atomic.AddInt64(&idx.searchCount, 1)

	return results, nil
}

// BatchSearch performs batch similarity search
func (idx *GPUSearchIndex) BatchSearch(queries []string, k int) ([][]*GPUSearchResult, error) {
	if len(queries) == 0 {
		return nil, nil
	}

	// Embed all queries
	dim := idx.config.EmbeddingDim
	allQueryEmbeddings := make([]int8, len(queries)*dim)

	for i, query := range queries {
		embedding, err := idx.embedder.Encode(query)
		if err != nil {
			log.Printf("Failed to embed query %d: %v", i, err)
			continue
		}

		queryInt8, _ := quantizeToInt8(embedding)
		copy(allQueryEmbeddings[i*dim:(i+1)*dim], queryInt8)
	}

	// Search on GPU
	totalResults := len(queries) * k
	indices := make([]int32, totalResults)
	scores := make([]float32, totalResults)

	result := C.search_batch_cuda(
		idx.cudaHandle,
		(*C.int8_t)(unsafe.Pointer(&allQueryEmbeddings[0])),
		C.int(len(queries)),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("CUDA batch search failed with code %d", result)
	}

	// Build results for each query
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	allResults := make([][]*GPUSearchResult, len(queries))
	for q := 0; q < len(queries); q++ {
		queryResults := make([]*GPUSearchResult, 0, k)

		for i := 0; i < k; i++ {
			resultIdx := q*k + i
			chunkIdx := indices[resultIdx]

			if chunkIdx >= 0 && int(chunkIdx) < len(idx.chunks) {
				chunk := idx.chunks[chunkIdx]

				preview := chunk.Text
				if len(preview) > 200 {
					preview = preview[:200] + "..."
				}

				queryResults = append(queryResults, &GPUSearchResult{
					Chunk:    chunk,
					Score:    scores[resultIdx],
					FilePath: chunk.FilePath,
					Preview:  preview,
					Lines:    fmt.Sprintf("%d-%d", chunk.LineStart+1, chunk.LineEnd+1),
				})
			}
		}

		allResults[q] = queryResults
	}

	return allResults, nil
}

// GetStats returns index statistics
func (idx *GPUSearchIndex) GetStats() map[string]interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var avgQPS, peakQPS C.float
	var memoryBytes C.size_t

	C.get_stats_cuda(idx.cudaHandle, &avgQPS, &peakQPS, &memoryBytes)

	return map[string]interface{}{
		"num_chunks":     len(idx.chunks),
		"num_files":      len(idx.fileIndex),
		"index_time_sec": idx.indexTime.Seconds(),
		"search_count":   atomic.LoadInt64(&idx.searchCount),
		"avg_qps":        float32(avgQPS),
		"peak_qps":       float32(peakQPS),
		"memory_mb":      float64(memoryBytes) / (1024 * 1024),
		"ivf_trained":    idx.ivfTrained,
		"embedding_dim":  idx.config.EmbeddingDim,
	}
}

// SaveIndex saves the index to disk
func (idx *GPUSearchIndex) SaveIndex(path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Create directory
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}

	// Save chunks
	chunksFile, err := os.Create(filepath.Join(path, "chunks.gob"))
	if err != nil {
		return err
	}
	defer chunksFile.Close()

	encoder := gob.NewEncoder(chunksFile)
	if err := encoder.Encode(idx.chunks); err != nil {
		return err
	}

	// Save file index
	indexFile, err := os.Create(filepath.Join(path, "file_index.gob"))
	if err != nil {
		return err
	}
	defer indexFile.Close()

	if err := gob.NewEncoder(indexFile).Encode(idx.fileIndex); err != nil {
		return err
	}

	// Save config
	configFile, err := os.Create(filepath.Join(path, "config.gob"))
	if err != nil {
		return err
	}
	defer configFile.Close()

	if err := gob.NewEncoder(configFile).Encode(idx.config); err != nil {
		return err
	}

	log.Printf("💾 Saved index to %s (%d chunks)", path, len(idx.chunks))
	return nil
}

// LoadIndex loads the index from disk
func (idx *GPUSearchIndex) LoadIndex(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Load chunks
	chunksFile, err := os.Open(filepath.Join(path, "chunks.gob"))
	if err != nil {
		return err
	}
	defer chunksFile.Close()

	decoder := gob.NewDecoder(chunksFile)
	if err := decoder.Decode(&idx.chunks); err != nil {
		return err
	}

	// Rebuild chunk map
	idx.chunkMap = make(map[uint64]*Chunk)
	for _, chunk := range idx.chunks {
		idx.chunkMap[chunk.ID] = chunk
	}

	// Load file index
	indexFile, err := os.Open(filepath.Join(path, "file_index.gob"))
	if err != nil {
		return err
	}
	defer indexFile.Close()

	if err := gob.NewDecoder(indexFile).Decode(&idx.fileIndex); err != nil {
		return err
	}

	// Upload to GPU
	if err := idx.uploadToGPU(); err != nil {
		return fmt.Errorf("failed to upload to GPU: %w", err)
	}

	atomic.StoreInt32(&idx.numVectors, int32(len(idx.chunks)))

	log.Printf("📂 Loaded index from %s (%d chunks)", path, len(idx.chunks))
	return nil
}