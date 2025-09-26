package src

import (
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/lee101/gobed"
)

// IndexVersion defines the current index format version
const IndexVersion = 1

// EmbeddingIndex represents the main semantic index
type EmbeddingIndex struct {
	// Metadata
	Version     int               `json:"version"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	BasePath    string            `json:"base_path"`
	TotalFiles  int               `json:"total_files"`
	TotalLines  int               `json:"total_lines"`
	IndexSize   int64             `json:"index_size"`
	
	// Configuration used to build this index
	Config      IndexConfig       `json:"config"`
	
	// File tracking
	Files       map[string]*FileEntry `json:"files"`
	Lines       []*LineEntry          `json:"lines"`
	
	// Embeddings and search structures
	FileEmbeddings map[string][]float32 `json:"file_embeddings"`
	LineEmbeddings [][]float32          `json:"line_embeddings"`
	
	// Fast lookup structures
	PathToID    map[string]int        `json:"path_to_id"`
	IDToPath    map[int]string        `json:"id_to_path"`
	
	// GPU-optimized structures (when GPU is available)
	GPUIndex    *GPUIndexData         `json:"gpu_index,omitempty"`
	
	// Concurrent access protection
	mu          sync.RWMutex          `json:"-"`
	dirty       bool                  `json:"-"`
	indexPath   string                `json:"-"`
}

// IndexConfig stores configuration used to build the index
type IndexConfig struct {
	MaxFileSize       int64    `json:"max_file_size"`
	IncludeExtensions []string `json:"include_extensions"`
	ExcludeExtensions []string `json:"exclude_extensions"`
	EmbeddingModel    string   `json:"embedding_model"`
	UseGPU            bool     `json:"use_gpu"`
	CompressionLevel  int      `json:"compression_level"`
	LineBasedIndex    bool     `json:"line_based_index"`
	ContextWindow     int      `json:"context_window"`
}

// FileEntry represents metadata about an indexed file
type FileEntry struct {
	ID           int       `json:"id"`
	Path         string    `json:"path"`
	RelativePath string    `json:"relative_path"`
	Size         int64     `json:"size"`
	ModTime      time.Time `json:"mod_time"`
	LineCount    int       `json:"line_count"`
	Language     string    `json:"language"`
	Checksum     uint32    `json:"checksum"`
	
	// Line range in the global line index
	LineStart    int       `json:"line_start"`
	LineEnd      int       `json:"line_end"`
}

// LineEntry represents an individual line with its embedding
type LineEntry struct {
	ID           int     `json:"id"`
	FileID       int     `json:"file_id"`
	LineNumber   int     `json:"line_number"`
	Content      string  `json:"content"`
	Embedding    []float32 `json:"embedding,omitempty"` // Omitted for space in JSON
	EmbeddingID  int     `json:"embedding_id"`         // Index into LineEmbeddings
	
	// Context for better search results
	ContextBefore string `json:"context_before,omitempty"`
	ContextAfter  string `json:"context_after,omitempty"`
}

// GPUIndexData holds GPU-optimized index structures
type GPUIndexData struct {
	DeviceID         int      `json:"device_id"`
	BatchSize        int      `json:"batch_size"`
	MemoryUsage      int64    `json:"memory_usage"`
	CompressedData   []byte   `json:"compressed_data,omitempty"`
	IndexType        string   `json:"index_type"` // "flat", "ivf", "hnsw"
	SearchParameters []byte   `json:"search_parameters,omitempty"`
}

// SearchResult represents a single search result
type SearchResult struct {
	FileID       int     `json:"file_id"`
	FilePath     string  `json:"file_path"`
	LineID       int     `json:"line_id"`
	LineNumber   int     `json:"line_number"`
	Content      string  `json:"content"`
	Similarity   float32 `json:"similarity"`
	Distance     float32 `json:"distance"`
	
	// Context for display
	ContextBefore string `json:"context_before"`
	ContextAfter  string `json:"context_after"`
	
	// Highlighting information
	Highlights   []HighlightRange `json:"highlights,omitempty"`
}

// HighlightRange represents a range to highlight in search results
type HighlightRange struct {
	Start  int `json:"start"`
	End    int `json:"end"`
	Type   string `json:"type"` // "match", "context", etc.
}

// NewEmbeddingIndex creates a new embedding index
func NewEmbeddingIndex(basePath, indexPath string, config IndexConfig) *EmbeddingIndex {
	return &EmbeddingIndex{
		Version:        IndexVersion,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
		BasePath:       basePath,
		Config:         config,
		Files:          make(map[string]*FileEntry),
		Lines:          make([]*LineEntry, 0),
		FileEmbeddings: make(map[string][]float32),
		LineEmbeddings: make([][]float32, 0),
		PathToID:       make(map[string]int),
		IDToPath:       make(map[int]string),
		indexPath:      indexPath,
	}
}

// LoadEmbeddingIndex loads an existing index from disk
func LoadEmbeddingIndex(indexPath string) (*EmbeddingIndex, error) {
	// Check for binary format first (more efficient)
	binaryPath := indexPath + ".bin"
	if _, err := os.Stat(binaryPath); err == nil {
		return loadBinaryIndex(binaryPath)
	}
	
	// Fall back to JSON format
	jsonPath := indexPath + ".json"
	if _, err := os.Stat(jsonPath); err == nil {
		return loadJSONIndex(jsonPath)
	}
	
	return nil, fmt.Errorf("no index found at %s", indexPath)
}

// Save persists the index to disk
func (idx *EmbeddingIndex) Save() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	
	idx.UpdatedAt = time.Now()
	
	// Create directory if it doesn't exist
	dir := filepath.Dir(idx.indexPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create index directory: %w", err)
	}
	
	// Save in binary format for performance
	if err := idx.saveBinary(); err != nil {
		return fmt.Errorf("failed to save binary index: %w", err)
	}
	
	// Also save JSON for debugging/portability
	if err := idx.saveJSON(); err != nil {
		// Non-fatal, log but don't fail
		fmt.Printf("Warning: failed to save JSON index: %v\n", err)
	}
	
	idx.dirty = false
	return nil
}

// AddFile adds a file to the index
func (idx *EmbeddingIndex) AddFile(fileInfo *FileInfo, embeddings [][]float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	
	// Calculate checksum
	checksum := crc32.ChecksumIEEE([]byte(fileInfo.Content))
	
	// Check if file already exists and hasn't changed
	if existing, exists := idx.Files[fileInfo.RelativePath]; exists {
		if existing.Checksum == checksum && existing.ModTime.Equal(fileInfo.ModTime) {
			return nil // File hasn't changed
		}
		
		// Remove old entries
		idx.removeFileEntries(existing)
	}
	
	// Create file entry
	fileID := len(idx.Files)
	fileEntry := &FileEntry{
		ID:           fileID,
		Path:         fileInfo.Path,
		RelativePath: fileInfo.RelativePath,
		Size:         fileInfo.Size,
		ModTime:      fileInfo.ModTime,
		LineCount:    fileInfo.LineCount,
		Language:     detectLanguage(fileInfo.Path),
		Checksum:     checksum,
		LineStart:    len(idx.Lines),
		LineEnd:      len(idx.Lines) + len(embeddings),
	}
	
	// Add file entry
	idx.Files[fileInfo.RelativePath] = fileEntry
	idx.PathToID[fileInfo.RelativePath] = fileID
	idx.IDToPath[fileID] = fileInfo.RelativePath
	
	// Add file-level embedding (average of line embeddings)
	if len(embeddings) > 0 {
		fileEmbedding := averageEmbeddings(embeddings)
		idx.FileEmbeddings[fileInfo.RelativePath] = fileEmbedding
	}
	
	// Add line entries and embeddings
	lines := splitIntoLines(fileInfo.Content)
	for i, embedding := range embeddings {
		if i >= len(lines) {
			break
		}
		
		lineID := len(idx.Lines)
		embeddingID := len(idx.LineEmbeddings)
		
		lineEntry := &LineEntry{
			ID:          lineID,
			FileID:      fileID,
			LineNumber:  i + 1,
			Content:     lines[i],
			EmbeddingID: embeddingID,
		}
		
		// Add context if configured
		if idx.Config.ContextWindow > 0 {
			lineEntry.ContextBefore = getContextBefore(lines, i, idx.Config.ContextWindow)
			lineEntry.ContextAfter = getContextAfter(lines, i, idx.Config.ContextWindow)
		}
		
		idx.Lines = append(idx.Lines, lineEntry)
		idx.LineEmbeddings = append(idx.LineEmbeddings, embedding)
	}
	
	idx.TotalFiles = len(idx.Files)
	idx.TotalLines = len(idx.Lines)
	idx.dirty = true
	
	return nil
}

// Search performs semantic search in the index
func (idx *EmbeddingIndex) Search(query string, model *gobed.EmbeddingModel, options SearchOptions) ([]*SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	
	// Generate query embedding
	queryEmbedding, err := model.Encode(query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %w", err)
	}
	
	// Use GPU search if available and enabled
	if idx.GPUIndex != nil && options.UseGPU {
		return idx.searchGPU(queryEmbedding, options)
	}
	
	// CPU-based search
	return idx.searchCPU(queryEmbedding, options)
}

// searchCPU performs CPU-based semantic search
func (idx *EmbeddingIndex) searchCPU(queryEmbedding []float32, options SearchOptions) ([]*SearchResult, error) {
	type scoredResult struct {
		lineID     int
		similarity float32
	}
	
	results := make([]scoredResult, 0, len(idx.Lines))
	
	// Calculate similarities for all lines
	for i, lineEmbedding := range idx.LineEmbeddings {
		similarity := cosineSimilarity(queryEmbedding, lineEmbedding)
		
		if similarity >= options.Threshold {
			results = append(results, scoredResult{
				lineID:     i,
				similarity: similarity,
			})
		}
	}
	
	// Sort by similarity (descending)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].similarity > results[i].similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
	
	// Limit results
	if len(results) > options.Limit {
		results = results[:options.Limit]
	}
	
	// Convert to SearchResults
	searchResults := make([]*SearchResult, len(results))
	for i, result := range results {
		line := idx.Lines[result.lineID]
		file := idx.Files[idx.IDToPath[line.FileID]]
		
		searchResults[i] = &SearchResult{
			FileID:        line.FileID,
			FilePath:      file.RelativePath,
			LineID:        line.ID,
			LineNumber:    line.LineNumber,
			Content:       line.Content,
			Similarity:    result.similarity,
			Distance:      1.0 - result.similarity,
			ContextBefore: line.ContextBefore,
			ContextAfter:  line.ContextAfter,
		}
	}
	
	return searchResults, nil
}

// searchGPU performs GPU-accelerated semantic search
func (idx *EmbeddingIndex) searchGPU(queryEmbedding []float32, options SearchOptions) ([]*SearchResult, error) {
	// This would integrate with the existing GPU search infrastructure
	// For now, fall back to CPU search
	return idx.searchCPU(queryEmbedding, options)
}

// Stats returns index statistics
func (idx *EmbeddingIndex) Stats() map[string]interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	
	memoryUsage := idx.calculateMemoryUsage()
	
	return map[string]interface{}{
		"version":       idx.Version,
		"created_at":    idx.CreatedAt,
		"updated_at":    idx.UpdatedAt,
		"base_path":     idx.BasePath,
		"total_files":   idx.TotalFiles,
		"total_lines":   idx.TotalLines,
		"index_size":    idx.IndexSize,
		"memory_usage":  memoryUsage,
		"has_gpu_index": idx.GPUIndex != nil,
		"is_dirty":      idx.dirty,
	}
}

// Helper functions

func (idx *EmbeddingIndex) removeFileEntries(file *FileEntry) {
	// Remove line entries for this file
	newLines := make([]*LineEntry, 0, len(idx.Lines)-file.LineCount)
	newEmbeddings := make([][]float32, 0, len(idx.LineEmbeddings)-file.LineCount)
	
	for i, line := range idx.Lines {
		if line.FileID != file.ID {
			newLines = append(newLines, line)
			newEmbeddings = append(newEmbeddings, idx.LineEmbeddings[i])
		}
	}
	
	idx.Lines = newLines
	idx.LineEmbeddings = newEmbeddings
	
	// Remove file embedding
	delete(idx.FileEmbeddings, file.RelativePath)
}

func (idx *EmbeddingIndex) calculateMemoryUsage() int64 {
	// Rough calculation of memory usage
	var size int64
	
	// File entries
	size += int64(len(idx.Files)) * 256 // Approximate size per file entry
	
	// Line entries
	size += int64(len(idx.Lines)) * 128 // Approximate size per line entry
	
	// Embeddings (assuming 1024 dimensions, 4 bytes per float32)
	size += int64(len(idx.LineEmbeddings)) * 1024 * 4
	size += int64(len(idx.FileEmbeddings)) * 1024 * 4
	
	return size
}

func averageEmbeddings(embeddings [][]float32) []float32 {
	if len(embeddings) == 0 {
		return nil
	}
	
	dims := len(embeddings[0])
	result := make([]float32, dims)
	
	for _, embedding := range embeddings {
		for i, val := range embedding {
			result[i] += val
		}
	}
	
	count := float32(len(embeddings))
	for i := range result {
		result[i] /= count
	}
	
	return result
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	
	var dotProduct, normA, normB float32
	
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float32) float32 {
	// Simple approximation for square root
	if x == 0 {
		return 0
	}
	
	guess := x / 2
	for i := 0; i < 10; i++ {
		guess = (guess + x/guess) / 2
	}
	
	return guess
}

func splitIntoLines(content string) []string {
	if content == "" {
		return []string{}
	}
	
	lines := make([]string, 0)
	current := ""
	
	for _, char := range content {
		if char == '\n' {
			lines = append(lines, current)
			current = ""
		} else {
			current += string(char)
		}
	}
	
	if current != "" {
		lines = append(lines, current)
	}
	
	return lines
}

func getContextBefore(lines []string, index, window int) string {
	start := index - window
	if start < 0 {
		start = 0
	}
	
	context := ""
	for i := start; i < index; i++ {
		if context != "" {
			context += "\n"
		}
		context += lines[i]
	}
	
	return context
}

func getContextAfter(lines []string, index, window int) string {
	end := index + window + 1
	if end > len(lines) {
		end = len(lines)
	}
	
	context := ""
	for i := index + 1; i < end; i++ {
		if context != "" {
			context += "\n"
		}
		context += lines[i]
	}
	
	return context
}

func detectLanguage(path string) string {
	ext := filepath.Ext(path)
	
	langMap := map[string]string{
		".go":   "go",
		".py":   "python",
		".js":   "javascript",
		".ts":   "typescript",
		".java": "java",
		".c":    "c",
		".cpp":  "cpp",
		".h":    "c",
		".hpp":  "cpp",
		".rs":   "rust",
		".rb":   "ruby",
		".php":  "php",
		".html": "html",
		".css":  "css",
		".json": "json",
		".yaml": "yaml",
		".yml":  "yaml",
		".xml":  "xml",
		".md":   "markdown",
		".sh":   "shell",
		".sql":  "sql",
	}
	
	if lang, exists := langMap[ext]; exists {
		return lang
	}
	
	return "text"
}