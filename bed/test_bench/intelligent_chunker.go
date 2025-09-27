package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_simple -L/usr/local/cuda/lib64 -lcudart
// #include <stdlib.h>
// extern void* simple_search_create(int max_docs, int dim);
// extern void simple_search_destroy(void* handle);
// extern int simple_search_add_vectors(void* handle, const signed char* docs, int num_docs);
// extern int simple_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"

// ChunkedDocument represents a document that may span multiple lines
type ChunkedDocument struct {
	FilePath   string
	StartLine  int
	EndLine    int
	Content    string
	Embedding  []int8
	LineCount  int
}

// ChunkingConfig defines how to chunk documents
type ChunkingConfig struct {
	MaxChunkSize      int  // Maximum characters per chunk
	MinChunkSize      int  // Minimum characters before considering a new chunk
	MaxLinesPerChunk  int  // Maximum lines to combine
	RespectParagraphs bool // Split on empty lines
	SmartBoundaries   bool // Try to split at sentence/paragraph boundaries
}

// DefaultChunkingConfig returns sensible defaults
func DefaultChunkingConfig() ChunkingConfig {
	return ChunkingConfig{
		MaxChunkSize:      500,  // ~100 tokens typical
		MinChunkSize:      50,   // Don't create tiny chunks
		MaxLinesPerChunk:  10,   // Don't combine too many lines
		RespectParagraphs: true, // Split on empty lines
		SmartBoundaries:   true, // Smart splitting
	}
}

// IntelligentChunker processes files into semantically meaningful chunks
type IntelligentChunker struct {
	config ChunkingConfig
	model  *FastModel
}

// NewIntelligentChunker creates a new chunker
func NewIntelligentChunker(model *FastModel, config ChunkingConfig) *IntelligentChunker {
	return &IntelligentChunker{
		config: config,
		model:  model,
	}
}

// ChunkFile processes a file into chunks with line tracking
func (ic *IntelligentChunker) ChunkFile(path string) ([]*ChunkedDocument, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var chunks []*ChunkedDocument
	scanner := bufio.NewScanner(file)

	currentChunk := &strings.Builder{}
	startLine := 1
	currentLine := 0
	linesInChunk := 0

	flushChunk := func() {
		if currentChunk.Len() > 0 {
			content := strings.TrimSpace(currentChunk.String())
			if content != "" {
				embedding, err := ic.model.EmbedInt8(content)
				if err == nil {
					chunks = append(chunks, &ChunkedDocument{
						FilePath:  path,
						StartLine: startLine,
						EndLine:   currentLine,
						Content:   content,
						Embedding: embedding,
						LineCount: linesInChunk,
					})
				}
			}
			currentChunk.Reset()
			startLine = currentLine + 1
			linesInChunk = 0
		}
	}

	for scanner.Scan() {
		currentLine++
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Check if we should start a new chunk
		shouldSplit := false

		// Empty line - respect paragraph boundaries
		if ic.config.RespectParagraphs && trimmed == "" && currentChunk.Len() > ic.config.MinChunkSize {
			shouldSplit = true
		}

		// Exceeded max chunk size
		if currentChunk.Len()+len(line) > ic.config.MaxChunkSize {
			shouldSplit = true
		}

		// Exceeded max lines per chunk
		if linesInChunk >= ic.config.MaxLinesPerChunk {
			shouldSplit = true
		}

		// Smart boundary detection (end of sentence, code block, etc.)
		if ic.config.SmartBoundaries && currentChunk.Len() > ic.config.MinChunkSize {
			if ic.isNaturalBoundary(trimmed) {
				shouldSplit = true
			}
		}

		if shouldSplit && currentChunk.Len() > 0 {
			flushChunk()
		}

		// Add line to current chunk (unless it's empty and we respect paragraphs)
		if !(ic.config.RespectParagraphs && trimmed == "" && currentChunk.Len() == 0) {
			if currentChunk.Len() > 0 {
				currentChunk.WriteString(" ")
			}
			currentChunk.WriteString(trimmed)
			linesInChunk++

			// Update start line if this is the first line of a new chunk
			if linesInChunk == 1 {
				startLine = currentLine
			}
		}
	}

	// Flush any remaining chunk
	flushChunk()

	return chunks, nil
}

// isNaturalBoundary detects natural splitting points
func (ic *IntelligentChunker) isNaturalBoundary(line string) bool {
	if line == "" {
		return true
	}

	// Common code/config boundaries
	if strings.HasPrefix(line, "---") || strings.HasPrefix(line, "===") {
		return true
	}

	// Function/class definitions (common programming patterns)
	keywords := []string{"func ", "def ", "class ", "public ", "private ", "package ", "import "}
	for _, keyword := range keywords {
		if strings.HasPrefix(line, keyword) {
			return true
		}
	}

	// Markdown headers
	if strings.HasPrefix(line, "#") {
		return true
	}

	return false
}

// ChunkDirectory processes all text files in a directory
func (ic *IntelligentChunker) ChunkDirectory(dir string, extensions []string) ([]*ChunkedDocument, error) {
	var allChunks []*ChunkedDocument

	// Walk directory
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}

		// Check file extension
		ext := strings.ToLower(filepath.Ext(path))
		validExt := false
		for _, allowedExt := range extensions {
			if ext == allowedExt {
				validExt = true
				break
			}
		}

		if !validExt {
			return nil
		}

		// Skip very large files
		if info.Size() > 50*1024*1024 { // 50MB
			return nil
		}

		// Chunk the file
		chunks, err := ic.ChunkFile(path)
		if err == nil {
			allChunks = append(allChunks, chunks...)
		}

		return nil
	})

	return allChunks, err
}

// FormatChunkResult formats a chunk result for display
func FormatChunkResult(chunk *ChunkedDocument, score float32, query string) string {
	var result strings.Builder

	relPath, _ := filepath.Rel(".", chunk.FilePath)

	// Format line range
	lineRange := fmt.Sprintf("L%d", chunk.StartLine)
	if chunk.EndLine > chunk.StartLine {
		lineRange = fmt.Sprintf("L%d-%d", chunk.StartLine, chunk.EndLine)
	}

	// Check if query appears in content
	contains := strings.Contains(strings.ToLower(chunk.Content), strings.ToLower(query))
	marker := "  "
	if contains {
		marker = "✓ "
	}

	result.WriteString(fmt.Sprintf("%s%s:%s (%.3f)\n", marker, relPath, lineRange, score))

	// Show content preview (truncate if needed)
	content := chunk.Content
	if len(content) > 200 {
		content = content[:197] + "..."
	}

	// Indent content
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		result.WriteString(fmt.Sprintf("     %s\n", line))
	}

	return result.String()
}

// ChunkedSearch performs search on chunked documents
func ChunkedSearch(chunks []*ChunkedDocument, query string, k int) ([]*ChunkedDocument, []float32, error) {
	if len(chunks) == 0 {
		return nil, nil, fmt.Errorf("no documents to search")
	}

	// Use the scalable GPU search
	numChunks := len(chunks)

	handle := C.simple_search_create(C.int(numChunks), C.int(512))
	defer C.simple_search_destroy(handle)

	// Prepare embeddings
	flatEmbeddings := make([]int8, numChunks*512)
	for i, chunk := range chunks {
		copy(flatEmbeddings[i*512:(i+1)*512], chunk.Embedding)
	}

	C.simple_search_add_vectors(
		handle,
		(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
		C.int(numChunks),
	)

	// Generate query embedding (assuming benchModel is available)
	queryEmb, err := benchModel.EmbedInt8(query)
	if err != nil {
		return nil, nil, err
	}

	// Search
	indices := make([]int32, k)
	scores := make([]float32, k)

	C.simple_search_query(
		handle,
		(*C.schar)(unsafe.Pointer(&queryEmb[0])),
		C.int(k),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	// Collect results
	results := make([]*ChunkedDocument, 0, k)
	resultScores := make([]float32, 0, k)

	for i := 0; i < k; i++ {
		idx := indices[i]
		if idx >= 0 && int(idx) < len(chunks) {
			results = append(results, chunks[idx])
			resultScores = append(resultScores, scores[i])
		}
	}

	return results, resultScores, nil
}