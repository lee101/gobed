package src

import (
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"
)

// GPUChunkedSearcher uses GPU-accelerated semantic search with chunked output
type GPUChunkedSearcher struct {
	engine         *SimpleSearchEngine
	queryProcessor *QueryProcessor
	maxLineLen     int
	contextChars   int
	mu             sync.Mutex
}

// NewGPUChunkedSearcher creates a new GPU-accelerated chunked searcher
func NewGPUChunkedSearcher() (*GPUChunkedSearcher, error) {
	engine, err := NewSimpleSearchEngine()
	if err != nil {
		return nil, err
	}

	return &GPUChunkedSearcher{
		engine:         engine,
		queryProcessor: NewQueryProcessor(),
		maxLineLen:     MaxLineLength,
		contextChars:   MaxContextChars,
	}, nil
}

// SearchWithChunkedOutput performs GPU-accelerated semantic search with chunked output
func (gcs *GPUChunkedSearcher) SearchWithChunkedOutput(options BedSearchOptions) error {
	// Process query for semantic search
	processed := gcs.queryProcessor.Process(options.Query)

	if options.Verbose {
		fmt.Printf("🚀 GPU-Accelerated Semantic Search\n")
		fmt.Printf("Query: %s\n", processed.Original)
		fmt.Printf("Enhanced: %s\n", processed.Enhanced)
		fmt.Printf("Type: %s\n\n", queryTypeString(processed.QueryType))
	}

	// Update search settings
	gcs.maxLineLen = options.Limit * 20  // Adjust based on limit
	if gcs.maxLineLen > MaxLineLength {
		gcs.maxLineLen = MaxLineLength
	}

	// Build index with GPU acceleration
	if !options.NoIndex {
		if err := gcs.engine.IndexDirectory(".", options); err != nil {
			return fmt.Errorf("indexing failed: %w", err)
		}
	}

	// Perform semantic search
	matches, err := gcs.engine.Search(processed.Enhanced, options.Limit, options.Threshold)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}

	// Display results with chunked formatting
	gcs.displayChunkedResults(matches, options)

	return nil
}

// displayChunkedResults shows search results with chunked output
func (gcs *GPUChunkedSearcher) displayChunkedResults(matches []SearchMatch, options BedSearchOptions) {
	if len(matches) == 0 {
		fmt.Println("No results found")
		return
	}

	fmt.Printf("Found %d result(s) (GPU-accelerated)\n\n", len(matches))

	// Group results by file
	fileGroups := make(map[string][]SearchMatch)
	for _, match := range matches {
		doc := match.Document
		fileGroups[doc.Path] = append(fileGroups[doc.Path], match)
	}

	useColor := shouldUseColor(options.ColorMode)

	// Display each file group with chunked content
	for path, fileMatches := range fileGroups {
		// File header
		if useColor {
			fmt.Printf("\033[35m%s\033[0m\n", path)
		} else {
			fmt.Printf("%s\n", path)
		}

		for _, match := range fileMatches {
			doc := match.Document

			// Handle binary files
			if doc.IsBinary {
				fmt.Printf("  Binary file (similarity: %.3f)\n", match.Similarity)
				continue
			}

			// Create chunked preview of content
			preview := gcs.createChunkedPreview(doc.Content, options.Query)

			// Display with line number and similarity score
			if useColor {
				fmt.Printf("  \033[32m%d\033[0m: %s \033[90m[%.3f]\033[0m\n",
					doc.LineNumber, preview, match.Similarity)
			} else {
				fmt.Printf("  %d: %s [%.3f]\n",
					doc.LineNumber, preview, match.Similarity)
			}
		}
		fmt.Println()
	}

	// Show GPU performance stats if verbose
	if options.Verbose {
		fmt.Printf("\n⚡ GPU Acceleration: Enabled\n")
		fmt.Printf("📊 Embeddings processed: %d\n", len(gcs.engine.documents))
		fmt.Printf("🎯 Similarity threshold: %.2f\n", options.Threshold)
	}
}

// createChunkedPreview creates a limited preview with ellipsis for long content
func (gcs *GPUChunkedSearcher) createChunkedPreview(content, query string) string {
	lineLen := utf8.RuneCountInString(content)

	// If content is short enough, return as-is
	if lineLen <= gcs.maxLineLen {
		return highlightMatch(content, query)
	}

	// Find query position for context-aware chunking
	lowerContent := strings.ToLower(content)
	lowerQuery := strings.ToLower(query)
	queryIdx := strings.Index(lowerContent, lowerQuery)

	var preview string

	if queryIdx >= 0 {
		// Query found - show context around it
		matchStart := queryIdx
		matchEnd := queryIdx + len(query)

		// Calculate context window
		contextStart := matchStart - gcs.contextChars
		contextEnd := matchEnd + gcs.contextChars

		// Adjust boundaries
		if contextStart < 0 {
			contextStart = 0
		}
		if contextEnd > len(content) {
			contextEnd = len(content)
		}

		// Find word boundaries for cleaner truncation
		if contextStart > 0 {
			// Find next space after context start
			for i := contextStart; i < matchStart && i < len(content); i++ {
				if content[i] == ' ' || content[i] == '\t' {
					contextStart = i + 1
					break
				}
			}
			preview = "..." + content[contextStart:contextEnd]
		} else {
			preview = content[contextStart:contextEnd]
		}

		if contextEnd < len(content) {
			// Find last space before context end
			for i := contextEnd - 1; i > matchEnd && i >= 0; i-- {
				if content[i] == ' ' || content[i] == '\t' {
					contextEnd = i
					break
				}
			}
			preview = preview[:len(preview)-(len(content)-contextEnd)] + "..."
		}

	} else {
		// Query not found literally - use semantic match context
		// Show beginning and end of line
		halfLen := gcs.maxLineLen / 2 - 5

		if halfLen > 0 {
			runes := []rune(content)
			if len(runes) > gcs.maxLineLen {
				start := string(runes[:halfLen])
				end := string(runes[len(runes)-halfLen:])
				preview = start + " ... " + end
			} else {
				preview = content
			}
		} else {
			preview = content[:gcs.maxLineLen-3] + "..."
		}
	}

	// Highlight the match if found
	return highlightMatch(preview, query)
}

// highlightMatch highlights the query in the content if color is enabled
func highlightMatch(content, query string) string {
	if !shouldUseColor("auto") {
		return content
	}

	lowerContent := strings.ToLower(content)
	lowerQuery := strings.ToLower(query)

	idx := strings.Index(lowerContent, lowerQuery)
	if idx < 0 {
		return content
	}

	// Highlight with yellow
	before := content[:idx]
	match := content[idx : idx+len(query)]
	after := content[idx+len(query):]

	return before + "\033[33m" + match + "\033[0m" + after
}


// Close releases resources
func (gcs *GPUChunkedSearcher) Close() error {
	// The engine will handle GPU cleanup
	return nil
}