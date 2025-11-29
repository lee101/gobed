package src

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"
)

const (
	MaxLineLength   = 200   // Maximum chars to show per line
	MaxContextChars = 80    // Chars to show around match
	MaxFileSize     = 10 << 20 // 10MB default max file size
	ChunkOverlap    = 50    // Overlap between chunks for context
)

// ChunkedMatch represents a search result with limited context
type ChunkedMatch struct {
	Path       string
	LineNumber int
	Column     int
	Preview    string // Limited preview with ... ellipsis
	FullMatch  bool   // Whether the entire match is shown
}

// ChunkedSearcher performs token-friendly searches
type ChunkedSearcher struct {
	ignoreFilter *EnhancedIgnoreFilter
	maxFileSize  int64
	maxLineLen   int
	contextChars int
	results      []ChunkedMatch
	mu           sync.Mutex
}

// NewChunkedSearcher creates a new chunked searcher
func NewChunkedSearcher() (*ChunkedSearcher, error) {
	ignoreFilter, err := NewEnhancedIgnoreFilter(".",
		WithMaxFileSize(MaxFileSize),
		WithBinarySearch(false),
	)
	if err != nil {
		return nil, err
	}

	return &ChunkedSearcher{
		ignoreFilter: ignoreFilter,
		maxFileSize:  MaxFileSize,
		maxLineLen:   MaxLineLength,
		contextChars: MaxContextChars,
		results:      make([]ChunkedMatch, 0),
	}, nil
}

// SearchDirectory searches for pattern in directory with chunked output
func (cs *ChunkedSearcher) SearchDirectory(dir, pattern string, caseSensitive bool) ([]ChunkedMatch, error) {
	cs.results = make([]ChunkedMatch, 0)

	var wg sync.WaitGroup
	fileChan := make(chan string, 100)

	// Start workers
	numWorkers := 4
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go cs.searchWorker(&wg, fileChan, pattern, caseSensitive)
	}

	// Walk directory
	go func() {
		defer close(fileChan)
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}

			// Check if we should process this file
			shouldProcess, fileType := cs.ignoreFilter.ShouldProcess(path)
			if !shouldProcess {
				return nil
			}

			// Skip binary files
			if fileType == FileTypeBinary {
				return nil
			}

			// Skip files that are too large
			if info.Size() > cs.maxFileSize {
				return nil
			}

			fileChan <- path
			return nil
		})
	}()

	wg.Wait()
	return cs.results, nil
}

// searchWorker processes files from the channel
func (cs *ChunkedSearcher) searchWorker(wg *sync.WaitGroup, fileChan <-chan string, pattern string, caseSensitive bool) {
	defer wg.Done()

	searchPattern := pattern
	if !caseSensitive {
		searchPattern = strings.ToLower(pattern)
	}

	for path := range fileChan {
		cs.searchFile(path, searchPattern, pattern, caseSensitive)
	}
}

// searchFile searches a single file for the pattern
func (cs *ChunkedSearcher) searchFile(path, searchPattern, originalPattern string, caseSensitive bool) {
	file, err := os.Open(path)
	if err != nil {
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		searchLine := line

		if !caseSensitive {
			searchLine = strings.ToLower(line)
		}

		// Find all matches in the line
		if idx := strings.Index(searchLine, searchPattern); idx >= 0 {
			// Create chunked preview
			preview := cs.createChunkedPreview(line, idx, len(originalPattern))

			cs.mu.Lock()
			cs.results = append(cs.results, ChunkedMatch{
				Path:       path,
				LineNumber: lineNum,
				Column:     idx + 1,
				Preview:    preview,
				FullMatch:  len(line) <= cs.maxLineLen,
			})
			cs.mu.Unlock()
		}
	}
}

// createChunkedPreview creates a limited preview with ellipsis
func (cs *ChunkedSearcher) createChunkedPreview(line string, matchIdx, matchLen int) string {
	lineLen := utf8.RuneCountInString(line)

	// If line is short enough, return it as-is
	if lineLen <= cs.maxLineLen {
		return line
	}

	// Calculate context window
	matchEnd := matchIdx + matchLen
	contextStart := matchIdx - cs.contextChars
	contextEnd := matchEnd + cs.contextChars

	// Adjust boundaries
	if contextStart < 0 {
		contextStart = 0
	}
	if contextEnd > len(line) {
		contextEnd = len(line)
	}

	// Build preview with ellipsis
	var preview strings.Builder

	if contextStart > 0 {
		preview.WriteString("...")
		// Find a good break point (space, punctuation)
		for contextStart < matchIdx && contextStart < len(line) {
			if line[contextStart] == ' ' || line[contextStart] == '\t' {
				contextStart++
				break
			}
			contextStart++
		}
	}

	// Extract the chunk
	chunk := line[contextStart:contextEnd]
	preview.WriteString(chunk)

	if contextEnd < len(line) {
		// Find a good break point at the end
		for contextEnd > matchEnd && contextEnd > 0 {
			if line[contextEnd-1] == ' ' || line[contextEnd-1] == '\t' {
				break
			}
			contextEnd--
		}
		preview.WriteString("...")
	}

	result := preview.String()

	// Final length check
	if utf8.RuneCountInString(result) > cs.maxLineLen {
		// Truncate more aggressively
		runes := []rune(result)
		halfLen := cs.maxLineLen / 2

		// Keep start and end, add ellipsis in middle
		if len(runes) > cs.maxLineLen {
			start := string(runes[:halfLen-2])
			end := string(runes[len(runes)-(halfLen-2):])
			result = start + " ... " + end
		}
	}

	return result
}

// PrintResults displays results in ripgrep-like format
func (cs *ChunkedSearcher) PrintResults(results []ChunkedMatch, useColor bool) {
	if len(results) == 0 {
		fmt.Println("No matches found")
		return
	}

	// Group results by file
	fileGroups := make(map[string][]ChunkedMatch)
	for _, match := range results {
		fileGroups[match.Path] = append(fileGroups[match.Path], match)
	}

	// Print grouped results
	for path, matches := range fileGroups {
		if useColor {
			fmt.Printf("\033[35m%s\033[0m\n", path)
		} else {
			fmt.Printf("%s\n", path)
		}

		for _, match := range matches {
			if useColor {
				fmt.Printf("\033[32m%d\033[0m:%d: %s\n",
					match.LineNumber, match.Column, match.Preview)
			} else {
				fmt.Printf("%d:%d: %s\n",
					match.LineNumber, match.Column, match.Preview)
			}
		}
		fmt.Println()
	}

	// Summary
	fmt.Printf("Found %d matches in %d files\n", len(results), len(fileGroups))
}