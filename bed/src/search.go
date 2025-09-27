package src

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/lee101/gobed"
)

// Searcher handles semantic search operations
type Searcher struct {
	index     *EmbeddingIndex
	model     *gobed.EmbeddingModel
	config    *Config
	indexPath string
}

// SearchOptions configures search behavior
type SearchOptions struct {
	Query         string
	Limit         int
	Context       int
	Threshold     float32
	NoIndex       bool
	ForceIndex    bool
	UseGPU        bool
	IgnoreCase    bool
	ColorMode     string
	Verbose       bool
}

// IndexStatus provides information about the current index
type IndexStatus struct {
	Exists       bool                   `json:"exists"`
	Path         string                 `json:"path"`
	Stats        map[string]interface{} `json:"stats,omitempty"`
	LastUpdated  time.Time              `json:"last_updated"`
	IsUpToDate   bool                   `json:"is_up_to_date"`
	Size         int64                  `json:"size"`
}

// NewSearcher creates a new searcher instance
func NewSearcher() (*Searcher, error) {
	// Load configuration
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}
	
	// Load embedding model
	model, err := gobed.LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load embedding model: %w", err)
	}
	
	searcher := &Searcher{
		model:     model,
		config:    config,
		indexPath: config.IndexPath,
	}
	
	return searcher, nil
}

// Search performs semantic search
func (s *Searcher) Search(options SearchOptions) error {
	// Load or create index
	if err := s.ensureIndex(options); err != nil {
		return fmt.Errorf("failed to ensure index: %w", err)
	}
	
	// Perform search
	results, err := s.index.Search(options.Query, s.model, options)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}
	
	// Display results
	return s.displayResults(results, options)
}

// ensureIndex loads existing index or creates new one if needed
func (s *Searcher) ensureIndex(options SearchOptions) error {
	// Check if we should skip indexing
	if options.NoIndex {
		return s.loadExistingIndex()
	}
	
	// Check if we should force reindex
	if options.ForceIndex {
		return s.createIndex(options)
	}
	
	// Try to load existing index first
	if err := s.loadExistingIndex(); err == nil {
		// Check if index is up to date
		if s.isIndexUpToDate() {
			return nil
		}
		
		if options.Verbose {
			fmt.Println("Index appears outdated, rebuilding...")
		}
	}
	
	// Create new index
	return s.createIndex(options)
}

// loadExistingIndex loads an existing index from disk
func (s *Searcher) loadExistingIndex() error {
	index, err := LoadEmbeddingIndex(s.indexPath)
	if err != nil {
		return fmt.Errorf("failed to load index: %w", err)
	}
	
	s.index = index
	return nil
}

// createIndex creates a new index
func (s *Searcher) createIndex(options SearchOptions) error {
	if options.Verbose {
		fmt.Println("Creating semantic index...")
	}
	
	// Create indexer
	indexer, err := NewIndexer()
	if err != nil {
		return fmt.Errorf("failed to create indexer: %w", err)
	}
	defer indexer.Close()
	
	// Configure indexing
	indexOptions := IndexOptions{
		Path:       ".",
		Force:      options.ForceIndex,
		BatchSize:  s.config.BatchSize,
		UseGPU:     options.UseGPU || s.config.UseGPU,
		Verbose:    options.Verbose,
	}
	
	// Build index
	if err := indexer.Index(indexOptions); err != nil {
		return fmt.Errorf("indexing failed: %w", err)
	}
	
	// Load the newly created index
	return s.loadExistingIndex()
}

// isIndexUpToDate checks if the current index is up to date
func (s *Searcher) isIndexUpToDate() bool {
	if s.index == nil {
		return false
	}
	
	// Simple heuristic: check if any files in the base path are newer than the index
	indexTime := s.index.UpdatedAt
	
	var hasNewerFiles bool
	filepath.WalkDir(s.index.BasePath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // Continue walking
		}
		
		if d.IsDir() {
			return nil
		}
		
		// Check if file should be ignored
		if s.shouldIgnoreFile(path) {
			return nil
		}
		
		info, err := d.Info()
		if err != nil {
			return nil
		}
		
		if info.ModTime().After(indexTime) {
			hasNewerFiles = true
			return filepath.SkipAll // Stop walking
		}
		
		return nil
	})
	
	return !hasNewerFiles
}

// shouldIgnoreFile checks if a file should be ignored
func (s *Searcher) shouldIgnoreFile(path string) bool {
	// Create a temporary ignore filter to check
	filter, err := NewIgnoreFilter(".", s.config.RespectGitignore)
	if err != nil {
		return false
	}
	
	return filter.ShouldIgnore(path) || !filter.IsTextFile(path)
}

// displayResults displays search results in a formatted way
func (s *Searcher) displayResults(results []*SearchResult, options SearchOptions) error {
	if len(results) == 0 {
		fmt.Printf("No results found for: %s\n", options.Query)
		return nil
	}
	
	fmt.Printf("Found %d result(s) for: %s\n\n", len(results), options.Query)
	
	for i, result := range results {
		// Display file path and line number
		fmt.Printf("%s:%d\n", result.FilePath, result.LineNumber)
		
		// Display similarity score
		fmt.Printf("  Similarity: %.3f\n", result.Similarity)
		
		// Display context before
		if options.Context > 0 && result.ContextBefore != "" {
			lines := splitIntoLines(result.ContextBefore)
			for j, line := range lines {
				lineNum := result.LineNumber - len(lines) + j
				fmt.Printf("  %d: %s\n", lineNum, line)
			}
		}
		
		// Display the matching line (highlighted if color is enabled)
		if shouldUseColor(options.ColorMode) {
			fmt.Printf("→ %d: \033[1;31m%s\033[0m\n", result.LineNumber, result.Content)
		} else {
			fmt.Printf("→ %d: %s\n", result.LineNumber, result.Content)
		}
		
		// Display context after
		if options.Context > 0 && result.ContextAfter != "" {
			lines := splitIntoLines(result.ContextAfter)
			for j, line := range lines {
				lineNum := result.LineNumber + 1 + j
				fmt.Printf("  %d: %s\n", lineNum, line)
			}
		}
		
		// Add separator between results
		if i < len(results)-1 {
			fmt.Println()
		}
	}
	
	return nil
}

// Close releases resources
func (s *Searcher) Close() error {
	// Index is closed automatically
	return nil
}

// GetIndexStatus returns information about the current index
func GetIndexStatus() (*IndexStatus, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}
	
	status := &IndexStatus{
		Path: config.IndexPath,
	}
	
	// Check if index exists
	if _, err := os.Stat(config.IndexPath + ".bin"); err == nil {
		status.Exists = true
		
		// Get file info
		if info, err := os.Stat(config.IndexPath + ".bin"); err == nil {
			status.LastUpdated = info.ModTime()
			status.Size = info.Size()
		}
		
		// Try to load and get stats
		if index, err := LoadEmbeddingIndex(config.IndexPath); err == nil {
			status.Stats = index.Stats()
			status.IsUpToDate = checkIndexUpToDate(index)
		}
	} else if _, err := os.Stat(config.IndexPath + ".json"); err == nil {
		status.Exists = true
		
		// Get file info for JSON format
		if info, err := os.Stat(config.IndexPath + ".json"); err == nil {
			status.LastUpdated = info.ModTime()
			status.Size = info.Size()
		}
		
		// Try to load and get stats
		if index, err := LoadEmbeddingIndex(config.IndexPath); err == nil {
			status.Stats = index.Stats()
			status.IsUpToDate = checkIndexUpToDate(index)
		}
	}
	
	return status, nil
}

// Display prints the index status in a readable format
func (status *IndexStatus) Display() error {
	fmt.Printf("Bed Index Status\n")
	fmt.Printf("================\n\n")
	
	if !status.Exists {
		fmt.Printf("Status: No index found\n")
		fmt.Printf("Path:   %s\n", status.Path)
		fmt.Printf("\nRun 'bed index' to create an index.\n")
		return nil
	}
	
	fmt.Printf("Status: Index exists\n")
	fmt.Printf("Path:   %s\n", status.Path)
	fmt.Printf("Size:   %.2f MB\n", float64(status.Size)/1024/1024)
	fmt.Printf("Updated: %s\n", status.LastUpdated.Format(time.RFC3339))
	
	if status.IsUpToDate {
		fmt.Printf("State:  Up to date ✓\n")
	} else {
		fmt.Printf("State:  May need updating\n")
	}
	
	if status.Stats != nil {
		fmt.Printf("\nIndex Statistics:\n")
		if totalFiles, ok := status.Stats["total_files"].(int); ok {
			fmt.Printf("  Files indexed: %d\n", totalFiles)
		}
		if totalLines, ok := status.Stats["total_lines"].(int); ok {
			fmt.Printf("  Lines indexed: %d\n", totalLines)
		}
		if memUsage, ok := status.Stats["memory_usage"].(int64); ok {
			fmt.Printf("  Memory usage:  %.2f MB\n", float64(memUsage)/1024/1024)
		}
		if hasGPU, ok := status.Stats["has_gpu_index"].(bool); ok && hasGPU {
			fmt.Printf("  GPU accelerated: Yes \n")
		}
	}
	
	return nil
}

// Helper functions

func checkIndexUpToDate(index *EmbeddingIndex) bool {
	// Simple check: if index was updated in last hour, consider it current
	return time.Since(index.UpdatedAt) < time.Hour
}

func shouldUseColor(colorMode string) bool {
	switch colorMode {
	case "always":
		return true
	case "never":
		return false
	case "auto":
		// Check if output is a terminal
		return isTerminal()
	default:
		return false
	}
}

func isTerminal() bool {
	// Simple check if stdout is a terminal
	if fileInfo, err := os.Stdout.Stat(); err == nil {
		return (fileInfo.Mode() & os.ModeCharDevice) != 0
	}
	return false
}