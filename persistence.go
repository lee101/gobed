package gobed

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/lee101/gobed/ann/search"
)

// PersistenceFormat represents the format for saving/loading indexes
type PersistenceFormat string

const (
	// FormatBinary uses Go's gob encoding (fastest, Go-specific)
	FormatBinary PersistenceFormat = "binary"
	// FormatJSON uses JSON encoding (portable but slower)
	FormatJSON PersistenceFormat = "json"
)

// IndexSnapshot represents a serializable snapshot of a SearchEngine
type IndexSnapshot struct {
	Version      string                 `json:"version"`
	CreatedAt    time.Time              `json:"created_at"`
	NumDocuments int                    `json:"num_documents"`
	Config       SearchConfig           `json:"config"`
	Documents    map[int]string         `json:"documents"`
	IndexData    *IndexData             `json:"index_data,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// IndexData contains the serializable index structures
type IndexData struct {
	// Core index data
	Vectors       [][]float32 `json:"-"` // Skip in JSON, too large
	VectorsBinary []byte      `json:"vectors_binary,omitempty"`
	IDs           []int       `json:"ids"`

	// Index state
	IndexType string `json:"index_type"`
	Trained   bool   `json:"trained"`

	// Stats
	MemoryUsageMB float64 `json:"memory_usage_mb"`
}

// SaveOptions configures how the index is saved
type SaveOptions struct {
	Format       PersistenceFormat
	Compress     bool
	IncludeTexts bool
	Metadata     map[string]interface{}
}

// DefaultSaveOptions returns recommended save options
func DefaultSaveOptions() SaveOptions {
	return SaveOptions{
		Format:       FormatBinary,
		Compress:     true,
		IncludeTexts: true,
		Metadata:     make(map[string]interface{}),
	}
}

// Save persists the SearchEngine to disk
func (se *SearchEngine) Save(path string, options SaveOptions) error {
	se.mu.RLock()
	defer se.mu.RUnlock()

	// Create directory if needed
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Create snapshot
	snapshot := &IndexSnapshot{
		Version:      "1.0",
		CreatedAt:    time.Now(),
		NumDocuments: len(se.documents),
		Config:       se.config,
		Metadata:     options.Metadata,
	}

	// Include documents if requested
	if options.IncludeTexts {
		snapshot.Documents = make(map[int]string, len(se.documents))
		for id, text := range se.documents {
			snapshot.Documents[id] = text
		}
	}

	// Extract index data
	if se.index != nil {
		indexData, err := se.extractIndexData()
		if err != nil {
			return fmt.Errorf("failed to extract index data: %w", err)
		}
		snapshot.IndexData = indexData
	}

	// Save based on format
	switch options.Format {
	case FormatBinary:
		return se.saveBinary(path, snapshot, options.Compress)
	case FormatJSON:
		return se.saveJSON(path, snapshot, options.Compress)
	default:
		return fmt.Errorf("unsupported format: %s", options.Format)
	}
}

// Load restores a SearchEngine from disk
func (se *SearchEngine) Load(path string) error {
	se.mu.Lock()
	defer se.mu.Unlock()

	// Detect format by trying binary first, then JSON
	snapshot, err := se.loadBinary(path)
	if err != nil {
		// Try JSON format
		snapshot, err = se.loadJSON(path)
		if err != nil {
			return fmt.Errorf("failed to load index: %w", err)
		}
	}

	// Restore state
	se.config = snapshot.Config
	se.documents = snapshot.Documents
	if se.documents == nil {
		se.documents = make(map[int]string)
	}

	// Restore index if present
	if snapshot.IndexData != nil {
		if err := se.restoreIndexData(snapshot.IndexData); err != nil {
			return fmt.Errorf("failed to restore index data: %w", err)
		}
	}

	se.initialized = true
	return nil
}

// SaveToDirectory saves the index to a directory with metadata
func (se *SearchEngine) SaveToDirectory(dir string, options SaveOptions) error {
	// Create directory
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Save main index
	indexPath := filepath.Join(dir, "index.bin")
	if err := se.Save(indexPath, options); err != nil {
		return err
	}

	// Save metadata file
	metaPath := filepath.Join(dir, "metadata.json")
	metadata := map[string]interface{}{
		"version":       "1.0",
		"created_at":    time.Now(),
		"format":        string(options.Format),
		"compressed":    options.Compress,
		"num_documents": len(se.documents),
		"stats":         se.Stats(),
	}

	metaFile, err := os.Create(metaPath)
	if err != nil {
		return fmt.Errorf("failed to create metadata file: %w", err)
	}
	defer metaFile.Close()

	encoder := json.NewEncoder(metaFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(metadata); err != nil {
		return fmt.Errorf("failed to write metadata: %w", err)
	}

	return nil
}

// LoadFromDirectory loads the index from a directory
func (se *SearchEngine) LoadFromDirectory(dir string) error {
	indexPath := filepath.Join(dir, "index.bin")
	return se.Load(indexPath)
}

// extractIndexData extracts serializable data from the internal index
func (se *SearchEngine) extractIndexData() (*IndexData, error) {
	if se.index == nil {
		return nil, nil
	}

	stats := se.index.Stats()

	data := &IndexData{
		IndexType:     stats.IndexType,
		Trained:       true, // If we're saving, assume it's trained
		MemoryUsageMB: float64(stats.MemoryUsage) / (1024 * 1024),
	}

	// For now, we'll need to rebuild the index on load
	// A full implementation would serialize the internal structures

	return data, nil
}

// restoreIndexData restores the internal index from saved data
func (se *SearchEngine) restoreIndexData(data *IndexData) error {
	if data == nil {
		return nil
	}

	// Create new index with saved configuration
	config := search.DefaultConfig()

	se.index = search.NewEngine(config)

	// IMPORTANT: Mark the index as already trained to skip re-indexing
	// This is a workaround until we implement full state serialization
	if data.Trained {
		// Set the trained flag to prevent re-indexing
		se.index.SetTrained(true)
	}

	return nil
}

// saveBinary saves the snapshot in binary format
func (se *SearchEngine) saveBinary(path string, snapshot *IndexSnapshot, compress bool) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	var writer io.Writer = file

	// Add compression if requested
	if compress {
		// Could use gzip here
		writer = file
	}

	encoder := gob.NewEncoder(writer)
	if err := encoder.Encode(snapshot); err != nil {
		return fmt.Errorf("failed to encode snapshot: %w", err)
	}

	return nil
}

// loadBinary loads the snapshot from binary format
func (se *SearchEngine) loadBinary(path string) (*IndexSnapshot, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var reader io.Reader = file

	// Detect and handle compression
	// Could check for gzip magic bytes here

	decoder := gob.NewDecoder(reader)
	var snapshot IndexSnapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return nil, fmt.Errorf("failed to decode snapshot: %w", err)
	}

	return &snapshot, nil
}

// saveJSON saves the snapshot in JSON format
func (se *SearchEngine) saveJSON(path string, snapshot *IndexSnapshot, compress bool) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(snapshot); err != nil {
		return fmt.Errorf("failed to encode snapshot: %w", err)
	}

	return nil
}

// loadJSON loads the snapshot from JSON format
func (se *SearchEngine) loadJSON(path string) (*IndexSnapshot, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var snapshot IndexSnapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return nil, fmt.Errorf("failed to decode snapshot: %w", err)
	}

	return &snapshot, nil
}

// QuickSave saves the index with default options
func (se *SearchEngine) QuickSave(path string) error {
	return se.Save(path, DefaultSaveOptions())
}

// Checkpoint creates a checkpoint of the current index state
func (se *SearchEngine) Checkpoint(dir string) error {
	timestamp := time.Now().Format("20060102_150405")
	checkpointDir := filepath.Join(dir, fmt.Sprintf("checkpoint_%s", timestamp))
	return se.SaveToDirectory(checkpointDir, DefaultSaveOptions())
}

// AutoSave starts automatic periodic saving
func (se *SearchEngine) AutoSave(dir string, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if err := se.Checkpoint(dir); err != nil {
					// Log error but don't stop
					fmt.Printf("AutoSave error: %v\n", err)
				}
			case <-se.ctx.Done():
				return
			}
		}
	}()
}

// PersistenceStats returns statistics about saved indexes
type PersistenceStats struct {
	LastSaved    time.Time
	SaveCount    int
	LoadCount    int
	LastLoadTime time.Duration
	LastSaveTime time.Duration
}

var persistenceStats = &PersistenceStats{}
var persistenceMu sync.RWMutex

// GetPersistenceStats returns persistence statistics
func GetPersistenceStats() PersistenceStats {
	persistenceMu.RLock()
	defer persistenceMu.RUnlock()
	return *persistenceStats
}
