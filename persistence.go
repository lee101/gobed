package gobed

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/lee101/gobed/pkg/ann/search"
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

// Snapshot files written since v2 start with snapshotMagic, then a u32
// version, a u64 length and the gob IndexSnapshot, followed by the ann engine
// state (search.Engine.Save) so Load restores the trained index without
// re-embedding or retraining. Files without the magic are the v1 gob-only
// format: their documents are re-embedded and the index rebuilt once.
const (
	snapshotMagic   = "GBSEARCH"
	snapshotVersion = 2
)

// ErrSnapshotMismatch is returned by LoadMatching when the saved documents
// differ from the expected ones.
var ErrSnapshotMismatch = errors.New("gobed: saved index documents do not match")

// Save persists the SearchEngine to disk atomically.
func (se *SearchEngine) Save(path string, options SaveOptions) error {
	se.mu.RLock()
	defer se.mu.RUnlock()

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	snapshot := &IndexSnapshot{
		Version:      "2.0",
		CreatedAt:    time.Now(),
		NumDocuments: len(se.documents),
		Config:       se.config,
		Metadata:     options.Metadata,
	}
	if options.IncludeTexts {
		snapshot.Documents = make(map[int]string, len(se.documents))
		for id, text := range se.documents {
			snapshot.Documents[id] = text
		}
	}
	if se.index != nil {
		indexData, err := se.extractIndexData()
		if err != nil {
			return fmt.Errorf("failed to extract index data: %w", err)
		}
		snapshot.IndexData = indexData
	}

	if options.Format == FormatJSON {
		return se.saveJSON(path, snapshot, options.Compress)
	}
	if options.Format != FormatBinary {
		return fmt.Errorf("unsupported format: %s", options.Format)
	}

	var meta bytes.Buffer
	if err := gob.NewEncoder(&meta).Encode(snapshot); err != nil {
		return fmt.Errorf("failed to encode snapshot: %w", err)
	}

	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp*")
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer os.Remove(tmp.Name())
	tmp.Chmod(0o644)
	w := bufio.NewWriterSize(tmp, 1<<20)
	w.WriteString(snapshotMagic)
	binary.Write(w, binary.LittleEndian, uint32(snapshotVersion))
	binary.Write(w, binary.LittleEndian, uint64(meta.Len()))
	w.Write(meta.Bytes())
	hasIndex := se.index != nil
	binary.Write(w, binary.LittleEndian, hasIndex)
	if hasIndex {
		if err := se.index.Save(w); err != nil {
			tmp.Close()
			return fmt.Errorf("failed to save index: %w", err)
		}
	}
	if err := w.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// Load restores a SearchEngine from disk. Snapshots with saved index state
// load without training; v1 snapshots (documents only) are re-embedded and
// indexed once, then rewritten in place in the current format (best effort).
func (se *SearchEngine) Load(path string) error {
	return se.load(path, nil, nil)
}

// LoadMatching is Load, but returns ErrSnapshotMismatch (leaving the engine
// untouched) unless the snapshot holds exactly these id/text documents.
func (se *SearchEngine) LoadMatching(path string, ids []int, texts []string) error {
	if len(ids) != len(texts) {
		return fmt.Errorf("ids and texts must have the same length")
	}
	return se.load(path, ids, texts)
}

func (se *SearchEngine) load(path string, ids []int, texts []string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to load index: %w", err)
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	head, _ := r.Peek(len(snapshotMagic))
	if string(head) != snapshotMagic {
		snapshot, err := se.loadBinary(path)
		if err != nil {
			snapshot, err = se.loadJSON(path)
			if err != nil {
				return fmt.Errorf("failed to load index: %w", err)
			}
		}
		if ids != nil && !documentsMatch(snapshot.Documents, ids, texts) {
			return ErrSnapshotMismatch
		}
		if err := se.rebuildFromSnapshot(snapshot); err != nil {
			return err
		}
		f.Close()
		if err := se.QuickSave(path); err != nil {
			log.Printf("gobed: rebuilt v1 snapshot %s but could not upgrade it: %v", path, err)
		}
		return nil
	}

	r.Discard(len(snapshotMagic))
	var version uint32
	var metaLen uint64
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil || version != snapshotVersion {
		return fmt.Errorf("failed to load index: unsupported snapshot version %d", version)
	}
	if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil || metaLen > 1<<34 {
		return fmt.Errorf("failed to load index: bad header")
	}
	var snapshot IndexSnapshot
	if err := gob.NewDecoder(io.LimitReader(r, int64(metaLen))).Decode(&snapshot); err != nil {
		return fmt.Errorf("failed to decode snapshot: %w", err)
	}
	if ids != nil && !documentsMatch(snapshot.Documents, ids, texts) {
		return ErrSnapshotMismatch
	}
	var hasIndex bool
	if err := binary.Read(r, binary.LittleEndian, &hasIndex); err != nil {
		return fmt.Errorf("failed to load index: %w", err)
	}
	var index *search.Engine
	if hasIndex {
		if index, err = search.ReadEngine(r); err != nil {
			return fmt.Errorf("failed to restore index data: %w", err)
		}
	}

	se.mu.Lock()
	defer se.mu.Unlock()
	se.config = snapshot.Config
	se.documents = snapshot.Documents
	if se.documents == nil {
		se.documents = make(map[int]string)
	}
	if index != nil {
		se.index = index
	} else {
		se.index = search.NewEngine(mapSearchConfigToAnnConfig(se.config))
	}
	se.initialized = true
	return nil
}

func documentsMatch(docs map[int]string, ids []int, texts []string) bool {
	if len(docs) != len(ids) {
		return false
	}
	for i, id := range ids {
		if t, ok := docs[id]; !ok || t != texts[i] {
			return false
		}
	}
	return true
}

// rebuildFromSnapshot indexes a v1 snapshot's documents from scratch: v1
// files never stored vectors, so restoring them used to yield an empty index.
func (se *SearchEngine) rebuildFromSnapshot(snapshot *IndexSnapshot) error {
	se.mu.Lock()
	defer se.mu.Unlock()
	se.config = snapshot.Config
	se.documents = make(map[int]string)
	se.index = search.NewEngine(mapSearchConfigToAnnConfig(se.config))
	se.initialized = false
	if len(snapshot.Documents) == 0 {
		se.initialized = true
		return nil
	}
	if se.model == nil {
		return fmt.Errorf("failed to restore index data: v1 snapshot needs an embedding model to rebuild")
	}
	ids := make([]int, 0, len(snapshot.Documents))
	for id := range snapshot.Documents {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	texts := make([]string, len(ids))
	for i, id := range ids {
		texts[i] = snapshot.Documents[id]
	}
	if err := se.indexBatchInternal(ids, texts); err != nil {
		return fmt.Errorf("failed to rebuild v1 snapshot: %w", err)
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
