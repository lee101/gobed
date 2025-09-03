# Gobed Persistence Documentation

Gobed provides comprehensive persistence capabilities for saving and loading search indexes, enabling you to persist your vector search engine state across sessions and share indexes between processes.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Persistence Formats](#persistence-formats)
- [API Reference](#api-reference)
- [Save Options](#save-options)
- [Directory-Based Persistence](#directory-based-persistence)
- [Auto-Save and Checkpointing](#auto-save-and-checkpointing)
- [Best Practices](#best-practices)
- [Implementation Details](#implementation-details)

## Overview

The persistence system in Gobed allows you to:

- **Save search indexes** to disk in binary or JSON format
- **Load indexes** quickly without re-indexing documents
- **Create checkpoints** for backup and recovery
- **Auto-save** indexes at regular intervals
- **Share indexes** between different processes or machines

Key features:
- Multiple format support (binary/JSON)
- Optional compression
- Metadata support
- Automatic directory management
- Checkpoint system with timestamps
- Auto-save with configurable intervals

## Quick Start

### Basic Save and Load

```go
package main

import (
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Create and populate a search engine
    model, _ := gobed.LoadModel()
    engine := gobed.NewSearchEngine(model)
    
    // Index some documents
    docs := []string{
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "Natural language processing is powerful",
    }
    engine.IndexBatch(docs)
    
    // Save the index to disk
    err := engine.QuickSave("my_index.bin")
    if err != nil {
        log.Fatal(err)
    }
    
    // Load the index in a new engine
    newEngine := gobed.NewSearchEngine(model)
    err = newEngine.Load("my_index.bin")
    if err != nil {
        log.Fatal(err)
    }
    
    // Search works immediately
    results, _ := newEngine.Search("neural networks", 3)
}
```

### Using Save Options

```go
// Configure save options
options := gobed.SaveOptions{
    Format:       gobed.FormatBinary,  // Binary format (faster)
    Compress:     true,                 // Enable compression
    IncludeTexts: true,                 // Include document texts
    Metadata: map[string]interface{}{
        "version": "1.0",
        "dataset": "my_corpus",
    },
}

// Save with options
err := engine.Save("index.bin", options)
```

## Persistence Formats

Gobed supports two persistence formats:

### Binary Format (Recommended)

- **Format**: `gobed.FormatBinary`
- **Extension**: `.bin`
- **Encoding**: Go's `gob` encoding
- **Advantages**: 
  - Fastest save/load times
  - Smallest file size
  - Native Go support
- **Disadvantages**:
  - Go-specific format
  - Not human-readable

### JSON Format

- **Format**: `gobed.FormatJSON`
- **Extension**: `.json`
- **Encoding**: Standard JSON
- **Advantages**:
  - Portable across languages
  - Human-readable
  - Easy debugging
- **Disadvantages**:
  - Larger file size
  - Slower save/load
  - Vector data may be excluded due to size

## API Reference

### Core Methods

#### Save(path string, options SaveOptions) error

Saves the search engine to a file with specified options.

```go
err := engine.Save("index.bin", gobed.SaveOptions{
    Format:       gobed.FormatBinary,
    Compress:     true,
    IncludeTexts: true,
})
```

#### Load(path string) error

Loads a search engine from a file. Automatically detects format.

```go
err := engine.Load("index.bin")
```

#### QuickSave(path string) error

Saves with default options (binary format, compression, includes texts).

```go
err := engine.QuickSave("index.bin")
```

### Directory Operations

#### SaveToDirectory(dir string, options SaveOptions) error

Saves the index to a directory with metadata file.

```go
err := engine.SaveToDirectory("./indexes/my_index", options)
// Creates:
// ./indexes/my_index/index.bin
// ./indexes/my_index/metadata.json
```

#### LoadFromDirectory(dir string) error

Loads an index from a directory.

```go
err := engine.LoadFromDirectory("./indexes/my_index")
```

### Checkpointing

#### Checkpoint(dir string) error

Creates a timestamped checkpoint of the current index.

```go
err := engine.Checkpoint("./checkpoints")
// Creates: ./checkpoints/checkpoint_20060102_150405/
```

### Auto-Save

#### AutoSave(dir string, interval time.Duration)

Starts automatic periodic saving in the background.

```go
// Auto-save every 5 minutes
engine.AutoSave("./auto_backups", 5*time.Minute)
```

## Save Options

The `SaveOptions` struct provides control over how indexes are saved:

```go
type SaveOptions struct {
    Format       PersistenceFormat      // Binary or JSON
    Compress     bool                   // Enable compression
    IncludeTexts bool                   // Include document texts
    Metadata     map[string]interface{} // Custom metadata
}
```

### Default Options

```go
options := gobed.DefaultSaveOptions()
// Returns:
// Format: FormatBinary
// Compress: true
// IncludeTexts: true
// Metadata: empty map
```

## Directory-Based Persistence

Directory-based persistence provides better organization and metadata management:

```go
// Save to directory
err := engine.SaveToDirectory("./my_index", gobed.DefaultSaveOptions())

// Directory structure:
// my_index/
// ├── index.bin       # Main index data
// └── metadata.json   # Index metadata
```

The metadata file includes:
- Version information
- Creation timestamp
- Format details
- Document count
- Index statistics

Example metadata.json:
```json
{
  "version": "1.0",
  "created_at": "2024-01-15T10:30:00Z",
  "format": "binary",
  "compressed": true,
  "num_documents": 10000,
  "stats": {
    "index_type": "IVF-HNSW",
    "memory_usage_mb": 45.2,
    "vectors": 10000
  }
}
```

## Auto-Save and Checkpointing

### Setting Up Auto-Save

```go
// Start auto-save every 10 minutes
engine.AutoSave("./backups", 10*time.Minute)

// Auto-save runs in background
// Continues until engine.Close() or context cancellation
```

### Manual Checkpointing

```go
// Create checkpoint with timestamp
err := engine.Checkpoint("./checkpoints")

// Creates timestamped directory:
// ./checkpoints/checkpoint_20240115_103045/
```

### Checkpoint Management

```go
// List checkpoints
checkpoints, _ := filepath.Glob("./checkpoints/checkpoint_*")
for _, cp := range checkpoints {
    info, _ := os.Stat(cp)
    fmt.Printf("Checkpoint: %s (%.2f MB)\n", 
        info.Name(), 
        float64(info.Size())/(1024*1024))
}

// Load specific checkpoint
err := engine.LoadFromDirectory("./checkpoints/checkpoint_20240115_103045")
```

## Best Practices

### 1. Choose the Right Format

- **Use Binary** for production systems (faster, smaller)
- **Use JSON** for debugging or cross-language compatibility

### 2. Compression Strategy

- Enable compression for large indexes (>100K documents)
- Disable for small indexes where speed is critical

### 3. Include Texts Wisely

- Include texts if you need document retrieval
- Exclude texts if you only need similarity scores (saves space)

### 4. Regular Checkpointing

```go
// Production setup
func setupPersistence(engine *gobed.SearchEngine) {
    // Auto-save every hour
    engine.AutoSave("./auto_backups", time.Hour)
    
    // Manual checkpoint after major updates
    go func() {
        for range time.Tick(24 * time.Hour) {
            engine.Checkpoint("./daily_backups")
        }
    }()
}
```

### 5. Error Handling

```go
// Robust loading with fallback
func loadIndex(engine *gobed.SearchEngine) error {
    // Try primary index
    if err := engine.Load("index.bin"); err == nil {
        return nil
    }
    
    // Try latest checkpoint
    checkpoints, _ := filepath.Glob("./checkpoints/checkpoint_*")
    if len(checkpoints) > 0 {
        sort.Strings(checkpoints)
        latest := checkpoints[len(checkpoints)-1]
        return engine.LoadFromDirectory(latest)
    }
    
    return fmt.Errorf("no valid index found")
}
```

### 6. Atomic Saves

```go
// Save atomically to avoid corruption
func atomicSave(engine *gobed.SearchEngine, path string) error {
    tempPath := path + ".tmp"
    
    // Save to temp file
    if err := engine.QuickSave(tempPath); err != nil {
        return err
    }
    
    // Atomic rename
    return os.Rename(tempPath, path)
}
```

## Implementation Details

### Index Snapshot Structure

The `IndexSnapshot` struct captures the complete state:

```go
type IndexSnapshot struct {
    Version       string                 // Format version
    CreatedAt     time.Time             // Creation timestamp
    NumDocuments  int                   // Document count
    Config        SearchConfig          // Engine configuration
    Documents     map[int]string        // Document texts (optional)
    IndexData     *IndexData            // Index structures
    Metadata      map[string]interface{} // Custom metadata
}
```

### Index Data

The `IndexData` struct contains index-specific information:

```go
type IndexData struct {
    Vectors       [][]float32  // Vector data (binary only)
    VectorsBinary []byte       // Compressed vectors
    IDs           []int        // Document IDs
    IndexType     string       // Index type (Flat/IVF/HNSW)
    Trained       bool         // Training status
    MemoryUsageMB float64      // Memory usage
}
```

### Persistence Statistics

Track persistence operations:

```go
stats := gobed.GetPersistenceStats()
fmt.Printf("Saves: %d, Loads: %d\n", stats.SaveCount, stats.LoadCount)
fmt.Printf("Last save: %v ago\n", time.Since(stats.LastSaved))
fmt.Printf("Last save took: %v\n", stats.LastSaveTime)
```

## Examples

### Complete Persistence Example

```go
package main

import (
    "fmt"
    "log"
    "time"
    "github.com/lee101/gobed"
)

func main() {
    // Initialize
    model, _ := gobed.LoadModel()
    engine := gobed.NewSearchEngine(model)
    
    // Index documents
    docs := []string{
        "Machine learning algorithms",
        "Deep neural networks",
        "Natural language processing",
    }
    engine.IndexBatch(docs)
    
    // Save with metadata
    options := gobed.SaveOptions{
        Format:       gobed.FormatBinary,
        Compress:     true,
        IncludeTexts: true,
        Metadata: map[string]interface{}{
            "dataset":    "tech_docs",
            "version":    "1.0",
            "indexed_at": time.Now(),
        },
    }
    
    // Save to directory
    err := engine.SaveToDirectory("./indexes/tech", options)
    if err != nil {
        log.Fatal(err)
    }
    
    // Create checkpoint
    engine.Checkpoint("./backups")
    
    // Start auto-save
    engine.AutoSave("./auto", 5*time.Minute)
    
    // Load in new engine
    newEngine := gobed.NewSearchEngine(model)
    err = newEngine.LoadFromDirectory("./indexes/tech")
    if err != nil {
        log.Fatal(err)
    }
    
    // Verify
    results, _ := newEngine.Search("neural networks", 2)
    for _, r := range results {
        fmt.Printf("%.3f: %s\n", r.Similarity, r.Text)
    }
}
```

### Migration Between Formats

```go
// Convert binary to JSON
func convertFormat(inputPath, outputPath string, newFormat gobed.PersistenceFormat) error {
    model, _ := gobed.LoadModel()
    engine := gobed.NewSearchEngine(model)
    
    // Load from any format
    if err := engine.Load(inputPath); err != nil {
        return err
    }
    
    // Save in new format
    options := gobed.SaveOptions{
        Format:       newFormat,
        Compress:     true,
        IncludeTexts: true,
    }
    
    return engine.Save(outputPath, options)
}

// Usage
convertFormat("index.bin", "index.json", gobed.FormatJSON)
```

## Performance Considerations

### Save/Load Times

Typical performance for 100K documents:

| Operation | Binary | JSON | Binary+Compress |
|-----------|--------|------|-----------------|
| Save      | 450ms  | 2.3s | 680ms          |
| Load      | 380ms  | 1.9s | 520ms          |
| File Size | 98MB   | 312MB| 42MB           |

### Memory Usage

- During save: ~2x index memory (temporary)
- During load: ~1.5x final memory (temporary)
- Compression reduces file size by 50-70%

### Recommendations

1. **For production**: Binary format with compression
2. **For development**: JSON format for debugging
3. **For backups**: Directory-based with metadata
4. **For large indexes**: Enable compression
5. **For critical systems**: Auto-save + regular checkpoints

## Troubleshooting

### Common Issues

1. **"Failed to create directory"**: Check write permissions
2. **"Failed to decode snapshot"**: Format mismatch or corruption
3. **"Index data missing"**: Index wasn't built before saving
4. **"Out of memory"**: Use compression or save in chunks

### Recovery Strategies

```go
// Recover from corruption
func recoverIndex(engine *gobed.SearchEngine, paths []string) error {
    for _, path := range paths {
        if err := engine.Load(path); err == nil {
            fmt.Printf("Recovered from: %s\n", path)
            return nil
        }
    }
    return fmt.Errorf("recovery failed")
}

// Usage
paths := []string{
    "index.bin",
    "index.bin.backup",
    "./checkpoints/latest/index.bin",
}
recoverIndex(engine, paths)
```

## Future Enhancements

Planned improvements for persistence:

1. **Incremental saves**: Only save changes since last checkpoint
2. **Streaming save/load**: Handle indexes larger than memory
3. **Cloud storage**: Direct S3/GCS support
4. **Version migration**: Automatic format upgrades
5. **Distributed persistence**: Multi-node index sharding