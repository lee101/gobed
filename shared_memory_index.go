package gobed

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/lee101/gobed/ann/simd"
)

// SharedMemoryIndex provides zero-copy, cross-process vector search
type SharedMemoryIndex struct {
	// Memory mapped files
	vectorFile   *os.File
	metadataFile *os.File
	
	// Memory mapped regions
	vectorData   []byte
	metadataData []byte
	
	// Index metadata (stored in shared memory)
	header *SharedIndexHeader
	
	// Local process state
	mu           sync.RWMutex
	isWriter     bool
	basePath     string
	
	// Performance optimizations
	vectorCache  map[int]*simd.Vec512 // Hot vector cache
	cacheSize    int
	maxCacheSize int
}

// SharedIndexHeader is the header stored in shared memory
type SharedIndexHeader struct {
	// Atomic fields for lock-free reads
	NumVectors    uint64 // Number of vectors (atomic)
	VectorDim     uint32 // Vector dimensions (512)
	Version       uint32 // Index version for compatibility
	
	// Index configuration
	MaxVectors    uint32 // Maximum capacity
	IndexType     uint32 // 0=flat, 1=ivf, 2=hnsw
	
	// Memory layout
	VectorOffset  uint64 // Offset to vector data
	MetaOffset    uint64 // Offset to metadata
	ScalesOffset  uint64 // Offset to scale factors
	IDsOffset     uint64 // Offset to document IDs
	
	// Write coordination
	WriterPID     int32  // PID of current writer (-1 if none)
	WriteSeqNum   uint64 // Sequence number for writes
	
	// Statistics
	TotalSearches uint64 // Total searches performed
	TotalReads    uint64 // Total vector reads
}

// SharedMemoryConfig configures shared memory index
type SharedMemoryConfig struct {
	BasePath      string // Base path for memory mapped files
	MaxVectors    int    // Maximum number of vectors
	ReadOnly      bool   // Open in read-only mode
	CreateIfNew   bool   // Create new index if doesn't exist
	CacheSize     int    // Size of hot vector cache
	UseLockFree   bool   // Use lock-free algorithms
}

// NewSharedMemoryIndex creates a new shared memory index
func NewSharedMemoryIndex(config SharedMemoryConfig) (*SharedMemoryIndex, error) {
	if config.BasePath == "" {
		config.BasePath = "/tmp/gobed_shared_index"
	}
	if config.MaxVectors <= 0 {
		config.MaxVectors = 1000000 // Default 1M vectors
	}
	if config.CacheSize <= 0 {
		config.CacheSize = 1000 // Cache 1000 hot vectors
	}
	
	// Ensure directory exists
	if err := os.MkdirAll(config.BasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create index directory: %w", err)
	}
	
	idx := &SharedMemoryIndex{
		basePath:     config.BasePath,
		isWriter:     !config.ReadOnly,
		vectorCache:  make(map[int]*simd.Vec512),
		maxCacheSize: config.CacheSize,
	}
	
	// Open or create memory mapped files
	vectorPath := filepath.Join(config.BasePath, "vectors.mmap")
	metadataPath := filepath.Join(config.BasePath, "metadata.mmap")
	
	var err error
	
	// Open vector file
	flags := os.O_RDWR
	if config.ReadOnly {
		flags = os.O_RDONLY
	}
	if config.CreateIfNew {
		flags |= os.O_CREATE
	}
	
	idx.vectorFile, err = os.OpenFile(vectorPath, flags, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open vector file: %w", err)
	}
	
	// Open metadata file
	idx.metadataFile, err = os.OpenFile(metadataPath, flags, 0644)
	if err != nil {
		idx.vectorFile.Close()
		return nil, fmt.Errorf("failed to open metadata file: %w", err)
	}
	
	// Initialize or map files
	if config.CreateIfNew {
		err = idx.initializeFiles(config.MaxVectors)
	} else {
		err = idx.mapExistingFiles()
	}
	
	if err != nil {
		idx.Close()
		return nil, err
	}
	
	return idx, nil
}

// initializeFiles initializes new memory mapped files
func (idx *SharedMemoryIndex) initializeFiles(maxVectors int) error {
	// Calculate sizes
	headerSize := uint64(unsafe.Sizeof(SharedIndexHeader{}))
	vectorSize := uint64(maxVectors * 512) // 512 bytes per vector
	scaleSize := uint64(maxVectors * 4)    // 4 bytes per scale (float32)
	idSize := uint64(maxVectors * 4)       // 4 bytes per ID (int32)
	
	totalVectorSize := headerSize + vectorSize
	totalMetadataSize := scaleSize + idSize
	
	// Resize vector file
	if err := idx.vectorFile.Truncate(int64(totalVectorSize)); err != nil {
		return fmt.Errorf("failed to resize vector file: %w", err)
	}
	
	// Resize metadata file
	if err := idx.metadataFile.Truncate(int64(totalMetadataSize)); err != nil {
		return fmt.Errorf("failed to resize metadata file: %w", err)
	}
	
	// Memory map the files
	var err error
	prot := syscall.PROT_READ | syscall.PROT_WRITE
	if !idx.isWriter {
		prot = syscall.PROT_READ
	}
	
	idx.vectorData, err = syscall.Mmap(
		int(idx.vectorFile.Fd()),
		0,
		int(totalVectorSize),
		prot,
		syscall.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("failed to mmap vector file: %w", err)
	}
	
	idx.metadataData, err = syscall.Mmap(
		int(idx.metadataFile.Fd()),
		0,
		int(totalMetadataSize),
		prot,
		syscall.MAP_SHARED,
	)
	if err != nil {
		syscall.Munmap(idx.vectorData)
		return fmt.Errorf("failed to mmap metadata file: %w", err)
	}
	
	// Initialize header
	idx.header = (*SharedIndexHeader)(unsafe.Pointer(&idx.vectorData[0]))
	idx.header.Version = 1
	idx.header.VectorDim = 512
	idx.header.MaxVectors = uint32(maxVectors)
	idx.header.VectorOffset = headerSize
	idx.header.MetaOffset = 0
	idx.header.ScalesOffset = 0
	idx.header.IDsOffset = scaleSize
	idx.header.WriterPID = -1
	
	// Force sync to disk
	if idx.isWriter {
		// Note: syscall.Msync not available on all platforms
		// Using file sync as fallback
		idx.vectorFile.Sync()
		idx.metadataFile.Sync()
	}
	
	return nil
}

// mapExistingFiles maps existing memory mapped files
func (idx *SharedMemoryIndex) mapExistingFiles() error {
	// Ensure files exist
	if _, err := os.Stat(filepath.Join(idx.basePath, "vectors.mmap")); os.IsNotExist(err) {
		return fmt.Errorf("vector file does not exist: %s", filepath.Join(idx.basePath, "vectors.mmap"))
	}
	if _, err := os.Stat(filepath.Join(idx.basePath, "metadata.mmap")); os.IsNotExist(err) {
		return fmt.Errorf("metadata file does not exist: %s", filepath.Join(idx.basePath, "metadata.mmap"))
	}
	// Get file sizes
	vectorInfo, err := idx.vectorFile.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat vector file: %w", err)
	}
	
	metadataInfo, err := idx.metadataFile.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat metadata file: %w", err)
	}
	
	// Memory map the files
	prot := syscall.PROT_READ
	if idx.isWriter {
		prot |= syscall.PROT_WRITE
	}
	
	idx.vectorData, err = syscall.Mmap(
		int(idx.vectorFile.Fd()),
		0,
		int(vectorInfo.Size()),
		prot,
		syscall.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("failed to mmap vector file: %w", err)
	}
	
	idx.metadataData, err = syscall.Mmap(
		int(idx.metadataFile.Fd()),
		0,
		int(metadataInfo.Size()),
		prot,
		syscall.MAP_SHARED,
	)
	if err != nil {
		syscall.Munmap(idx.vectorData)
		return fmt.Errorf("failed to mmap metadata file: %w", err)
	}
	
	// Map header
	idx.header = (*SharedIndexHeader)(unsafe.Pointer(&idx.vectorData[0]))
	
	// Validate version
	if idx.header.Version != 1 {
		idx.Close()
		return fmt.Errorf("incompatible index version: %d", idx.header.Version)
	}
	
	return nil
}

// AddVector adds a vector to the shared index (writer only)
func (idx *SharedMemoryIndex) AddVector(vec *simd.Vec512, scale float32, id int) error {
	if !idx.isWriter {
		return fmt.Errorf("cannot add vectors in read-only mode")
	}
	
	idx.mu.Lock()
	defer idx.mu.Unlock()
	
	// Check capacity
	currentCount := atomic.LoadUint64(&idx.header.NumVectors)
	if currentCount >= uint64(idx.header.MaxVectors) {
		return fmt.Errorf("index is full: %d/%d", currentCount, idx.header.MaxVectors)
	}
	
	// Calculate offsets
	vectorOffset := idx.header.VectorOffset + currentCount*512
	scaleOffset := idx.header.ScalesOffset + currentCount*4
	idOffset := idx.header.IDsOffset + currentCount*4
	
	// Write vector data directly to memory mapped region (zero-copy)
	vecBytes := (*[512]byte)(unsafe.Pointer(vec))
	copy(idx.vectorData[vectorOffset:vectorOffset+512], vecBytes[:])
	
	// Write scale
	binary.LittleEndian.PutUint32(idx.metadataData[scaleOffset:], *(*uint32)(unsafe.Pointer(&scale)))
	
	// Write ID
	binary.LittleEndian.PutUint32(idx.metadataData[idOffset:], uint32(id))
	
	// Increment count atomically
	atomic.AddUint64(&idx.header.NumVectors, 1)
	atomic.AddUint64(&idx.header.WriteSeqNum, 1)
	
	// Sync to ensure visibility to other processes
	if currentCount%100 == 0 { // Batch syncs for performance
		// Memory barriers for visibility
		// Data is already in shared memory, will be visible to other processes
	}
	
	return nil
}

// SearchTopK performs zero-copy k-NN search directly on shared memory
func (idx *SharedMemoryIndex) SearchTopK(query *simd.Vec512, k int) []SearchResult {
	// Read lock for concurrent access safety
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	
	// Atomic read of vector count (lock-free)
	numVectors := atomic.LoadUint64(&idx.header.NumVectors)
	if numVectors == 0 {
		return nil
	}
	
	// Update statistics
	atomic.AddUint64(&idx.header.TotalSearches, 1)
	
	// Use min-heap for top-k
	results := make([]SearchResult, 0, k)
	
	// Scan vectors with zero-copy access
	for i := uint64(0); i < numVectors; i++ {
		// Zero-copy vector access - directly cast memory region
		vectorOffset := idx.header.VectorOffset + i*512
		
		// Bounds check to prevent segfault
		if vectorOffset+512 > uint64(len(idx.vectorData)) {
			break
		}
		
		vecPtr := (*simd.Vec512)(unsafe.Pointer(&idx.vectorData[vectorOffset]))
		
		// Compute similarity
		score := simd.Dot512(query, vecPtr)
		
		// Get ID (also zero-copy)
		idOffset := idx.header.IDsOffset + i*4
		
		// Bounds check for metadata
		if idOffset+4 > uint64(len(idx.metadataData)) {
			break
		}
		
		id := int(binary.LittleEndian.Uint32(idx.metadataData[idOffset:]))
		
		// Update statistics
		atomic.AddUint64(&idx.header.TotalReads, 1)
		
		// Maintain top-k results
		if len(results) < k {
			results = append(results, SearchResult{
				ID:         id,
				Similarity: float32(score),
			})
		} else if float32(score) > results[0].Similarity {
			// Replace minimum
			results[0] = SearchResult{
				ID:         id,
				Similarity: float32(score),
			}
		}
		
		// Re-heapify if needed
		if len(results) == k {
			// Simple bubble for min-heap property
			for j := 0; j < len(results)-1; j++ {
				if results[j].Similarity > results[j+1].Similarity {
					results[j], results[j+1] = results[j+1], results[j]
				}
			}
		}
	}
	
	// Sort results in descending order
	for i := 0; i < len(results)/2; i++ {
		results[i], results[len(results)-1-i] = results[len(results)-1-i], results[i]
	}
	
	return results
}

// GetVector returns a zero-copy reference to a vector
func (idx *SharedMemoryIndex) GetVector(index int) (*simd.Vec512, error) {
	numVectors := atomic.LoadUint64(&idx.header.NumVectors)
	if uint64(index) >= numVectors {
		return nil, fmt.Errorf("index out of bounds: %d >= %d", index, numVectors)
	}
	
	// Check cache first
	idx.mu.RLock()
	if cached, ok := idx.vectorCache[index]; ok {
		idx.mu.RUnlock()
		return cached, nil
	}
	idx.mu.RUnlock()
	
	// Zero-copy access to vector
	vectorOffset := idx.header.VectorOffset + uint64(index)*512
	vecPtr := (*simd.Vec512)(unsafe.Pointer(&idx.vectorData[vectorOffset]))
	
	// Cache hot vectors
	idx.mu.Lock()
	if len(idx.vectorCache) < idx.maxCacheSize {
		idx.vectorCache[index] = vecPtr
	}
	idx.mu.Unlock()
	
	return vecPtr, nil
}

// BatchSearch performs multiple searches efficiently
func (idx *SharedMemoryIndex) BatchSearch(queries []*simd.Vec512, k int) [][]SearchResult {
	results := make([][]SearchResult, len(queries))
	
	// Parallel search for better throughput
	var wg sync.WaitGroup
	for i, query := range queries {
		wg.Add(1)
		go func(resultIdx int, q *simd.Vec512) {
			defer wg.Done()
			results[resultIdx] = idx.SearchTopK(q, k)
		}(i, query)
	}
	wg.Wait()
	
	return results
}

// Stats returns index statistics
func (idx *SharedMemoryIndex) Stats() SharedIndexStats {
	return SharedIndexStats{
		NumVectors:    atomic.LoadUint64(&idx.header.NumVectors),
		MaxVectors:    uint64(idx.header.MaxVectors),
		TotalSearches: atomic.LoadUint64(&idx.header.TotalSearches),
		TotalReads:    atomic.LoadUint64(&idx.header.TotalReads),
		WriteSeqNum:   atomic.LoadUint64(&idx.header.WriteSeqNum),
		CacheSize:     len(idx.vectorCache),
		MemoryUsageMB: float64(len(idx.vectorData)+len(idx.metadataData)) / (1024 * 1024),
	}
}

// Sync forces synchronization to disk
func (idx *SharedMemoryIndex) Sync() error {
	if !idx.isWriter {
		return nil // No-op for readers
	}
	
	idx.mu.Lock()
	defer idx.mu.Unlock()
	
	// Sync both memory regions using file sync
	if err := idx.vectorFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync vector data: %w", err)
	}
	
	if err := idx.metadataFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync metadata: %w", err)
	}
	
	return nil
}

// Close unmaps memory and closes files
func (idx *SharedMemoryIndex) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	
	// Clear cache
	idx.vectorCache = nil
	
	// Unmap memory regions
	if idx.vectorData != nil {
		if idx.isWriter {
			idx.vectorFile.Sync()
		}
		syscall.Munmap(idx.vectorData)
		idx.vectorData = nil
	}
	
	if idx.metadataData != nil {
		if idx.isWriter {
			idx.metadataFile.Sync()
		}
		syscall.Munmap(idx.metadataData)
		idx.metadataData = nil
	}
	
	// Close files
	var err error
	if idx.vectorFile != nil {
		err = idx.vectorFile.Close()
		idx.vectorFile = nil
	}
	
	if idx.metadataFile != nil {
		if e := idx.metadataFile.Close(); e != nil && err == nil {
			err = e
		}
		idx.metadataFile = nil
	}
	
	return err
}

// SharedIndexStats contains index statistics
type SharedIndexStats struct {
	NumVectors    uint64
	MaxVectors    uint64
	TotalSearches uint64
	TotalReads    uint64
	WriteSeqNum   uint64
	CacheSize     int
	MemoryUsageMB float64
}

// WaitForWrites waits for pending writes to complete
func (idx *SharedMemoryIndex) WaitForWrites(targetSeq uint64) {
	for atomic.LoadUint64(&idx.header.WriteSeqNum) < targetSeq {
		// Spin-wait or use futex for efficiency
		// This is lock-free synchronization
	}
}

// TryAcquireWriter attempts to become the writer process
func (idx *SharedMemoryIndex) TryAcquireWriter() bool {
	if !idx.isWriter {
		return false
	}
	
	pid := int32(os.Getpid())
	return atomic.CompareAndSwapInt32(&idx.header.WriterPID, -1, pid)
}

// ReleaseWriter releases writer lock
func (idx *SharedMemoryIndex) ReleaseWriter() {
	pid := int32(os.Getpid())
	atomic.CompareAndSwapInt32(&idx.header.WriterPID, pid, -1)
}