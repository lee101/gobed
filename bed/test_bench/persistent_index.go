package main

import (
	"crypto/md5"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_persistent -L/usr/local/cuda/lib64 -lcudart -lcublas
// #include <stdlib.h>
// extern void* cuda_persistent_index_create(int max_docs, int dim);
// extern void cuda_persistent_index_destroy(void* handle);
// extern int cuda_persistent_index_update(void* handle, const signed char* vectors, int* doc_ids, int num_docs);
// extern int cuda_persistent_index_remove(void* handle, int* doc_ids, int num_docs);
// extern int cuda_persistent_index_search(void* handle, const signed char* query, int k, int* indices, float* scores);
// extern int cuda_persistent_index_get_stats(void* handle, int* active_docs, int* capacity, float* gpu_memory_mb);
import "C"

// FileInfo tracks file metadata for change detection
type FileInfo struct {
	Path         string
	Size         int64
	ModTime      time.Time
	Hash         [16]byte // MD5 hash of content
	DocIDs       []int32  // Document IDs in index for this file
	LastIndexed  time.Time
	NeedsReindex bool
}

// PersistentIndex maintains a GPU-resident search index
type PersistentIndex struct {
	// GPU state
	gpuHandle    unsafe.Pointer
	maxDocs      int
	activeDocs   int32 // atomic counter
	docIDCounter int32 // atomic counter

	// File tracking
	fileIndex map[string]*FileInfo
	fileMutex sync.RWMutex

	// Document mapping
	docToFile map[int32]string // docID -> filepath
	docMutex  sync.RWMutex

	// Processing queues
	indexQueue   chan *IndexJob
	searchQueue  chan *SearchRequest
	removeQueue  chan string

	// Worker pools
	fileWorkers   int
	indexWorkers  int
	searchWorkers int

	// Model
	model *FastModel

	// Stats
	stats struct {
		FilesProcessed  uint64
		DocsIndexed     uint64
		SearchesHandled uint64
		ReindexCount    uint64
	}

	// Control
	shutdown chan struct{}
	wg       sync.WaitGroup
}

// IndexJob represents a file to index
type IndexJob struct {
	Path      string
	Info      fs.FileInfo
	Force     bool // Force reindex even if unchanged
	Callback  func(error)
}

// SearchRequest represents a search query
type SearchRequest struct {
	Query    string
	TopK     int
	Response chan *SearchResponse
}

// SearchResponse contains search results
type SearchResponse struct {
	Results []SearchResult
	Time    time.Duration
	Error   error
}

// SearchResult represents a single search result
type SearchResult struct {
	FilePath  string
	StartLine int
	EndLine   int
	Content   string
	Score     float32
}

// NewPersistentIndex creates a new persistent GPU index
func NewPersistentIndex(maxDocs int, model *FastModel) *PersistentIndex {
	idx := &PersistentIndex{
		maxDocs:       maxDocs,
		fileIndex:     make(map[string]*FileInfo),
		docToFile:     make(map[int32]string),
		indexQueue:    make(chan *IndexJob, 1000),
		searchQueue:   make(chan *SearchRequest, 100),
		removeQueue:   make(chan string, 100),
		fileWorkers:   runtime.NumCPU(),
		indexWorkers:  4,
		searchWorkers: 2,
		model:         model,
		shutdown:      make(chan struct{}),
	}

	// Create GPU index
	idx.gpuHandle = C.cuda_persistent_index_create(C.int(maxDocs), C.int(512))
	if idx.gpuHandle == nil {
		panic("Failed to create GPU persistent index")
	}

	// Start worker pools
	idx.startWorkers()

	return idx
}

// startWorkers launches all worker goroutines
func (idx *PersistentIndex) startWorkers() {
	// File reading workers (CPU-bound, parallel)
	for i := 0; i < idx.fileWorkers; i++ {
		idx.wg.Add(1)
		go idx.fileWorker(i)
	}

	// Indexing worker (GPU-bound, single)
	idx.wg.Add(1)
	go idx.indexingWorker()

	// Search workers (GPU-bound but can use CUDA streams)
	for i := 0; i < idx.searchWorkers; i++ {
		idx.wg.Add(1)
		go idx.searchWorker(i)
	}

	// File watcher
	idx.wg.Add(1)
	go idx.fileWatcher()

	// Stats reporter
	idx.wg.Add(1)
	go idx.statsReporter()
}

// fileWorker processes files in parallel
func (idx *PersistentIndex) fileWorker(id int) {
	defer idx.wg.Done()

	chunker := NewIntelligentChunker(idx.model, DefaultChunkingConfig())

	for {
		select {
		case job := <-idx.indexQueue:
			idx.processFile(job, chunker)
		case <-idx.shutdown:
			return
		}
	}
}

// processFile reads and chunks a file
func (idx *PersistentIndex) processFile(job *IndexJob, chunker *IntelligentChunker) {
	// Check if file needs reindexing
	idx.fileMutex.RLock()
	fileInfo, exists := idx.fileIndex[job.Path]
	idx.fileMutex.RUnlock()

	if exists && !job.Force {
		// Check if file has changed
		if !fileInfo.NeedsReindex && fileInfo.ModTime == job.Info.ModTime() &&
			fileInfo.Size == job.Info.Size() {
			if job.Callback != nil {
				job.Callback(nil)
			}
			return
		}
	}

	// Read and chunk file
	chunks, err := chunker.ChunkFile(job.Path)
	if err != nil {
		if job.Callback != nil {
			job.Callback(err)
		}
		return
	}

	// Remove old documents if reindexing
	if exists && len(fileInfo.DocIDs) > 0 {
		idx.removeDocuments(fileInfo.DocIDs)
	}

	// Prepare batch for GPU indexing
	numChunks := len(chunks)
	if numChunks == 0 {
		return
	}

	// Allocate document IDs
	startID := atomic.AddInt32(&idx.docIDCounter, int32(numChunks)) - int32(numChunks)
	docIDs := make([]int32, numChunks)
	embeddings := make([]int8, numChunks*512)

	for i, chunk := range chunks {
		docID := startID + int32(i)
		docIDs[i] = docID

		// Copy embedding
		copy(embeddings[i*512:(i+1)*512], chunk.Embedding)

		// Update document mapping
		idx.docMutex.Lock()
		idx.docToFile[docID] = job.Path
		idx.docMutex.Unlock()
	}

	// Send to GPU indexing queue
	idx.addDocuments(embeddings, docIDs)

	// Update file info
	idx.fileMutex.Lock()
	if fileInfo == nil {
		fileInfo = &FileInfo{Path: job.Path}
		idx.fileIndex[job.Path] = fileInfo
	}
	fileInfo.Size = job.Info.Size()
	fileInfo.ModTime = job.Info.ModTime()
	fileInfo.DocIDs = docIDs
	fileInfo.LastIndexed = time.Now()
	fileInfo.NeedsReindex = false
	idx.fileMutex.Unlock()

	// Update stats
	atomic.AddUint64(&idx.stats.FilesProcessed, 1)
	atomic.AddUint64(&idx.stats.DocsIndexed, uint64(numChunks))

	if job.Callback != nil {
		job.Callback(nil)
	}
}

// indexingWorker handles GPU indexing operations
func (idx *PersistentIndex) indexingWorker() {
	defer idx.wg.Done()

	// Batch indexing for efficiency
	const batchSize = 100
	const batchTimeout = 100 * time.Millisecond

	embedBatch := make([]int8, 0, batchSize*512)
	idBatch := make([]int32, 0, batchSize)
	timer := time.NewTimer(batchTimeout)

	flushBatch := func() {
		if len(idBatch) == 0 {
			return
		}

		// Add to GPU index
		C.cuda_persistent_index_update(
			idx.gpuHandle,
			(*C.schar)(unsafe.Pointer(&embedBatch[0])),
			(*C.int)(unsafe.Pointer(&idBatch[0])),
			C.int(len(idBatch)),
		)

		// Update active document count
		atomic.AddInt32(&idx.activeDocs, int32(len(idBatch)))

		// Clear batch
		embedBatch = embedBatch[:0]
		idBatch = idBatch[:0]
	}

	for {
		select {
		case <-timer.C:
			flushBatch()
			timer.Reset(batchTimeout)

		case <-idx.shutdown:
			flushBatch()
			return
		}
	}
}

// searchWorker handles search queries
func (idx *PersistentIndex) searchWorker(id int) {
	defer idx.wg.Done()

	for {
		select {
		case req := <-idx.searchQueue:
			idx.handleSearch(req)
		case <-idx.shutdown:
			return
		}
	}
}

// handleSearch processes a search request
func (idx *PersistentIndex) handleSearch(req *SearchRequest) {
	start := time.Now()

	// Generate query embedding
	queryEmb, err := idx.model.EmbedInt8(req.Query)
	if err != nil {
		req.Response <- &SearchResponse{Error: err}
		return
	}

	// Perform GPU search
	indices := make([]int32, req.TopK)
	scores := make([]float32, req.TopK)

	numResults := C.cuda_persistent_index_search(
		idx.gpuHandle,
		(*C.schar)(unsafe.Pointer(&queryEmb[0])),
		C.int(req.TopK),
		(*C.int)(unsafe.Pointer(&indices[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)

	// Convert results
	results := make([]SearchResult, 0, numResults)
	for i := 0; i < int(numResults); i++ {
		docID := indices[i]
		if docID < 0 {
			break
		}

		idx.docMutex.RLock()
		filePath, exists := idx.docToFile[docID]
		idx.docMutex.RUnlock()

		if exists {
			results = append(results, SearchResult{
				FilePath: filePath,
				Score:    scores[i],
			})
		}
	}

	// Update stats
	atomic.AddUint64(&idx.stats.SearchesHandled, 1)

	req.Response <- &SearchResponse{
		Results: results,
		Time:    time.Since(start),
		Error:   nil,
	}
}

// fileWatcher monitors filesystem changes
func (idx *PersistentIndex) fileWatcher() {
	defer idx.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			idx.checkForChanges()
		case <-idx.shutdown:
			return
		}
	}
}

// checkForChanges scans for modified files
func (idx *PersistentIndex) checkForChanges() {
	idx.fileMutex.RLock()
	filesToCheck := make([]string, 0, len(idx.fileIndex))
	for path := range idx.fileIndex {
		filesToCheck = append(filesToCheck, path)
	}
	idx.fileMutex.RUnlock()

	for _, path := range filesToCheck {
		info, err := os.Stat(path)
		if err != nil {
			if os.IsNotExist(err) {
				// File deleted, remove from index
				idx.RemoveFile(path)
			}
			continue
		}

		idx.fileMutex.RLock()
		fileInfo := idx.fileIndex[path]
		idx.fileMutex.RUnlock()

		if fileInfo.ModTime != info.ModTime() || fileInfo.Size != info.Size() {
			// File changed, mark for reindex
			idx.fileMutex.Lock()
			fileInfo.NeedsReindex = true
			idx.fileMutex.Unlock()

			// Queue for reindexing
			idx.IndexFile(path, false)
			atomic.AddUint64(&idx.stats.ReindexCount, 1)
		}
	}
}

// statsReporter periodically reports statistics
func (idx *PersistentIndex) statsReporter() {
	defer idx.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			idx.printStats()
		case <-idx.shutdown:
			return
		}
	}
}

// Helper methods for queue operations
func (idx *PersistentIndex) addDocuments(embeddings []int8, docIDs []int32) {
	// This would send to the indexing worker's batch queue
	// Implementation depends on batching strategy
}

func (idx *PersistentIndex) removeDocuments(docIDs []int32) {
	C.cuda_persistent_index_remove(
		idx.gpuHandle,
		(*C.int)(unsafe.Pointer(&docIDs[0])),
		C.int(len(docIDs)),
	)
	atomic.AddInt32(&idx.activeDocs, -int32(len(docIDs)))
}

// Public API methods

// IndexDirectory indexes all files in a directory
func (idx *PersistentIndex) IndexDirectory(dir string, extensions []string) error {
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}

		// Check extension
		ext := filepath.Ext(path)
		valid := false
		for _, allowedExt := range extensions {
			if ext == allowedExt {
				valid = true
				break
			}
		}

		if valid {
			info, _ := d.Info()
			if info.Size() < 50*1024*1024 { // 50MB limit
				idx.indexQueue <- &IndexJob{
					Path: path,
					Info: info,
				}
			}
		}

		return nil
	})
}

// IndexFile queues a file for indexing
func (idx *PersistentIndex) IndexFile(path string, force bool) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}

	idx.indexQueue <- &IndexJob{
		Path:  path,
		Info:  info,
		Force: force,
	}

	return nil
}

// RemoveFile removes a file from the index
func (idx *PersistentIndex) RemoveFile(path string) {
	idx.fileMutex.Lock()
	fileInfo, exists := idx.fileIndex[path]
	if exists {
		delete(idx.fileIndex, path)
	}
	idx.fileMutex.Unlock()

	if exists && len(fileInfo.DocIDs) > 0 {
		idx.removeDocuments(fileInfo.DocIDs)

		// Clean up document mapping
		idx.docMutex.Lock()
		for _, docID := range fileInfo.DocIDs {
			delete(idx.docToFile, docID)
		}
		idx.docMutex.Unlock()
	}
}

// Search performs a semantic search
func (idx *PersistentIndex) Search(query string, topK int) (*SearchResponse, error) {
	req := &SearchRequest{
		Query:    query,
		TopK:     topK,
		Response: make(chan *SearchResponse, 1),
	}

	idx.searchQueue <- req

	select {
	case resp := <-req.Response:
		return resp, resp.Error
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("search timeout")
	}
}

// GetStats returns current statistics
func (idx *PersistentIndex) GetStats() map[string]interface{} {
	var activeDocs, capacity C.int
	var gpuMemory C.float

	C.cuda_persistent_index_get_stats(
		idx.gpuHandle,
		&activeDocs,
		&capacity,
		&gpuMemory,
	)

	return map[string]interface{}{
		"active_docs":      int(activeDocs),
		"capacity":         int(capacity),
		"gpu_memory_mb":    float32(gpuMemory),
		"files_processed":  atomic.LoadUint64(&idx.stats.FilesProcessed),
		"docs_indexed":     atomic.LoadUint64(&idx.stats.DocsIndexed),
		"searches_handled": atomic.LoadUint64(&idx.stats.SearchesHandled),
		"reindex_count":    atomic.LoadUint64(&idx.stats.ReindexCount),
		"files_tracked":    len(idx.fileIndex),
	}
}

// printStats outputs current statistics
func (idx *PersistentIndex) printStats() {
	stats := idx.GetStats()
	fmt.Printf("\n📊 Persistent Index Stats:\n")
	fmt.Printf("  Active Docs: %d / %d\n", stats["active_docs"], stats["capacity"])
	fmt.Printf("  GPU Memory: %.1f MB\n", stats["gpu_memory_mb"])
	fmt.Printf("  Files: %d tracked, %d processed\n", stats["files_tracked"], stats["files_processed"])
	fmt.Printf("  Operations: %d searches, %d reindexes\n", stats["searches_handled"], stats["reindex_count"])
}

// Shutdown gracefully stops the index
func (idx *PersistentIndex) Shutdown() {
	close(idx.shutdown)
	idx.wg.Wait()
	C.cuda_persistent_index_destroy(idx.gpuHandle)
}