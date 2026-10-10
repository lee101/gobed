package gobed

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/lee101/gobed/pkg/ann/simd"
)

type PreparedAppend struct {
	owner   *SearchEngine
	ids     []int
	texts   []string
	vectors []simd.Vec512
	scales  []float32
}

func (se *SearchEngine) PrepareAppendBatchWithIDs(ids []int, texts []string) (*PreparedAppend, error) {
	if len(ids) != len(texts) {
		return nil, fmt.Errorf("ids and texts must have the same length")
	}
	prepared := &PreparedAppend{owner: se}
	if len(ids) == 0 {
		return prepared, nil
	}

	se.mu.RLock()
	err := se.validateAppendIDsLocked(ids)
	se.mu.RUnlock()
	if err != nil {
		return nil, err
	}

	prepared.ids = append([]int(nil), ids...)
	prepared.texts = append([]string(nil), texts...)
	prepared.vectors = make([]simd.Vec512, len(texts))
	prepared.scales = make([]float32, len(texts))

	workers := min(runtime.NumCPU(), min(8, len(texts)))
	type result struct {
		position  int
		embedding *EmbedInt8Result
		err       error
	}
	jobs := make(chan int, len(texts))
	results := make(chan result, len(texts))
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for position := range jobs {
				text := texts[position]
				if cached, found := se.embeddingCache.Get(text); found {
					results <- result{position: position, embedding: cached}
					continue
				}
				embedding, err := se.model.EmbedInt8(text)
				if err == nil {
					se.embeddingCache.Put(text, embedding)
				}
				results <- result{position: position, embedding: embedding, err: err}
			}
		}()
	}
	for i := range texts {
		jobs <- i
	}
	close(jobs)
	go func() {
		wg.Wait()
		close(results)
	}()

	var firstErr error
	for result := range results {
		if result.err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("failed to embed text %d: %v", result.position, result.err)
			}
			continue
		}
		copy(prepared.vectors[result.position][:], result.embedding.Vector)
		prepared.scales[result.position] = result.embedding.Scale
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return prepared, nil
}

func (se *SearchEngine) CommitPreparedAppend(prepared *PreparedAppend) error {
	if prepared == nil || prepared.owner != se {
		return fmt.Errorf("prepared append belongs to another search engine")
	}
	if len(prepared.ids) == 0 {
		return nil
	}

	se.mu.Lock()
	defer se.mu.Unlock()
	if err := se.validateAppendIDsLocked(prepared.ids); err != nil {
		return err
	}

	for i, id := range prepared.ids {
		se.documents[id] = prepared.texts[i]
	}
	rollback := func() {
		for _, id := range prepared.ids {
			delete(se.documents, id)
		}
	}

	if !se.initialized {
		if err := se.initializeIndex(len(se.documents)); err != nil {
			rollback()
			return fmt.Errorf("failed to initialize index: %v", err)
		}
	}
	if err := se.index.AddBatch(prepared.vectors, prepared.scales, prepared.ids); err != nil {
		rollback()
		return err
	}
	return nil
}

func (se *SearchEngine) validateAppendIDsLocked(ids []int) error {
	seen := make(map[int]struct{}, len(ids))
	for _, id := range ids {
		if _, exists := seen[id]; exists {
			return fmt.Errorf("%w: %d", ErrDocumentIDExists, id)
		}
		seen[id] = struct{}{}
		if _, exists := se.documents[id]; exists {
			return fmt.Errorf("%w: %d", ErrDocumentIDExists, id)
		}
	}
	return nil
}
