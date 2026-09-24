package gobed

import (
	"errors"
	"sync"
	"testing"
)

func TestSearchEngineAppendWithID(t *testing.T) {
	model := mustLoadModelForTest(t)
	config := DefaultSearchConfig()
	config.AutoMode = false
	config.EnableGPU = false
	config.MaxExactSearchSize = 50000
	engine := NewSearchEngineWithConfig(model, config)

	if err := engine.AppendWithID(41, "calm piano ambience"); err != nil {
		t.Fatal(err)
	}
	if err := engine.AppendWithID(42, "heavy metal guitar"); err != nil {
		t.Fatal(err)
	}
	if err := engine.AppendWithID(41, "replacement"); !errors.Is(err, ErrDocumentIDExists) {
		t.Fatalf("expected ErrDocumentIDExists, got %v", err)
	}
	if got := engine.Stats().NumDocuments; got != 2 {
		t.Fatalf("expected 2 documents, got %d", got)
	}

	results, err := engine.Search("piano", 2)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, result := range results {
		if result.ID == 41 {
			found = true
		}
	}
	if !found {
		t.Fatalf("appended document missing from results: %+v", results)
	}
}

func TestSearchEngineAppendWithIDConcurrent(t *testing.T) {
	model := mustLoadModelForTest(t)
	config := DefaultSearchConfig()
	config.AutoMode = false
	config.EnableGPU = false
	config.MaxExactSearchSize = 50000
	engine := NewSearchEngineWithConfig(model, config)

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for id, text := range map[int]string{10: "ocean waves", 11: "forest birds"} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- engine.AppendWithID(id, text)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := engine.Stats().NumDocuments; got != 2 {
		t.Fatalf("expected 2 documents, got %d", got)
	}
}

func TestSearchEngineAppendBatchRejectsDuplicateIDs(t *testing.T) {
	engine := &SearchEngine{documents: make(map[int]string)}
	if err := engine.AppendBatchWithIDs([]int{7, 7}, []string{"one", "two"}); !errors.Is(err, ErrDocumentIDExists) {
		t.Fatalf("expected ErrDocumentIDExists, got %v", err)
	}
	if engine.initialized || len(engine.documents) != 0 {
		t.Fatal("duplicate batch changed engine state")
	}
	if err := engine.AppendBatchWithIDs(nil, nil); err != nil {
		t.Fatal(err)
	}
	if engine.initialized {
		t.Fatal("empty batch initialized engine")
	}
}
