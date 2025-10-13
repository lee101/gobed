package src

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var (
	benchOnce   sync.Once
	benchEngine *SimpleSearchEngine
	benchErr    error
)

func prepareBenchmarkEngine() (*SimpleSearchEngine, error) {
	benchOnce.Do(func() {
		engine, err := NewSimpleSearchEngine()
		if err != nil {
			benchErr = err
			return
		}

		testDir := filepath.Join("..", "testdata")
		if err := populateEngineWithTestdata(engine, testDir); err != nil {
			benchErr = err
			return
		}

		benchEngine = engine
	})

	return benchEngine, benchErr
}

func populateEngineWithTestdata(engine *SimpleSearchEngine, dir string) error {
	files, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	engine.mu.Lock()
	defer engine.mu.Unlock()

	engine.documents = engine.documents[:0]
	engine.embeddings = engine.embeddings[:0]

	for _, entry := range files {
		if entry.IsDir() {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		lines := strings.Split(string(data), "\n")
		for idx, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			embedding, err := engine.model.Encode(line)
			if err != nil {
				return err
			}

			doc := Document{
				ID:         len(engine.documents),
				Path:       path,
				LineNumber: idx + 1,
				Content:    line,
			}

			engine.documents = append(engine.documents, doc)
			engine.embeddings = append(engine.embeddings, embedding)
		}
	}

	return nil
}

func BenchmarkBedSearchTestdata(b *testing.B) {
	engine, err := prepareBenchmarkEngine()
	if err != nil {
		b.Fatalf("failed to prepare benchmark engine: %v", err)
	}

	queries := []string{
		"machine learning algorithms",
		"anime characters",
		"database connections",
		"error handling",
		"neural networks",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		matches, err := engine.Search(query, 10, 0.2)
		if err != nil {
			b.Fatalf("search failed: %v", err)
		}
		if len(matches) == 0 {
			b.Fatalf("expected at least one match for query %q", query)
		}
	}

	if b.N > 0 {
		qps := float64(b.N) / b.Elapsed().Seconds()
		b.ReportMetric(qps, "queries_per_second")
	}
}
