package src

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func BenchmarkInt8SearchEngine(b *testing.B) {
	engine, err := NewInt8SearchEngine()
	if err != nil {
		b.Skipf("Could not create engine: %v", err)
	}

	// Generate test documents
	numDocs := 10000
	docs := make([]Document, numDocs)
	for i := 0; i < numDocs; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("file%d.go", i),
			LineNumber: i + 1,
			Content:    generateRandomContent(i),
		}
	}

	// Batch add documents
	b.Run("BatchAdd_10k", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			engine.Clear()
			err := engine.BatchAddDocuments(docs, 0)
			if err != nil {
				b.Fatalf("BatchAddDocuments failed: %v", err)
			}
		}
	})

	// Add docs for search benchmarks
	engine.Clear()
	err = engine.BatchAddDocuments(docs, 0)
	if err != nil {
		b.Fatalf("Setup failed: %v", err)
	}

	queries := []string{
		"function to handle errors",
		"database connection pool",
		"authentication and login",
		"memory allocation",
		"http request handler",
	}

	b.Run("Search_top10_10k_docs", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := queries[i%len(queries)]
			_, err := engine.Search(query, 10, 0.5)
			if err != nil {
				b.Fatalf("Search failed: %v", err)
			}
		}
	})

	b.Run("Search_top100_10k_docs", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := queries[i%len(queries)]
			_, err := engine.Search(query, 100, 0.3)
			if err != nil {
				b.Fatalf("Search failed: %v", err)
			}
		}
	})
}

func BenchmarkFastIndexer(b *testing.B) {
	config := DefaultFastIndexerConfig()
	config.MaxFileSize = 1 << 20 // 1MB
	config.BatchSize = 100

	b.Run("IndexDirectory_small", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			indexer, err := NewFastIndexer(config)
			if err != nil {
				b.Skipf("Could not create indexer: %v", err)
			}
			// Index a small directory
			_ = indexer.IndexDirectory(".", false)
			indexer.Clear()
		}
	})
}

func TestInt8SearchEngine(t *testing.T) {
	engine, err := NewInt8SearchEngine()
	if err != nil {
		t.Skipf("Could not create engine: %v", err)
	}

	// Test basic functionality
	docs := []Document{
		{Path: "main.go", LineNumber: 1, Content: "func main() { fmt.Println(\"Hello\") }"},
		{Path: "handler.go", LineNumber: 10, Content: "func handleRequest(w http.ResponseWriter, r *http.Request) error"},
		{Path: "db.go", LineNumber: 5, Content: "func connectDatabase(url string) (*sql.DB, error)"},
		{Path: "auth.go", LineNumber: 20, Content: "func authenticateUser(username, password string) (*User, error)"},
		{Path: "error.go", LineNumber: 15, Content: "func handleError(err error) { log.Printf(\"error: %v\", err) }"},
	}

	err = engine.BatchAddDocuments(docs, 0)
	if err != nil {
		t.Fatalf("BatchAddDocuments failed: %v", err)
	}

	if engine.NumDocuments() != len(docs) {
		t.Errorf("Expected %d documents, got %d", len(docs), engine.NumDocuments())
	}

	// Test search
	results, err := engine.Search("handle errors", 3, 0.3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one result")
	}

	t.Logf("Search results for 'handle errors':")
	for i, r := range results {
		t.Logf("  %d. %s:%d (%.3f) - %s", i+1, r.Document.Path, r.Document.LineNumber, r.Similarity, r.Document.Content)
	}

	// Test database query
	results, err = engine.Search("database connection", 3, 0.3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	t.Logf("Search results for 'database connection':")
	for i, r := range results {
		t.Logf("  %d. %s:%d (%.3f) - %s", i+1, r.Document.Path, r.Document.LineNumber, r.Similarity, r.Document.Content)
	}
}

func TestSearchPerformance(t *testing.T) {
	engine, err := NewInt8SearchEngine()
	if err != nil {
		t.Skipf("Could not create engine: %v", err)
	}

	// Generate larger dataset
	numDocs := 50000
	docs := make([]Document, numDocs)
	for i := 0; i < numDocs; i++ {
		docs[i] = Document{
			Path:       fmt.Sprintf("src/file%d.go", i),
			LineNumber: i + 1,
			Content:    generateRandomContent(i),
		}
	}

	// Time batch indexing
	start := time.Now()
	err = engine.BatchAddDocuments(docs, 0)
	indexTime := time.Since(start)
	if err != nil {
		t.Fatalf("BatchAddDocuments failed: %v", err)
	}

	t.Logf("Indexed %d documents in %v (%.0f docs/sec)",
		numDocs, indexTime, float64(numDocs)/indexTime.Seconds())

	// Time search
	queries := []string{
		"function to handle errors",
		"database connection pool",
		"authentication and login",
		"memory allocation",
		"http request handler",
	}

	numSearches := 100
	start = time.Now()
	for i := 0; i < numSearches; i++ {
		query := queries[i%len(queries)]
		_, err := engine.Search(query, 10, 0.5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}
	searchTime := time.Since(start)

	t.Logf("Performed %d searches over %d documents in %v (%.0f searches/sec)",
		numSearches, numDocs, searchTime, float64(numSearches)/searchTime.Seconds())
	t.Logf("Average search latency: %v", searchTime/time.Duration(numSearches))
}

// generateRandomContent creates varied test content
func generateRandomContent(seed int) string {
	rand.Seed(int64(seed))

	templates := []string{
		"func process%d() error { return nil }",
		"type Handler%d struct { db *sql.DB }",
		"const MaxRetries%d = 3",
		"var config%d = Config{Timeout: 30}",
		"if err != nil { return fmt.Errorf(\"failed %d: %%w\", err) }",
		"log.Printf(\"processing item %d\")",
		"ctx, cancel := context.WithTimeout(parent, time.Second*%d)",
		"http.HandleFunc(\"/api/%d\", handler)",
		"db.Query(\"SELECT * FROM table%d\")",
		"for i := 0; i < %d; i++ { process(i) }",
	}

	return fmt.Sprintf(templates[seed%len(templates)], seed)
}
