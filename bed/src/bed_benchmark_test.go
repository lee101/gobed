package src

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/lee101/gobed"
	"github.com/lee101/gobed/metrics"
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

// BedEvalQuery represents a test query with known relevant documents
type BedEvalQuery struct {
	Query     string
	Relevance map[string]float64 // docID -> relevance score
}

// BedEvalDataset is a collection of queries for NDCG evaluation
type BedEvalDataset struct {
	Docs    []Document
	Queries []BedEvalQuery
}

// CreateBedEvalDataset creates a small eval dataset for bed search
func CreateBedEvalDataset() *BedEvalDataset {
	docs := []Document{
		{ID: 0, Path: "sort_impl.go", LineNumber: 1, Content: "func quickSort(arr []int) { if len(arr) < 2 { return } pivot := arr[0]; left, right := partition(arr, pivot); quickSort(left); quickSort(right) }"},
		{ID: 1, Path: "sort_merge.go", LineNumber: 1, Content: "func mergeSort(data []int) []int { if len(data) <= 1 { return data } mid := len(data)/2; return merge(mergeSort(data[:mid]), mergeSort(data[mid:])) }"},
		{ID: 2, Path: "http_handler.go", LineNumber: 1, Content: "func handleRequest(w http.ResponseWriter, r *http.Request) { if r.Method != POST { http.Error(w, method not allowed, 405); return } }"},
		{ID: 3, Path: "http_server.go", LineNumber: 1, Content: "func startServer(port int) { http.ListenAndServe(fmt.Sprintf(:%d, port), nil) }"},
		{ID: 4, Path: "db_query.go", LineNumber: 1, Content: "func queryDatabase(ctx context.Context, sql string) (*sql.Rows, error) { return db.QueryContext(ctx, sql) }"},
		{ID: 5, Path: "db_insert.go", LineNumber: 1, Content: "func insertRecord(ctx context.Context, table string, data map[string]any) error { return db.ExecContext(ctx, buildInsert(table, data)) }"},
		{ID: 6, Path: "auth_jwt.go", LineNumber: 1, Content: "func validateJWT(token string) (*Claims, error) { claims := &Claims{}; _, err := jwt.ParseWithClaims(token, claims, keyFunc); return claims, err }"},
		{ID: 7, Path: "auth_password.go", LineNumber: 1, Content: "func hashPassword(password string) (string, error) { hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost); return string(hash), err }"},
		{ID: 8, Path: "cache_get.go", LineNumber: 1, Content: "func getFromCache(key string) ([]byte, bool) { val, ok := cache.Get(key); if !ok { return nil, false }; return val.([]byte), true }"},
		{ID: 9, Path: "cache_set.go", LineNumber: 1, Content: "func setInCache(key string, value []byte, ttl time.Duration) { cache.Set(key, value, ttl) }"},
		{ID: 10, Path: "ml_embedding.go", LineNumber: 1, Content: "func generateEmbedding(text string) []float32 { tokens := tokenizer.Encode(text); return model.Forward(tokens) }"},
		{ID: 11, Path: "ml_inference.go", LineNumber: 1, Content: "func runInference(input []float32) []float32 { output := neural_network.Predict(input); return softmax(output) }"},
		{ID: 12, Path: "vector_search.go", LineNumber: 1, Content: "func searchVectors(query []float32, corpus [][]float32, k int) []int { scores := computeDotProducts(query, corpus); return topK(scores, k) }"},
		{ID: 13, Path: "vector_index.go", LineNumber: 1, Content: "func buildIndex(vectors [][]float32) *IVFIndex { centroids := kmeans(vectors, nlist); return &IVFIndex{centroids: centroids} }"},
		{ID: 14, Path: "file_read.go", LineNumber: 1, Content: "func readFile(path string) ([]byte, error) { return os.ReadFile(path) }"},
		{ID: 15, Path: "file_write.go", LineNumber: 1, Content: "func writeFile(path string, data []byte) error { return os.WriteFile(path, data, 0644) }"},
		{ID: 16, Path: "json_parse.go", LineNumber: 1, Content: "func parseJSON(data []byte, v any) error { return json.Unmarshal(data, v) }"},
		{ID: 17, Path: "json_encode.go", LineNumber: 1, Content: "func encodeJSON(v any) ([]byte, error) { return json.Marshal(v) }"},
		{ID: 18, Path: "gpu_kernel.go", LineNumber: 1, Content: "func launchKernel(ctx *cuda.Context, grid, block dim3) { ctx.Launch(kernel, grid, block) }"},
		{ID: 19, Path: "gpu_memory.go", LineNumber: 1, Content: "func allocGPUMemory(size int) unsafe.Pointer { ptr, _ := cuda.Malloc(size); return ptr }"},
	}

	queries := []BedEvalQuery{
		{Query: "sorting algorithm implementation", Relevance: map[string]float64{"sort_impl.go": 3.0, "sort_merge.go": 3.0}},
		{Query: "HTTP request handling", Relevance: map[string]float64{"http_handler.go": 3.0, "http_server.go": 2.0}},
		{Query: "database operations SQL", Relevance: map[string]float64{"db_query.go": 3.0, "db_insert.go": 3.0}},
		{Query: "authentication and security", Relevance: map[string]float64{"auth_jwt.go": 3.0, "auth_password.go": 3.0}},
		{Query: "caching data storage", Relevance: map[string]float64{"cache_get.go": 3.0, "cache_set.go": 3.0}},
		{Query: "machine learning embeddings neural network", Relevance: map[string]float64{"ml_embedding.go": 3.0, "ml_inference.go": 3.0, "vector_search.go": 2.0}},
		{Query: "vector similarity search index", Relevance: map[string]float64{"vector_search.go": 3.0, "vector_index.go": 3.0, "ml_embedding.go": 1.0}},
		{Query: "file I/O read write", Relevance: map[string]float64{"file_read.go": 3.0, "file_write.go": 3.0}},
		{Query: "JSON serialization parsing", Relevance: map[string]float64{"json_parse.go": 3.0, "json_encode.go": 3.0}},
		{Query: "GPU CUDA kernel memory", Relevance: map[string]float64{"gpu_kernel.go": 3.0, "gpu_memory.go": 3.0}},
	}

	return &BedEvalDataset{Docs: docs, Queries: queries}
}

// TestBedSearchNDCG tests search quality using NDCG@10
func TestBedSearchNDCG(t *testing.T) {
	engine, err := NewOptimizedSearchEngine()
	if err != nil {
		t.Skipf("Engine not available: %v", err)
	}

	ds := CreateBedEvalDataset()

	err = engine.BatchAddDocuments(ds.Docs, 8)
	if err != nil {
		t.Fatalf("Failed to add documents: %v", err)
	}

	t.Logf("Indexed %d documents", engine.NumDocuments())

	var totalNDCG, totalRecall float64
	k := 10

	for _, q := range ds.Queries {
		results, err := engine.Search(q.Query, k, 0.0)
		if err != nil {
			t.Errorf("Search failed for %q: %v", q.Query, err)
			continue
		}

		resultPaths := make([]string, len(results))
		for i, r := range results {
			resultPaths[i] = r.Document.Path
		}

		ndcg := metrics.NDCGAtK(resultPaths, q.Relevance, k)
		totalNDCG += ndcg

		relevant := 0
		for _, path := range resultPaths {
			if q.Relevance[path] > 0 {
				relevant++
			}
		}
		totalRecall += float64(relevant) / float64(len(q.Relevance))

		t.Logf("Query: %q  NDCG: %.3f  Recall: %.2f", q.Query, ndcg, float64(relevant)/float64(len(q.Relevance)))
	}

	n := float64(len(ds.Queries))
	avgNDCG := totalNDCG / n
	avgRecall := totalRecall / n

	t.Logf("Average NDCG@10: %.4f", avgNDCG)
	t.Logf("Average Recall@10: %.4f", avgRecall)

	if avgNDCG < 0.5 {
		t.Errorf("NDCG@10 too low: %.4f (expected >= 0.5)", avgNDCG)
	}
	if avgRecall < 0.5 {
		t.Errorf("Recall@10 too low: %.4f (expected >= 0.5)", avgRecall)
	}
}

// BenchmarkOptimizedSearch benchmarks the optimized search engine
func BenchmarkOptimizedSearch(b *testing.B) {
	engine, err := NewOptimizedSearchEngine()
	if err != nil {
		b.Skipf("Engine not available: %v", err)
	}

	ds := CreateBedEvalDataset()
	engine.BatchAddDocuments(ds.Docs, 8)

	queries := []string{
		"sorting algorithm",
		"HTTP handler",
		"database query",
		"authentication",
		"cache storage",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		_, _ = engine.Search(q, 10, 0.0)
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "QPS")
}

// BenchmarkBedIndexing benchmarks document indexing
func BenchmarkBedIndexing(b *testing.B) {
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}

	docs := CreateBedEvalDataset().Docs

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc := docs[i%len(docs)]
		_, release := model.EmbedFast(doc.Content)
		release()
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "docs/sec")
}

// BenchmarkOptimizedSearchLargeCorpus benchmarks search over larger corpus
func BenchmarkOptimizedSearchLargeCorpus(b *testing.B) {
	engine, err := NewOptimizedSearchEngine()
	if err != nil {
		b.Skipf("Engine not available: %v", err)
	}

	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}

	docs := make([]Document, 1000)
	contents := []string{
		"func sortArray(arr []int) { sort.Ints(arr) }",
		"func handleHTTP(w http.ResponseWriter) { w.Write(ok) }",
		"func queryDB(sql string) error { return db.Query(sql) }",
		"func hash(s string) []byte { return sha256.Sum256(s) }",
		"func readConfig(path string) Config { return parseYAML(path) }",
	}

	for i := 0; i < 1000; i++ {
		docs[i] = Document{
			ID:         i,
			Path:       "file" + string(rune('0'+i%10)) + ".go",
			LineNumber: i + 1,
			Content:    contents[i%len(contents)],
		}
	}

	for i := 0; i < len(docs); i++ {
		emb, err := model.EmbedInt8(docs[i].Content)
		if err == nil {
			engine.AddDocument(docs[i], emb)
		}
	}

	b.Logf("Indexed %d documents", engine.NumDocuments())

	query := "sorting algorithm implementation"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.Search(query, 10, 0.0)
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "QPS")
}
