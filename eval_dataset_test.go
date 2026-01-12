package gobed

import (
	"testing"

	"github.com/lee101/gobed/metrics"
)

// EvalQuery represents a test query with known relevant documents
type EvalQuery struct {
	Query     string
	Relevance map[string]float64 // docID -> relevance score (higher = more relevant)
}

// EvalDataset is a collection of queries for NDCG evaluation
type EvalDataset struct {
	Name    string
	Queries []EvalQuery
	Docs    map[string]string // docID -> text
}

// CreateCodeSearchDataset creates a small eval dataset for code search
func CreateCodeSearchDataset() *EvalDataset {
	docs := map[string]string{
		"sort_impl":      "func quickSort(arr []int) { if len(arr) < 2 { return } pivot := arr[0]; left, right := partition(arr, pivot); quickSort(left); quickSort(right) }",
		"sort_merge":     "func mergeSort(data []int) []int { if len(data) <= 1 { return data } mid := len(data)/2; return merge(mergeSort(data[:mid]), mergeSort(data[mid:])) }",
		"http_handler":   "func handleRequest(w http.ResponseWriter, r *http.Request) { if r.Method != POST { http.Error(w, method not allowed, 405); return } }",
		"http_server":    "func startServer(port int) { http.ListenAndServe(fmt.Sprintf(:%d, port), nil) }",
		"db_query":       "func queryDatabase(ctx context.Context, sql string) (*sql.Rows, error) { return db.QueryContext(ctx, sql) }",
		"db_insert":      "func insertRecord(ctx context.Context, table string, data map[string]any) error { return db.ExecContext(ctx, buildInsert(table, data)) }",
		"auth_jwt":       "func validateJWT(token string) (*Claims, error) { claims := &Claims{}; _, err := jwt.ParseWithClaims(token, claims, keyFunc); return claims, err }",
		"auth_password":  "func hashPassword(password string) (string, error) { hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost); return string(hash), err }",
		"cache_get":      "func getFromCache(key string) ([]byte, bool) { val, ok := cache.Get(key); if !ok { return nil, false }; return val.([]byte), true }",
		"cache_set":      "func setInCache(key string, value []byte, ttl time.Duration) { cache.Set(key, value, ttl) }",
		"ml_embedding":   "func generateEmbedding(text string) []float32 { tokens := tokenizer.Encode(text); return model.Forward(tokens) }",
		"ml_inference":   "func runInference(input []float32) []float32 { output := neural_network.Predict(input); return softmax(output) }",
		"vector_search":  "func searchVectors(query []float32, corpus [][]float32, k int) []int { scores := computeDotProducts(query, corpus); return topK(scores, k) }",
		"vector_index":   "func buildIndex(vectors [][]float32) *IVFIndex { centroids := kmeans(vectors, nlist); return &IVFIndex{centroids: centroids} }",
		"file_read":      "func readFile(path string) ([]byte, error) { return os.ReadFile(path) }",
		"file_write":     "func writeFile(path string, data []byte) error { return os.WriteFile(path, data, 0644) }",
		"json_parse":     "func parseJSON(data []byte, v any) error { return json.Unmarshal(data, v) }",
		"json_encode":    "func encodeJSON(v any) ([]byte, error) { return json.Marshal(v) }",
		"gpu_kernel":     "func launchKernel(ctx *cuda.Context, grid, block dim3) { ctx.Launch(kernel, grid, block) }",
		"gpu_memory":     "func allocGPUMemory(size int) unsafe.Pointer { ptr, _ := cuda.Malloc(size); return ptr }",
	}

	queries := []EvalQuery{
		{
			Query: "sorting algorithm implementation",
			Relevance: map[string]float64{
				"sort_impl":  3.0, // highly relevant
				"sort_merge": 3.0, // highly relevant
			},
		},
		{
			Query: "HTTP request handling",
			Relevance: map[string]float64{
				"http_handler": 3.0,
				"http_server":  2.0,
			},
		},
		{
			Query: "database operations SQL",
			Relevance: map[string]float64{
				"db_query":  3.0,
				"db_insert": 3.0,
			},
		},
		{
			Query: "authentication and security",
			Relevance: map[string]float64{
				"auth_jwt":      3.0,
				"auth_password": 3.0,
			},
		},
		{
			Query: "caching data storage",
			Relevance: map[string]float64{
				"cache_get": 3.0,
				"cache_set": 3.0,
			},
		},
		{
			Query: "machine learning embeddings neural network",
			Relevance: map[string]float64{
				"ml_embedding":  3.0,
				"ml_inference":  3.0,
				"vector_search": 2.0,
			},
		},
		{
			Query: "vector similarity search index",
			Relevance: map[string]float64{
				"vector_search": 3.0,
				"vector_index":  3.0,
				"ml_embedding":  1.0,
			},
		},
		{
			Query: "file I/O read write",
			Relevance: map[string]float64{
				"file_read":  3.0,
				"file_write": 3.0,
			},
		},
		{
			Query: "JSON serialization parsing",
			Relevance: map[string]float64{
				"json_parse":  3.0,
				"json_encode": 3.0,
			},
		},
		{
			Query: "GPU CUDA kernel memory",
			Relevance: map[string]float64{
				"gpu_kernel": 3.0,
				"gpu_memory": 3.0,
			},
		},
	}

	return &EvalDataset{
		Name:    "CodeSearch",
		Queries: queries,
		Docs:    docs,
	}
}

// EvalResult contains evaluation metrics
type EvalResult struct {
	NDCG10     float64
	Recall10   float64
	AvgLatency float64 // microseconds
	QPS        float64
}

// RunEvaluation runs NDCG@10 evaluation on the dataset
func (ds *EvalDataset) RunEvaluation(searchFn func(query string, k int) []string) EvalResult {
	var totalNDCG, totalRecall float64
	k := 10

	for _, q := range ds.Queries {
		results := searchFn(q.Query, k)

		// Compute NDCG@10
		ndcg := metrics.NDCGAtK(results, q.Relevance, k)
		totalNDCG += ndcg

		// Compute Recall@10
		relevant := 0
		for _, r := range results {
			if q.Relevance[r] > 0 {
				relevant++
			}
		}
		totalRecall += float64(relevant) / float64(len(q.Relevance))
	}

	n := float64(len(ds.Queries))
	return EvalResult{
		NDCG10:   totalNDCG / n,
		Recall10: totalRecall / n,
	}
}

func TestCodeSearchEvaluation(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}

	ds := CreateCodeSearchDataset()

	// Build corpus embeddings
	type docEmbed struct {
		id   string
		text string
		emb  []float32
	}
	corpus := make([]docEmbed, 0, len(ds.Docs))
	for id, text := range ds.Docs {
		emb, _ := model.Embed(text)
		corpus = append(corpus, docEmbed{id: id, text: text, emb: emb})
	}

	// Search function using brute-force
	searchFn := func(query string, k int) []string {
		qEmb, release := model.EmbedFast(query)
		defer release()

		type scored struct {
			id    string
			score float32
		}
		scores := make([]scored, len(corpus))
		for i, doc := range corpus {
			scores[i] = scored{id: doc.id, score: dotProductF32(qEmb, doc.emb)}
		}

		// Sort by score descending
		for i := 0; i < k && i < len(scores); i++ {
			maxIdx := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[maxIdx].score {
					maxIdx = j
				}
			}
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}

		results := make([]string, 0, k)
		for i := 0; i < k && i < len(scores); i++ {
			results = append(results, scores[i].id)
		}
		return results
	}

	result := ds.RunEvaluation(searchFn)

	t.Logf("NDCG@10: %.4f", result.NDCG10)
	t.Logf("Recall@10: %.4f", result.Recall10)

	if result.NDCG10 < 0.5 {
		t.Errorf("NDCG@10 too low: %.4f (expected >= 0.5)", result.NDCG10)
	}
	if result.Recall10 < 0.5 {
		t.Errorf("Recall@10 too low: %.4f (expected >= 0.5)", result.Recall10)
	}
}

func dotProductF32(a, b []float32) float32 {
	var sum float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		sum += a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	return sum
}

func BenchmarkCodeSearchEval(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}

	ds := CreateCodeSearchDataset()

	type docEmbed struct {
		id  string
		emb []float32
	}
	corpus := make([]docEmbed, 0, len(ds.Docs))
	for id, text := range ds.Docs {
		emb, _ := model.Embed(text)
		corpus = append(corpus, docEmbed{id: id, emb: emb})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := ds.Queries[i%len(ds.Queries)]
		qEmb, release := model.EmbedFast(q.Query)

		var bestScore float32 = -1
		for _, doc := range corpus {
			if score := dotProductF32(qEmb, doc.emb); score > bestScore {
				bestScore = score
			}
		}
		_ = bestScore
		release()
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "QPS")
}
