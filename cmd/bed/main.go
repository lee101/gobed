package main

import (
	"flag"
	"fmt"
	"log"

	gobed "github.com/lee101/gobed"
	"github.com/lee101/gobed/bed/lib"
)

func main() {
	dirPath := flag.String("dir", ".", "directory to index")
	query := flag.String("q", "", "search query")
	k := flag.Int("k", 10, "number of results")
	benchmark := flag.Bool("bench", false, "run benchmark with NDCG@10")
	numQueries := flag.Int("queries", 100, "number of queries for benchmark")
	flag.Parse()

	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	cfg := gobed.DefaultVectorIndexConfig()
	cfg.NList = 1024
	cfg.NProbe = 8

	fmt.Printf("Indexing directory: %s\n", *dirPath)
	indexer, err := lib.NewFSIndexer(*dirPath, model, cfg)
	if err != nil {
		log.Fatalf("Failed to create indexer: %v", err)
	}

	count, err := indexer.IndexAll()
	if err != nil {
		log.Fatalf("Failed to index directory: %v", err)
	}
	fmt.Printf("Indexed %d document chunks\n", count)

	if *benchmark {
		runBenchmark(model, *numQueries, *k)
	} else if *query != "" {
		runSearch(indexer, *query, *k)
	} else {
		fmt.Println("Use -q for search or -bench for benchmark")
		flag.Usage()
	}
}

func runSearch(indexer *lib.FSIndexer, query string, k int) {
	results, err := indexer.Index().Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nTop %d results for: %s\n", k, query)
	fmt.Println("---")
	for i, r := range results {
		preview := r.Text
		if len(preview) > 200 { preview = preview[:200] + "..." }
		fmt.Printf("[%d] Score: %.4f\n%s\n\n", i+1, r.Similarity, preview)
	}
}

func runBenchmark(model *gobed.EmbeddingModel, numQueries, k int) {
	fmt.Printf("\nRunning benchmark: %d queries, k=%d\n", numQueries, k)

	docs := make([]gobed.Document, numQueries*2)
	for i := range docs {
		docs[i] = gobed.Document{ID: i, Text: fmt.Sprintf("benchmark doc %d content", i)}
	}

	evalCfg := lib.EvalConfig{K: k, NumQueries: numQueries, Warmup: 10}
	result, err := lib.RunEval(model, docs, evalCfg)
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	fmt.Println("\n=== Benchmark Results ===")
	fmt.Printf("K: %d, Queries: %d\n", result.K, result.NumQueries)
	fmt.Printf("P50 Latency: %.2f ms\n", result.P50LatencyMs)
	fmt.Printf("P95 Latency: %.2f ms\n", result.P95LatencyMs)
	fmt.Printf("QPS: %.2f\n", result.QPS)
	fmt.Printf("Recall@%d: %.4f\n", result.K, result.RecallAtK)
	fmt.Printf("NDCG@%d: %.4f\n", result.K, result.NDCGAtK)
}
