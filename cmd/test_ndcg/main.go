package main

import (
	"fmt"
	"log"

	gobed "github.com/lee101/gobed"
	"github.com/lee101/gobed/bed"
)

func main() {
	fmt.Println("Testing NDCG@10 evaluation...")

	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Load model: %v", err)
	}

	docs := []gobed.Document{
		{ID: 0, Text: "GPU acceleration deep learning neural networks"},
		{ID: 1, Text: "CUDA programming parallel computing"},
		{ID: 2, Text: "Transformer models natural language"},
		{ID: 3, Text: "Vector search CAGRA index"},
		{ID: 4, Text: "Database SQL queries"},
	}

	cfg := bed.EvalConfig{K: 5, NumQueries: 3, Warmup: 1}
	result, err := bed.RunEval(model, docs, cfg)
	if err != nil {
		log.Fatalf("Eval: %v", err)
	}

	fmt.Printf("\n✓ NDCG@%d: %.4f\n", result.K, result.NDCGAtK)
	fmt.Printf("✓ Recall@%d: %.4f\n", result.K, result.RecallAtK)
	fmt.Printf("✓ P50: %.2fms, QPS: %.1f\n", result.P50LatencyMs, result.QPS)
}
