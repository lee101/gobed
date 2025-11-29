//go:build legacy && gpu && cuda

package src

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/lee101/gobed/metrics"
)

type qualityCase struct {
	Query     string             `json:"query"`
	Relevance map[string]float64 `json:"relevance"`
}

func TestBedGPUSearchQualityNDCG10(t *testing.T) {
	if os.Getenv("BED_GPU_TESTS") == "" {
		t.Skip("set BED_GPU_TESTS=1 to enable GPU search quality validation")
	}

	corpusDir, cases := loadQualityCorpus(t)

	config := DefaultSearchConfig()
	config.MaxVectors = 4096
	config.ChunkSize = 256
	config.ChunkOverlap = 32
	config.BatchSize = 64
	config.IVFClusters = 16
	config.ProbeLists = 4
	config.NumWorkers = minInt(runtime.NumCPU(), 4)

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Skipf("GPU index unavailable: %v", err)
	}
	defer index.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	if err := index.IndexDirectory(ctx, corpusDir, []string{".txt"}); err != nil {
		t.Fatalf("failed to index test corpus: %v", err)
	}

	if len(index.fileIndex) < len(cases) {
		t.Fatalf("expected at least %d files, got %d", len(cases), len(index.fileIndex))
	}

	const k = 10
	var sumNDCG float64

	for _, qc := range cases {
		results, err := index.Search(qc.Query, k)
		if err != nil {
			t.Fatalf("search for %q failed: %v", qc.Query, err)
		}

		ranking := make([]string, 0, len(results))
		for _, res := range results {
			ranking = append(ranking, filepath.Base(res.FilePath))
		}

		score := metrics.NDCGAtK(ranking, qc.Relevance, k)
		sumNDCG += score

		if score < 0.35 {
			t.Errorf("ndcg@10 for query %q too low: %.3f", qc.Query, score)
		}
	}

	avgNDCG := sumNDCG / float64(len(cases))
	if avgNDCG < 0.6 {
		t.Fatalf("average ndcg@10 too low: got %.3f, need >= 0.60", avgNDCG)
	}
	t.Logf("average ndcg@10 across %d queries: %.3f", len(cases), avgNDCG)
}

func loadQualityCorpus(t *testing.T) (string, []qualityCase) {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("unable to detect caller path")
	}

	root := filepath.Join(filepath.Dir(file), "..", "testsearches")
	corpusDir := filepath.Join(root, "docs")
	metaPath := filepath.Join(root, "queries.json")

	if _, err := os.Stat(corpusDir); err != nil {
		t.Fatalf("missing corpus directory %s: %v", corpusDir, err)
	}
	data, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("failed to read queries: %v", err)
	}

	var cases []qualityCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to parse queries: %v", err)
	}

	if len(cases) == 0 {
		t.Fatal("no query cases loaded")
	}

	return corpusDir, cases
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
