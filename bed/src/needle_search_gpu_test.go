//go:build gpu && cuda

package src

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBedGPUNeedleSearch(t *testing.T) {
	if os.Getenv("BED_GPU_NEEDLE") == "" {
		t.Skip("set BED_GPU_NEEDLE=1 to enable repository needle search test")
	}

	root := repoRoot(t)

	config := DefaultSearchConfig()
	config.MaxVectors = 500000
	config.MaxFileSize = 2 * 1024 * 1024
	config.BatchSize = 128
	config.ChunkSize = 256
	config.ChunkOverlap = 48
	config.NumWorkers = minInt(runtime.NumCPU(), 6)

	index, err := NewGPUSearchIndex(config)
	if err != nil {
		t.Skipf("GPU index unavailable: %v", err)
	}
	defer index.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	if err := index.IndexDirectory(ctx, root, nil); err != nil {
		t.Fatalf("failed to index repo %s: %v", root, err)
	}

	type queryCheck struct {
		Query string
	}

	queries := []queryCheck{
		{Query: "gobed"},
		{Query: "rohs"},
		{Query: "powerbank"},
	}

	scores := make(map[string]float32, len(queries))

	for _, qc := range queries {
		results, err := index.Search(qc.Query, 10)
		if err != nil {
			t.Fatalf("search for %q failed: %v", qc.Query, err)
		}
		if len(results) == 0 {
			t.Fatalf("expected at least one result for %q", qc.Query)
		}

		scores[qc.Query] = results[0].Score
		t.Logf("top result for %q: %s (score=%.4f)", qc.Query, results[0].FilePath, results[0].Score)

		switch qc.Query {
		case "gobed":
			if !containsLiteral(results, "gobed") {
				t.Errorf("expected at least one top result preview or path containing 'gobed'")
			}
		case "rohs", "powerbank":
			if containsLiteral(results, qc.Query) {
				t.Errorf("did not expect literal match for query %q but found one", qc.Query)
			}
		}
	}

	if scores["gobed"] <= scores["rohs"] {
		t.Errorf("expected score for 'gobed' to exceed 'rohs' (%.4f <= %.4f)", scores["gobed"], scores["rohs"])
	}

	if scorePowerbank, ok := scores["powerbank"]; ok && scorePowerbank > scores["gobed"] {
		t.Logf("warning: powerbank score %.4f exceeded gobed %.4f", scorePowerbank, scores["gobed"])
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("unable to resolve caller for repo root")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	if _, err := os.Stat(root); err != nil {
		t.Fatalf("repo root %s not accessible: %v", root, err)
	}
	return root
}

func containsLiteral(results []*GPUSearchResult, needle string) bool {
	needle = strings.ToLower(needle)
	max := minInt(len(results), 5)
	for i := 0; i < max; i++ {
		if strings.Contains(strings.ToLower(results[i].Preview), needle) ||
			strings.Contains(strings.ToLower(results[i].FilePath), needle) {
			return true
		}
	}
	return false
}
