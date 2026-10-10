package src

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestTopUniqueDocKeysUsesRelativePaths(t *testing.T) {
	root := filepath.Join(string(os.PathSeparator), "tmp", "quality-docs")
	matches := []SearchMatch{
		{Document: Document{Path: filepath.Join(root, "pkg", "a.txt")}},
		{Document: Document{Path: filepath.Join(root, "pkg", "a.txt")}},
		{Document: Document{Path: filepath.Join(root, "cmd", "a.txt")}},
	}

	got := topUniqueDocKeys(matches, 3, root)
	if len(got) != 2 {
		t.Fatalf("expected 2 unique doc keys, got %d", len(got))
	}
	if got[0] != "pkg/a.txt" || got[1] != "cmd/a.txt" {
		t.Fatalf("unexpected ranking: %#v", got)
	}
}

func TestEvaluateQualityCases(t *testing.T) {
	root := filepath.Join(string(os.PathSeparator), "tmp", "quality-docs")
	cases := []benchQualityCase{
		{
			Query: "q1",
			Relevance: map[string]float64{
				"pkg/a.txt": 3,
				"pkg/b.txt": 2,
			},
		},
		{
			Query: "q2",
			Relevance: map[string]float64{
				"pkg/c.txt": 3,
			},
		},
	}

	searchFn := func(opts BedSearchOptions) ([]SearchMatch, error) {
		switch opts.Query {
		case "q1":
			return []SearchMatch{
				{Document: Document{Path: filepath.Join(root, "pkg", "a.txt")}},
				{Document: Document{Path: filepath.Join(root, "pkg", "b.txt")}},
			}, nil
		case "q2":
			return []SearchMatch{
				{Document: Document{Path: filepath.Join(root, "pkg", "c.txt")}},
			}, nil
		default:
			return nil, nil
		}
	}

	ndcg, recall, err := evaluateQualityCases(root, cases, searchFn, 10)
	if err != nil {
		t.Fatalf("evaluateQualityCases failed: %v", err)
	}
	if ndcg < 0.99 {
		t.Fatalf("expected ndcg close to 1.0, got %.4f", ndcg)
	}
	if recall < 0.99 {
		t.Fatalf("expected recall close to 1.0, got %.4f", recall)
	}
}

func TestLoadQualityDatasetExplicitDir(t *testing.T) {
	root := t.TempDir()
	docsDir := filepath.Join(root, "docs")
	if err := os.MkdirAll(docsDir, 0o755); err != nil {
		t.Fatalf("mkdir docs failed: %v", err)
	}
	if err := os.WriteFile(filepath.Join(docsDir, "a.txt"), []byte("alpha"), 0o644); err != nil {
		t.Fatalf("write docs failed: %v", err)
	}

	cases := []benchQualityCase{
		{
			Query: "q1",
			Relevance: map[string]float64{
				"a.txt": 3,
			},
		},
	}
	data, err := json.Marshal(cases)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "queries.json"), data, 0o644); err != nil {
		t.Fatalf("write queries failed: %v", err)
	}

	prev := flagBenchQuality
	flagBenchQuality = root
	defer func() { flagBenchQuality = prev }()

	got, err := loadQualityDataset()
	if err != nil {
		t.Fatalf("loadQualityDataset failed: %v", err)
	}
	if got.DocsDir != docsDir {
		t.Fatalf("unexpected docs dir: %s", got.DocsDir)
	}
	if len(got.Cases) != 1 || got.Cases[0].Query != "q1" {
		t.Fatalf("unexpected cases: %#v", got.Cases)
	}
}

func TestLoadBenchmarkQueriesFromFile(t *testing.T) {
	queryFile := filepath.Join(t.TempDir(), "queries.txt")
	content := "# comment\nfirst query\n\nsecond query\n"
	if err := os.WriteFile(queryFile, []byte(content), 0o644); err != nil {
		t.Fatalf("write query file failed: %v", err)
	}

	prev := flagBenchQueryFile
	flagBenchQueryFile = queryFile
	defer func() { flagBenchQueryFile = prev }()

	got, source, err := loadBenchmarkQueries()
	if err != nil {
		t.Fatalf("loadBenchmarkQueries failed: %v", err)
	}
	if source != queryFile {
		t.Fatalf("unexpected query source: %s", source)
	}
	if len(got) != 2 || got[0] != "first query" || got[1] != "second query" {
		t.Fatalf("unexpected queries: %#v", got)
	}
}

func TestPercentileDuration(t *testing.T) {
	sorted := []time.Duration{
		1 * time.Millisecond,
		2 * time.Millisecond,
		3 * time.Millisecond,
		4 * time.Millisecond,
		5 * time.Millisecond,
	}
	if got := percentileDuration(sorted, 50); got != 3*time.Millisecond {
		t.Fatalf("unexpected p50: %s", got)
	}
	if got := percentileDuration(sorted, 95); got != 5*time.Millisecond {
		t.Fatalf("unexpected p95: %s", got)
	}
	if got := percentileDuration(sorted, 99); got != 5*time.Millisecond {
		t.Fatalf("unexpected p99: %s", got)
	}
}

func TestApplyCAGRABenchEnv(t *testing.T) {
	prevItopk := flagBenchCAGRAItopk
	prevAlgo := flagBenchCAGRAAlgo
	flagBenchCAGRAItopk = 128
	flagBenchCAGRAAlgo = "multi_cta"
	defer func() {
		flagBenchCAGRAItopk = prevItopk
		flagBenchCAGRAAlgo = prevAlgo
	}()

	restore, err := applyCAGRABenchEnv()
	if err != nil {
		t.Fatalf("applyCAGRABenchEnv failed: %v", err)
	}
	defer restore()

	if got := os.Getenv("CAGRA_ITOPK_SIZE"); got != "128" {
		t.Fatalf("unexpected itopk override: %q", got)
	}
	if got := os.Getenv("CAGRA_ALGO"); got != "1" {
		t.Fatalf("unexpected algo override: %q", got)
	}
}
