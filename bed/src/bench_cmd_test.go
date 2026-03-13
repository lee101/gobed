package src

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTopUniqueFileNames(t *testing.T) {
	matches := []SearchMatch{
		{Document: Document{Path: "/tmp/a.txt"}},
		{Document: Document{Path: "/tmp/a.txt"}},
		{Document: Document{Path: "/tmp/b.txt"}},
		{Document: Document{Path: "/tmp/c.txt"}},
	}

	got := topUniqueFileNames(matches, 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
	if got[0] != "a.txt" || got[1] != "b.txt" {
		t.Fatalf("unexpected ranking: %#v", got)
	}
}

func TestEvaluateQualityCases(t *testing.T) {
	cases := []benchQualityCase{
		{
			Query: "q1",
			Relevance: map[string]float64{
				"a.txt": 3,
				"b.txt": 2,
			},
		},
		{
			Query: "q2",
			Relevance: map[string]float64{
				"c.txt": 3,
			},
		},
	}

	searchFn := func(opts BedSearchOptions) ([]SearchMatch, error) {
		switch opts.Query {
		case "q1":
			return []SearchMatch{
				{Document: Document{Path: "/tmp/a.txt"}},
				{Document: Document{Path: "/tmp/b.txt"}},
			}, nil
		case "q2":
			return []SearchMatch{
				{Document: Document{Path: "/tmp/c.txt"}},
			}, nil
		default:
			return nil, nil
		}
	}

	ndcg, recall, err := evaluateQualityCases(cases, searchFn, 10)
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
	if err := os.MkdirAll(docsDir, 0755); err != nil {
		t.Fatalf("mkdir docs failed: %v", err)
	}
	if err := os.WriteFile(filepath.Join(docsDir, "a.txt"), []byte("alpha"), 0644); err != nil {
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
	if err := os.WriteFile(filepath.Join(root, "queries.json"), data, 0644); err != nil {
		t.Fatalf("write queries failed: %v", err)
	}

	prev := flagBenchQuality
	flagBenchQuality = root
	defer func() { flagBenchQuality = prev }()

	gotDocs, gotCases, err := loadQualityDataset()
	if err != nil {
		t.Fatalf("loadQualityDataset failed: %v", err)
	}
	if gotDocs != docsDir {
		t.Fatalf("unexpected docs dir: %s", gotDocs)
	}
	if len(gotCases) != 1 || gotCases[0].Query != "q1" {
		t.Fatalf("unexpected cases: %#v", gotCases)
	}
}
