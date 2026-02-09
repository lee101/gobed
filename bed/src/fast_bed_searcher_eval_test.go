package src

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/lee101/gobed/metrics"
)

type bedQualityCase struct {
	Query     string             `json:"query"`
	Relevance map[string]float64 `json:"relevance"`
}

func prepareQualityCorpus(tb testing.TB) (string, []bedQualityCase) {
	tb.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("unable to resolve caller path")
	}

	root := filepath.Join(filepath.Dir(file), "..", "testsearches")
	docsDir := filepath.Join(root, "docs")
	metaPath := filepath.Join(root, "queries.json")

	data, err := os.ReadFile(metaPath)
	if err != nil {
		tb.Fatalf("failed to read queries: %v", err)
	}

	var cases []bedQualityCase
	if err := json.Unmarshal(data, &cases); err != nil {
		tb.Fatalf("failed to parse queries: %v", err)
	}
	if len(cases) == 0 {
		tb.Fatal("no quality cases found")
	}

	tempDir := tb.TempDir()
	entries, err := os.ReadDir(docsDir)
	if err != nil {
		tb.Fatalf("failed to read corpus dir: %v", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		srcPath := filepath.Join(docsDir, entry.Name())
		dstPath := filepath.Join(tempDir, entry.Name())
		data, err := os.ReadFile(srcPath)
		if err != nil {
			tb.Fatalf("failed to read corpus file %s: %v", entry.Name(), err)
		}
		if err := os.WriteFile(dstPath, data, 0644); err != nil {
			tb.Fatalf("failed to write corpus file %s: %v", entry.Name(), err)
		}
	}

	return tempDir, cases
}

func evaluateFastSearcher(tb testing.TB, searcher *FastBedSearcher, cases []bedQualityCase, k int) (float64, float64) {
	tb.Helper()

	var totalNDCG, totalRecall float64
	for _, qc := range cases {
		matches, err := searcher.SearchMatches(BedSearchOptions{
			Query:     qc.Query,
			Limit:     k,
			Threshold: 0.0,
			NoIndex:   true,
		})
		if err != nil {
			tb.Fatalf("search failed for %q: %v", qc.Query, err)
		}

		ranking := make([]string, len(matches))
		for i, match := range matches {
			ranking[i] = filepath.Base(match.Document.Path)
		}

		ndcg := metrics.NDCGAtK(ranking, qc.Relevance, k)
		totalNDCG += ndcg

		relevant := 0
		seen := make(map[string]struct{}, len(ranking))
		for _, id := range ranking {
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			if qc.Relevance[id] > 0 {
				relevant++
			}
			if len(seen) == k {
				break
			}
		}
		totalRecall += float64(relevant) / float64(len(qc.Relevance))

		if tb, ok := tb.(*testing.T); ok {
			tb.Logf("Query: %q  NDCG@%d: %.3f  Recall@%d: %.2f",
				qc.Query, k, ndcg, k, float64(relevant)/float64(len(qc.Relevance)))
		}
	}

	n := float64(len(cases))
	return totalNDCG / n, totalRecall / n
}

func TestFastBedSearcherNDCG10(t *testing.T) {
	searcher, err := NewFastBedSearcher()
	if err != nil {
		t.Skipf("Fast searcher unavailable: %v", err)
	}
	defer searcher.Close()

	corpusDir, cases := prepareQualityCorpus(t)
	if err := searcher.IndexDirectory(corpusDir, BedSearchOptions{ForceIndex: true}); err != nil {
		t.Fatalf("failed to index corpus: %v", err)
	}

	avgNDCG, avgRecall := evaluateFastSearcher(t, searcher, cases, 10)
	t.Logf("Average NDCG@10: %.4f", avgNDCG)
	t.Logf("Average Recall@10: %.4f", avgRecall)

	if avgNDCG < 0.5 {
		t.Errorf("NDCG@10 too low: %.4f (expected >= 0.5)", avgNDCG)
	}
	if avgRecall < 0.5 {
		t.Errorf("Recall@10 too low: %.4f (expected >= 0.5)", avgRecall)
	}
}

func BenchmarkFastBedSearcher_Search(b *testing.B) {
	searcher, err := NewFastBedSearcher()
	if err != nil {
		b.Skipf("Fast searcher unavailable: %v", err)
	}
	b.Cleanup(func() { _ = searcher.Close() })

	corpusDir, cases := prepareQualityCorpus(b)
	if err := searcher.IndexDirectory(corpusDir, BedSearchOptions{ForceIndex: true}); err != nil {
		b.Fatalf("failed to index corpus: %v", err)
	}

	avgNDCG, _ := evaluateFastSearcher(b, searcher, cases, 10)
	b.ReportMetric(avgNDCG, "ndcg@10")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qc := cases[i%len(cases)]
		_, err := searcher.SearchMatches(BedSearchOptions{
			Query:     qc.Query,
			Limit:     10,
			Threshold: 0.0,
			NoIndex:   true,
		})
		if err != nil {
			b.Fatalf("search failed for %q: %v", qc.Query, err)
		}
	}

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
}
