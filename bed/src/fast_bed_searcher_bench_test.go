package src

import (
	"fmt"
	"strings"
	"testing"
)

func BenchmarkFastBedSearcherBuildLineInfos(b *testing.B) {
	opts := defaultFastSearcherOptions()
	opts.maxLineLength = 128
	opts.ignoreLongLines = true
	opts.minLineLength = 3

	searcher, err := newFastBedSearcherWithModel(fakeFastModel{}, b.TempDir(), opts)
	if err != nil {
		b.Fatalf("failed to create searcher: %v", err)
	}
	defer searcher.Close()

	var sb strings.Builder
	for i := 0; i < 2000; i++ {
		sb.WriteString("short useful line content\n")
	}
	for i := 0; i < 200; i++ {
		sb.WriteString(strings.Repeat("x", 512))
		sb.WriteString("\n")
	}
	data := []byte(sb.String())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = searcher.buildLineInfos(data)
	}
}

func BenchmarkFastBedSearcherSearchMatches(b *testing.B) {
	opts := defaultFastSearcherOptions()
	searcher, err := newFastBedSearcherWithModel(fakeFastModel{}, b.TempDir(), opts)
	if err != nil {
		b.Fatalf("failed to create searcher: %v", err)
	}
	defer searcher.Close()

	const docsCount = 12000
	docs := make([]Document, 0, docsCount)
	embs := make([][]float32, 0, docsCount)
	norms := make([]float32, 0, docsCount)

	for i := 0; i < docsCount; i++ {
		content := fmt.Sprintf("handler_%d processes authentication and cache updates", i)
		emb, release := searcher.model.EmbedFast(content)
		embCopy := make([]float32, len(emb))
		var normSq float32
		for j, v := range emb {
			embCopy[j] = v
			normSq += v * v
		}
		release()

		docs = append(docs, Document{
			Path:       fmt.Sprintf("/tmp/file_%d.go", i),
			LineNumber: i%200 + 1,
			Content:    content,
		})
		embs = append(embs, embCopy)
		norms = append(norms, normSq)
	}

	searcher.mu.Lock()
	searcher.appendDocumentsLocked(docs, embs, norms)
	searcher.mu.Unlock()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := searcher.SearchMatches(BedSearchOptions{
			Query:     "authentication handler cache logic",
			Limit:     10,
			Threshold: 0.0,
			NoIndex:   true,
		})
		if err != nil {
			b.Fatalf("search failed: %v", err)
		}
	}
}
