package gobed

import (
	"strings"
	"testing"

	"github.com/lee101/gobed/pkg/ann/simd"
)

func TestEstimateTokens(t *testing.T) {
	cases := []struct {
		text string
		min  int
		max  int
	}{
		{"", 0, 0},
		{"hello", 1, 3},
		{"The quick brown fox jumps over the lazy dog.", 8, 16},
		{strings.Repeat("a", 100), 20, 35},
	}
	for _, c := range cases {
		got := EstimateTokens(c.text)
		if got < c.min || got > c.max {
			t.Errorf("EstimateTokens(%q) = %d, want [%d, %d]", c.text, got, c.min, c.max)
		}
	}
}

func TestSentenceSplit(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected []string
	}{
		{"empty", "", nil},
		{"single", "Hello world", []string{"Hello world"}},
		{"period", "First sentence. Second sentence.", []string{"First sentence.", "Second sentence."}},
		{"newline", "Line one\nLine two", []string{"Line one", "Line two"}},
		{"exclaim_question", "Really! Are you sure?", []string{"Really!", "Are you sure?"}},
		{"semicolon", "Part one; part two", []string{"Part one;", "part two"}},
		{"decimal_preserved", "The value is 3.5 units.", []string{"The value is 3.5 units."}},
		{"multiple_newlines", "A\n\nB\nC", []string{"A", "B", "C"}},
		{"mixed", "Hello. How are you? Fine; thanks!", []string{"Hello.", "How are you?", "Fine;", "thanks!"}},
		{"trailing_space", "Hello.  World.", []string{"Hello.", "World."}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := SentenceSplit(c.input)
			if len(got) != len(c.expected) {
				t.Fatalf("SentenceSplit(%q) got %d sentences %v, want %d %v", c.input, len(got), got, len(c.expected), c.expected)
			}
			for i := range got {
				if got[i] != c.expected[i] {
					t.Errorf("sentence[%d] = %q, want %q", i, got[i], c.expected[i])
				}
			}
		})
	}
}

func TestEstimateTokensSlice(t *testing.T) {
	input := []string{"hello", "world foo bar"}
	got := EstimateTokensSlice(input)
	if len(got) != 2 {
		t.Fatalf("expected 2, got %d", len(got))
	}
	if got[0] < 1 {
		t.Errorf("first estimate too low: %d", got[0])
	}
}

func TestCompactBatchInt8Cosine(t *testing.T) {
	// synthetic embeddings: 3 sentences, budget for ~2
	emb1 := &Int8Result512{Vector: make([]int8, Int8EmbeddingDim), Scale: 1.0}
	emb2 := &Int8Result512{Vector: make([]int8, Int8EmbeddingDim), Scale: 1.0}
	emb3 := &Int8Result512{Vector: make([]int8, Int8EmbeddingDim), Scale: 1.0}

	// make emb1 and emb3 similar to ref, emb2 different
	for i := 0; i < Int8EmbeddingDim; i++ {
		emb1.Vector[i] = 10
		emb2.Vector[i] = -10
		emb3.Vector[i] = 8
	}

	var ref simd.Vec512
	for i := 0; i < 512; i++ {
		ref[i] = 10
	}

	tokenCounts := []int{100, 100, 100}
	selected := CompactBatchInt8Cosine(
		[]*Int8Result512{emb1, emb2, emb3},
		&ref, 1.0,
		tokenCounts,
		250, // budget for 2 of 3
	)

	if len(selected) != 2 {
		t.Fatalf("expected 2 selected, got %d: %v", len(selected), selected)
	}
	// should select indices 0 and 2 (highest similarity), in order
	if selected[0] != 0 || selected[1] != 2 {
		t.Errorf("expected [0,2], got %v", selected)
	}
}

func TestCompactBatchInt8Cosine_AllFit(t *testing.T) {
	embs := make([]*Int8Result512, 3)
	for i := range embs {
		embs[i] = &Int8Result512{Vector: make([]int8, Int8EmbeddingDim), Scale: 1.0}
		for j := range embs[i].Vector {
			embs[i].Vector[j] = int8(i + 1)
		}
	}

	var ref simd.Vec512
	for i := range ref {
		ref[i] = 5
	}

	selected := CompactBatchInt8Cosine(embs, &ref, 1.0, []int{10, 10, 10}, 1000)
	if len(selected) != 3 {
		t.Errorf("expected all 3 selected, got %d", len(selected))
	}
}

func TestCompactBatchInt8Cosine_Empty(t *testing.T) {
	selected := CompactBatchInt8Cosine(nil, nil, 0, nil, 100)
	if selected != nil {
		t.Errorf("expected nil, got %v", selected)
	}
}

func TestVec512FromInt8(t *testing.T) {
	emb := &Int8Result512{Vector: make([]int8, Int8EmbeddingDim), Scale: 1.0}
	emb.Vector[0] = 42
	emb.Vector[511] = -7

	v := Vec512FromInt8(emb)
	if v == nil {
		t.Fatal("expected non-nil")
	}
	if v[0] != 42 || v[511] != -7 {
		t.Errorf("values mismatch: [0]=%d [511]=%d", v[0], v[511])
	}
}

func TestVec512FromInt8_WrongSize(t *testing.T) {
	emb := &Int8Result512{Vector: make([]int8, 100), Scale: 1.0}
	v := Vec512FromInt8(emb)
	if v != nil {
		t.Error("expected nil for wrong-sized vector")
	}
}

func TestSentenceSplit_LongText(t *testing.T) {
	// realistic paragraph
	text := "Machine learning is a subset of artificial intelligence. It enables computers to learn from data. Deep learning uses neural networks with many layers. These networks can process images, text, and audio. The field has grown rapidly since 2012."
	sentences := SentenceSplit(text)
	if len(sentences) != 5 {
		t.Errorf("expected 5 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestSentenceSplit_CodeSnippet(t *testing.T) {
	text := "func main() { fmt.Println(\"hello\") }\nfunc other() { return 3.14 }"
	sentences := SentenceSplit(text)
	if len(sentences) < 1 {
		t.Error("expected at least 1 sentence from code")
	}
}

func TestCompactSentences_NilModel(t *testing.T) {
	// empty input should work without model
	result, err := CompactSentences(nil, nil, 100, "")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestCompactSentences_AlreadyFits(t *testing.T) {
	// nil model is fine when content already fits
	sentences := []string{"Short."}
	result, err := CompactSentences(nil, sentences, 10000, "")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 1 || result[0] != "Short." {
		t.Errorf("expected passthrough, got %v", result)
	}
}
