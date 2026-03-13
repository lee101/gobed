package gobed

import (
	"strings"
	"testing"
)

func TestCompactIntegration(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("model not available: %v", err)
	}

	text := "Machine learning is a subset of artificial intelligence. " +
		"It enables computers to learn from data without explicit programming. " +
		"Deep learning uses neural networks with many layers. " +
		"These networks excel at image recognition and natural language processing. " +
		"The field has grown rapidly since 2012 when AlexNet won ImageNet. " +
		"Transfer learning allows reusing pretrained models on new tasks. " +
		"Transformers revolutionized NLP with attention mechanisms. " +
		"GPT and BERT are examples of transformer-based models. " +
		"Reinforcement learning trains agents through reward signals. " +
		"Computer vision applies deep learning to visual data."

	sentences := SentenceSplit(text)
	if len(sentences) != 10 {
		t.Fatalf("expected 10 sentences, got %d", len(sentences))
	}

	totalTokens := 0
	for _, s := range sentences {
		totalTokens += EstimateTokens(s)
	}
	t.Logf("total estimated tokens: %d", totalTokens)

	// compact to ~half
	budget := totalTokens / 2
	result, err := Compact(model, text, budget, "deep learning and neural networks")
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	resultTokens := EstimateTokens(result)
	t.Logf("compacted to %d tokens (budget %d): %s", resultTokens, budget, result)

	if resultTokens > budget+5 { // small tolerance
		t.Errorf("result %d tokens exceeds budget %d", resultTokens, budget)
	}
	if len(result) == 0 {
		t.Error("result is empty")
	}

	// verify it contains relevant content about deep learning
	lower := strings.ToLower(result)
	if !strings.Contains(lower, "deep learning") && !strings.Contains(lower, "neural") {
		t.Logf("warning: compacted text may not contain most relevant sentences for query")
	}
}

func TestCompactIntegration_NoQuery(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("model not available: %v", err)
	}

	text := "Apples are red. Bananas are yellow. Oranges are orange. Grapes are purple. Strawberries are red."
	result, err := Compact(model, text, 10, "")
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	t.Logf("compacted (no query): %s", result)
	if len(result) == 0 {
		t.Error("result is empty")
	}
}

func TestCompactIntegration_AlreadyFits(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("model not available: %v", err)
	}

	text := "Short text."
	result, err := Compact(model, text, 10000, "anything")
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}
	if result != "Short text." {
		t.Errorf("expected passthrough, got %q", result)
	}
}

func TestCompactIntegration_VeryTightBudget(t *testing.T) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		t.Skipf("model not available: %v", err)
	}

	text := "First sentence is here. Second sentence follows. Third one too. Fourth is last."
	result, err := Compact(model, text, 5, "first")
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	t.Logf("tight budget result: %q", result)
	if len(result) == 0 {
		t.Error("should return at least something")
	}
}

func BenchmarkCompact(b *testing.B) {
	model, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skipf("model not available: %v", err)
	}

	text := strings.Repeat("Machine learning enables computers to learn. Deep learning uses neural networks. "+
		"Transformers use attention. Models can be pretrained. Transfer learning is powerful. ", 20)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Compact(model, text, 50, "neural networks")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSentenceSplit(b *testing.B) {
	text := strings.Repeat("This is a sentence. Another one follows! Is this a question? Yes; it is. ", 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SentenceSplit(text)
	}
}
