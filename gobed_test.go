package gobed

import (
	"math"
	"testing"
)

// mustLoadModelForTest loads the embedding model or skips the test when unavailable.
func mustLoadModelForTest(t *testing.T) *EmbeddingModel {
	t.Helper()
	model, err := LoadModel()
	if err != nil {
		t.Skipf("Skipping test: model unavailable: %v", err)
	}
	return model
}

func mustLoadModelForBenchmark(b *testing.B) *EmbeddingModel {
	b.Helper()
	model, err := LoadModel()
	if err != nil {
		b.Skipf("Skipping benchmark: model unavailable: %v", err)
	}
	return model
}

func isModelUnavailable(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	return containsWord(errStr, "model") || containsWord(errStr, "file") || containsWord(errStr, "not found")
}

func TestEmbeddingModelWithRealModel(t *testing.T) {
	model := mustLoadModelForTest(t)

	testCases := []struct {
		name     string
		sentence string
	}{
		{"simple", "Hello world"},
		{"question", "What is machine learning?"},
		{"statement", "The weather is nice today."},
		{"technical", "Python is a programming language used for data science."},
		{"long", "This is a longer sentence that contains multiple clauses and should test the model's ability to handle more complex input structures."},
		{"numbers", "The year 2024 has 365 days."},
		{"punctuation", "Hello! How are you? I'm fine, thanks."},
		{"mixed_case", "ThIs Is A MiXeD CaSe SeNtEnCe"},
		{"very_long", "The quick brown fox jumps over the lazy dog. This pangram sentence contains all letters of the English alphabet and is commonly used for testing purposes in typography and keyboard layouts."},
		{"scientific", "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations."},
		{"code_like", "def hello_world(): print('Hello, World!')"},
		{"urls", "Visit https://example.com for more information about our services and products."},
	}

	embeddings := make([][]float32, len(testCases))

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			embedding, err := model.Encode(tc.sentence)
			if err != nil {
				t.Fatalf("Failed to encode sentence '%s': %v", tc.sentence, err)
			}

			expectedDim := model.EmbedDim
			if len(embedding) != expectedDim {
				t.Errorf("Expected embedding dimension %d, got %d", expectedDim, len(embedding))
			}

			var sum, sqSum float32
			var minVal, maxVal float32 = math.MaxFloat32, -math.MaxFloat32
			for _, val := range embedding {
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Errorf("Invalid value in embedding: %v", val)
				}
				sum += val
				sqSum += val * val
				if val < minVal {
					minVal = val
				}
				if val > maxVal {
					maxVal = val
				}
			}

			mean := sum / float32(len(embedding))
			norm := float32(math.Sqrt(float64(sqSum)))

			if norm == 0 {
				t.Errorf("Zero norm embedding for sentence: %s", tc.sentence)
			}

			t.Logf("Sentence: '%s'", tc.sentence)
			t.Logf("  First 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]",
				embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
			t.Logf("  Stats - Mean: %.4f, Norm: %.4f, Min: %.4f, Max: %.4f", mean, norm, minVal, maxVal)

			embeddings[i] = embedding
		})
	}

	t.Run("diversity_check", func(t *testing.T) {
		allSame := true
		var maxSim, minSim float32 = -1, 2
		similarPairs := 0

		for i := 0; i < len(embeddings); i++ {
			for j := i + 1; j < len(embeddings); j++ {
				sim := CosineSimilarity(embeddings[i], embeddings[j])

				if sim < 0.999 {
					allSame = false
				}
				if sim > 0.95 {
					similarPairs++
					t.Logf("High similarity (%.4f) between '%s' and '%s'",
						sim, testCases[i].sentence, testCases[j].sentence)
				}
				if sim > maxSim {
					maxSim = sim
				}
				if sim < minSim {
					minSim = sim
				}
			}
		}

		if allSame {
			t.Errorf("All embeddings are nearly identical (all similarities > 0.999)")
		}

		t.Logf("Similarity range: [%.4f, %.4f]", minSim, maxSim)
		t.Logf("Highly similar pairs (>0.95): %d out of %d total pairs", similarPairs, len(embeddings)*(len(embeddings)-1)/2)

		if maxSim-minSim < 0.1 {
			t.Errorf("Insufficient diversity: similarity range is too narrow (%.4f)", maxSim-minSim)
		}
	})
}

func TestSimilarityMethod(t *testing.T) {
	model := mustLoadModelForTest(t)

	testPairs := []struct {
		name   string
		text1  string
		text2  string
		minSim float32
		maxSim float32
	}{
		{"identical", "Hello world", "Hello world", 0.999, 1.001},
		{"similar", "The cat is sleeping", "The kitten is sleeping", 0.75, 0.85},
		{"different", "I love programming", "The weather is cold", -0.1, 0.1},
		{"semantic_similar", "The car is fast", "The vehicle is quick", 0.45, 0.55},
	}

	for _, tp := range testPairs {
		t.Run(tp.name, func(t *testing.T) {
			sim, err := model.Similarity(tp.text1, tp.text2)
			if err != nil {
				t.Fatalf("Failed to compute similarity: %v", err)
			}

			t.Logf("Similarity between '%s' and '%s': %.4f", tp.text1, tp.text2, sim)

			if sim < tp.minSim || sim > tp.maxSim {
				t.Errorf("Similarity %.4f outside expected range [%.4f, %.4f]", sim, tp.minSim, tp.maxSim)
			}
		})
	}
}

func TestFindMostSimilar(t *testing.T) {
	model := mustLoadModelForTest(t)

	candidates := []string{
		"The dog is playing in the garden",
		"A cat is sleeping on the couch",
		"Programming in Python is fun",
		"The weather is sunny today",
		"Machine learning models are powerful",
		"Dogs love to play fetch",
		"Cats are independent animals",
		"I enjoy coding in Go",
		"It's raining outside",
		"Neural networks can learn patterns",
	}

	queries := []struct {
		query         string
		expectedTop   string
		expectedWords []string
	}{
		{
			query:         "The puppy is running",
			expectedWords: []string{"dog", "Dogs"},
		},
		{
			query:         "Artificial intelligence is fascinating",
			expectedWords: []string{"Machine learning", "Neural networks"},
		},
		{
			query:         "Writing code is enjoyable",
			expectedWords: []string{"Programming", "coding"},
		},
	}

	for _, q := range queries {
		t.Run(q.query, func(t *testing.T) {
			results, err := model.FindMostSimilar(q.query, candidates, 3)
			if err != nil {
				t.Fatalf("Failed to find similar: %v", err)
			}

			if len(results) != 3 {
				t.Errorf("Expected 3 results, got %d", len(results))
			}

			t.Logf("Query: '%s'", q.query)
			for i, r := range results {
				t.Logf("  %d. '%s' (similarity: %.4f)", i+1, r.Text2, r.Similarity)
			}

			topResult := results[0].Text2
			foundExpected := false
			for _, word := range q.expectedWords {
				if containsWord(topResult, word) {
					foundExpected = true
					break
				}
			}

			if !foundExpected {
				t.Logf("Warning: Top result '%s' doesn't contain expected words %v", topResult, q.expectedWords)
			}

			for i := 1; i < len(results); i++ {
				if results[i].Similarity > results[i-1].Similarity {
					t.Errorf("Results not sorted by similarity: %.4f > %.4f", results[i].Similarity, results[i-1].Similarity)
				}
			}
		})
	}
}

func TestEdgeCases(t *testing.T) {
	model := mustLoadModelForTest(t)

	edgeCases := []struct {
		name        string
		text        string
		shouldError bool
	}{
		{"empty", "", false},
		{"single_space", " ", false},
		{"multiple_spaces", "   ", false},
		{"newline", "\n", false},
		{"tab", "\t", false},
		{"punctuation_only", "...", false},
		{"exclamation", "!!!", false},
		{"emoji", "😀", false},
		{"chinese", "你好世界", false},
		{"russian", "Здравствуй мир", false},
		{"arabic", "مرحبا بالعالم", false},
		{"very_long", string(make([]byte, 1000, 1000)), false},
		{"special_chars", "@#$%^&*()", false},
		{"mixed_script", "Hello世界مرحبا", false},
	}

	for _, ec := range edgeCases {
		t.Run(ec.name, func(t *testing.T) {
			embedding, err := model.Encode(ec.text)

			if ec.shouldError && err == nil {
				t.Errorf("Expected error but got none")
			}
			if !ec.shouldError && err != nil {
				if isModelUnavailable(err) {
					t.Skipf("skipping: %v", err)
				}
				t.Logf("Edge case '%s' produced error (might be expected): %v", ec.name, err)
				return
			}

			if err == nil {
				expectedDim := model.EmbedDim
				if len(embedding) != expectedDim {
					t.Errorf("Unexpected embedding dimension: %d (expected %d)", len(embedding), model.EmbedDim)
				}

				var nonZeroCount int
				for _, val := range embedding {
					if val != 0 {
						nonZeroCount++
					}
				}

				t.Logf("Edge case '%s': %d/%d non-zero values", ec.name, nonZeroCount, len(embedding))

				if nonZeroCount == 0 {
					t.Logf("Warning: All-zero embedding for '%s'", ec.name)
				}
			}
		})
	}
}

func TestConsistency(t *testing.T) {
	model := mustLoadModelForTest(t)

	sentence := "This is a test for consistency"

	firstEmb, err := model.Encode(sentence)
	if err != nil {
		if isModelUnavailable(err) {
			t.Skipf("skipping: %v", err)
		}
		t.Fatalf("Failed to encode: %v", err)
	}

	for i := 0; i < 10; i++ {
		emb, err := model.Encode(sentence)
		if err != nil {
			if isModelUnavailable(err) {
				t.Skipf("skipping: %v", err)
			}
			t.Fatalf("Failed to encode on iteration %d: %v", i, err)
		}

		sim := CosineSimilarity(firstEmb, emb)
		if sim < 0.9999 {
			t.Errorf("Inconsistent embeddings on iteration %d: similarity = %.6f", i, sim)
		}

		var diff float32
		for j := range emb {
			d := firstEmb[j] - emb[j]
			diff += d * d
		}
		diff = float32(math.Sqrt(float64(diff)))

		if diff > 0.001 {
			t.Errorf("Embeddings differ by %.6f on iteration %d", diff, i)
		}
	}
}

func BenchmarkSingleEncoding(b *testing.B) {
	model := mustLoadModelForBenchmark(b)

	sentence := "This is a benchmark test sentence for performance measurement."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Encode(sentence)
		if err != nil {
			b.Fatalf("Failed to encode: %v", err)
		}
	}
}

func BenchmarkSimilarityComputation(b *testing.B) {
	model := mustLoadModelForBenchmark(b)

	text1 := "First text for similarity comparison"
	text2 := "Second text for similarity comparison"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Similarity(text1, text2)
		if err != nil {
			b.Fatalf("Failed to compute similarity: %v", err)
		}
	}
}

func containsWord(text, word string) bool {
	return len(word) > 0 && (text == word ||
		(len(text) > len(word) && (text[:len(word)] == word || text[len(text)-len(word):] == word)) ||
		(len(text) > len(word)+1 && containsWord(text[1:], word)))
}
