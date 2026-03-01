package gobed

import (
	"math"
	"sort"
	"strings"
	"unsafe"

	"github.com/lee101/gobed/pkg/ann/simd"
)

type CompactConfig struct {
	MaxTokens     int
	Query         string
	PreserveOrder bool // re-sort selected sentences by original position
}

type scoredSentence struct {
	index    int
	text     string
	tokens   int
	score    float32
	selected bool
}

// EstimateTokens returns a fast heuristic token count (~3.5 chars/token for English).
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return int(math.Ceil(float64(len(text)) / 3.5))
}

// EstimateTokensSlice returns per-sentence token estimates.
func EstimateTokensSlice(sentences []string) []int {
	out := make([]int, len(sentences))
	for i, s := range sentences {
		out[i] = EstimateTokens(s)
	}
	return out
}

// SentenceSplit splits text into sentences on ., !, ?, newlines, and semicolons.
// Keeps the delimiter attached to the preceding sentence.
func SentenceSplit(text string) []string {
	if len(text) == 0 {
		return nil
	}

	var sentences []string
	start := 0
	runes := []rune(text)
	n := len(runes)

	for i := 0; i < n; i++ {
		r := runes[i]
		isSplit := false

		switch r {
		case '\n':
			isSplit = true
		case '.', '!', '?':
			// dont split on decimals like 3.5
			if r == '.' && i > 0 && i < n-1 {
				prev := runes[i-1]
				next := runes[i+1]
				if prev >= '0' && prev <= '9' && next >= '0' && next <= '9' {
					continue
				}
			}
			isSplit = true
		case ';':
			isSplit = true
		}

		if isSplit {
			end := i + 1
			// skip trailing whitespace after delimiter for cleaner splits
			for end < n && (runes[end] == ' ' || runes[end] == '\t' || runes[end] == '\r') {
				end++
			}
			seg := strings.TrimSpace(string(runes[start:end]))
			if len(seg) > 0 {
				sentences = append(sentences, seg)
			}
			start = end
			i = end - 1 // loop increments
		}
	}

	// trailing text
	if start < n {
		seg := strings.TrimSpace(string(runes[start:]))
		if len(seg) > 0 {
			sentences = append(sentences, seg)
		}
	}

	return sentences
}

// Compact compresses text to fit within maxTokens by embedding all sentences,
// ranking by similarity to query (or mean embedding if query is empty),
// and greedily selecting top sentences that fit the budget.
// Returns the compacted text with sentences in original order.
func Compact(model *SimpleInt8Model512, text string, maxTokens int, query string) (string, error) {
	sentences := SentenceSplit(text)
	if len(sentences) == 0 {
		return "", nil
	}

	result, err := CompactSentences(model, sentences, maxTokens, query)
	if err != nil {
		return "", err
	}

	return strings.Join(result, " "), nil
}

// CompactSentences takes pre-split sentences, embeds them all in batch,
// ranks by similarity, and returns the subset that fits within maxTokens
// in their original order.
func CompactSentences(model *SimpleInt8Model512, sentences []string, maxTokens int, query string) ([]string, error) {
	n := len(sentences)
	if n == 0 {
		return nil, nil
	}

	// if everything fits, return as-is
	totalTokens := 0
	tokenCounts := make([]int, n)
	for i, s := range sentences {
		t := EstimateTokens(s)
		tokenCounts[i] = t
		totalTokens += t
	}
	if totalTokens <= maxTokens {
		return sentences, nil
	}

	// batch embed all sentences
	embeddings, err := model.EmbedBatchInt8(sentences)
	if err != nil {
		return nil, err
	}

	// compute reference embedding
	var refVec simd.Vec512
	var refScale float32

	if query != "" {
		qEmb, err := model.EmbedInt8(query)
		if err != nil {
			return nil, err
		}
		copy(refVec[:], qEmb.Vector)
		refScale = qEmb.Scale
	} else {
		// mean of all sentence embeddings (in float32 space, then quantize)
		mean := make([]float32, Int8EmbeddingDim)
		for _, emb := range embeddings {
			for j := 0; j < Int8EmbeddingDim; j++ {
				mean[j] += float32(emb.Vector[j]) * emb.Scale
			}
		}
		invN := 1.0 / float32(n)
		maxAbs := float32(0)
		for j := range mean {
			mean[j] *= invN
			a := float32(math.Abs(float64(mean[j])))
			if a > maxAbs {
				maxAbs = a
			}
		}
		refScale = maxAbs / 127.0
		if refScale == 0 {
			refScale = 1.0
		}
		invScale := 1.0 / refScale
		for j := range mean {
			q := int8(math.Round(float64(mean[j] * invScale)))
			refVec[j] = q
		}
	}

	// score all sentences by cosine similarity to reference
	scored := make([]scoredSentence, n)
	for i, emb := range embeddings {
		var v simd.Vec512
		copy(v[:], emb.Vector)
		// use SIMD dot product for speed
		dot := simd.Dot512(&refVec, &v)
		// compute norms for proper cosine
		normRef := simd.Dot512(&refVec, &refVec)
		normV := simd.Dot512(&v, &v)
		denom := float32(math.Sqrt(float64(normRef))) * float32(math.Sqrt(float64(normV)))
		var sim float32
		if denom > 0 {
			sim = float32(dot) * refScale * emb.Scale / (float32(math.Sqrt(float64(float32(normRef)*refScale*refScale))) * float32(math.Sqrt(float64(float32(normV)*emb.Scale*emb.Scale))))
		}
		scored[i] = scoredSentence{
			index:  i,
			text:   sentences[i],
			tokens: tokenCounts[i],
			score:  sim,
		}
	}

	// sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// greedily select top sentences within budget
	budget := maxTokens
	for i := range scored {
		if scored[i].tokens <= budget {
			scored[i].selected = true
			budget -= scored[i].tokens
		}
		if budget <= 0 {
			break
		}
	}

	// re-sort by original index to preserve order
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].index < scored[j].index
	})

	var result []string
	for _, s := range scored {
		if s.selected {
			result = append(result, s.text)
		}
	}

	if len(result) == 0 && n > 0 {
		// at minimum return the highest-scored sentence truncated
		sort.Slice(scored, func(i, j int) bool {
			return scored[i].score > scored[j].score
		})
		best := scored[0].text
		// rough truncate to fit
		charLimit := int(float64(maxTokens) * 3.5)
		if charLimit < len(best) {
			best = best[:charLimit]
		}
		result = append(result, best)
	}

	return result, nil
}

// CompactWithConfig provides full configuration control.
func CompactWithConfig(model *SimpleInt8Model512, text string, cfg CompactConfig) (string, error) {
	sentences := SentenceSplit(text)
	if len(sentences) == 0 {
		return "", nil
	}

	result, err := CompactSentencesWithConfig(model, sentences, cfg)
	if err != nil {
		return "", err
	}

	return strings.Join(result, " "), nil
}

// CompactSentencesWithConfig is the full-featured compaction entry point.
func CompactSentencesWithConfig(model *SimpleInt8Model512, sentences []string, cfg CompactConfig) ([]string, error) {
	return CompactSentences(model, sentences, cfg.MaxTokens, cfg.Query)
}

// CompactBatchInt8Cosine is the low-level hot path: given pre-computed embeddings,
// a reference vector, and token counts, returns selected indices within budget.
// This is useful when you already have embeddings (e.g. from GPU pipeline).
func CompactBatchInt8Cosine(
	embeddings []*Int8Result512,
	refVec *simd.Vec512,
	refScale float32,
	tokenCounts []int,
	maxTokens int,
) []int {
	n := len(embeddings)
	if n == 0 {
		return nil
	}

	type scored struct {
		idx   int
		score float32
		toks  int
	}

	items := make([]scored, n)
	for i, emb := range embeddings {
		var v simd.Vec512
		copy(v[:], emb.Vector)
		dot := simd.Dot512(refVec, &v)
		normRef := simd.Dot512(refVec, refVec)
		normV := simd.Dot512(&v, &v)

		var sim float32
		denomRef := float32(normRef) * refScale * refScale
		denomV := float32(normV) * emb.Scale * emb.Scale
		if denomRef > 0 && denomV > 0 {
			sim = float32(dot) * refScale * emb.Scale / (float32(math.Sqrt(float64(denomRef))) * float32(math.Sqrt(float64(denomV))))
		}
		items[i] = scored{idx: i, score: sim, toks: tokenCounts[i]}
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	budget := maxTokens
	var selected []int
	for _, it := range items {
		if it.toks <= budget {
			selected = append(selected, it.idx)
			budget -= it.toks
		}
		if budget <= 0 {
			break
		}
	}

	sort.Ints(selected)
	return selected
}

// Vec512FromInt8 converts an Int8Result512 vector into a simd.Vec512 without allocation
// by reinterpreting the underlying memory.
func Vec512FromInt8(emb *Int8Result512) *simd.Vec512 {
	if len(emb.Vector) != Int8EmbeddingDim {
		return nil
	}
	return (*simd.Vec512)(unsafe.Pointer(&emb.Vector[0]))
}
