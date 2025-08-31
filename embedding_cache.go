package gobed

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// TokenPatternCache stores precomputed embeddings for common token patterns
type TokenPatternCache struct {
	mu sync.RWMutex

	// Stopwords to filter out
	stopwordSet map[int]bool
	stopwords   []int

	// Single token embeddings (most common tokens)
	singleTokens map[int]*CachedEmbedding

	// N-gram pattern embeddings
	bigrams   map[string]*CachedEmbedding // key: "token1_token2"
	trigrams  map[string]*CachedEmbedding // key: "token1_token2_token3"
	fourgrams map[string]*CachedEmbedding // key: "token1_token2_token3_token4"

	// Statistics
	hits   uint64
	misses uint64

	// GPU-friendly batch buffers
	batchBuffer   [][]int    // For batching token lookups
	batchMu       sync.Mutex
	batchCapacity int
}

// CachedEmbedding stores a precomputed embedding
type CachedEmbedding struct {
	Vector    []float32
	VectorI8  []int8  // Quantized version
	Scale     float32 // Quantization scale
	UseCount  uint32  // Track usage for cache eviction
	LastUsed  int64   // Unix timestamp
}

// TokenFrequencyData represents the frequency analysis results
type TokenFrequencyData struct {
	TokenizerName string                `json:"tokenizer_name"`
	VocabSize     int                   `json:"vocab_size"`
	Stopwords     []int                 `json:"stopwords"`
	SingleTokens  TokenFrequencySection `json:"single_tokens"`
	Bigrams       TokenPatternSection   `json:"bigrams"`
	Trigrams      TokenPatternSection   `json:"trigrams"`
	Fourgrams     TokenPatternSection   `json:"fourgrams"`
	Stats         map[string]int        `json:"stats"`
}

type TokenFrequencySection struct {
	IDs    []int `json:"ids"`
	Counts []int `json:"counts"`
}

type TokenPatternSection struct {
	Patterns [][]int `json:"patterns"`
	Counts   []int   `json:"counts"`
}

// PrecomputedEmbeddings stores actual embedding vectors
type PrecomputedEmbeddings struct {
	Single   map[string][]float32 `json:"single"`
	Bigram   map[string][]float32 `json:"bigram"`
	Trigram  map[string][]float32 `json:"trigram"`
	Fourgram map[string][]float32 `json:"fourgram"`
}

// NewTokenPatternCache creates a new cache with precomputed embeddings
func NewTokenPatternCache(freqFile, embeddingFile string) (*TokenPatternCache, error) {
	cache := &TokenPatternCache{
		stopwordSet:   make(map[int]bool),
		singleTokens:  make(map[int]*CachedEmbedding),
		bigrams:       make(map[string]*CachedEmbedding),
		trigrams:      make(map[string]*CachedEmbedding),
		fourgrams:     make(map[string]*CachedEmbedding),
		batchCapacity: 1000,
		batchBuffer:   make([][]int, 0, 1000),
	}

	// Load frequency data
	if err := cache.loadFrequencyData(freqFile); err != nil {
		return nil, fmt.Errorf("failed to load frequency data: %v", err)
	}

	// Load precomputed embeddings if available
	if embeddingFile != "" {
		if err := cache.loadEmbeddings(embeddingFile); err != nil {
			log.Printf("Warning: Could not load precomputed embeddings: %v", err)
			// Continue without precomputed embeddings
		}
	}

	return cache, nil
}

func (c *TokenPatternCache) loadFrequencyData(filename string) error {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	var freqData TokenFrequencyData
	if err := json.Unmarshal(data, &freqData); err != nil {
		return err
	}

	// Load stopwords
	c.stopwords = freqData.Stopwords
	for _, sw := range freqData.Stopwords {
		c.stopwordSet[sw] = true
	}

	log.Printf("Loaded %d stopwords", len(c.stopwords))

	// Initialize structures for top tokens (we'll populate embeddings later)
	for i, tokenID := range freqData.SingleTokens.IDs {
		if i >= 5000 { // Limit to top 5000
			break
		}
		c.singleTokens[tokenID] = &CachedEmbedding{
			UseCount: uint32(freqData.SingleTokens.Counts[i]),
		}
	}

	// Store pattern keys for later embedding lookup
	for i, pattern := range freqData.Bigrams.Patterns {
		if i >= 5000 {
			break
		}
		key := makePatternKey(pattern)
		c.bigrams[key] = &CachedEmbedding{
			UseCount: uint32(freqData.Bigrams.Counts[i]),
		}
	}

	for i, pattern := range freqData.Trigrams.Patterns {
		if i >= 3000 {
			break
		}
		key := makePatternKey(pattern)
		c.trigrams[key] = &CachedEmbedding{
			UseCount: uint32(freqData.Trigrams.Counts[i]),
		}
	}

	for i, pattern := range freqData.Fourgrams.Patterns {
		if i >= 1000 {
			break
		}
		key := makePatternKey(pattern)
		c.fourgrams[key] = &CachedEmbedding{
			UseCount: uint32(freqData.Fourgrams.Counts[i]),
		}
	}

	log.Printf("Loaded frequency data: %d singles, %d bigrams, %d trigrams, %d fourgrams",
		len(c.singleTokens), len(c.bigrams), len(c.trigrams), len(c.fourgrams))

	return nil
}

func (c *TokenPatternCache) loadEmbeddings(filename string) error {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	var embeddings PrecomputedEmbeddings
	if err := json.Unmarshal(data, &embeddings); err != nil {
		return err
	}

	// Load single token embeddings
	for tokenIDStr, vec := range embeddings.Single {
		var tokenID int
		fmt.Sscanf(tokenIDStr, "%d", &tokenID)
		if cached, exists := c.singleTokens[tokenID]; exists {
			cached.Vector = vec
			cached.VectorI8, cached.Scale = quantizeEmbedding(vec)
		}
	}

	// Load bigram embeddings
	for key, vec := range embeddings.Bigram {
		if cached, exists := c.bigrams[key]; exists {
			cached.Vector = vec
			cached.VectorI8, cached.Scale = quantizeEmbedding(vec)
		}
	}

	// Load trigram embeddings
	for key, vec := range embeddings.Trigram {
		if cached, exists := c.trigrams[key]; exists {
			cached.Vector = vec
			cached.VectorI8, cached.Scale = quantizeEmbedding(vec)
		}
	}

	log.Printf("Loaded precomputed embeddings")
	return nil
}

// FilterStopwords removes stopwords from token sequence if text is long
func (c *TokenPatternCache) FilterStopwords(tokens []int, textLength int) []int {
	// Only filter stopwords for longer texts
	if textLength < 10 || len(tokens) < 15 {
		return tokens
	}

	filtered := make([]int, 0, len(tokens))
	for _, token := range tokens {
		if !c.stopwordSet[token] {
			filtered = append(filtered, token)
		}
	}

	// Keep at least 30% of original tokens
	minTokens := len(tokens) * 3 / 10
	if len(filtered) < minTokens {
		// Add back some stopwords from the end
		for i := len(tokens) - 1; i >= 0 && len(filtered) < minTokens; i-- {
			if c.stopwordSet[tokens[i]] {
				filtered = append(filtered, tokens[i])
			}
		}
	}

	return filtered
}

// GetCachedEmbedding tries to retrieve cached embedding for token pattern
func (c *TokenPatternCache) GetCachedEmbedding(tokens []int) (*CachedEmbedding, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	now := time.Now().Unix()

	// Try different n-gram sizes
	switch len(tokens) {
	case 1:
		if emb, exists := c.singleTokens[tokens[0]]; exists && emb.Vector != nil {
			atomic.AddUint64(&c.hits, 1)
			atomic.AddUint32(&emb.UseCount, 1)
			atomic.StoreInt64(&emb.LastUsed, now)
			return emb, true
		}
	case 2:
		key := makePatternKey(tokens)
		if emb, exists := c.bigrams[key]; exists && emb.Vector != nil {
			atomic.AddUint64(&c.hits, 1)
			atomic.AddUint32(&emb.UseCount, 1)
			atomic.StoreInt64(&emb.LastUsed, now)
			return emb, true
		}
	case 3:
		key := makePatternKey(tokens)
		if emb, exists := c.trigrams[key]; exists && emb.Vector != nil {
			atomic.AddUint64(&c.hits, 1)
			atomic.AddUint32(&emb.UseCount, 1)
			atomic.StoreInt64(&emb.LastUsed, now)
			return emb, true
		}
	case 4:
		key := makePatternKey(tokens)
		if emb, exists := c.fourgrams[key]; exists && emb.Vector != nil {
			atomic.AddUint64(&c.hits, 1)
			atomic.AddUint32(&emb.UseCount, 1)
			atomic.StoreInt64(&emb.LastUsed, now)
			return emb, true
		}
	}

	// Try to find partial matches for longer sequences
	if len(tokens) > 4 {
		// Check first 4 tokens
		if emb, found := c.GetCachedEmbedding(tokens[:4]); found {
			return emb, true
		}
		// Check last 4 tokens
		if emb, found := c.GetCachedEmbedding(tokens[len(tokens)-4:]); found {
			return emb, true
		}
	}

	atomic.AddUint64(&c.misses, 1)
	return nil, false
}

// ComputeEmbeddingWithCache computes embedding using cache where possible
func (c *TokenPatternCache) ComputeEmbeddingWithCache(tokens []int, computeFn func([]int) ([]float32, error)) ([]float32, error) {
	// Check if entire sequence is cached
	if cached, found := c.GetCachedEmbedding(tokens); found {
		return cached.Vector, nil
	}

	// For longer sequences, try to use cached sub-patterns
	if len(tokens) > 4 {
		return c.computeWithPartialCache(tokens, computeFn)
	}

	// No cache hit, compute normally
	return computeFn(tokens)
}

// computeWithPartialCache combines cached n-grams with computed embeddings
func (c *TokenPatternCache) computeWithPartialCache(tokens []int, computeFn func([]int) ([]float32, error)) ([]float32, error) {
	const embDim = 1024
	// Use buffer pool for result
	result := GetEmbedBuffer()
	if len(result) < embDim {
		result = make([]float32, embDim)
	} else {
		result = result[:embDim]
	}
	defer PutEmbedBuffer(result)
	covered := make([]bool, len(tokens))
	partialCount := 0

	// Try to cover tokens with cached n-grams (greedy approach)
	i := 0
	for i < len(tokens) {
		bestMatch := 0
		var bestEmb *CachedEmbedding

		// Try 4-gram, then 3-gram, then 2-gram, then single
		for n := 4; n >= 1; n-- {
			if i+n > len(tokens) {
				continue
			}

			subTokens := tokens[i : i+n]
			if emb, found := c.GetCachedEmbedding(subTokens); found {
				bestMatch = n
				bestEmb = emb
				break
			}
		}

		if bestMatch > 0 {
			// Add cached embedding to result
			for j := 0; j < embDim; j++ {
				result[j] += bestEmb.Vector[j]
			}
			partialCount++

			// Mark tokens as covered
			for j := i; j < i+bestMatch; j++ {
				covered[j] = true
			}
			i += bestMatch
		} else {
			i++
		}
	}

	// Compute embeddings for uncovered tokens
	uncoveredTokens := make([]int, 0)
	for i, token := range tokens {
		if !covered[i] {
			uncoveredTokens = append(uncoveredTokens, token)
		}
	}

	if len(uncoveredTokens) > 0 {
		uncoveredEmb, err := computeFn(uncoveredTokens)
		if err != nil {
			return nil, err
		}
		for j := 0; j < embDim; j++ {
			result[j] += uncoveredEmb[j]
		}
		partialCount++
	}

	// Average the embeddings
	if partialCount > 0 {
		scale := 1.0 / float32(partialCount)
		for j := 0; j < embDim; j++ {
			result[j] *= scale
		}
	}

	return result, nil
}

// PrecomputeCommonPatterns adds embeddings for common patterns
func (c *TokenPatternCache) PrecomputeCommonPatterns(model *EmbeddingModel, patterns [][]int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Precomputing embeddings for %d patterns...", len(patterns))
	computed := 0

	for _, pattern := range patterns {
		key := makePatternKey(pattern)
		
		// Skip if already computed
		var targetMap map[string]*CachedEmbedding
		switch len(pattern) {
		case 2:
			targetMap = c.bigrams
		case 3:
			targetMap = c.trigrams
		case 4:
			targetMap = c.fourgrams
		default:
			continue
		}

		if existing, exists := targetMap[key]; exists && existing.Vector != nil {
			continue
		}

		// Compute embedding
		embedding, err := model.computeEmbedding(pattern)
		if err != nil {
			continue
		}

		// Store in cache
		quantized, scale := quantizeEmbedding(embedding)
		cached := &CachedEmbedding{
			Vector:   embedding,
			VectorI8: quantized,
			Scale:    scale,
			UseCount: 1,
			LastUsed: time.Now().Unix(),
		}

		targetMap[key] = cached
		computed++
	}

	log.Printf("Precomputed %d new embeddings", computed)
}

// BatchGetEmbeddings retrieves embeddings for multiple patterns in parallel
func (c *TokenPatternCache) BatchGetEmbeddings(tokenBatches [][]int) ([]*CachedEmbedding, []bool) {
	results := make([]*CachedEmbedding, len(tokenBatches))
	found := make([]bool, len(tokenBatches))

	// Use goroutines for parallel lookup (good for GPU scenarios)
	var wg sync.WaitGroup
	for i := range tokenBatches {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			if emb, ok := c.GetCachedEmbedding(tokenBatches[idx]); ok {
				results[idx] = emb
				found[idx] = true
			}
		}(i)
	}
	wg.Wait()

	return results, found
}

// GetStats returns cache statistics
func (c *TokenPatternCache) GetStats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadUint64(&c.hits)
	misses := atomic.LoadUint64(&c.misses)
	total := hits + misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(hits) / float64(total) * 100
	}

	return map[string]interface{}{
		"hits":          hits,
		"misses":        misses,
		"hit_rate":      hitRate,
		"singles":       len(c.singleTokens),
		"bigrams":       len(c.bigrams),
		"trigrams":      len(c.trigrams),
		"fourgrams":     len(c.fourgrams),
		"stopwords":     len(c.stopwords),
		"batch_capacity": c.batchCapacity,
	}
}

// Helper function to create pattern key
func makePatternKey(tokens []int) string {
	parts := make([]string, len(tokens))
	for i, t := range tokens {
		parts[i] = fmt.Sprintf("%d", t)
	}
	return strings.Join(parts, "_")
}

// OptimizedEmbedding wraps the embedding computation with caching
func (m *EmbeddingModel) OptimizedEmbedding(text string, cache *TokenPatternCache) ([]float32, error) {
	// Tokenize
	cleanText := normalizeText(text)
	if strings.TrimSpace(cleanText) == "" {
		return make([]float32, m.EmbedDim), nil
	}

	encoding, err := m.tokenizer.EncodeSingle(cleanText, false)
	if err != nil {
		return nil, err
	}

	// Use buffer pool for tokens to reduce allocations
	tokens := GetTokenBuffer()
	defer PutTokenBuffer(tokens)
	for _, id := range encoding.Ids {
		tokens = append(tokens, int(id))
	}

	// Filter stopwords if needed
	wordCount := len(strings.Fields(cleanText))
	if cache != nil && wordCount > 10 {
		tokens = cache.FilterStopwords(tokens, wordCount)
	}

	// Use cache if available
	if cache != nil {
		return cache.ComputeEmbeddingWithCache(tokens, m.computeEmbedding)
	}

	// Fallback to normal computation
	return m.computeEmbedding(tokens)
}