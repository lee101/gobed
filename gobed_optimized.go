package gobed

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// OptimizedEmbeddingModel extends the base model with caching and batch processing
type OptimizedEmbeddingModel struct {
	*EmbeddingModel
	cache          *TokenPatternCache
	batchProcessor *GPUBatchProcessor
	mu             sync.RWMutex
	
	// Performance metrics
	totalQueries   uint64
	cacheHits      uint64
	avgLatency     time.Duration
}

// LoadOptimizedModel loads the model with all optimizations enabled
func LoadOptimizedModel() (*OptimizedEmbeddingModel, error) {
	// Load base model
	baseModel, err := LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load base model: %v", err)
	}
	
	// Initialize cache (try to load precomputed data)
	cache, err := NewTokenPatternCache("token_frequency_cache.json", "precomputed_embeddings.json")
	if err != nil {
		log.Printf("Warning: Could not load token cache: %v. Creating empty cache.", err)
		cache = &TokenPatternCache{
			stopwordSet:  make(map[int]bool),
			singleTokens: make(map[int]*CachedEmbedding),
			bigrams:      make(map[string]*CachedEmbedding),
			trigrams:     make(map[string]*CachedEmbedding),
			fourgrams:    make(map[string]*CachedEmbedding),
		}
	}
	
	optimized := &OptimizedEmbeddingModel{
		EmbeddingModel: baseModel,
		cache:          cache,
	}
	
	// Initialize GPU batch processor if available
	optimized.batchProcessor = NewGPUBatchProcessor(baseModel, cache)
	
	// Warm up cache with common queries
	optimized.WarmupCache()
	
	log.Printf("Optimized model loaded with cache (%d stopwords, %d singles, %d bigrams)",
		len(cache.stopwords), len(cache.singleTokens), len(cache.bigrams))
	
	return optimized, nil
}

// WarmupCache preloads common search patterns
func (m *OptimizedEmbeddingModel) WarmupCache() {
	commonQueries := []string{
		// Common AI image search terms
		"anime", "portrait", "landscape", "abstract", "realistic",
		"fantasy", "cyberpunk", "cute", "beautiful", "epic",
		"girl", "woman", "man", "character", "person",
		"cat", "dog", "dragon", "robot", "alien",
		"sunset", "mountain", "ocean", "forest", "city",
		"neon", "dark", "bright", "colorful", "monochrome",
		
		// Common bigrams
		"anime girl", "portrait woman", "abstract art",
		"landscape sunset", "fantasy dragon", "cute cat",
		"cyberpunk city", "beautiful woman", "epic fantasy",
		
		// Common trigrams
		"anime girl portrait", "beautiful sunset landscape",
		"cute cat playing", "fantasy dragon flying",
	}
	
	log.Printf("Warming up cache with %d common patterns...", len(commonQueries))
	start := time.Now()
	
	for _, query := range commonQueries {
		_, _ = m.EmbedOptimized(query)
	}
	
	elapsed := time.Since(start)
	avgMs := float64(elapsed.Milliseconds()) / float64(len(commonQueries))
	log.Printf("Cache warmed up in %v (%.2fms per query)", elapsed, avgMs)
	
	if m.cache != nil {
		stats := m.cache.GetStats()
		log.Printf("Cache stats after warmup: hits=%d, hit_rate=%.1f%%",
			stats["hits"], stats["hit_rate"])
	}
}

// EmbedOptimized generates embeddings with all optimizations
func (m *OptimizedEmbeddingModel) EmbedOptimized(text string) ([]float32, error) {
	start := time.Now()
	defer func() {
		m.mu.Lock()
		m.totalQueries++
		m.avgLatency = (m.avgLatency + time.Since(start)) / 2
		m.mu.Unlock()
	}()
	
	// Use the optimized embedding function with cache
	return m.OptimizedEmbedding(text, m.cache)
}

// EmbedInt8Optimized generates quantized embeddings with optimizations
func (m *OptimizedEmbeddingModel) EmbedInt8Optimized(text string) (*EmbedInt8Result, error) {
	embedding, err := m.EmbedOptimized(text)
	if err != nil {
		return nil, err
	}
	
	quantized, scale := quantizeEmbedding(embedding)
	return &EmbedInt8Result{
		Vector: quantized,
		Scale:  scale,
	}, nil
}

// BatchEmbed processes multiple texts efficiently
func (m *OptimizedEmbeddingModel) BatchEmbed(texts []string) ([]*EmbedInt8Result, error) {
	if m.batchProcessor != nil {
		return m.batchProcessor.ProcessBatch(texts)
	}
	
	// Fallback to sequential processing
	results := make([]*EmbedInt8Result, len(texts))
	for i, text := range texts {
		result, err := m.EmbedInt8Optimized(text)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}

// PrecomputePatterns adds new patterns to the cache
func (m *OptimizedEmbeddingModel) PrecomputePatterns(patterns []string) {
	tokenizedPatterns := make([][]int, 0, len(patterns))
	
	for _, pattern := range patterns {
		cleanText := normalizeText(pattern)
		if cleanText == "" {
			continue
		}
		
		encoding, err := m.tokenizer.EncodeSingle(cleanText, false)
		if err != nil {
			continue
		}
		
		tokens := make([]int, len(encoding.Ids))
		for i, id := range encoding.Ids {
			tokens[i] = int(id)
		}
		
		if len(tokens) > 0 && len(tokens) <= 4 {
			tokenizedPatterns = append(tokenizedPatterns, tokens)
		}
	}
	
	if len(tokenizedPatterns) > 0 {
		m.cache.PrecomputeCommonPatterns(m.EmbeddingModel, tokenizedPatterns)
	}
}

// GetStats returns performance statistics
func (m *OptimizedEmbeddingModel) GetStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	stats := map[string]interface{}{
		"total_queries": m.totalQueries,
		"avg_latency_ms": m.avgLatency.Milliseconds(),
	}
	
	if m.cache != nil {
		cacheStats := m.cache.GetStats()
		stats["cache_hits"] = cacheStats["hits"]
		stats["cache_hit_rate"] = cacheStats["hit_rate"]
		stats["cache_size"] = map[string]int{
			"singles":   len(m.cache.singleTokens),
			"bigrams":   len(m.cache.bigrams),
			"trigrams":  len(m.cache.trigrams),
			"fourgrams": len(m.cache.fourgrams),
			"stopwords": len(m.cache.stopwords),
		}
	}
	
	if m.batchProcessor != nil {
		batchStats := m.batchProcessor.GetStats()
		stats["batch_processor"] = batchStats
	}
	
	return stats
}

// OptimizeForProduction applies production-ready optimizations
func (m *OptimizedEmbeddingModel) OptimizeForProduction(maxCacheSize int, gpuEnabled bool) {
	// Adjust cache size
	if m.cache != nil {
		m.cache.mu.Lock()
		// Evict least recently used entries if needed
		if maxCacheSize > 0 && len(m.cache.singleTokens) > maxCacheSize {
			// Simple eviction: remove entries with low use count
			for key, entry := range m.cache.singleTokens {
				if entry.UseCount < 2 {
					delete(m.cache.singleTokens, key)
				}
				if len(m.cache.singleTokens) <= maxCacheSize {
					break
				}
			}
		}
		m.cache.mu.Unlock()
	}
	
	// Enable/disable GPU processing
	if !gpuEnabled && m.batchProcessor != nil {
		m.batchProcessor = nil
		log.Println("GPU processing disabled")
	}
	
	log.Printf("Production optimizations applied: maxCache=%d, GPU=%v", maxCacheSize, gpuEnabled)
}

// FastSearch performs optimized search for AI images
func (m *OptimizedEmbeddingModel) FastSearch(query string, limit int) ([]float32, error) {
	// Check if query is in top frequent patterns
	if m.cache != nil {
		// Try exact match first
		cleanQuery := normalizeText(query)
		encoding, err := m.tokenizer.EncodeSingle(cleanQuery, false)
		if err == nil && len(encoding.Ids) <= 4 {
			tokens := make([]int, len(encoding.Ids))
			for i, id := range encoding.Ids {
				tokens[i] = int(id)
			}
			
			if cached, found := m.cache.GetCachedEmbedding(tokens); found {
				m.mu.Lock()
				m.cacheHits++
				m.mu.Unlock()
				return cached.Vector, nil
			}
		}
	}
	
	// Fall back to optimized embedding
	return m.EmbedOptimized(query)
}