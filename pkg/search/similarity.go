package search

import (
	"math"
)

// CosineSimilarity calculates cosine similarity between two float32 vectors
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}
	
	dotProduct := float32(0.0)
	normA := float32(0.0)
	normB := float32(0.0)
	
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}
	
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// CosineSimilarityFloat64 calculates cosine similarity between two float64 vectors
func CosineSimilarityFloat64(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}
	
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0
	
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}