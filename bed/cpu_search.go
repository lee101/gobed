package main

import (
	"math"
	"sort"
)

// cpuSearch performs brute-force cosine similarity search on CPU
func cpuSearch(query []int8, documents []int8, numDocs, dim, topK int, outIndices []int32, outScores []float32) int {
	type Result struct {
		Index int
		Score float32
	}

	results := make([]Result, 0, numDocs)

	// Compute cosine similarity for each document
	for i := 0; i < numDocs; i++ {
		docStart := i * dim
		docEnd := docStart + dim

		score := cosineSimilarityInt8(query, documents[docStart:docEnd])
		results = append(results, Result{
			Index: i,
			Score: score,
		})
	}

	// Sort by score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Copy top K results
	n := topK
	if n > len(results) {
		n = len(results)
	}

	for i := 0; i < n; i++ {
		outIndices[i] = int32(results[i].Index)
		outScores[i] = results[i].Score
	}

	return n
}

// cosineSimilarityInt8 computes cosine similarity between two int8 vectors
func cosineSimilarityInt8(a, b []int8) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct int32
	var normA, normB int32

	for i := range a {
		ai := int32(a[i])
		bi := int32(b[i])

		dotProduct += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct) / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}