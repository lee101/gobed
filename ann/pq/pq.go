package pq

import (
	"math"
	"math/rand"
)

// ProductQuantizer performs product quantization on vectors
type ProductQuantizer struct {
	M           int             // Number of subquantizers
	KSub        int             // Codebook size per subquantizer (typically 256 for 8-bit)
	D           int             // Vector dimension
	DSub        int             // Dimension per subquantizer
	Codebooks   [][][]float32   // Codebooks[m][k][d] - m subquantizers, k codes, d dims
	Trained     bool
}

// NewProductQuantizer creates a new product quantizer
func NewProductQuantizer(d, m, ksub int) *ProductQuantizer {
	if d%m != 0 {
		panic("dimension must be divisible by number of subquantizers")
	}
	
	dsub := d / m
	codebooks := make([][][]float32, m)
	for i := range codebooks {
		codebooks[i] = make([][]float32, ksub)
		for j := range codebooks[i] {
			codebooks[i][j] = make([]float32, dsub)
		}
	}
	
	return &ProductQuantizer{
		M:         m,
		KSub:      ksub,
		D:         d,
		DSub:      dsub,
		Codebooks: codebooks,
	}
}

// Train trains the product quantizer on a set of vectors
func (pq *ProductQuantizer) Train(vectors [][]float32) {
	n := len(vectors)
	if n < pq.KSub*pq.M {
		panic("not enough training data for PQ")
	}
	
	// Train each subquantizer independently
	for m := 0; m < pq.M; m++ {
		startDim := m * pq.DSub
		endDim := startDim + pq.DSub
		
		// Extract subvectors for this subquantizer
		subvectors := make([][]float32, n)
		for i := range subvectors {
			subvectors[i] = vectors[i][startDim:endDim]
		}
		
		// Run k-means on subvectors
		pq.trainSubquantizer(m, subvectors)
	}
	
	pq.Trained = true
}

// trainSubquantizer trains a single subquantizer using k-means
func (pq *ProductQuantizer) trainSubquantizer(m int, subvectors [][]float32) {
	n := len(subvectors)
	k := pq.KSub
	maxIters := 25
	
	// Initialize centroids with k-means++
	centroids := pq.Codebooks[m]
	pq.initializeKMeansPlusPlus(subvectors, centroids)
	
	assignments := make([]int, n)
	
	for iter := 0; iter < maxIters; iter++ {
		// Assignment step
		changed := false
		for i, vec := range subvectors {
			minDist := float32(math.MaxFloat32)
			bestK := 0
			
			for ki := 0; ki < k; ki++ {
				dist := euclideanDistance(vec, centroids[ki])
				if dist < minDist {
					minDist = dist
					bestK = ki
				}
			}
			
			if assignments[i] != bestK {
				assignments[i] = bestK
				changed = true
			}
		}
		
		if !changed && iter > 0 {
			break
		}
		
		// Update step
		counts := make([]int, k)
		for i := range centroids {
			for j := range centroids[i] {
				centroids[i][j] = 0
			}
		}
		
		for i, vec := range subvectors {
			ki := assignments[i]
			counts[ki]++
			for d := range vec {
				centroids[ki][d] += vec[d]
			}
		}
		
		for ki := 0; ki < k; ki++ {
			if counts[ki] > 0 {
				for d := range centroids[ki] {
					centroids[ki][d] /= float32(counts[ki])
				}
			} else {
				// Reinitialize empty cluster
				idx := rand.Intn(n)
				copy(centroids[ki], subvectors[idx])
			}
		}
	}
}

// Encode encodes a vector into PQ codes
func (pq *ProductQuantizer) Encode(vec []float32) []uint8 {
	if !pq.Trained {
		panic("PQ must be trained before encoding")
	}
	
	codes := make([]uint8, pq.M)
	
	for m := 0; m < pq.M; m++ {
		startDim := m * pq.DSub
		endDim := startDim + pq.DSub
		subvec := vec[startDim:endDim]
		
		// Find nearest centroid
		minDist := float32(math.MaxFloat32)
		bestK := uint8(0)
		
		for k := 0; k < pq.KSub; k++ {
			dist := euclideanDistance(subvec, pq.Codebooks[m][k])
			if dist < minDist {
				minDist = dist
				bestK = uint8(k)
			}
		}
		
		codes[m] = bestK
	}
	
	return codes
}

// EncodeBatch encodes multiple vectors efficiently
func (pq *ProductQuantizer) EncodeBatch(vectors [][]float32) [][]uint8 {
	codes := make([][]uint8, len(vectors))
	for i := range vectors {
		codes[i] = pq.Encode(vectors[i])
	}
	return codes
}

// Decode reconstructs a vector from PQ codes
func (pq *ProductQuantizer) Decode(codes []uint8) []float32 {
	if !pq.Trained {
		panic("PQ must be trained before decoding")
	}
	
	vec := make([]float32, pq.D)
	
	for m := 0; m < pq.M; m++ {
		startDim := m * pq.DSub
		code := codes[m]
		centroid := pq.Codebooks[m][code]
		
		for d := 0; d < pq.DSub; d++ {
			vec[startDim+d] = centroid[d]
		}
	}
	
	return vec
}

// ComputeDistance computes approximate distance between PQ codes and a query
func (pq *ProductQuantizer) ComputeDistance(codes []uint8, query []float32) float32 {
	if !pq.Trained {
		panic("PQ must be trained before computing distances")
	}
	
	distance := float32(0)
	
	for m := 0; m < pq.M; m++ {
		startDim := m * pq.DSub
		endDim := startDim + pq.DSub
		subquery := query[startDim:endDim]
		
		code := codes[m]
		centroid := pq.Codebooks[m][code]
		
		distance += euclideanDistance(subquery, centroid)
	}
	
	return distance
}

// initializeKMeansPlusPlus initializes centroids using k-means++
func (pq *ProductQuantizer) initializeKMeansPlusPlus(vectors [][]float32, centroids [][]float32) {
	n := len(vectors)
	k := len(centroids)
	
	// Choose first centroid randomly
	first := rand.Intn(n)
	copy(centroids[0], vectors[first])
	
	distances := make([]float32, n)
	
	for ki := 1; ki < k; ki++ {
		// Compute distances to nearest centroid
		totalDist := float32(0)
		for i, vec := range vectors {
			minDist := float32(math.MaxFloat32)
			for j := 0; j < ki; j++ {
				dist := euclideanDistance(vec, centroids[j])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist * minDist // Square for k-means++
			totalDist += distances[i]
		}
		
		// Choose next centroid
		threshold := rand.Float32() * totalDist
		cumSum := float32(0)
		for i, vec := range vectors {
			cumSum += distances[i]
			if cumSum >= threshold {
				copy(centroids[ki], vec)
				break
			}
		}
	}
}

// euclideanDistance computes L2 distance between two vectors
func euclideanDistance(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// ADCTable stores precomputed distances for Asymmetric Distance Computation
type ADCTable struct {
	Distances [][]float32 // [M][KSub] distances
}

// ComputeADCTable precomputes distance table for a query
func (pq *ProductQuantizer) ComputeADCTable(query []float32) *ADCTable {
	if !pq.Trained {
		panic("PQ must be trained before computing ADC table")
	}
	
	table := &ADCTable{
		Distances: make([][]float32, pq.M),
	}
	
	for m := 0; m < pq.M; m++ {
		table.Distances[m] = make([]float32, pq.KSub)
		startDim := m * pq.DSub
		endDim := startDim + pq.DSub
		subquery := query[startDim:endDim]
		
		for k := 0; k < pq.KSub; k++ {
			table.Distances[m][k] = euclideanDistance(subquery, pq.Codebooks[m][k])
		}
	}
	
	return table
}

// ADCDistance computes distance using precomputed ADC table
func (table *ADCTable) Distance(codes []uint8) float32 {
	distance := float32(0)
	for m, code := range codes {
		distance += table.Distances[m][code]
	}
	return distance
}

// PQCode represents a product quantized vector
type PQCode struct {
	ID    int
	Codes []uint8
}

// EncodedIndex stores PQ-encoded vectors for search
type EncodedIndex struct {
	PQ     *ProductQuantizer
	Codes  []PQCode
}

// NewEncodedIndex creates a new PQ-encoded index
func NewEncodedIndex(pq *ProductQuantizer) *EncodedIndex {
	return &EncodedIndex{
		PQ:    pq,
		Codes: make([]PQCode, 0),
	}
}

// Add adds a PQ-encoded vector to the index
func (idx *EncodedIndex) Add(id int, codes []uint8) {
	idx.Codes = append(idx.Codes, PQCode{ID: id, Codes: codes})
}

// Search performs approximate search using ADC
func (idx *EncodedIndex) Search(query []float32, k int) []SearchResult {
	// Precompute ADC table
	table := idx.PQ.ComputeADCTable(query)
	
	// Compute distances for all codes
	type scorePair struct {
		id    int
		score float32
	}
	
	scores := make([]scorePair, len(idx.Codes))
	for i, code := range idx.Codes {
		scores[i] = scorePair{
			id:    code.ID,
			score: table.Distance(code.Codes),
		}
	}
	
	// Select top-k (partial sort)
	for i := 0; i < k && i < len(scores); i++ {
		minIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score < scores[minIdx].score {
				minIdx = j
			}
		}
		scores[i], scores[minIdx] = scores[minIdx], scores[i]
	}
	
	// Return results
	results := make([]SearchResult, k)
	for i := 0; i < k && i < len(scores); i++ {
		results[i] = SearchResult{
			ID:    scores[i].id,
			Score: scores[i].score,
		}
	}
	
	return results
}

// SearchResult represents a search result
type SearchResult struct {
	ID    int
	Score float32
}