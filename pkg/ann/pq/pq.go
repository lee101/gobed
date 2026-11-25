package pq

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// ProductQuantizer performs product quantization on vectors
type ProductQuantizer struct {
	M         int           // Number of subquantizers
	KSub      int           // Codebook size per subquantizer (typically 256 for 8-bit)
	D         int           // Vector dimension
	DSub      int           // Dimension per subquantizer
	Codebooks [][][]float32 // Codebooks[m][k][d] - m subquantizers, k codes, d dims
	Trained   bool
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

// Train trains the product quantizer on a set of vectors. Subquantizers are
// independent, so they are trained concurrently across GOMAXPROCS workers.
func (pq *ProductQuantizer) Train(vectors [][]float32) {
	n := len(vectors)
	if n < pq.KSub {
		panic("not enough training data for PQ")
	}

	workers := min(runtime.GOMAXPROCS(0), pq.M)
	next := make(chan int, pq.M)
	for m := 0; m < pq.M; m++ {
		next <- m
	}
	close(next)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sub := make([]float32, n*pq.DSub)
			for m := range next {
				off := m * pq.DSub
				for i, v := range vectors {
					copy(sub[i*pq.DSub:(i+1)*pq.DSub], v[off:off+pq.DSub])
				}
				pq.trainSubquantizer(m, sub, rand.New(rand.NewSource(int64(m)+1)))
			}
		}()
	}
	wg.Wait()

	pq.Trained = true
}

// trainSubquantizer runs k-means++ then Lloyd on one subspace. sub holds n
// contiguous DSub-dim subvectors. Stops when <=0.1% of points move.
func (pq *ProductQuantizer) trainSubquantizer(m int, sub []float32, rng *rand.Rand) {
	ds := pq.DSub
	n := len(sub) / ds
	k := pq.KSub
	centroids := pq.Codebooks[m]
	pq.seedPlusPlus(sub, centroids, rng)

	assignments := make([]int, n)
	for i := range assignments {
		assignments[i] = -1
	}
	sums := make([]float32, k*ds)
	counts := make([]int, k)
	tol := n / 1000

	for iter := 0; iter < 25; iter++ {
		changed := 0
		for i := 0; i < n; i++ {
			v := sub[i*ds : (i+1)*ds]
			best, bestD := 0, float32(math.MaxFloat32)
			for ki := 0; ki < k; ki++ {
				if d := squaredDistance(v, centroids[ki]); d < bestD {
					bestD, best = d, ki
				}
			}
			if assignments[i] != best {
				assignments[i] = best
				changed++
			}
		}
		if iter > 0 && changed <= tol {
			break
		}

		clear(sums)
		clear(counts)
		for i, a := range assignments {
			counts[a]++
			acc := sums[a*ds : (a+1)*ds]
			for d, x := range sub[i*ds : (i+1)*ds] {
				acc[d] += x
			}
		}
		for ki := 0; ki < k; ki++ {
			if counts[ki] == 0 {
				idx := rng.Intn(n)
				copy(centroids[ki], sub[idx*ds:(idx+1)*ds])
				continue
			}
			inv := 1 / float32(counts[ki])
			for d := range centroids[ki] {
				centroids[ki][d] = sums[ki*ds+d] * inv
			}
		}
	}
}

// seedPlusPlus is incremental k-means++: each step folds only the newest
// centroid into the per-point nearest distance (O(n*k) instead of O(n*k^2)).
func (pq *ProductQuantizer) seedPlusPlus(sub []float32, centroids [][]float32, rng *rand.Rand) {
	ds := pq.DSub
	n := len(sub) / ds
	first := rng.Intn(n)
	copy(centroids[0], sub[first*ds:(first+1)*ds])
	dist := make([]float32, n)
	for i := range dist {
		dist[i] = math.MaxFloat32
	}
	for ki := 1; ki < len(centroids); ki++ {
		c := centroids[ki-1]
		total := 0.0
		for i := 0; i < n; i++ {
			if d := squaredDistance(sub[i*ds:(i+1)*ds], c); d < dist[i] {
				dist[i] = d
			}
			total += float64(dist[i])
		}
		chosen := rng.Intn(n)
		if total > 0 {
			threshold := rng.Float64() * total
			cum := 0.0
			for i := 0; i < n; i++ {
				cum += float64(dist[i])
				if cum >= threshold {
					chosen = i
					break
				}
			}
		}
		copy(centroids[ki], sub[chosen*ds:(chosen+1)*ds])
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
			dist := squaredDistance(subvec, pq.Codebooks[m][k])
			if dist < minDist {
				minDist = dist
				bestK = uint8(k)
			}
		}

		codes[m] = bestK
	}

	return codes
}

// EncodeBatch encodes multiple vectors across GOMAXPROCS workers.
func (pq *ProductQuantizer) EncodeBatch(vectors [][]float32) [][]uint8 {
	codes := make([][]uint8, len(vectors))
	workers := max(1, min(runtime.GOMAXPROCS(0), len(vectors)))
	chunk := (len(vectors) + workers - 1) / workers
	var wg sync.WaitGroup
	for s := 0; s < len(vectors); s += chunk {
		e := min(s+chunk, len(vectors))
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				codes[i] = pq.Encode(vectors[i])
			}
		}(s, e)
	}
	wg.Wait()
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

// euclideanDistance computes L2 distance between two vectors
func euclideanDistance(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// squaredDistance omits the sqrt; nearest-codeword selection only compares.
func squaredDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
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

// ComputeIPTable precomputes per-subspace inner products of the query with
// every codeword. ADCTable.Distance then returns the approximate dot product
// of the query with the encoded vector (higher is more similar).
func (pq *ProductQuantizer) ComputeIPTable(query []float32) *ADCTable {
	if !pq.Trained {
		panic("PQ must be trained before computing IP table")
	}
	table := &ADCTable{Distances: make([][]float32, pq.M)}
	flat := make([]float32, pq.M*pq.KSub)
	for m := 0; m < pq.M; m++ {
		row := flat[m*pq.KSub : (m+1)*pq.KSub]
		q := query[m*pq.DSub : (m+1)*pq.DSub]
		for k, c := range pq.Codebooks[m] {
			var dot float32
			for d, x := range q {
				dot += x * c[d]
			}
			row[k] = dot
		}
		table.Distances[m] = row
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
	PQ    *ProductQuantizer
	Codes []PQCode
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
