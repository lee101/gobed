package gobed

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// refCosineInt8 transcribes the scalar C cosine_similarity_int8_avx512
// (int sums, then dot / (sqrtf(na) * sqrtf(nb)) in float32).
func refCosineInt8(a, b []uint8) float32 {
	var dot, na, nb int64
	for i := range a {
		x, y := int64(a[i])-128, int64(b[i])-128
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	sa := float32(math.Sqrt(float64(float32(na))))
	sb := float32(math.Sqrt(float64(float32(nb))))
	return float32(dot) / float32(sa*sb)
}

// refEmbedInt8 transcribes the scalar C compute_embedding_int8_avx512.
func refEmbedInt8(flat []int8, ids []int, dim, vocab int, scale float32) []uint8 {
	acc := make([]int32, dim)
	valid := 0
	for _, id := range ids {
		if id < 0 || id >= vocab {
			continue
		}
		for d := 0; d < dim; d++ {
			acc[d] += int32(flat[id*dim+d])
		}
		valid++
	}
	out := make([]uint8, dim)
	for d := range out {
		if valid == 0 {
			out[d] = 128
			continue
		}
		inv := float32(1) / float32(valid)
		n := float32(float32(float32(float32(float32(acc[d])*inv)/scale)+1) * 127.5)
		if n < 0 {
			n = 0
		}
		if n > 255 {
			n = 255
		}
		out[d] = uint8(n)
	}
	return out
}

func fillInt8Edge(r *rand.Rand, v []uint8, mode int) {
	for i := range v {
		switch mode {
		case 0:
			v[i] = uint8(r.Intn(256))
		case 1:
			v[i] = 128 // centered zero vector
		case 2:
			v[i] = 0 // all -128
		case 3:
			v[i] = 255 // all +127
		default:
			v[i] = []uint8{0, 1, 127, 128, 129, 255}[r.Intn(6)]
		}
	}
}

func TestInt8KernelCosineProperty(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 384, 511, 512, 513, 768, 1000, 1024} {
		for ma := 0; ma < 5; ma++ {
			for mb := 0; mb < 5; mb++ {
				for rep := 0; rep < 3; rep++ {
					a, b := make([]uint8, n), make([]uint8, n)
					fillInt8Edge(r, a, ma)
					fillInt8Edge(r, b, mb)
					want := refCosineInt8(a, b)
					got := CosineSimilarityInt8(a, b)
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("CosineSimilarityInt8 n=%d modes=%d/%d: %v != ref %v", n, ma, mb, got, want)
					}
					if g := cosineInt8Go(a, b); math.Float32bits(g) != math.Float32bits(want) {
						t.Fatalf("cosineInt8Go n=%d modes=%d/%d: %v != ref %v", n, ma, mb, g, want)
					}
					if haveCgoInt8 {
						if g := cosineInt8Cgo(a, b); math.Float32bits(g) != math.Float32bits(want) {
							t.Fatalf("cosineInt8Cgo n=%d modes=%d/%d: %v != ref %v", n, ma, mb, g, want)
						}
					}
					// The Go fallback takes sqrt of the exact int64 norm, so it can
					// differ in the last bits once a norm exceeds 2^24.
					if fb := CosineSimilarityInt8Fallback(a, b); math.Abs(float64(fb-want)) > 1e-6 {
						t.Fatalf("fallback n=%d modes=%d/%d: %v vs %v", n, ma, mb, fb, want)
					}
					if math.IsNaN(float64(got)) || got > 1.0001 || got < -1.0001 {
						t.Fatalf("out of range n=%d: %v", n, got)
					}
				}
			}
		}
	}
	if CosineSimilarityInt8(nil, nil) != 0 || CosineSimilarityInt8([]uint8{1}, []uint8{1, 2}) != 0 {
		t.Fatal("empty/mismatched input must return 0")
	}
}

func TestInt8KernelEmbedProperty(t *testing.T) {
	r := rand.New(rand.NewSource(8))
	const vocab = 64
	for _, dim := range []int{1, 7, 15, 16, 17, 33, 384, 512, 513, 1100} {
		flat := make([]int8, vocab*dim)
		for i := range flat {
			flat[i] = []int8{-128, 127, 0, int8(r.Intn(256) - 128)}[r.Intn(4)]
		}
		for _, scale := range []float32{0.0037, 0.5, 1, 40} {
			for _, nTok := range []int{0, 1, 2, 5, 32, 200} {
				ids := make([]int, nTok)
				for i := range ids {
					ids[i] = r.Intn(vocab+4) - 2 // includes invalid -2,-1,vocab,vocab+1
				}
				want := refEmbedInt8(flat, ids, dim, vocab, scale)
				check := func(name string, got []uint8) {
					for d := range want {
						if got[d] != want[d] {
							t.Fatalf("%s dim=%d scale=%v tokens=%d d=%d: %d != %d", name, dim, scale, nTok, d, got[d], want[d])
						}
					}
				}
				got := make([]uint8, dim)
				embedInt8Go(flat, ids, dim, vocab, got, scale)
				check("go", got)
				if nTok > 0 {
					got = make([]uint8, dim)
					embedInt8(flat, ids, dim, vocab, got, scale)
					check("dispatch", got)
					if haveCgoInt8 {
						got = make([]uint8, dim)
						embedInt8Cgo(flat, ids, dim, vocab, got, scale)
						check("cgo", got)
					}
				}
			}
		}
	}
}

func TestInt8KernelQuantizeRow(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	in := make([]float32, 1037)
	for i := range in {
		in[i] = float32(r.NormFloat64() * 3)
	}
	in[0], in[1], in[2], in[3] = 0.5, -0.5, 2.5, 1e30
	for _, zp := range []int8{-128, -3, 0, 127} {
		for _, scale := range []float32{1, 7.3, 100} {
			want := make([]int8, len(in))
			for i, x := range in {
				q := math.Round(float64(x * scale))
				q = math.Max(-1024, math.Min(1024, q)) + float64(zp)
				want[i] = int8(math.Max(-128, math.Min(127, q)))
			}
			got := make([]int8, len(in))
			quantizeRowInt8Go(in, got, scale, zp)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("go zp=%d scale=%v i=%d (%v): %d != %d", zp, scale, i, in[i], got[i], want[i])
				}
			}
			if haveCgoInt8 {
				quantizeRowInt8Cgo(in, got, scale, zp)
				for i := range want {
					if math.Abs(float64(in[i]*scale)) > 1<<30 {
						continue // C's (int32_t) cast of an out-of-range float is UB
					}
					d := int(got[i]) - int(want[i])
					// AVX-512 rounds exact .5 ties to even; scalar C/Go round away.
					if d != 0 && !(useCgoInt8 && (d == 1 || d == -1)) {
						t.Fatalf("cgo zp=%d scale=%v i=%d (%v): %d != %d", zp, scale, i, in[i], got[i], want[i])
					}
				}
			}
		}
	}
}

func TestInt8KernelModelUsesFlatWeights(t *testing.T) {
	m := benchInt8Model(100, 64)
	ids := []int{3, 99, -1, 100, 42, 3}
	want := refEmbedInt8(m.weightsInt8Flat, ids, m.EmbedDim, m.VocabSize, m.scale)
	got, err := m.ComputeEmbeddingFromTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	m2 := *m
	m2.weightsInt8Flat = nil // row-by-row model: temporary flatten path
	got2, _ := m2.ComputeEmbeddingFromTokens(ids)
	for d := range want {
		if got[d] != want[d] || got2[d] != want[d] {
			t.Fatalf("d=%d: %d %d want %d", d, got[d], got2[d], want[d])
		}
	}
}

// --- benchmarks ---------------------------------------------------------
//
// go test -run '^$' -bench 'Int8Kernel' -benchmem .
// On arm64, add -tags gobed_cgo_int8 to also build the (scalar, clang
// auto-vectorized) cgo kernels for comparison; on amd64 they are always built.

func benchInt8Model(vocab, dim int) *EmbeddingModelInt8 {
	r := rand.New(rand.NewSource(1))
	m := &EmbeddingModelInt8{VocabSize: vocab, EmbedDim: dim, scale: 0.0123, useInt8: true}
	m.weightsInt8Flat = make([]int8, vocab*dim)
	for i := range m.weightsInt8Flat {
		m.weightsInt8Flat[i] = int8(r.Intn(256) - 128)
	}
	m.weightsInt8 = make([][]int8, vocab)
	for i := range m.weightsInt8 {
		m.weightsInt8[i] = m.weightsInt8Flat[i*dim : (i+1)*dim]
	}
	return m
}

func randUint8s(r *rand.Rand, n int) []uint8 {
	v := make([]uint8, n)
	for i := range v {
		v[i] = uint8(r.Intn(256))
	}
	return v
}

type cosineImpl struct {
	name string
	fn   func(a, b []uint8) float32
}

func cosineImpls() []cosineImpl {
	impls := []cosineImpl{
		{"api", CosineSimilarityInt8},
		{"go_simd", cosineInt8Go}, // NEON SDOT Go asm on arm64, unrolled Go elsewhere
		{"go_scalar_fallback", CosineSimilarityInt8Fallback},
	}
	if haveCgoInt8 {
		impls = append(impls, cosineImpl{"cgo", cosineInt8Cgo})
	}
	return impls
}

var int8BenchSink float32

func BenchmarkInt8KernelCosine(b *testing.B) {
	r := rand.New(rand.NewSource(3))
	x, y := randUint8s(r, 512), randUint8s(r, 512)
	for _, impl := range cosineImpls() {
		b.Run("single512/"+impl.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				int8BenchSink += impl.fn(x, y)
			}
		})
	}
	// len 1 isolates per-call overhead (cgo transition vs Go call).
	for _, impl := range cosineImpls() {
		b.Run("len1_overhead/"+impl.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				int8BenchSink += impl.fn(x[:1], y[:1])
			}
		})
	}
}

func BenchmarkInt8KernelCosineBatch(b *testing.B) {
	r := rand.New(rand.NewSource(4))
	const dim = 512
	q := randUint8s(r, dim)
	for _, n := range []int{1000, 100000} {
		db := randUint8s(r, n*dim)
		for _, impl := range cosineImpls() {
			b.Run(fmt.Sprintf("n%d/%s", n, impl.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					for j := 0; j < n; j++ {
						int8BenchSink += impl.fn(q, db[j*dim:(j+1)*dim])
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/cmp")
			})
		}
	}
}

func BenchmarkInt8KernelEmbed(b *testing.B) {
	m := benchInt8Model(30522, 512) // real vocab size, 512 dims
	r := rand.New(rand.NewSource(5))
	ids := make([]int, 32)
	for i := range ids {
		ids[i] = r.Intn(m.VocabSize)
	}
	out := make([]uint8, m.EmbedDim)
	b.Run("tok32/api", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = m.ComputeEmbeddingFromTokens(ids)
		}
	})
	b.Run("tok32/go_simd", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			embedInt8Go(m.weightsInt8Flat, ids, m.EmbedDim, m.VocabSize, out, m.scale)
		}
	})
	if haveCgoInt8 {
		b.Run("tok32/cgo", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				embedInt8Cgo(m.weightsInt8Flat, ids, m.EmbedDim, m.VocabSize, out, m.scale)
			}
		})
	}
	b.Run("tok32/old_api_reflatten_per_call", func(b *testing.B) {
		// What ComputeEmbeddingFromTokens did before: copy the whole
		// 30522x512 table into a fresh slab on every call.
		m2 := *m
		m2.weightsInt8Flat = nil
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = m2.ComputeEmbeddingFromTokens(ids)
		}
	})
}
