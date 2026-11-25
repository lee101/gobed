package i8simd

import (
	"fmt"
	"math/rand"
	"testing"
)

// Set by the GOEXPERIMENT=simd test files (archsimd_*_test.go) when available.
var (
	archsimdDot512   func(a, b *[512]int8) int32
	archsimdCentered func(a, b []uint8) (dot, na, nb int64)
)

func refDot(a, b []int8) int64 {
	var s int64
	for i := range a {
		s += int64(a[i]) * int64(b[i])
	}
	return s
}

func refCentered(a, b []uint8) (d, x, y int64) {
	for i := range a {
		p, q := int64(a[i])-128, int64(b[i])-128
		d += p * q
		x += p * p
		y += q * q
	}
	return
}

// fillU8 fills with a mix of random and extreme values depending on mode.
func fillU8(r *rand.Rand, v []uint8, mode int) {
	for i := range v {
		switch mode {
		case 0:
			v[i] = uint8(r.Intn(256))
		case 1:
			v[i] = 128 // centered zero
		case 2:
			v[i] = 0 // -128
		case 3:
			v[i] = 255 // +127
		default:
			v[i] = []uint8{0, 1, 127, 128, 129, 254, 255}[r.Intn(7)]
		}
	}
}

func TestCenteredDotNormsProperty(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	lengths := []int{0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 511, 512, 513, 1023, 4097, 32768 + 9, 131072 + 33}
	for _, n := range lengths {
		for mode := 0; mode < 5; mode++ {
			for mb := 0; mb < 5; mb++ {
				a, b := make([]uint8, n), make([]uint8, n)
				fillU8(r, a, mode)
				fillU8(r, b, mb)
				wd, wx, wy := refCentered(a, b)
				check := func(name string, d, x, y int64) {
					if d != wd || x != wx || y != wy {
						t.Fatalf("%s n=%d modes=%d/%d: got (%d,%d,%d) want (%d,%d,%d)", name, n, mode, mb, d, x, y, wd, wx, wy)
					}
				}
				d, x, y := CenteredDotNorms(a, b)
				check("dispatch", d, x, y)
				d, x, y = CenteredDotNormsGo(a, b)
				check("go", d, x, y)
				if archsimdCentered != nil {
					d, x, y = archsimdCentered(a, b)
					check("archsimd", d, x, y)
				}
			}
		}
	}
}

func TestCenteredDotNormsUnalignedAndMismatch(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	buf := make([]uint8, 600)
	fillU8(r, buf, 0)
	for off := 0; off < 17; off++ {
		a, b := buf[off:off+512], buf[80+off/2:]
		wd, wx, wy := refCentered(a, b[:512])
		if d, x, y := CenteredDotNorms(a, b); d != wd || x != wx || y != wy {
			t.Fatalf("off=%d mismatch", off)
		}
	}
}

func TestDot512Property(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for iter := 0; iter < 2000; iter++ {
		var a, b [512]int8
		for i := range a {
			switch iter % 4 {
			case 0:
				a[i], b[i] = int8(r.Intn(256)-128), int8(r.Intn(256)-128)
			case 1:
				a[i], b[i] = -128, -128
			case 2:
				a[i], b[i] = 127, -128
			case 3:
				a[i], b[i] = []int8{-128, -127, -1, 0, 1, 127}[r.Intn(6)], []int8{-128, 127, 0}[r.Intn(3)]
			}
		}
		want := int32(refDot(a[:], b[:]))
		if g := Dot512(&a, &b); g != want {
			t.Fatalf("Dot512 iter %d: %d != %d", iter, g, want)
		}
		if g := Dot512Go(&a, &b); g != want {
			t.Fatalf("Dot512Go iter %d: %d != %d", iter, g, want)
		}
		if archsimdDot512 != nil {
			if g := archsimdDot512(&a, &b); g != want {
				t.Fatalf("archsimd iter %d: %d != %d", iter, g, want)
			}
		}
	}
}

func TestAccumulateS8Property(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for _, n := range []int{0, 1, 7, 15, 16, 17, 31, 64, 511, 512, 513} {
		acc, ref := make([]int32, n+3), make([]int32, n+3)
		for i := range acc {
			acc[i] = int32(r.Intn(1 << 20))
			ref[i] = acc[i]
		}
		for rep := 0; rep < 40; rep++ {
			row := make([]int8, n)
			for i := range row {
				row[i] = []int8{-128, 127, int8(r.Intn(256) - 128)}[r.Intn(3)]
			}
			AccumulateS8(acc, row)
			for i, v := range row {
				ref[i] += int32(v)
			}
		}
		for i := range acc {
			if acc[i] != ref[i] {
				t.Fatalf("n=%d i=%d: %d != %d", n, i, acc[i], ref[i])
			}
		}
	}
}

func TestAccumulateRowsProperty(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for _, dim := range []int{0, 1, 15, 63, 64, 65, 127, 128, 384, 512, 513, 1000} {
		const vocab = 40
		flat := make([]int8, vocab*dim)
		for i := range flat {
			flat[i] = []int8{-128, 127, int8(r.Intn(256) - 128)}[r.Intn(3)]
		}
		for _, ntok := range []int{0, 1, 3, 32, 255, 256, 257, 600} {
			offs := make([]int, ntok)
			for i := range offs {
				offs[i] = r.Intn(vocab) * dim
			}
			if ntok >= 256 { // worst case for the int16 lanes
				for i := range offs[:256] {
					offs[i] = 0
				}
				for d := 0; d < dim; d++ {
					flat[d] = -128
				}
			}
			acc, ref := make([]int32, dim), make([]int32, dim)
			for i := range acc {
				acc[i] = int32(r.Intn(1000)) - 500
				ref[i] = acc[i]
			}
			AccumulateRows(acc, flat, offs)
			for _, off := range offs {
				for d := 0; d < dim; d++ {
					ref[d] += int32(flat[off+d])
				}
			}
			for d := range acc {
				if acc[d] != ref[d] {
					t.Fatalf("dim=%d ntok=%d d=%d: %d != %d", dim, ntok, d, acc[d], ref[d])
				}
			}
		}
	}
}

func TestFinishU8Property(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	for _, n := range []int{0, 1, 15, 16, 17, 512, 1000} {
		for _, scale := range []float32{0.0037, 0.0123, 1, 3.3e-5, 77} {
			for _, valid := range []int{1, 3, 32, 255} {
				acc := make([]int32, n)
				for i := range acc {
					switch r.Intn(4) {
					case 0:
						acc[i] = int32(r.Intn(1<<16)) - 1<<15
					case 1:
						acc[i] = int32(r.Intn(256)) - 128
					case 2:
						acc[i] = int32(r.Intn(100000000)) - 50000000
					default:
						acc[i] = []int32{0, -1, 1, 127 * int32(valid), -128 * int32(valid)}[r.Intn(5)]
					}
				}
				inv := 1 / float32(valid)
				got, want := make([]uint8, n), make([]uint8, n)
				FinishU8(got, acc, inv, scale)
				FinishU8Go(want, acc, inv, scale)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("n=%d scale=%v valid=%d i=%d acc=%d: %d != %d", n, scale, valid, i, acc[i], got[i], want[i])
					}
				}
			}
		}
	}
}

var sink32 int32
var sink64 int64

func randU8(n int, seed int64) []uint8 {
	v := make([]uint8, n)
	fillU8(rand.New(rand.NewSource(seed)), v, 0)
	return v
}

func BenchmarkKernelDot512(b *testing.B) {
	var x, y [512]int8
	for i := range x {
		x[i], y[i] = int8(i*7-100), int8(i*13+5)
	}
	b.Run("go_unrolled", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink32 += Dot512Go(&x, &y)
		}
	})
	b.Run("dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink32 += Dot512(&x, &y)
		}
	})
	if archsimdDot512 != nil {
		b.Run("archsimd", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 += archsimdDot512(&x, &y)
			}
		})
	}
}

func BenchmarkKernelCentered(b *testing.B) {
	for _, n := range []int{512, 384, 1024} {
		x, y := randU8(n, 1), randU8(n, 2)
		b.Run(fmt.Sprintf("n%d/go_unrolled", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				d, _, _ := CenteredDotNormsGo(x, y)
				sink64 += d
			}
		})
		b.Run(fmt.Sprintf("n%d/dispatch", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				d, _, _ := CenteredDotNorms(x, y)
				sink64 += d
			}
		})
		if archsimdCentered != nil {
			b.Run(fmt.Sprintf("n%d/archsimd", n), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					d, _, _ := archsimdCentered(x, y)
					sink64 += d
				}
			})
		}
	}
}

func BenchmarkKernelAccumulateRows32x512(b *testing.B) {
	flat := make([]int8, 30522*512)
	offs := make([]int, 32)
	r := rand.New(rand.NewSource(1))
	for i := range offs {
		offs[i] = r.Intn(30522) * 512
	}
	acc := make([]int32, 512)
	b.Run("dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AccumulateRows(acc, flat, offs)
		}
	})
	b.Run("go_unrolled", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, off := range offs {
				AccumulateS8Go(acc, flat[off:off+512])
			}
		}
	})
}

func BenchmarkKernelAccumulate512(b *testing.B) {
	row := make([]int8, 512)
	acc := make([]int32, 512)
	b.Run("go_unrolled", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AccumulateS8Go(acc, row)
		}
	})
	b.Run("dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AccumulateS8(acc, row)
		}
	})
}

// chunked keeps a kernel's int32 accumulators from overflowing on long inputs.
func chunked(f func(a, b []uint8) (int64, int64, int64)) func(a, b []uint8) (int64, int64, int64) {
	return func(a, b []uint8) (dot, na, nb int64) {
		n := min(len(a), len(b))
		for i := 0; i < n || i == 0; i += 65536 {
			j := min(i+65536, n)
			d, x, y := f(a[i:j], b[i:j])
			dot, na, nb = dot+d, na+x, nb+y
		}
		return
	}
}
