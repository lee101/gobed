package gobed

import (
	"strings"
	"testing"
)

var fastBenchText = strings.Repeat("The quick brown fox jumps over the lazy dog while embedding retrieval benchmarks run on commodity hardware. ", 3)

func benchModel(b *testing.B) *SimpleInt8Model512 {
	m, err := LoadSimpleInt8Model512()
	if err != nil {
		b.Skip(err)
	}
	return m
}

func BenchmarkTextEmbed(b *testing.B) {
	m := benchModel(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Embed(fastBenchText); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTextSimpleTokenize(b *testing.B) {
	m := benchModel(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.SimpleTokenize(fastBenchText)
	}
}

func BenchmarkTextEmbedFast(b *testing.B) {
	m := benchModel(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, done := m.EmbedFast(fastBenchText)
		done()
	}
}

func BenchmarkTextTokenizeInto(b *testing.B) {
	m := benchModel(b)
	buf := make([]int16, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.TokenizeInto(fastBenchText, buf[:0], 256)
	}
}

func BenchmarkTextEmbedTextInto(b *testing.B) {
	m := benchModel(b)
	buf := make([]int16, 0, 512)
	res := make([]float32, Int8EmbeddingDim)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.EmbedTextInto(fastBenchText, 256, res, buf)
	}
}
