package main

import (
	"testing"
	"unsafe"
)

// #cgo LDFLAGS: -L. -lcuda_similarity -L/usr/local/cuda/lib64 -lcudart -lcublas
// #include <stdlib.h>
// extern void* cuda_fast_search_create(int max_vectors, int dim);
// extern void cuda_fast_search_destroy(void* handle);
// extern int cuda_fast_search_add_vectors(void* handle, const signed char* vectors, int num_vectors);
// extern int cuda_fast_search_query(void* handle, const signed char* query, int k, int* indices, float* scores);
import "C"

var (
	benchModel   *FastModel
	benchQueries = []string{
		"anime",
		"Studio Ghibli films",
		"Dragon Ball Z",
		"neural networks machine learning",
		"semantic search optimization",
		"CUDA GPU acceleration performance",
	}
	benchTexts = []string{
		"anime is great",
		"Dragon Ball Z is a popular anime series",
		"Studio Ghibli makes beautiful anime films",
		"Neural networks are machine learning models",
		"CUDA provides GPU acceleration for compute workloads",
		"Semantic search finds meaning in text",
		"Optimization improves performance significantly",
		"Vectorized operations enable parallelism",
		"Memory access patterns affect cache performance",
		"Quantization reduces model size and improves speed",
	}
)

func init() {
	var err error
	benchModel, err = LoadFastModel("../model/modelint8_512dim.safetensors", "../model/tokenizer.json")
	if err != nil {
		panic("Failed to load benchmark model: " + err.Error())
	}
}

// BenchmarkTokenizer tests zero-allocation tokenization performance
func BenchmarkTokenizer(b *testing.B) {
	tokenizer := benchModel.tokenizer

	b.Run("Short", func(b *testing.B) {
		text := "anime"
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tokenizer.Tokenize(text)
		}
	})

	b.Run("Medium", func(b *testing.B) {
		text := "Studio Ghibli makes beautiful anime films"
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tokenizer.Tokenize(text)
		}
	})

	b.Run("Long", func(b *testing.B) {
		text := "Neural networks are machine learning models that use CUDA GPU acceleration for fast training and inference"
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tokenizer.Tokenize(text)
		}
	})
}

// BenchmarkEmbedding tests int8 embedding generation performance
func BenchmarkEmbedding(b *testing.B) {
	b.Run("EmbedInt8", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			text := benchQueries[i%len(benchQueries)]
			_, err := benchModel.EmbedInt8(text)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("EmbedInt8_Batch", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, text := range benchTexts {
				_, err := benchModel.EmbedInt8(text)
				if err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

// BenchmarkGPUOperations tests CUDA kernel performance
func BenchmarkGPUOperations(b *testing.B) {
	const numVectors = 10000
	const dim = 512

	// Create test embeddings
	embeddings := make([]int8, numVectors*dim)
	for i := range embeddings {
		embeddings[i] = int8(i % 256 - 128)
	}

	queryEmb, _ := benchModel.EmbedInt8("anime test query")

	b.Run("GPUCreate", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			handle := C.cuda_fast_search_create(C.int(numVectors), C.int(dim))
			if handle != nil {
				C.cuda_fast_search_destroy(handle)
			}
		}
	})

	b.Run("GPUAddVectors", func(b *testing.B) {
		handle := C.cuda_fast_search_create(C.int(numVectors), C.int(dim))
		defer C.cuda_fast_search_destroy(handle)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			C.cuda_fast_search_add_vectors(
				handle,
				(*C.schar)(unsafe.Pointer(&embeddings[0])),
				C.int(numVectors),
			)
		}
	})

	b.Run("GPUQuery", func(b *testing.B) {
		handle := C.cuda_fast_search_create(C.int(numVectors), C.int(dim))
		defer C.cuda_fast_search_destroy(handle)

		C.cuda_fast_search_add_vectors(
			handle,
			(*C.schar)(unsafe.Pointer(&embeddings[0])),
			C.int(numVectors),
		)

		indices := make([]int32, 10)
		scores := make([]float32, 10)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			C.cuda_fast_search_query(
				handle,
				(*C.schar)(unsafe.Pointer(&queryEmb[0])),
				C.int(10),
				(*C.int)(unsafe.Pointer(&indices[0])),
				(*C.float)(unsafe.Pointer(&scores[0])),
			)
		}
	})
}

// BenchmarkEndToEnd tests complete search pipeline
func BenchmarkEndToEnd(b *testing.B) {
	const numDocs = 1000

	// Create test documents
	documents := make([]*Document, numDocs)
	for i := 0; i < numDocs; i++ {
		text := benchTexts[i%len(benchTexts)]
		emb, _ := benchModel.EmbedInt8(text)
		documents[i] = &Document{
			FilePath:  "test.txt",
			LineNum:   i + 1,
			Content:   text,
			Embedding: emb,
		}
	}

	b.Run("Complete_1K_docs", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// Setup GPU
			handle := C.cuda_fast_search_create(C.int(numDocs+100), C.int(512))

			// Add embeddings
			flatEmbeddings := make([]int8, numDocs*512)
			for j, doc := range documents {
				copy(flatEmbeddings[j*512:(j+1)*512], doc.Embedding)
			}

			C.cuda_fast_search_add_vectors(
				handle,
				(*C.schar)(unsafe.Pointer(&flatEmbeddings[0])),
				C.int(numDocs),
			)

			// Query
			query := benchQueries[i%len(benchQueries)]
			queryEmb, _ := benchModel.EmbedInt8(query)

			indices := make([]int32, 10)
			scores := make([]float32, 10)

			C.cuda_fast_search_query(
				handle,
				(*C.schar)(unsafe.Pointer(&queryEmb[0])),
				C.int(10),
				(*C.int)(unsafe.Pointer(&indices[0])),
				(*C.float)(unsafe.Pointer(&scores[0])),
			)

			C.cuda_fast_search_destroy(handle)
		}
	})
}

// BenchmarkMemoryOperations tests memory allocation patterns
func BenchmarkMemoryOperations(b *testing.B) {
	b.Run("SliceAllocation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = make([]int8, 512)
		}
	})

	b.Run("EmbeddingCopy", func(b *testing.B) {
		src := make([]int8, 512)
		for i := 0; i < b.N; i++ {
			dst := make([]int8, 512)
			copy(dst, src)
		}
	})

	b.Run("VectorizedAccumulation", func(b *testing.B) {
		result := make([]float32, 512)
		embedding := make([]int8, 512)
		scale := float32(0.1)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Simulate vectorized accumulation from model.go
			for j := 0; j < 512; j += 4 {
				result[j] += float32(embedding[j]) * scale
				result[j+1] += float32(embedding[j+1]) * scale
				result[j+2] += float32(embedding[j+2]) * scale
				result[j+3] += float32(embedding[j+3]) * scale
			}
		}
	})
}