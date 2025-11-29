package gobed

import (
	"reflect"
	"testing"
)

func TestSimpleInt8ModelEmbeddingAccessors(t *testing.T) {
	embeddings := [][]int8{
		{1, -2, 3},
		{4, 5, -6},
	}
	scales := []float32{0.5, 0.25}

	model := &SimpleInt8Model512{
		embeddings: embeddings,
		scales:     scales,
	}

	if got := model.EmbeddingTable(); !reflect.DeepEqual(got, embeddings) {
		t.Fatalf("EmbeddingTable mismatch, got %v want %v", got, embeddings)
	}

	if got := model.ScaleTable(); !reflect.DeepEqual(got, scales) {
		t.Fatalf("ScaleTable mismatch, got %v want %v", got, scales)
	}
}
