//go:build routebench

package gobed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

var (
	loadServerModelOnce sync.Once
	serverModel         *EmbeddingModel
	serverModelErr      error
)

func getServerModel(b *testing.B) *EmbeddingModel {
	loadServerModelOnce.Do(func() {
		serverModel, serverModelErr = LoadModel()
	})
	if serverModelErr != nil {
		b.Fatalf("LoadModel failed: %v", serverModelErr)
	}
	return serverModel
}

// BenchmarkHandleGPUSearch exercises the HTTP handler to profile per-request allocs/latency.
func BenchmarkHandleGPUSearch(b *testing.B) {
	model := getServerModel(b)

	config := DefaultGPUServerConfig()
	config.Port = 0 // We invoke handler directly; no listen needed.
	config.EnableMetrics = false
	config.EnableProfiling = false
	config.SharedIndexPath = b.TempDir()
	config.MaxVectors = 1024
	config.GPUBatchSize = 128

	server, err := NewGPUSearchServer(model, config)
	if err != nil {
		b.Fatalf("NewGPUSearchServer failed: %v", err)
	}
	defer server.Stop()

	payloads := make([][]byte, b.N)
	for i := range payloads {
		req := SearchRequest{
			Query: fmt.Sprintf("anime cyberpunk hero %d", i),
			K:     50,
		}
		buf, err := json.Marshal(&req)
		if err != nil {
			b.Fatalf("json.Marshal failed: %v", err)
		}
		payloads[i] = buf
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/search", bytes.NewReader(payloads[i]))
		rec := httptest.NewRecorder()
		server.handleGPUSearch(rec, req)

		if rec.Code != http.StatusOK {
			b.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
		}
	}
}
