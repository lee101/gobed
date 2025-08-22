// client.go - GPU search client
package gpu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// SearchClient handles communication with GPU search server
type SearchClient struct {
	BaseURL string
	Client  *http.Client
}

// Health checks GPU server health
func (c *SearchClient) Health() (*HealthResponse, error) {
	resp, err := c.Client.Get(c.BaseURL + "/health")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// LoadDatabase loads embeddings to GPU
func (c *SearchClient) LoadDatabase(embeddings [][]int8) (*LoadResponse, error) {
	req := LoadRequest{Embeddings: embeddings}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.Post(c.BaseURL+"/load", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("load failed: %s", body)
	}

	var result LoadResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Search performs single query search
func (c *SearchClient) Search(query []int8, k int) (*SearchResponse, error) {
	req := SearchRequest{Query: query, K: k}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.Post(c.BaseURL+"/search", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search failed: %s", body)
	}

	var result SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// BatchSearch performs batch search
func (c *SearchClient) BatchSearch(queries [][]int8, k int) (*BatchSearchResponse, error) {
	req := BatchSearchRequest{Queries: queries, K: k}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.Post(c.BaseURL+"/batch_search", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("batch search failed: %s", body)
	}

	var result BatchSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// EmbedTexts generates embeddings using GPU
func (c *SearchClient) EmbedTexts(texts []string) (*EmbedResponse, error) {
	req := EmbedRequest{Texts: texts}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.Post(c.BaseURL+"/embed", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed failed: %s", body)
	}

	var result EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Benchmark runs performance benchmark
func (c *SearchClient) Benchmark() (*BenchmarkResult, error) {
	resp, err := c.Client.Get(c.BaseURL + "/benchmark")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result BenchmarkResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// API Response types
type HealthResponse struct {
	Status         string `json:"status"`
	Device         string `json:"device"`
	CudaAvailable  bool   `json:"cuda_available"`
	DatabaseLoaded bool   `json:"database_loaded"`
	DatabaseSize   int    `json:"database_size"`
}

type LoadRequest struct {
	Embeddings [][]int8 `json:"embeddings"`
}

type LoadResponse struct {
	Status   string  `json:"status"`
	Count    int     `json:"count"`
	Shape    []int   `json:"shape"`
	Device   string  `json:"device"`
	MemoryMB float64 `json:"memory_mb"`
}

type SearchRequest struct {
	Query []int8 `json:"query"`
	K     int    `json:"k"`
}

type SearchResponse struct {
	IDs          []int     `json:"ids"`
	Scores       []float32 `json:"scores"`
	SearchTimeMs float64   `json:"search_time_ms"`
	K            int       `json:"k"`
}

type BatchSearchRequest struct {
	Queries [][]int8 `json:"queries"`
	K       int      `json:"k"`
}

type BatchSearchResponse struct {
	BatchIDs     [][]int     `json:"batch_ids"`
	BatchScores  [][]float32 `json:"batch_scores"`
	BatchSize    int         `json:"batch_size"`
	SearchTimeMs float64     `json:"search_time_ms"`
	QPS          float64     `json:"qps"`
	K            int         `json:"k"`
}

type EmbedRequest struct {
	Texts []string `json:"texts"`
}

type EmbedResponse struct {
	Embeddings  [][]int8 `json:"embeddings"`
	Count       int      `json:"count"`
	EmbedTimeMs float64  `json:"embed_time_ms"`
	TextsPerSec float64  `json:"texts_per_sec"`
	Dimensions  int      `json:"dimensions"`
}

type BenchmarkResult struct {
	SingleQuery struct {
		AvgLatencyMs float64 `json:"avg_latency_ms"`
		QPS          float64 `json:"qps"`
		Iterations   int     `json:"iterations"`
	} `json:"single_query"`
	Batch struct {
		BatchSize      int     `json:"batch_size"`
		BatchLatencyMs float64 `json:"batch_latency_ms"`
		QPS            float64 `json:"qps"`
		Iterations     int     `json:"iterations"`
	} `json:"batch"`
	Database struct {
		Size       int     `json:"size"`
		Dimensions int     `json:"dimensions"`
		Device     string  `json:"device"`
		MemoryMB   float64 `json:"memory_mb"`
	} `json:"database"`
}
