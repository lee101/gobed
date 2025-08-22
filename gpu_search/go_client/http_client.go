package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPClient implements GPU search via HTTP API
type HTTPClient struct {
	baseURL string
	client  *http.Client
}

// NewHTTPClient creates a new HTTP search client
func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// SearchRequest represents a search query
type SearchRequest struct {
	Query []int8 `json:"query"`
	K     int    `json:"k"`
}

// SearchResponse represents search results
type SearchResponse struct {
	IDs          []int     `json:"ids"`
	Scores       []float32 `json:"scores"`
	SearchTimeMs float64   `json:"search_time_ms"`
	K            int       `json:"k"`
}

// BatchSearchRequest represents multiple search queries
type BatchSearchRequest struct {
	Queries [][]int8 `json:"queries"`
	K       int      `json:"k"`
}

// BatchSearchResponse represents batch search results
type BatchSearchResponse struct {
	BatchIDs     [][]int     `json:"batch_ids"`
	BatchScores  [][]float32 `json:"batch_scores"`
	BatchSize    int         `json:"batch_size"`
	SearchTimeMs float64     `json:"search_time_ms"`
	QPS          float64     `json:"qps"`
	K            int         `json:"k"`
}

// LoadRequest for loading embeddings
type LoadRequest struct {
	Embeddings [][]int8 `json:"embeddings"`
}

// LoadResponse from loading embeddings
type LoadResponse struct {
	Status   string  `json:"status"`
	Count    int     `json:"count"`
	Shape    []int   `json:"shape"`
	Device   string  `json:"device"`
	MemoryMB float64 `json:"memory_mb"`
}

// HealthResponse from health check
type HealthResponse struct {
	Status         string `json:"status"`
	Device         string `json:"device"`
	CudaAvailable  bool   `json:"cuda_available"`
	DatabaseLoaded bool   `json:"database_loaded"`
	DatabaseSize   int    `json:"database_size"`
}

// BenchmarkResult from benchmark endpoint
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

// Health checks the server status
func (c *HTTPClient) Health() (*HealthResponse, error) {
	resp, err := c.client.Get(c.baseURL + "/health")
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

// LoadDatabase loads embeddings into GPU memory
func (c *HTTPClient) LoadDatabase(embeddings [][]int8) (*LoadResponse, error) {
	req := LoadRequest{Embeddings: embeddings}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Post(c.baseURL+"/load", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("load failed (status %d): %s", resp.StatusCode, body)
	}

	var result LoadResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Search performs similarity search for a single query
func (c *HTTPClient) Search(query []int8, k int) (*SearchResponse, error) {
	req := SearchRequest{Query: query, K: k}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Post(c.baseURL+"/search", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search failed (status %d): %s", resp.StatusCode, body)
	}

	var result SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// BatchSearch performs similarity search for multiple queries
func (c *HTTPClient) BatchSearch(queries [][]int8, k int) (*BatchSearchResponse, error) {
	req := BatchSearchRequest{Queries: queries, K: k}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Post(c.baseURL+"/batch_search", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("batch search failed (status %d): %s", resp.StatusCode, body)
	}

	var result BatchSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Benchmark runs performance benchmarks on the server
func (c *HTTPClient) Benchmark() (*BenchmarkResult, error) {
	resp, err := c.client.Get(c.baseURL + "/benchmark")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("benchmark failed (status %d): %s", resp.StatusCode, body)
	}

	var result BenchmarkResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Clear removes the database from memory
func (c *HTTPClient) Clear() error {
	resp, err := c.client.Post(c.baseURL+"/clear", "application/json", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("clear failed (status %d): %s", resp.StatusCode, body)
	}

	return nil
}
