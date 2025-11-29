// +build !gpu,!cuda

package gobed

import "time"

// GPU stubs for CPU-only builds - only types not declared elsewhere

// BatchResult represents the result of batch processing
type BatchResult struct {
	Embeddings [][]float32
	Time       time.Duration
}

// GPUStats represents GPU statistics
type GPUStats struct {
	DeviceID     int
	MemoryUsed   int64
	MemoryTotal  int64
	Utilization  float32
	Temperature  float32
}