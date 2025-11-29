package gobed

import (
	"fmt"
	"os"
	"path/filepath"
)

// UnifiedModelConfig configures model loading behavior
type UnifiedModelConfig struct {
	UseInt8     bool   // Use optimized int8 model (recommended)
	ModelDir    string // Optional custom model directory
	ForceFloat32 bool   // Force float32 for compatibility (slower)
}

// DefaultFastConfig returns optimized configuration for maximum performance
func DefaultFastConfig() *UnifiedModelConfig {
	return &UnifiedModelConfig{
		UseInt8:      true,  // Always use fast int16→int8 pipeline
		ForceFloat32: false, // Never use slow float32 unless required
	}
}

// LoadModelUnified loads the best available model with consistent path handling
func LoadModelUnified(config *UnifiedModelConfig) (interface{}, error) {
	if config == nil {
		config = DefaultFastConfig()
	}

	// Find model directory with robust path resolution
	modelDir, err := findModelDirectory(config.ModelDir)
	if err != nil {
		return nil, fmt.Errorf("model directory not found: %v", err)
	}

	// Always prefer int8 model for maximum performance unless forced otherwise
	if config.UseInt8 && !config.ForceFloat32 {
		return loadInt8ModelFromPath(modelDir)
	}

	// Fallback to float32 model (slower)
	return loadFloat32ModelFromPath(modelDir)
}

// LoadFastModel is a convenience function that always loads the fastest model
func LoadFastModel() (*Int8EmbeddingModel512, error) {
	model, err := LoadModelUnified(DefaultFastConfig())
	if err != nil {
		return nil, err
	}

	int8Model, ok := model.(*Int8EmbeddingModel512)
	if !ok {
		return nil, fmt.Errorf("failed to load int8 model, got %T", model)
	}

	return int8Model, nil
}

// findModelDirectory locates the model directory with robust path handling
func findModelDirectory(customDir string) (string, error) {
	// If custom directory specified, use it
	if customDir != "" {
		if _, err := os.Stat(customDir); err == nil {
			return customDir, nil
		}
	}

	// Standard model directory search paths
	searchPaths := []string{
		"model",                    // Current directory
		"../model",                 // Parent directory
		"../../model",              // Grandparent directory
		"./model",                  // Explicit current
		"/home/lee/code/gobed/model", // Absolute fallback
	}

	for _, path := range searchPaths {
		absPath, err := filepath.Abs(path)
		if err != nil {
			continue
		}

		if _, err := os.Stat(absPath); err == nil {
			// Verify it contains the required files
			int8ModelPath := filepath.Join(absPath, "modelint8_512dim.safetensors")
			tokenizerPath := filepath.Join(absPath, "tokenizer.json")

			if _, err := os.Stat(int8ModelPath); err == nil {
				if _, err := os.Stat(tokenizerPath); err == nil {
					return absPath, nil
				}
			}
		}
	}

	return "", fmt.Errorf("model directory not found in any search path")
}

// loadInt8ModelFromPath loads the optimized int8 model from a specific path
func loadInt8ModelFromPath(modelDir string) (*Int8EmbeddingModel512, error) {
	modelPath := filepath.Join(modelDir, "modelint8_512dim.safetensors")
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")

	fmt.Printf("🚀 Loading optimized INT8 512-dim model from %s\n", modelPath)

	// Check if model files exist
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("int8 model not found: %s", modelPath)
	}
	if _, err := os.Stat(tokenizerPath); err != nil {
		return nil, fmt.Errorf("tokenizer not found: %s", tokenizerPath)
	}

	// Load the quantized model using existing function but with explicit paths
	embeddings, scales, err := loadInt8Embeddings(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings: %v", err)
	}

	// Create a simplified int8 model without tokenizer dependency conflicts
	model := &Int8EmbeddingModel512{
		embeddings: embeddings,
		scales:     scales,
		tokenizer:  nil, // Skip tokenizer to avoid dependency conflicts
	}

	// TODO: Add tokenizer loading when dependency issues are resolved

	fmt.Printf("✅ INT8 model loaded: vocab=%d, dims=%d, memory=%.1f MB\n",
		len(embeddings), Int8EmbeddingDim,
		float64(len(embeddings)*Int8EmbeddingDim+len(scales)*4)/(1024*1024))

	return model, nil
}

// loadFloat32ModelFromPath loads the traditional float32 model (slower but compatible)
func loadFloat32ModelFromPath(modelDir string) (*EmbeddingModel, error) {
	fmt.Println("⚠️  Loading Float32 model (slower, consider using int8)")

	// Use existing LoadModel() but ensure it finds the right path
	oldWd, _ := os.Getwd()
	defer os.Chdir(oldWd)

	// Change to model parent directory to make relative paths work
	parentDir := filepath.Dir(modelDir)
	os.Chdir(parentDir)

	return LoadModel()
}

// ModelCompatibilityWrapper provides a common interface for both model types
type ModelCompatibilityWrapper struct {
	int8Model   *Int8EmbeddingModel512
	float32Model *EmbeddingModel
	isInt8      bool
}

// LoadCompatibleModel loads a model that works with all existing code
func LoadCompatibleModel() (*ModelCompatibilityWrapper, error) {
	// Try int8 first (faster)
	int8Model, err := LoadFastModel()
	if err == nil {
		return &ModelCompatibilityWrapper{
			int8Model: int8Model,
			isInt8:    true,
		}, nil
	}

	// Fallback to float32 (slower but more compatible)
	fmt.Printf("⚠️  Int8 model failed (%v), falling back to float32\n", err)
	float32Model, err := LoadModelUnified(&UnifiedModelConfig{UseInt8: false})
	if err != nil {
		return nil, fmt.Errorf("failed to load any model: %v", err)
	}

	return &ModelCompatibilityWrapper{
		float32Model: float32Model.(*EmbeddingModel),
		isInt8:       false,
	}, nil
}

// Encode provides a unified interface for both model types
func (w *ModelCompatibilityWrapper) Encode(text string) ([]float32, error) {
	if w.isInt8 {
		return w.int8Model.Embed(text)
	}
	return w.float32Model.Encode(text)
}

// EncodeInt8 provides fast int8 encoding when available
func (w *ModelCompatibilityWrapper) EncodeInt8(text string) ([]int8, error) {
	if w.isInt8 {
		result, err := w.int8Model.EmbedInt8(text)
		if err != nil {
			return nil, err
		}
		return result.Vector, nil
	}

	// Fallback: encode with float32 then quantize
	floatEmb, err := w.float32Model.Encode(text)
	if err != nil {
		return nil, err
	}

	// Take first 512 dimensions and quantize
	dim := 512
	if len(floatEmb) < dim {
		dim = len(floatEmb)
	}

	return quantizeToInt8Simple(floatEmb[:dim]), nil
}

// quantizeToInt8Simple provides basic quantization for compatibility
func quantizeToInt8Simple(embedding []float32) []int8 {
	// Find min/max for quantization
	minVal, maxVal := embedding[0], embedding[0]
	for _, v := range embedding {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Scale to INT8 range [-128, 127]
	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}

	quantized := make([]int8, len(embedding))
	for i, v := range embedding {
		// Map to [0, 255] then shift to [-128, 127]
		q := int((v-minVal)/scale) - 128
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		quantized[i] = int8(q)
	}

	return quantized
}