// torch_search.go - LibTorch integration for Go using gotch
package gpu

import (
	"fmt"
	"path/filepath"

	"github.com/lee101/gobed"
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// TorchSearchClient provides GPU search using LibTorch
type TorchSearchClient struct {
	module *ts.Module
	device gotch.Device
}

// NewTorchSearchClient creates a new LibTorch-based search client
func NewTorchSearchClient(modelPath string) (*TorchSearchClient, error) {
	// Determine device
	device := gotch.CPU
	if gotch.CudaIfAvailable() {
		device = gotch.CUDA(0)
		fmt.Println(" Using CUDA device")
	} else {
		fmt.Println("  CUDA not available, using CPU")
	}

	// Load the TorchScript module
	module, err := ts.ModuleLoad(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load TorchScript module: %w", err)
	}

	client := &TorchSearchClient{
		module: module,
		device: device,
	}

	fmt.Printf(" Loaded TorchScript search module from: %s\n", modelPath)
	return client, nil
}

// LoadDatabase loads embeddings into the GPU search module
func (c *TorchSearchClient) LoadDatabase(embeddings [][]int8) error {
	if len(embeddings) == 0 {
		return fmt.Errorf("empty embeddings slice")
	}

	dimSize := len(embeddings[0])
	numVectors := len(embeddings)

	// Convert Go slice to tensor
	data := make([]int8, numVectors*dimSize)
	for i, emb := range embeddings {
		copy(data[i*dimSize:(i+1)*dimSize], emb)
	}

	// Create tensor and move to device
	shape := []int64{int64(numVectors), int64(dimSize)}
	tensor := ts.MustOfSlice(data).MustView(shape, true).MustTo(c.device, true)

	// Call load_database method on the module
	_, err := c.module.Forward(ts.NewIValue(tensor), "load_database")
	if err != nil {
		tensor.MustDrop()
		return fmt.Errorf("failed to load database: %w", err)
	}

	tensor.MustDrop()
	fmt.Printf("📚 Loaded %d vectors (%d dims) to GPU\n", numVectors, dimSize)
	return nil
}

// Search performs similarity search using the GPU module
func (c *TorchSearchClient) Search(query []int8, k int) (*SearchResponse, error) {
	// Create query tensor
	queryTensor := ts.MustOfSlice(query).MustTo(c.device, true)
	kTensor := ts.MustOfSlice([]int64{int64(k)}).MustTo(c.device, true)

	// Call search method
	outputs, err := c.module.Forward(ts.NewIValue(queryTensor), ts.NewIValue(kTensor), "search")
	if err != nil {
		queryTensor.MustDrop()
		kTensor.MustDrop()
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Extract results
	outputList := outputs.ToTuple()
	if len(outputList) != 2 {
		queryTensor.MustDrop()
		kTensor.MustDrop()
		return nil, fmt.Errorf("unexpected output format")
	}

	scoresTensor := outputList[0].ToTensor()
	indicesTensor := outputList[1].ToTensor()

	// Convert to Go slices
	scoresData := scoresTensor.Float64Values(true)
	indicesData := indicesTensor.Int64Values(true)

	scores := make([]float32, len(scoresData))
	indices := make([]int, len(indicesData))

	for i := range scoresData {
		scores[i] = float32(scoresData[i])
		indices[i] = int(indicesData[i])
	}

	queryTensor.MustDrop()
	kTensor.MustDrop()

	return &SearchResponse{
		IDs:    indices,
		Scores: scores,
	}, nil
}

// BatchSearch performs batch similarity search
func (c *TorchSearchClient) BatchSearch(queries [][]int8, k int) (*BatchSearchResponse, error) {
	if len(queries) == 0 {
		return &BatchSearchResponse{}, nil
	}

	batchSize := len(queries)
	dimSize := len(queries[0])

	// Convert batch to tensor
	data := make([]int8, batchSize*dimSize)
	for i, query := range queries {
		copy(data[i*dimSize:(i+1)*dimSize], query)
	}

	shape := []int64{int64(batchSize), int64(dimSize)}
	queriesTensor := ts.MustOfSlice(data).MustView(shape, true).MustTo(c.device, true)
	kTensor := ts.MustOfSlice([]int64{int64(k)}).MustTo(c.device, true)

	// Call batch_search method
	outputs, err := c.module.Forward(ts.NewIValue(queriesTensor), ts.NewIValue(kTensor), "batch_search")
	if err != nil {
		queriesTensor.MustDrop()
		kTensor.MustDrop()
		return nil, fmt.Errorf("batch search failed: %w", err)
	}

	// Extract results
	outputList := outputs.ToTuple()
	if len(outputList) != 2 {
		queriesTensor.MustDrop()
		kTensor.MustDrop()
		return nil, fmt.Errorf("unexpected output format")
	}

	scoresTensor := outputList[0].ToTensor()
	indicesTensor := outputList[1].ToTensor()

	scoresShape := scoresTensor.MustSize()
	indicesShape := indicesTensor.MustSize()

	scoresData := scoresTensor.Float64Values(true)
	indicesData := indicesTensor.Int64Values(true)

	// Reshape results
	actualK := int(scoresShape[1])
	batchScores := make([][]float32, batchSize)
	batchIDs := make([][]int, batchSize)

	for i := 0; i < batchSize; i++ {
		batchScores[i] = make([]float32, actualK)
		batchIDs[i] = make([]int, actualK)

		for j := 0; j < actualK; j++ {
			idx := i*actualK + j
			batchScores[i][j] = float32(scoresData[idx])
			batchIDs[i][j] = int(indicesData[idx])
		}
	}

	queriesTensor.MustDrop()
	kTensor.MustDrop()

	return &BatchSearchResponse{
		BatchScores: batchScores,
		BatchIDs:    batchIDs,
	}, nil
}

// GetStats returns database statistics
func (c *TorchSearchClient) GetStats() (*TorchStats, error) {
	// Call get_stats method
	outputs, err := c.module.Forward("get_stats")
	if err != nil {
		return nil, fmt.Errorf("get_stats failed: %w", err)
	}

	// Extract results
	outputList := outputs.ToTuple()
	if len(outputList) != 2 {
		return nil, fmt.Errorf("unexpected stats output format")
	}

	numVectorsTensor := outputList[0].ToTensor()
	memoryMBTensor := outputList[1].ToTensor()

	numVectors := numVectorsTensor.Int64Values(true)[0]
	memoryMB := memoryMBTensor.Float64Values(true)[0]

	return &TorchStats{
		NumVectors: int(numVectors),
		MemoryMB:   float64(memoryMB),
		Device:     c.device.String(),
	}, nil
}

// Close releases resources
func (c *TorchSearchClient) Close() error {
	if c.module != nil {
		c.module.Drop()
		c.module = nil
	}
	return nil
}

// TorchStats holds LibTorch search statistics
type TorchStats struct {
	NumVectors int     `json:"num_vectors"`
	MemoryMB   float64 `json:"memory_mb"`
	Device     string  `json:"device"`
}

// TorchPipeline provides a complete GPU pipeline using LibTorch
type TorchPipeline struct {
	embedder     *gobed.EmbeddingModel
	searchClient *TorchSearchClient
	texts        []string
	config       Config
}

// NewTorchPipeline creates a GPU pipeline using LibTorch
func NewTorchPipeline(config Config) (*TorchPipeline, error) {
	// Initialize CPU embedder
	embedder, err := gobed.LoadModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load embedding model: %w", err)
	}

	// Initialize LibTorch search client
	modelPath := filepath.Join(config.ModelPath, "simple_gpu_search_module.pt")
	searchClient, err := NewTorchSearchClient(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create torch search client: %w", err)
	}

	return &TorchPipeline{
		embedder:     embedder,
		searchClient: searchClient,
		texts:        make([]string, 0),
		config:       config,
	}, nil
}

// IndexTexts processes and indexes texts using LibTorch
func (p *TorchPipeline) IndexTexts(texts []string) error {
	if len(texts) == 0 {
		return nil
	}

	// Generate embeddings using CPU model
	embeddings := make([][]int8, len(texts))

	batchSize := p.config.BatchSize
	if batchSize == 0 {
		batchSize = 128
	}

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		for j, text := range batch {
			// Get float32 embedding
			emb, err := p.embedder.Encode(text)
			if err != nil {
				return fmt.Errorf("failed to encode text %d: %w", i+j, err)
			}

			// Convert to int8
			embeddings[i+j] = Float32ToInt8(emb)
		}
	}

	// Store texts for search results
	p.texts = append(p.texts, texts...)

	// Load to GPU via LibTorch
	return p.searchClient.LoadDatabase(embeddings)
}

// Search performs similarity search using LibTorch
func (p *TorchPipeline) Search(query string, k int) ([]Result, error) {
	// Encode query
	queryEmb, err := p.embedder.Encode(query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %w", err)
	}

	queryInt8 := Float32ToInt8(queryEmb)

	// Search using LibTorch
	resp, err := p.searchClient.Search(queryInt8, k)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Build results
	results := make([]Result, len(resp.IDs))
	for i, id := range resp.IDs {
		if id < len(p.texts) {
			results[i] = Result{
				Text:  p.texts[id],
				Score: resp.Scores[i],
				ID:    id,
			}
		}
	}

	return results, nil
}

// BatchSearch performs batch search using LibTorch
func (p *TorchPipeline) BatchSearch(queries []string, k int) ([][]Result, error) {
	if len(queries) == 0 {
		return nil, nil
	}

	// Encode all queries
	queryEmbeddings := make([][]int8, len(queries))
	for i, query := range queries {
		emb, err := p.embedder.Encode(query)
		if err != nil {
			return nil, fmt.Errorf("failed to encode query %d: %w", i, err)
		}
		queryEmbeddings[i] = Float32ToInt8(emb)
	}

	// Batch search using LibTorch
	resp, err := p.searchClient.BatchSearch(queryEmbeddings, k)
	if err != nil {
		return nil, fmt.Errorf("batch search failed: %w", err)
	}

	// Build results
	allResults := make([][]Result, len(queries))
	for i := range queries {
		results := make([]Result, len(resp.BatchIDs[i]))
		for j, id := range resp.BatchIDs[i] {
			if id < len(p.texts) {
				results[j] = Result{
					Text:  p.texts[id],
					Score: resp.BatchScores[i][j],
					ID:    id,
				}
			}
		}
		allResults[i] = results
	}

	return allResults, nil
}

// GetStats returns pipeline statistics
func (p *TorchPipeline) GetStats() (*Stats, error) {
	torchStats, err := p.searchClient.GetStats()
	if err != nil {
		return nil, err
	}

	return &Stats{
		NumTexts:       len(p.texts),
		NumEmbeddings:  torchStats.NumVectors,
		GPUDevice:      torchStats.Device,
		GPUMemoryMB:    torchStats.MemoryMB,
		CPUMemoryMB:    0, // LibTorch handles memory
		SingleQueryMS:  0, // Would need benchmarking
		SingleQueryQPS: 0,
		BatchQPS:       0,
	}, nil
}

// Close releases all resources
func (p *TorchPipeline) Close() error {
	if p.searchClient != nil {
		return p.searchClient.Close()
	}
	return nil
}
