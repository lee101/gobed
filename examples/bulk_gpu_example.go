package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

// BulkGPUIndexingExample demonstrates how to use the bulk GPU indexing feature
func main() {
	fmt.Println("🚀 Bulk GPU Indexing Example")
	fmt.Println("============================")
	
	// Load the embedding model
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()
	
	// Create vector index with bulk GPU indexing enabled
	config := gobed.DefaultVectorIndexConfig()
	config.EnableBulkGPU = true    // Enable GPU acceleration
	config.BulkBatchSize = 5000    // Process 5k documents per GPU batch
	
	index := gobed.NewVectorIndex(model, config)
	
	// Create a large dataset
	fmt.Printf("📚 Generating test documents...\n")
	docs := make([]gobed.Document, 15000)
	for i := 0; i < len(docs); i++ {
		docs[i] = gobed.Document{
			ID: i,
			Text: fmt.Sprintf("Document %d: This document discusses topic %d with various keywords and content related to subject matter %d", 
				i, i%100, i%50),
		}
	}
	
	// Method 1: Automatic bulk indexing (uses GPU if dataset is large enough)
	fmt.Printf("\n🔄 Method 1: Automatic bulk indexing\n")
	start := time.Now()
	err = index.AddDocuments(docs)
	if err != nil {
		log.Fatalf("Automatic indexing failed: %v", err)
	}
	elapsed := time.Since(start)
	fmt.Printf("✅ Indexed %d documents in %.2fs (%.0f docs/sec)\n", 
		len(docs), elapsed.Seconds(), float64(len(docs))/elapsed.Seconds())
	
	// Method 2: Force GPU bulk indexing
	fmt.Printf("\n🚀 Method 2: Forced GPU bulk indexing\n")
	newIndex := gobed.NewVectorIndex(model, config)
	
	start = time.Now()
	err = newIndex.AddDocumentsBulkGPU(docs)
	if err != nil {
		log.Fatalf("Forced GPU indexing failed: %v", err)
	}
	elapsed = time.Since(start)
	fmt.Printf("✅ GPU indexed %d documents in %.2fs (%.0f docs/sec)\n", 
		len(docs), elapsed.Seconds(), float64(len(docs))/elapsed.Seconds())
	
	// Method 3: Bulk indexing with real-time monitoring
	fmt.Printf("\n📊 Method 3: Bulk indexing with GPU monitoring\n")
	monitoredIndex := gobed.NewVectorIndex(model, config)
	
	progressChan, err := monitoredIndex.AddDocumentsWithMonitoring(docs)
	if err != nil {
		log.Fatalf("Monitored indexing failed to start: %v", err)
	}
	
	// Monitor progress in real-time
	for progress := range progressChan {
		if progress.Error != nil {
			log.Printf("❌ Error: %v", progress.Error)
			break
		}
		
		if progress.Complete {
			fmt.Printf("✅ Monitoring complete! Final throughput: %.0f docs/sec\n", progress.Throughput)
			if progress.GPUStats != nil {
				progress.GPUStats.LogStats()
			}
			break
		} else {
			// Log progress updates
			progressPercent := float64(progress.Current) * 100 / float64(progress.Total)
			fmt.Printf("   Progress: %.1f%% (%d/%d) - %.0f docs/sec\n", 
				progressPercent, progress.Current, progress.Total, progress.Throughput)
		}
	}
	
	// Test search functionality
	fmt.Printf("\n🔍 Testing search functionality\n")
	query := "document topic keywords content"
	start = time.Now()
	results, err := monitoredIndex.Search(query, 5)
	searchTime := time.Since(start)
	
	if err != nil {
		log.Printf("❌ Search failed: %v", err)
	} else {
		fmt.Printf("✅ Search completed in %.2fms, found %d results\n", 
			float64(searchTime.Nanoseconds())/1e6, len(results))
		
		for i, result := range results {
			fmt.Printf("   %d. Doc %d (similarity: %.4f)\n", 
				i+1, result.ID, result.Similarity)
		}
	}
	
	fmt.Printf("\n🎯 Example completed successfully!\n")
	fmt.Printf("Key features demonstrated:\n")
	fmt.Printf("  • Automatic GPU bulk indexing for large datasets\n")
	fmt.Printf("  • Manual GPU bulk indexing control\n")
	fmt.Printf("  • Real-time GPU memory monitoring\n")
	fmt.Printf("  • High-throughput document processing (5k batch size)\n")
	fmt.Printf("  • Seamless integration with existing search functionality\n")
}