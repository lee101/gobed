package main

/*
#cgo CPPFLAGS: -I../../gpu -I../../libtorch/include -I../../libtorch/include/torch/csrc/api/include
#cgo LDFLAGS: -L../../libtorch/lib -L../../gpu -L/usr/local/cuda-12.0/targets/x86_64-linux/lib -ltorch_cgo_wrapper -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lcudart -ldl
#include "torch_cgo_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"
	"unsafe"
)

type TextDocument struct {
	ID      int
	Title   string
	Content string
	Vector  []int8
}

type InteractiveDemo struct {
	indexer   C.TorchIndexerHandle
	documents []TextDocument
	vectorDim int
}

func main() {
	fmt.Println("🔍 Interactive Text Search Demo - LibTorch GPU")
	fmt.Println("==============================================")

	// System info
	version := C.GoString(C.torch_get_version())
	cudaAvailable := C.torch_cuda_is_available() != 0
	deviceCount := int(C.torch_cuda_device_count())

	fmt.Printf("📊 System Status:\n")
	fmt.Printf("   LibTorch: %s\n", version)
	fmt.Printf("   CUDA Available: %v\n", cudaAvailable)
	fmt.Printf("   GPU Devices: %d\n", deviceCount)
	fmt.Printf("   Go Runtime: %s\n", runtime.Version())

	if !cudaAvailable {
		fmt.Println("⚠️  CUDA not available - running on CPU")
	}

	demo := &InteractiveDemo{vectorDim: 384} // Common embedding dimension
	
	fmt.Println("\n🔧 Building text corpus and embeddings...")
	demo.buildTextCorpus()
	
	fmt.Println("\n🚀 Initializing LibTorch indexer...")
	demo.initializeIndexer()
	
	fmt.Println("\n📚 Training index...")
	demo.trainIndex()
	
	fmt.Println("\n📝 Adding documents to index...")
	demo.indexDocuments()
	
	fmt.Println("\n✅ Ready for interactive search!")
	demo.runInteractiveSearch()
}

func (demo *InteractiveDemo) buildTextCorpus() {
	// Generate diverse text corpus
	categories := []struct {
		name  string
		texts []string
	}{
		{
			name: "Technology",
			texts: []string{
				"Artificial intelligence and machine learning are revolutionizing how we process data",
				"Cloud computing enables scalable infrastructure for modern applications",
				"Quantum computing promises to solve complex computational problems",
				"Blockchain technology provides decentralized and secure transactions",
				"Neural networks can learn complex patterns from large datasets",
				"Deep learning models achieve state-of-the-art performance in computer vision",
				"Natural language processing enables computers to understand human language",
				"Edge computing brings computation closer to data sources",
				"Internet of Things connects everyday devices to the network",
				"Cybersecurity protects digital assets from malicious attacks",
			},
		},
		{
			name: "Science",
			texts: []string{
				"DNA sequencing has revolutionized our understanding of genetics",
				"Climate change affects global weather patterns and ecosystems",
				"Space exploration helps us understand the universe and our place in it",
				"Renewable energy sources are crucial for sustainable development",
				"Vaccines have saved millions of lives throughout human history",
				"Photosynthesis converts sunlight into chemical energy in plants",
				"Evolution explains the diversity of life on Earth",
				"The periodic table organizes chemical elements by their properties",
				"Gravity is the fundamental force that governs celestial mechanics",
				"Antibiotics fight bacterial infections and have transformed medicine",
			},
		},
		{
			name: "Business",
			texts: []string{
				"Customer satisfaction drives long-term business success and growth",
				"Market research helps companies understand consumer needs and preferences",
				"Supply chain management optimizes the flow of goods and services",
				"Digital transformation modernizes business processes and operations",
				"Leadership skills are essential for managing teams and organizations",
				"Financial planning ensures sustainable business growth and profitability",
				"Brand building creates lasting connections with target audiences",
				"Innovation drives competitive advantage in rapidly changing markets",
				"Data analytics provides insights for informed business decisions",
				"Risk management protects businesses from unforeseen challenges",
			},
		},
		{
			name: "Health",
			texts: []string{
				"Regular exercise improves cardiovascular health and overall wellbeing",
				"Balanced nutrition provides essential nutrients for optimal body function",
				"Mental health awareness reduces stigma and promotes treatment",
				"Preventive care helps detect health issues early and saves lives",
				"Sleep quality significantly impacts cognitive performance and immune function",
				"Stress management techniques improve quality of life and productivity",
				"Hydration maintains proper bodily functions and energy levels",
				"Meditation and mindfulness reduce anxiety and improve focus",
				"Regular checkups help monitor health status and prevent diseases",
				"Healthy relationships contribute to emotional wellbeing and longevity",
			},
		},
		{
			name: "Environment",
			texts: []string{
				"Renewable energy reduces carbon emissions and environmental impact",
				"Recycling conserves natural resources and reduces waste",
				"Biodiversity preservation protects ecosystems and endangered species",
				"Sustainable agriculture ensures food security while protecting the environment",
				"Water conservation addresses scarcity and ensures future availability",
				"Forest conservation prevents deforestation and maintains carbon balance",
				"Clean transportation reduces air pollution and greenhouse gases",
				"Green building practices minimize environmental impact of construction",
				"Environmental education raises awareness about conservation issues",
				"Wildlife protection preserves natural habitats and ecosystem balance",
			},
		},
		{
			name: "Education",
			texts: []string{
				"Online learning provides flexible access to education worldwide",
				"Critical thinking skills help students analyze information effectively",
				"Collaborative learning encourages teamwork and knowledge sharing",
				"Educational technology enhances teaching methods and student engagement",
				"Lifelong learning ensures continuous skill development and adaptation",
				"Inclusive education provides equal opportunities for all students",
				"Research skills enable students to find and evaluate reliable information",
				"Creative problem solving develops innovative thinking abilities",
				"Assessment methods should measure understanding rather than memorization",
				"Teacher training improves educational quality and student outcomes",
			},
		},
		{
			name: "Arts",
			texts: []string{
				"Visual arts express emotions and ideas through creative mediums",
				"Music therapy helps patients heal and improve mental health",
				"Literature preserves cultural heritage and human experiences",
				"Theater brings stories to life through live performance",
				"Digital art combines technology with traditional artistic expression",
				"Cultural diversity enriches artistic expression and creativity",
				"Art education develops creativity and aesthetic appreciation",
				"Public art transforms communities and creates shared experiences",
				"Art conservation preserves masterpieces for future generations",
				"Contemporary art reflects current social and political issues",
			},
		},
		{
			name: "Travel",
			texts: []string{
				"Cultural immersion provides authentic travel experiences and understanding",
				"Sustainable tourism protects destinations while supporting local economies",
				"Adventure travel offers thrilling experiences in natural environments",
				"Food tourism explores local cuisines and culinary traditions",
				"Historical sites provide insights into past civilizations and cultures",
				"Travel photography captures memories and shares experiences with others",
				"Language learning enhances travel experiences and cultural connections",
				"Solo travel builds confidence and encourages personal growth",
				"Group travel creates shared memories and strengthens relationships",
				"Travel planning ensures smooth trips and maximizes experiences",
			},
		},
		{
			name: "Sports",
			texts: []string{
				"Team sports develop cooperation and communication skills",
				"Individual sports build personal discipline and self-motivation",
				"Athletic training improves physical fitness and performance",
				"Sports psychology helps athletes manage pressure and maintain focus",
				"Youth sports teach valuable life lessons and social skills",
				"Professional sports inspire millions of fans worldwide",
				"Sports medicine prevents injuries and optimizes athlete performance",
				"Olympic games celebrate international competition and unity",
				"Adaptive sports provide opportunities for athletes with disabilities",
				"Sports nutrition optimizes energy and recovery for athletes",
			},
		},
		{
			name: "History",
			texts: []string{
				"Ancient civilizations laid the foundation for modern society",
				"Historical events shape current political and social structures",
				"Archaeological discoveries reveal insights about past cultures",
				"World wars transformed global politics and international relations",
				"Renaissance period marked a revival of art, science, and learning",
				"Industrial revolution changed how humans work and live",
				"Cultural revolutions challenged existing social norms and values",
				"Historical preservation maintains connections to our heritage",
				"Oral traditions pass down knowledge through generations",
				"Historical research methods help us understand the past accurately",
			},
		},
	}

	docID := 0
	for _, category := range categories {
		for i, text := range category.texts {
			// Create synthetic embeddings (in real app, use actual embedding model)
			vector := demo.generateSyntheticEmbedding(text, category.name)
			
			doc := TextDocument{
				ID:      docID,
				Title:   fmt.Sprintf("%s Document %d", category.name, i+1),
				Content: text,
				Vector:  vector,
			}
			demo.documents = append(demo.documents, doc)
			docID++
		}
	}

	fmt.Printf("   📊 Generated %d documents across %d categories\n", len(demo.documents), len(categories))
	fmt.Printf("   🎯 Vector dimension: %d\n", demo.vectorDim)
}

func (demo *InteractiveDemo) generateSyntheticEmbedding(text, category string) []int8 {
	// Generate deterministic but varied embeddings based on text content
	// This simulates real embeddings with semantic similarity
	vector := make([]int8, demo.vectorDim)
	
	// Use text hash for deterministic generation
	hash := 0
	for _, char := range text + category {
		hash = hash*31 + int(char)
	}
	
	rng := rand.New(rand.NewSource(int64(hash)))
	
	// Generate base pattern based on category
	categorySeeds := map[string]int{
		"Technology":  1000,
		"Science":     2000,
		"Business":    3000,
		"Health":      4000,
		"Environment": 5000,
		"Education":   6000,
		"Arts":        7000,
		"Travel":      8000,
		"Sports":      9000,
		"History":     10000,
	}
	
	categorySeed := categorySeeds[category]
	
	// Create category-specific patterns
	for i := 0; i < demo.vectorDim; i++ {
		// Mix category pattern with text-specific variation
		categoryComponent := int8((categorySeed + i*17) % 256 - 128)
		textComponent := int8(rng.Intn(256) - 128)
		
		// Weight: 70% category pattern, 30% text variation
		vector[i] = int8((int(categoryComponent)*7 + int(textComponent)*3) / 10)
	}
	
	return vector
}

func (demo *InteractiveDemo) initializeIndexer() {
	config := C.IndexConfig{
		vector_dim:        C.int(demo.vectorDim),
		num_subquantizers: C.int(64),
		codebook_size:     C.int(256),
		ivf_clusters:      C.int(256), // Smaller for 1000 docs
		probe_lists:       C.int(16),
		rerank_k:         C.int(100),
		device_id:        C.int(0), // Force GPU usage
	}

	demo.indexer = C.torch_indexer_create(config)
	if demo.indexer == nil {
		log.Fatal("❌ Failed to create indexer")
	}

	fmt.Println("   ✅ Indexer initialized with GPU configuration")
}

func (demo *InteractiveDemo) trainIndex() {
	// Use first 100 documents for training
	trainingSize := 100
	if len(demo.documents) < trainingSize {
		trainingSize = len(demo.documents)
	}

	trainingData := make([]int8, trainingSize*demo.vectorDim)
	for i := 0; i < trainingSize; i++ {
		copy(trainingData[i*demo.vectorDim:], demo.documents[i].Vector)
	}

	start := time.Now()
	result := C.torch_indexer_train(
		demo.indexer,
		(*C.schar)(unsafe.Pointer(&trainingData[0])),
		C.int(trainingSize),
		C.int(demo.vectorDim),
	)
	trainTime := time.Since(start)

	if result == 0 {
		log.Fatal("❌ Training failed")
	}

	fmt.Printf("   ✅ Training completed in %v (%d vectors)\n", trainTime, trainingSize)
}

func (demo *InteractiveDemo) indexDocuments() {
	allVectors := make([]int8, len(demo.documents)*demo.vectorDim)
	for i, doc := range demo.documents {
		copy(allVectors[i*demo.vectorDim:], doc.Vector)
	}

	start := time.Now()
	result := C.torch_indexer_add_vectors(
		demo.indexer,
		(*C.schar)(unsafe.Pointer(&allVectors[0])),
		C.int(len(demo.documents)),
		C.int(demo.vectorDim),
	)
	indexTime := time.Since(start)

	if result == 0 {
		log.Fatal("❌ Indexing failed")
	}

	indexRate := float64(len(demo.documents)) / indexTime.Seconds()
	
	// Get memory stats
	stats := C.torch_indexer_get_stats(demo.indexer)
	
	fmt.Printf("   ✅ Indexed %d documents in %v\n", len(demo.documents), indexTime)
	fmt.Printf("   📊 Rate: %.1f docs/sec\n", indexRate)
	fmt.Printf("   💾 GPU Memory: %.1f MB\n", float64(stats.gpu_memory_mb))
}

func (demo *InteractiveDemo) runInteractiveSearch() {
	scanner := bufio.NewScanner(os.Stdin)
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🔍 Interactive Text Search")
	fmt.Println("Type your search query (or 'quit' to exit)")
	fmt.Println("Commands: 'stats', 'help', 'examples'")
	fmt.Println(strings.Repeat("=", 60))

	for {
		fmt.Print("\n🔍 Search> ")
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}

		switch strings.ToLower(query) {
		case "quit", "exit", "q":
			fmt.Println("👋 Goodbye!")
			return
		case "stats":
			demo.printStats()
			continue
		case "help":
			demo.printHelp()
			continue
		case "examples":
			demo.printExamples()
			continue
		}

		demo.performSearch(query)
	}
}

func (demo *InteractiveDemo) performSearch(query string) {
	start := time.Now()
	
	// Generate query embedding (same method as documents)
	queryVector := demo.generateSyntheticEmbedding(query, "Query")
	
	// Perform search
	searchResult := C.torch_indexer_search(
		demo.indexer,
		(*C.schar)(unsafe.Pointer(&queryVector[0])),
		C.int(demo.vectorDim),
		C.int(10), // Top 10 results
	)
	
	searchTime := time.Since(start)
	
	if searchResult.count == 0 {
		fmt.Println("❌ No results found")
		return
	}

	fmt.Printf("\n📊 Found %d results in %v (%.2f μs)\n", 
		searchResult.count, searchTime, float64(searchTime.Nanoseconds())/1000.0)
	fmt.Println(strings.Repeat("-", 60))

	// Convert C arrays to Go slices
	ids := (*[10]C.int)(unsafe.Pointer(searchResult.ids))[:searchResult.count:searchResult.count]
	scores := (*[10]C.float)(unsafe.Pointer(searchResult.scores))[:searchResult.count:searchResult.count]

	for i := 0; i < int(searchResult.count); i++ {
		docID := int(ids[i])
		score := float64(scores[i])
		
		if docID >= 0 && docID < len(demo.documents) {
			doc := demo.documents[docID]
			fmt.Printf("🏆 %d. [Score: %.1f] %s\n", i+1, score, doc.Title)
			
			// Highlight matching content
			content := doc.Content
			if len(content) > 80 {
				content = content[:77] + "..."
			}
			fmt.Printf("   📝 %s\n", content)
		}
	}

	// Free search results
	C.torch_search_result_free(&searchResult)
	
	// Show performance stats
	stats := C.torch_indexer_get_stats(demo.indexer)
	fmt.Printf("\n💾 GPU Memory: %.1f MB | 📚 Total Docs: %d\n", 
		float64(stats.gpu_memory_mb), len(demo.documents))
}

func (demo *InteractiveDemo) printStats() {
	stats := C.torch_indexer_get_stats(demo.indexer)
	
	fmt.Println("\n📊 System Statistics:")
	fmt.Printf("   📚 Documents: %d\n", len(demo.documents))
	fmt.Printf("   🎯 Vector Dimension: %d\n", demo.vectorDim)
	fmt.Printf("   💾 GPU Memory: %.1f MB\n", float64(stats.gpu_memory_mb))
	fmt.Printf("   🏗️  Index Built: %v\n", stats.index_built != 0)
	fmt.Printf("   🎓 Trained: %v\n", stats.is_trained != 0)
	fmt.Printf("   🔧 IVF Clusters: %d\n", int(stats.ivf_clusters))
}

func (demo *InteractiveDemo) printHelp() {
	fmt.Println("\n📖 Help:")
	fmt.Println("   🔍 Type any text to search for similar documents")
	fmt.Println("   📊 'stats' - Show system statistics")
	fmt.Println("   💡 'examples' - Show example queries")
	fmt.Println("   ❓ 'help' - Show this help")
	fmt.Println("   🚪 'quit' - Exit the demo")
}

func (demo *InteractiveDemo) printExamples() {
	examples := []string{
		"artificial intelligence and machine learning",
		"renewable energy and environment",
		"health and wellness",
		"space exploration",
		"financial planning",
		"online education",
		"travel and culture",
		"sports training",
		"art and creativity",
		"historical events",
	}
	
	fmt.Println("\n💡 Example Queries:")
	for i, example := range examples {
		fmt.Printf("   %d. %s\n", i+1, example)
	}
}