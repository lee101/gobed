package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

// Diverse corpus covering many domains
var baseCorpus = []string{
	// Technology (25)
	"Python is a versatile programming language widely used in data science and web development",
	"JavaScript powers interactive web applications and runs in browsers and Node.js environments",
	"Go provides efficient concurrency through goroutines and channels for system programming",
	"Rust ensures memory safety without garbage collection through its ownership system",
	"Java runs on billions of devices with platform independence through the JVM",
	"C++ offers high performance and low-level control for system and game development",
	"TypeScript adds static typing to JavaScript for better code maintainability",
	"Swift is Apple's modern language for iOS and macOS application development",
	"Kotlin is Google's preferred language for Android app development",
	"Ruby emphasizes developer happiness and convention over configuration",
	"Machine learning algorithms learn patterns from data to make predictions",
	"Deep learning uses multi-layer neural networks for complex pattern recognition",
	"Natural language processing enables computers to understand human language",
	"Computer vision allows machines to interpret and analyze visual information",
	"Reinforcement learning trains agents through reward and punishment signals",
	"Docker containers package applications with dependencies for portability",
	"Kubernetes orchestrates containerized applications with automatic scaling",
	"Microservices architecture decomposes applications into small independent services",
	"REST APIs enable communication between systems using HTTP protocols",
	"GraphQL provides flexible query language for APIs with exact data fetching",
	"PostgreSQL is a powerful open-source relational database with ACID compliance",
	"MongoDB stores data in flexible JSON-like documents for scalability",
	"Redis provides in-memory data structures for caching and messaging",
	"Elasticsearch enables full-text search and analytics at scale",
	"Apache Kafka handles real-time data streaming and event processing",

	// Science (25)
	"Quantum computing leverages superposition and entanglement for exponential speedup",
	"CRISPR-Cas9 enables precise gene editing by cutting DNA at specific locations",
	"Climate change causes rising temperatures and extreme weather events globally",
	"Artificial neural networks mimic biological neurons for pattern recognition",
	"Black holes have gravitational fields so strong that nothing can escape",
	"DNA stores genetic information in a double helix structure with base pairs",
	"Evolution occurs through natural selection acting on genetic variation",
	"Photosynthesis converts sunlight into chemical energy in plants and algae",
	"Stem cells can differentiate into specialized cell types for regeneration",
	"The Large Hadron Collider studies fundamental particles at high energies",
	"Dark matter makes up most of the universe's mass but doesn't emit light",
	"Vaccines train the immune system to recognize and fight pathogens",
	"Antibiotics kill bacteria but are ineffective against viral infections",
	"MRI scans use magnetic fields to create detailed images of soft tissues",
	"The human brain contains 86 billion neurons connected by synapses",
	"Renewable energy from solar and wind reduces carbon emissions",
	"Nuclear fusion powers the sun by combining hydrogen into helium",
	"Plate tectonics explains earthquakes and volcanic activity on Earth",
	"The greenhouse effect traps heat in Earth's atmosphere causing warming",
	"Biodiversity loss threatens ecosystem stability and human well-being",
	"Nanotechnology manipulates matter at the atomic and molecular scale",
	"The human genome contains about 3 billion DNA base pairs",
	"Antibody therapies use immune proteins to treat diseases",
	"Ocean acidification threatens marine ecosystems due to CO2 absorption",
	"Space exploration advances our understanding of the universe",

	// Business & Finance (25)
	"Stock markets enable trading of company shares and securities",
	"Bitcoin pioneered cryptocurrency using blockchain technology",
	"Supply chain management optimizes the flow of goods and services",
	"Digital marketing reaches customers through online channels",
	"E-commerce platforms facilitate online buying and selling",
	"Venture capital funds high-growth startups in exchange for equity",
	"Data analytics transforms raw data into actionable insights",
	"Project management ensures projects complete on time and budget",
	"Business intelligence systems support data-driven decision making",
	"Financial planning helps achieve long-term monetary goals",
	"Investment banking advises on mergers and capital raising",
	"Risk management identifies and mitigates potential threats",
	"Customer relationship management tracks all customer interactions",
	"Agile methodology promotes iterative development and flexibility",
	"Market research analyzes consumer behavior and preferences",
	"Entrepreneurship creates new businesses and innovations",
	"Corporate strategy defines competitive positioning and goals",
	"Human resources manages talent acquisition and development",
	"Accounting maintains financial records and ensures compliance",
	"Economics studies resource allocation and market behavior",
	"Blockchain creates tamper-proof distributed ledgers",
	"Fintech innovations disrupt traditional financial services",
	"Private equity invests in companies for restructuring",
	"Options trading provides leverage and hedging strategies",
	"ESG investing considers environmental and social factors",

	// Healthcare (25)
	"Telemedicine enables remote medical consultations via technology",
	"Personalized medicine tailors treatments to individual genetics",
	"Cancer immunotherapy harnesses the immune system to fight tumors",
	"Mental health awareness reduces stigma around psychological conditions",
	"Diabetes management involves blood sugar monitoring and lifestyle changes",
	"Heart disease remains the leading cause of death globally",
	"Alzheimer's research seeks treatments for cognitive decline",
	"Gene therapy replaces defective genes to treat diseases",
	"Robotic surgery provides precision and minimally invasive procedures",
	"Clinical trials test new treatments for safety and efficacy",
	"Epidemiology studies disease patterns in populations",
	"Preventive medicine focuses on disease prevention rather than treatment",
	"Regenerative medicine uses stem cells to repair damaged tissues",
	"Medical imaging technologies visualize internal body structures",
	"Pharmaceutical development creates new drugs through research",
	"Public health initiatives improve community wellness",
	"Nutrition science studies how food affects health",
	"Exercise physiology examines body responses to physical activity",
	"Sleep medicine addresses disorders affecting rest and recovery",
	"Addiction treatment combines therapy and medication",
	"Pediatric care specializes in children's health needs",
	"Geriatric medicine focuses on aging-related health issues",
	"Emergency medicine provides acute care for urgent conditions",
	"Precision diagnostics uses advanced testing for accurate diagnosis",
	"Healthcare informatics manages medical data and systems",
}

func expandCorpus(size int) []string {
	corpus := make([]string, 0, size)

	// Modifiers to create variations
	prefixes := []string{
		"", "Introduction to ", "Advanced ", "Modern ", "Essential ",
		"Practical ", "Understanding ", "The future of ", "Exploring ",
		"Fundamentals of ", "Mastering ", "Complete guide to ",
	}

	suffixes := []string{
		"", " explained", " in practice", " for beginners", " at scale",
		" best practices", " case study", " deep dive", " overview",
		" trends", " innovations", " strategies",
	}

	contexts := []string{
		"", " in 2024", " for enterprises", " for startups", " in production",
		" for developers", " for managers", " for students", " for professionals",
	}

	// Generate variations
	for len(corpus) < size {
		// Pick a random base text
		base := baseCorpus[rand.Intn(len(baseCorpus))]

		// Apply random modifications
		prefix := prefixes[rand.Intn(len(prefixes))]
		suffix := suffixes[rand.Intn(len(suffixes))]
		context := contexts[rand.Intn(len(contexts))]

		// Construct variation
		text := prefix
		if prefix != "" {
			text += strings.ToLower(base[:1]) + base[1:]
		} else {
			text = base
		}
		text += suffix + context

		corpus = append(corpus, text)

		// Sometimes combine two topics
		if rand.Float32() < 0.2 && len(corpus) < size {
			other := baseCorpus[rand.Intn(len(baseCorpus))]
			combined := base[:len(base)/2] + " combined with " + strings.ToLower(other[:len(other)/2])
			corpus = append(corpus, combined)
		}
	}

	return corpus[:size]
}

func main_disabled() {
	fmt.Println("=== Gobed Large-Scale Search Benchmark ===")
	fmt.Println("Testing with 100,000 documents\n")

	// Load model
	fmt.Println("Loading embedding model...")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ Model loaded in %v\n\n", time.Since(start))

	// Generate large corpus
	corpusSize := 100000
	fmt.Printf("Generating corpus of %d documents...\n", corpusSize)
	start = time.Now()
	corpus := expandCorpus(corpusSize)
	fmt.Printf("✓ Corpus generated in %v\n\n", time.Since(start))

	// Create search engine with optimized config for 100k
	config := gobed.SearchConfig{
		AutoMode:           false,
		MaxExactSearchSize: 10000, // Use IVF for this size
		NumClusters:        1000,  // sqrt(100k) ≈ 316, use 1000 for better distribution
		SearchClusters:     10,    // Search 1% of clusters
		UseCompression:     false, // Skip PQ for better accuracy at 100k
		UseGraphRouting:    false, // Skip HNSW for simplicity at 100k
		CandidatesToRerank: 100,   // Rerank top 100
	}
	engine := gobed.NewSearchEngineWithConfig(model, config)

	// Index documents in batches
	fmt.Println("=== INDEXING PHASE ===")
	batchSize := 2000
	indexStart := time.Now()

	for i := 0; i < corpusSize; i += batchSize {
		end := min(i+batchSize, corpusSize)
		batchStart := time.Now()

		_, err := engine.IndexBatch(corpus[i:end])
		if err != nil {
			log.Printf("Failed to index batch: %v", err)
			continue
		}

		if (i/batchSize+1)%10 == 0 {
			elapsed := time.Since(indexStart)
			rate := float64(i+batchSize) / elapsed.Seconds()
			fmt.Printf("  Progress: %d/%d documents (%.0f docs/sec)\n",
				min(i+batchSize, corpusSize), corpusSize, rate)
		}
	}

	indexTime := time.Since(indexStart)
	fmt.Printf("\n✓ Indexing completed\n")
	fmt.Printf("  Total time: %v\n", indexTime)
	fmt.Printf("  Throughput: %.0f docs/sec\n", float64(corpusSize)/indexTime.Seconds())
	fmt.Printf("  Latency: %.3f ms/doc\n\n", float64(indexTime.Milliseconds())/float64(corpusSize))

	// Optimize index
	fmt.Println("Optimizing index for search performance...")
	optStart := time.Now()
	err = engine.Optimize()
	if err != nil {
		fmt.Printf("  Optimization skipped: %v\n", err)
	} else {
		fmt.Printf("✓ Optimization completed in %v\n\n", time.Since(optStart))
	}

	// Get statistics
	stats := engine.Stats()
	fmt.Println("=== INDEX STATISTICS ===")
	fmt.Printf("Documents:    %d\n", stats.NumDocuments)
	fmt.Printf("Index Type:   %s\n", stats.IndexType)
	fmt.Printf("Memory:       %.2f MB\n", stats.MemoryUsageMB)
	fmt.Printf("Details:      %+v\n\n", stats.IndexDetails)

	// Test queries covering different domains
	testQueries := []struct {
		query  string
		domain string
	}{
		{"deep learning neural network architectures", "AI/ML"},
		{"kubernetes container orchestration deployment", "DevOps"},
		{"quantum computing cryptography applications", "Quantum"},
		{"gene editing CRISPR therapy", "Biotech"},
		{"cryptocurrency blockchain finance", "Fintech"},
		{"climate change renewable energy", "Environment"},
		{"stock market investment strategies", "Finance"},
		{"cancer immunotherapy treatment", "Healthcare"},
		{"microservices architecture patterns", "Software"},
		{"data science machine learning pipelines", "Data"},
		{"mobile app development frameworks", "Mobile"},
		{"cybersecurity threat detection", "Security"},
		{"cloud computing AWS Azure", "Cloud"},
		{"natural language processing transformers", "NLP"},
		{"database optimization PostgreSQL performance", "Database"},
	}

	fmt.Println("=== SEARCH PERFORMANCE ===")
	fmt.Printf("Running %d search queries...\n\n", len(testQueries))

	totalSearchTime := time.Duration(0)
	k := 5

	for i, tq := range testQueries {
		searchStart := time.Now()
		results, err := engine.Search(tq.query, k)
		searchTime := time.Since(searchStart)
		totalSearchTime += searchTime

		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("Query %d: \"%s\"\n", i+1, tq.query)
		fmt.Printf("Domain: %s | Latency: %v\n", tq.domain, searchTime)
		fmt.Println("Top 3 results:")

		for j := 0; j < min(3, len(results)); j++ {
			text := results[j].Text
			if len(text) > 75 {
				text = text[:72] + "..."
			}
			fmt.Printf("  %d. [%.3f] %s\n", j+1, results[j].Similarity, text)
		}
		fmt.Println()
	}

	// Calculate performance metrics
	avgSearchLatency := totalSearchTime / time.Duration(len(testQueries))
	searchQPS := float64(len(testQueries)) / totalSearchTime.Seconds()

	fmt.Println("=== BENCHMARK SUMMARY ===")
	fmt.Println("\nDataset:")
	fmt.Printf("  Size: %d documents\n", corpusSize)
	fmt.Printf("  Index type: %s\n", stats.IndexType)
	fmt.Printf("  Memory usage: %.2f MB\n", stats.MemoryUsageMB)

	fmt.Println("\nIndexing Performance:")
	fmt.Printf("  Total time: %v\n", indexTime)
	fmt.Printf("  Throughput: %.0f docs/sec\n", float64(corpusSize)/indexTime.Seconds())
	fmt.Printf("  Avg latency: %.3f ms/doc\n", float64(indexTime.Milliseconds())/float64(corpusSize))

	fmt.Println("\nSearch Performance:")
	fmt.Printf("  Avg latency: %v\n", avgSearchLatency)
	fmt.Printf("  P50 estimate: %v\n", avgSearchLatency*9/10)
	fmt.Printf("  P99 estimate: %v\n", avgSearchLatency*2)
	fmt.Printf("  Throughput: %.1f QPS\n", searchQPS)

	if avgSearchLatency < time.Millisecond {
		fmt.Println("\n✅ SUB-MILLISECOND SEARCH ACHIEVED!")
	} else if avgSearchLatency < 2*time.Millisecond {
		fmt.Println("\n✅ Target latency achieved (<2ms)")
	}

	fmt.Println("\n✓ Benchmark completed successfully!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
