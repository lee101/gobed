//go:build legacy

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/lee101/gobed"
)

func main() {
	// Command line flags
	var (
		port            = flag.Int("port", 8080, "Server port")
		sharedPath      = flag.String("shared-path", "/tmp/gobed_shared_index", "Path for shared memory index")
		maxVectors      = flag.Int("max-vectors", 1000000, "Maximum number of vectors")
		readOnly        = flag.Bool("readonly", false, "Start in read-only mode")
		workers         = flag.Int("workers", 0, "Number of worker threads (0=auto)")
		cacheSize       = flag.Int("cache", 10000, "Hot vector cache size")
		enableProfiling = flag.Bool("profiling", false, "Enable pprof profiling")
		enableMetrics   = flag.Bool("metrics", true, "Enable metrics endpoint")
		preload         = flag.Bool("preload", false, "Preload model embeddings")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Gobed High-Performance Search Server\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # Start server with shared memory index:\n")
		fmt.Fprintf(os.Stderr, "  %s --port 8080 --shared-path /tmp/search_index\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Start read-only replica:\n")
		fmt.Fprintf(os.Stderr, "  %s --port 8081 --readonly --shared-path /tmp/search_index\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Start with profiling:\n")
		fmt.Fprintf(os.Stderr, "  %s --profiling --metrics\n\n", os.Args[0])
	}

	flag.Parse()

	// Print banner
	printBanner()

	// Load embedding model
	log.Println("Loading embedding model...")
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	log.Println("✓ Model loaded successfully")

	// Configure server
	config := gobed.ServerConfig{
		Port:              *port,
		SharedIndexPath:   *sharedPath,
		MaxVectors:        *maxVectors,
		MaxConcurrency:    1000,
		EnableProfiling:   *enableProfiling,
		EnableMetrics:     *enableMetrics,
		ReadOnly:          *readOnly,
		PreloadEmbeddings: *preload,
		WorkerThreads:     *workers,
	}

	// Print configuration
	printConfig(config)

	// Create and start server
	server, err := gobed.NewSearchServer(model, config)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}

	// Print endpoints
	printEndpoints(config)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	log.Println("Server is ready. Press Ctrl+C to stop.")
	<-sigChan

	// Graceful shutdown
	log.Println("\nShutting down server...")
	if err := server.Stop(); err != nil {
		log.Printf("Error during shutdown: %v", err)
	}

	log.Println("Server stopped successfully")
}

func printBanner() {
	banner := `
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     ██████╗  ██████╗ ██████╗ ███████╗██████╗           ║
║    ██╔════╝ ██╔═══██╗██╔══██╗██╔════╝██╔══██╗          ║
║    ██║  ███╗██║   ██║██████╔╝█████╗  ██║  ██║          ║
║    ██║   ██║██║   ██║██╔══██╗██╔══╝  ██║  ██║          ║
║    ╚██████╔╝╚██████╔╝██████╔╝███████╗██████╔╝          ║
║     ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚═════╝           ║
║                                                          ║
║          High-Performance Vector Search Server          ║
║                  with Shared Memory Index               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
`
	fmt.Println(banner)
}

func printConfig(config gobed.ServerConfig) {
	fmt.Println("\n Configuration:")
	fmt.Println("═══════════════════════════════════════")
	fmt.Printf("  Port:              %d\n", config.Port)
	fmt.Printf("  Shared Index Path: %s\n", config.SharedIndexPath)
	fmt.Printf("  Max Vectors:       %d\n", config.MaxVectors)
	fmt.Printf("  Read-Only Mode:    %v\n", config.ReadOnly)
	fmt.Printf("  Worker Threads:    %d\n", config.WorkerThreads)
	fmt.Printf("  Profiling:         %v\n", config.EnableProfiling)
	fmt.Printf("  Metrics:           %v\n", config.EnableMetrics)
	fmt.Println()
}

func printEndpoints(config gobed.ServerConfig) {
	fmt.Println("\n🌐 Available Endpoints:")
	fmt.Println("═══════════════════════════════════════")

	baseURL := fmt.Sprintf("http://localhost:%d", config.Port)

	// Search endpoints
	fmt.Println("\n📍 Search Endpoints:")
	fmt.Printf("  POST %s/search        - Single search\n", baseURL)
	fmt.Printf("  POST %s/batch_search  - Batch search\n", baseURL)

	// Index endpoints (if not read-only)
	if !config.ReadOnly {
		fmt.Println("\n Indexing Endpoints:")
		fmt.Printf("  POST %s/index         - Index documents\n", baseURL)
		fmt.Printf("  POST %s/batch_index   - Batch index documents\n", baseURL)
	}

	// Monitoring endpoints
	fmt.Println("\n Monitoring Endpoints:")
	fmt.Printf("  GET  %s/health        - Health check\n", baseURL)

	if config.EnableMetrics {
		fmt.Printf("  GET  %s/metrics       - Performance metrics\n", baseURL)
	}

	if config.EnableProfiling {
		fmt.Printf("  GET  %s/debug/pprof/  - Profiling data\n", baseURL)
	}

	// Example requests
	fmt.Println("\n📖 Example Requests:")
	fmt.Println("═══════════════════════════════════════")

	fmt.Println("\n1⃣ Search for documents:")
	fmt.Println(`curl -X POST http://localhost:` + fmt.Sprint(config.Port) + `/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "k": 10
  }'`)

	if !config.ReadOnly {
		fmt.Println("\n2⃣ Index new documents:")
		fmt.Println(`curl -X POST http://localhost:` + fmt.Sprint(config.Port) + `/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": 1, "text": "Introduction to deep learning"},
      {"id": 2, "text": "Natural language processing"}
    ]
  }'`)
	}

	fmt.Println("\n3⃣ Check server health:")
	fmt.Printf("curl http://localhost:%d/health\n", config.Port)

	fmt.Println("\n4⃣ Get performance metrics:")
	fmt.Printf("curl http://localhost:%d/metrics\n", config.Port)

	fmt.Println("\n═══════════════════════════════════════")
}
