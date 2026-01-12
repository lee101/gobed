package main

import (
	"database/sql"
	"fmt"
	"log"
	"runtime"
	"sort"
	"time"

	_ "github.com/lib/pq"
	"github.com/lee101/gobed"
	"github.com/lee101/gobed/pkg/ann/simd"
)

func main() {
	log.Println("GPU Scale Benchmark - Real Data")

	db, err := sql.Open("postgres", "postgresql://lee:netwrck2024@localhost/netwrck_ai_characters")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	var count int
	db.QueryRow("SELECT COUNT(*) FROM ai_characters").Scan(&count)
	log.Printf("Total characters: %d", count)

	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Loading %d characters...", count)
	rows, _ := db.Query(`
		SELECT id, COALESCE(name, ''), COALESCE(title, ''),
		       COALESCE(description, ''), COALESCE(greeting, '')
		FROM ai_characters
		ORDER BY num_interactions DESC NULLS LAST`)

	type char struct {
		id   int
		text string
	}
	var chars []char
	for rows.Next() {
		var id int
		var name, title, desc, greet string
		rows.Scan(&id, &name, &title, &desc, &greet)
		chars = append(chars, char{id, name + " " + title + " " + greet + " " + desc})
	}
	rows.Close()
	log.Printf("Loaded %d characters", len(chars))

	log.Println("Embedding...")
	embedStart := time.Now()

	vectors := make([]simd.Vec512, len(chars))
	scales := make([]float32, len(chars))

	numWorkers := runtime.NumCPU()
	chunkSize := (len(chars) + numWorkers - 1) / numWorkers
	done := make(chan bool, numWorkers)

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(chars) {
			end = len(chars)
		}
		if start >= len(chars) {
			done <- true
			continue
		}
		go func(start, end int) {
			for i := start; i < end; i++ {
				result, _ := model.EmbedInt8(chars[i].text)
				if result != nil {
					copy(vectors[i][:], result.Vector)
					scales[i] = result.Scale
				}
			}
			done <- true
		}(start, end)
	}
	for i := 0; i < numWorkers; i++ {
		<-done
	}

	embedTime := time.Since(embedStart)
	log.Printf("Embedding: %v (%.0f/sec)", embedTime, float64(len(chars))/embedTime.Seconds())

	if gobed.FusedCAGRAAvailable() {
		log.Println("\nTesting GPU Fused CAGRA...")

		cfg := gobed.DefaultFusedCAGRAConfig()
		cfg.VocabSize = 30522
		cfg.MaxVectors = len(vectors)
		cfg.TopK = 10

		engine, err := gobed.NewFusedCAGRAEngine(cfg)
		if err != nil {
			log.Printf("Fused CAGRA error: %v", err)
		} else {
			embedWeights := make([]int8, 30522*512)
			embedScales := make([]float32, 30522)

			table := model.EmbeddingTable()
			scaleTable := model.ScaleTable()
			for i := 0; i < 30522 && i < len(table); i++ {
				copy(embedWeights[i*512:(i+1)*512], table[i])
				embedScales[i] = scaleTable[i]
			}

			log.Println("Building GPU index...")
			buildStart := time.Now()
			err = engine.BuildIndex(embedWeights, embedScales, vectors, scales)
			if err != nil {
				log.Printf("Build error: %v", err)
			} else {
				buildTime := time.Since(buildStart)
				log.Printf("GPU index built: %v (%.0f vec/sec)", buildTime, float64(len(vectors))/buildTime.Seconds())

				queries := []string{"anime girl", "dragon ball", "helpful assistant"}

				for _, q := range queries {
					tokens := model.SimpleTokenize(q)
					tokenU16 := make([]uint16, len(tokens))
					for i, t := range tokens {
						tokenU16[i] = uint16(t)
					}

					start := time.Now()
					results, err := engine.SearchBatch([][]uint16{tokenU16}, 20)
					elapsed := time.Since(start)

					if err != nil {
						log.Printf("Query %q error: %v", q, err)
					} else if len(results) > 0 {
						fmt.Printf("\nQuery: %q (%v)\n", q, elapsed)
						for i, r := range results[0][:min(5, len(results[0]))] {
							if r.ID >= 0 && r.ID < len(chars) {
								fmt.Printf("  %d. [%d] sim=%.1f\n", i+1, r.ID, r.Similarity)
							}
						}
					}
				}

				log.Println("\nGPU search latency benchmark...")
				tokens := model.SimpleTokenize("anime")
				tokenU16 := make([]uint16, len(tokens))
				for i, t := range tokens {
					tokenU16[i] = uint16(t)
				}

				for i := 0; i < 10; i++ {
					engine.SearchBatch([][]uint16{tokenU16}, 20)
				}

				numSearches := 1000
				latencies := make([]time.Duration, numSearches)
				for i := 0; i < numSearches; i++ {
					start := time.Now()
					engine.SearchBatch([][]uint16{tokenU16}, 20)
					latencies[i] = time.Since(start)
				}

				sort.Slice(latencies, func(i, j int) bool {
					return latencies[i] < latencies[j]
				})

				var total time.Duration
				for _, l := range latencies {
					total += l
				}

				fmt.Printf("\nGPU Fused CAGRA (%d vectors):\n", len(vectors))
				fmt.Printf("  Avg: %v\n", total/time.Duration(numSearches))
				fmt.Printf("  P50: %v\n", latencies[numSearches/2])
				fmt.Printf("  P95: %v\n", latencies[numSearches*95/100])
				fmt.Printf("  P99: %v\n", latencies[numSearches*99/100])
				fmt.Printf("  QPS: %.0f\n", float64(numSearches)/total.Seconds())
			}
			engine.Close()
		}
	} else {
		log.Println("Fused CAGRA not available")
	}

	log.Println("\nCPU brute-force comparison...")
	qResult, _ := model.EmbedInt8("anime")
	var qVec simd.Vec512
	copy(qVec[:], qResult.Vector)

	for i := 0; i < 10; i++ {
		simd.Dot512Batch(&qVec, vectors, nil)
	}

	numSearches := 100
	latencies := make([]time.Duration, numSearches)
	for i := 0; i < numSearches; i++ {
		start := time.Now()
		simd.Dot512Batch(&qVec, vectors, nil)
		latencies[i] = time.Since(start)
	}

	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})

	var total time.Duration
	for _, l := range latencies {
		total += l
	}

	fmt.Printf("\nCPU Brute-force (%d vectors):\n", len(vectors))
	fmt.Printf("  Avg: %v\n", total/time.Duration(numSearches))
	fmt.Printf("  P50: %v\n", latencies[numSearches/2])
	fmt.Printf("  P99: %v\n", latencies[numSearches*99/100])
	fmt.Printf("  QPS: %.0f\n", float64(numSearches)/total.Seconds())
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
