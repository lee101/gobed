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

type Character struct {
	ID          int
	Name        string
	Title       string
	Description string
	Greeting    string
}

func main() {
	log.Println("Real Data Benchmark - 213K Characters")

	db, err := sql.Open("postgres", "postgresql://lee:netwrck2024@localhost/netwrck_ai_characters")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	var count int
	db.QueryRow("SELECT COUNT(*) FROM ai_characters").Scan(&count)
	log.Printf("Total characters in DB: %d", count)

	log.Println("Loading int8 embedding model...")
	model, err := gobed.LoadSimpleInt8Model512()
	if err != nil {
		log.Fatal(err)
	}

	batchSize := 133000
	if count < batchSize {
		batchSize = count
	}

	log.Printf("Loading %d characters...", batchSize)
	rows, err := db.Query(`
		SELECT id, COALESCE(name, ''), COALESCE(title, ''),
		       COALESCE(description, ''), COALESCE(greeting, '')
		FROM ai_characters
		ORDER BY num_interactions DESC NULLS LAST
		LIMIT $1`, batchSize)
	if err != nil {
		log.Fatal(err)
	}

	var chars []Character
	for rows.Next() {
		var c Character
		rows.Scan(&c.ID, &c.Name, &c.Title, &c.Description, &c.Greeting)
		chars = append(chars, c)
	}
	rows.Close()
	log.Printf("Loaded %d characters", len(chars))

	log.Println("Embedding characters...")
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
				c := chars[i]
				text := c.Name + " " + c.Title + " " + c.Greeting + " " + c.Description
				result, _ := model.EmbedInt8(text)
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
	embedRate := float64(len(chars)) / embedTime.Seconds()
	log.Printf("Embedding: %v (%.0f chars/sec)", embedTime, embedRate)

	log.Println("Testing brute-force search...")
	queries := []string{
		"anime girl",
		"dragon ball",
		"helpful assistant",
		"fantasy warrior",
		"cute cat",
	}

	for _, q := range queries {
		qResult, _ := model.EmbedInt8(q)
		if qResult == nil {
			continue
		}
		var qVec simd.Vec512
		copy(qVec[:], qResult.Vector)

		start := time.Now()
		scores := simd.Dot512Batch(&qVec, vectors, nil)

		type scored struct {
			idx   int
			score int32
		}
		results := make([]scored, len(scores))
		for i, s := range scores {
			results[i] = scored{i, s}
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].score > results[j].score
		})
		elapsed := time.Since(start)

		fmt.Printf("\nQuery: %q (%v)\n", q, elapsed)
		for i := 0; i < 5 && i < len(results); i++ {
			c := chars[results[i].idx]
			fmt.Printf("  %d. %s - %s (score: %d)\n", i+1, c.Name, c.Title, results[i].score)
		}
	}

	log.Println("\nSearch latency benchmark...")
	numSearches := 1000
	qResult, _ := model.EmbedInt8("anime")
	var qVec simd.Vec512
	copy(qVec[:], qResult.Vector)

	for i := 0; i < 10; i++ {
		simd.Dot512Batch(&qVec, vectors, nil)
	}

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

	fmt.Printf("\nBrute-force search (%d vectors):\n", len(vectors))
	fmt.Printf("  Avg: %v\n", total/time.Duration(numSearches))
	fmt.Printf("  P50: %v\n", latencies[numSearches/2])
	fmt.Printf("  P95: %v\n", latencies[numSearches*95/100])
	fmt.Printf("  P99: %v\n", latencies[numSearches*99/100])
	fmt.Printf("  QPS: %.0f\n", float64(numSearches)/total.Seconds())

	rawMem := float64(len(vectors)*512) / 1024 / 1024
	fmt.Printf("\nMemory: %.1f MB for %d vectors (%.0f bytes/vec)\n",
		rawMem, len(vectors), float64(len(vectors)*512)/float64(len(vectors)))
}
