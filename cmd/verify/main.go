//go:build legacy

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/lee101/gobed"
)

func readTexts(inputPath string) ([]string, error) {
	if inputPath == "" {
		// Default small set if no input provided
		return []string{
			"Machine learning is fascinating.",
			"Deep learning models are powerful.",
			"The weather is nice today.",
		}, nil
	}
	data, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read input file: %v", err)
	}
	var texts []string
	if err := json.Unmarshal(data, &texts); err != nil {
		return nil, fmt.Errorf("failed to parse input JSON: %v", err)
	}
	return texts, nil
}

func writeEmbeddings(outputPath string, embeddings [][]float32) error {
	data, err := json.Marshal(embeddings)
	if err != nil {
		return fmt.Errorf("failed to marshal embeddings: %v", err)
	}
	if outputPath == "" {
		// Write to stdout
		_, err = os.Stdout.Write(append(data, '\n'))
		return err
	}
	return ioutil.WriteFile(outputPath, data, 0o644)
}

func main() {
	inputPath := flag.String("input", "", "Path to JSON file containing an array of texts")
	outputPath := flag.String("output", "", "Path to write embeddings JSON (defaults to stdout)")
	flag.Parse()

	model, err := gobed.LoadModel()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	texts, err := readTexts(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading texts: %v\n", err)
		os.Exit(1)
	}

	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := model.Encode(text)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to encode '%s': %v\n", text, err)
			embeddings[i] = []float32{} // keep alignment
			continue
		}
		embeddings[i] = emb
	}

	if err := writeEmbeddings(*outputPath, embeddings); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing embeddings: %v\n", err)
		os.Exit(1)
	}
}
