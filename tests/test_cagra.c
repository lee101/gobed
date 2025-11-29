// test_cagra.c - Direct C-level test for CAGRA
// Compile: gcc -o test_cagra test_cagra.c -L. -lfused_cagra -L/usr/local/cuda/lib64 -lcudart -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define NUM_VECTORS 100
#define NUM_QUERIES 30
#define EMBED_DIM 512
#define TOP_K 10
#define VOCAB_SIZE 50000

// External CAGRA functions
extern void* create_fused_context(
    int8_t* embed_weights,
    float* embed_scales_raw,
    int vocab_size,
    int embed_dim,
    int8_t* database,
    float* db_scales_raw,
    int num_vectors,
    int top_k);

extern void build_cagra_graph(void* context, int degree);

extern void fused_search(
    void* context,
    uint16_t* token_batch,
    int* token_lengths,
    int batch_size,
    int max_tokens,
    float* output_distances,
    int* output_indices);

extern void destroy_fused_context(void* context);

// Simple hash function for tokenization
uint16_t hash_string(const char* str) {
    uint16_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + *str++;
    }
    return hash % VOCAB_SIZE;
}

// Generate realistic int8 embedding from text
void generate_embedding(const char* text, int8_t* embedding, float* scale) {
    // Simple but deterministic embedding generation
    uint32_t seed = hash_string(text);
    srand(seed);

    // Generate embedding with pattern based on text
    float max_val = 0.0f;
    float temp_embed[EMBED_DIM];

    for (int i = 0; i < EMBED_DIM; i++) {
        // Mix of sin waves and random for realistic pattern
        float angle = (float)i * 0.1f + seed * 0.01f;
        temp_embed[i] = sinf(angle) * 50.0f + (rand() % 100 - 50) * 0.5f;
        if (fabsf(temp_embed[i]) > max_val) {
            max_val = fabsf(temp_embed[i]);
        }
    }

    // Quantize to int8
    *scale = max_val / 127.0f;
    for (int i = 0; i < EMBED_DIM; i++) {
        embedding[i] = (int8_t)(temp_embed[i] / (*scale));
    }
}

// Test strings from ai.txt style content
const char* test_strings[NUM_VECTORS] = {
    "time series forecasting with RNNs and LSTMs",
    "BERT and GPT model architectures",
    "reinforcement learning and policy gradient",
    "natural language processing and transformer models",
    "graph neural networks and geometric deep learning",
    "quantum machine learning algorithms",
    "federated learning and privacy-preserving ML",
    "transfer learning and fine-tuning strategies",
    "computer vision and convolutional neural networks",
    "edge AI and model compression techniques",
    "attention mechanisms and self-attention layers",
    "AutoML and neural architecture search",
    "clustering algorithms and dimensionality reduction",
    "support vector machines and kernel methods",
    "deep neural networks and backpropagation",
    "object detection and image segmentation",
    "generative adversarial networks and VAE architectures",
    "Bayesian optimization and hyperparameter tuning",
    "knowledge distillation and model pruning",
    "CUDA kernels and GPU acceleration",
    "cross-validation and regularization techniques",
    "confusion matrix and evaluation metrics",
    "Adam optimizer and learning rate scheduling",
    "batch normalization and dropout",
    "word embeddings and Word2Vec",
    "sentiment analysis and text classification",
    "sequence-to-sequence models and attention",
    "recurrent neural networks and LSTM cells",
    "gradient descent and optimization algorithms",
    "feature engineering and data preprocessing",
    // Add more diverse strings
    "distributed training and model parallelism",
    "tensor operations and automatic differentiation",
    "neural compression and quantization",
    "embedding layers and positional encoding",
    "multi-head attention and transformer blocks",
    "loss functions and gradient computation",
    "data parallelism and pipeline parallelism",
    "mixed precision training and tensor cores",
    "knowledge graphs and graph embeddings",
    "contrastive learning and self-supervised methods",
    "meta-learning and few-shot learning",
    "neural ODEs and continuous models",
    "graph attention networks and message passing",
    "capsule networks and dynamic routing",
    "evolutionary algorithms and neuroevolution",
    "spiking neural networks and neuromorphic computing",
    "adversarial examples and robustness",
    "explainable AI and interpretability methods",
    "active learning and data selection",
    "online learning and streaming algorithms",
    // Fill rest with variations
    "ensemble methods and boosting algorithms",
    "random forests and decision trees",
    "k-nearest neighbors and distance metrics",
    "principal component analysis and SVD",
    "t-SNE and UMAP visualization",
    "autoencoders and representation learning",
    "variational inference and probabilistic models",
    "Markov chains and hidden Markov models",
    "reinforcement learning with Q-learning",
    "policy networks and actor-critic methods",
    "curriculum learning and progressive training",
    "zero-shot learning and task adaptation",
    "neural style transfer and artistic AI",
    "speech recognition and audio processing",
    "time series analysis and forecasting",
    "anomaly detection and outlier analysis",
    "recommender systems and collaborative filtering",
    "natural language generation and GPT models",
    "machine translation and multilingual models",
    "question answering and reading comprehension",
    "named entity recognition and POS tagging",
    "dependency parsing and syntactic analysis",
    "semantic segmentation and instance segmentation",
    "3D vision and point cloud processing",
    "video understanding and action recognition",
    "medical image analysis and healthcare AI",
    "robotics and embodied AI",
    "simulation environments and synthetic data",
    "causal inference and counterfactual reasoning",
    "fairness in ML and bias mitigation",
    "differential privacy and secure computation",
    "federated analytics and edge computing",
    "model serving and inference optimization",
    "MLOps and continuous integration",
    "experiment tracking and model versioning",
    "data versioning and reproducibility",
    "hyperparameter optimization and AutoML",
    "neural architecture search strategies",
    "weight pruning and structured sparsity",
    "knowledge distillation techniques",
    "quantization-aware training methods",
    "low-rank approximation and compression",
    "gradient checkpointing and memory optimization",
    "distributed optimizer and gradient aggregation",
    "asynchronous training and parameter servers",
    "elastic training and fault tolerance",
    "profiling tools and performance analysis",
    "CUDA programming and kernel optimization",
    "tensor compiler and graph optimization",
    "hardware acceleration and custom ASICs"
};

int main() {
    printf("=== C-Level CAGRA Test ===\n");
    printf("Testing with %d vectors, %d queries\n\n", NUM_VECTORS, NUM_QUERIES);

    // Allocate memory
    int8_t* embed_weights = (int8_t*)calloc(VOCAB_SIZE * EMBED_DIM, sizeof(int8_t));
    float* embed_scales = (float*)calloc(VOCAB_SIZE, sizeof(float));
    int8_t* database = (int8_t*)calloc(NUM_VECTORS * EMBED_DIM, sizeof(int8_t));
    float* db_scales = (float*)calloc(NUM_VECTORS, sizeof(float));

    // Generate embeddings for vocabulary (simplified - just for used tokens)
    printf("Generating vocabulary embeddings...\n");
    for (int i = 0; i < NUM_VECTORS; i++) {
        uint16_t token = hash_string(test_strings[i]);
        generate_embedding(test_strings[i],
                         embed_weights + token * EMBED_DIM,
                         &embed_scales[token]);
    }

    // Generate database vectors
    printf("Generating database vectors...\n");
    for (int i = 0; i < NUM_VECTORS; i++) {
        generate_embedding(test_strings[i],
                         database + i * EMBED_DIM,
                         &db_scales[i]);
        if (i < 5) {
            printf("  Vector %d: '%s' (scale=%.4f)\n",
                   i, test_strings[i], db_scales[i]);
        }
    }

    // Create CAGRA context
    printf("\nCreating CAGRA context...\n");
    void* context = create_fused_context(
        embed_weights, embed_scales,
        VOCAB_SIZE, EMBED_DIM,
        database, db_scales,
        NUM_VECTORS, TOP_K
    );

    if (!context) {
        printf("ERROR: Failed to create CAGRA context\n");
        return 1;
    }

    // Build CAGRA graph
    printf("Building CAGRA graph...\n");
    clock_t build_start = clock();
    build_cagra_graph(context, 32);
    double build_time = (double)(clock() - build_start) / CLOCKS_PER_SEC;
    printf("Graph built in %.3f seconds\n", build_time);

    // Test queries
    printf("\n=== Testing %d Queries ===\n", NUM_QUERIES);

    int exact_matches = 0;
    int top3_matches = 0;
    double total_search_time = 0.0;

    for (int q = 0; q < NUM_QUERIES; q++) {
        // Use first 30 strings as queries
        const char* query_str = test_strings[q];
        uint16_t tokens[20] = {0};
        int token_lengths[1] = {1};

        // Simple tokenization
        tokens[0] = hash_string(query_str);

        // Prepare output
        float distances[TOP_K];
        int indices[TOP_K];

        // Search
        clock_t search_start = clock();
        fused_search(context, tokens, token_lengths, 1, 20, distances, indices);
        double search_time = (double)(clock() - search_start) / CLOCKS_PER_SEC;
        total_search_time += search_time;

        // Check results
        printf("\nQuery %d: '%s'\n", q, query_str);
        printf("  Search time: %.6f seconds\n", search_time);
        printf("  Top results:\n");

        int found_exact = 0;
        int found_in_top3 = 0;

        for (int i = 0; i < TOP_K && i < 5; i++) {
            if (indices[i] >= 0 && indices[i] < NUM_VECTORS) {
                printf("    %d. [%d] '%s' (dist=%.4f)\n",
                       i+1, indices[i], test_strings[indices[i]], distances[i]);

                if (indices[i] == q) {
                    if (i == 0) found_exact = 1;
                    if (i < 3) found_in_top3 = 1;
                }
            } else {
                printf("    %d. [%d] INVALID INDEX\n", i+1, indices[i]);
            }
        }

        // Check for duplicates
        int has_duplicates = 0;
        for (int i = 0; i < TOP_K - 1; i++) {
            for (int j = i + 1; j < TOP_K; j++) {
                if (indices[i] == indices[j]) {
                    has_duplicates = 1;
                    break;
                }
            }
        }

        if (has_duplicates) {
            printf("  WARNING: Duplicate indices detected!\n");
        }

        exact_matches += found_exact;
        top3_matches += found_in_top3;
    }

    // Print summary
    printf("\n=== RESULTS SUMMARY ===\n");
    printf("Total queries: %d\n", NUM_QUERIES);
    printf("Exact matches (rank #1): %d (%.1f%%)\n",
           exact_matches, 100.0 * exact_matches / NUM_QUERIES);
    printf("Top-3 matches: %d (%.1f%%)\n",
           top3_matches, 100.0 * top3_matches / NUM_QUERIES);
    printf("Average search time: %.6f seconds\n", total_search_time / NUM_QUERIES);
    printf("Total search time: %.3f seconds\n", total_search_time);

    if (exact_matches >= (NUM_QUERIES * 0.8)) {
        printf("\n✅ PASSED: Achieved 80%% exact match target\n");
    } else {
        printf("\n❌ FAILED: Below 80%% exact match target\n");
    }

    if (total_search_time < 1.0) {
        printf("✅ PASSED: Sub-second total search time\n");
    } else {
        printf("❌ FAILED: Search too slow (%.1fs > 1s)\n", total_search_time);
    }

    // Cleanup
    destroy_fused_context(context);
    free(embed_weights);
    free(embed_scales);
    free(database);
    free(db_scales);

    return 0;
}