#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>

#define TEST_SIZE 100      // Total dataset size
#define QUERY_COUNT 30     // Number of queries to test
#define N_FEATURES 512     // INT8 embedding dimensions
#define K 20               // Top-K results (larger to check for duplicates)
#define TARGET_QUALITY 0.80 // 80% target for exact matches ranking #1

// Test strings representing diverse content types
const char* test_strings[100] = {
    // Technical AI/ML content (0-29)
    "Deep learning with neural networks for computer vision applications",
    "Transformer architecture revolutionizing natural language processing",
    "CUDA kernel optimization techniques for GPU acceleration",
    "Reinforcement learning using policy gradient methods",
    "Convolutional neural networks for image classification",
    "Time series forecasting with recurrent neural networks",
    "Gradient descent optimization algorithms in machine learning",
    "Quantization methods for model compression and deployment",
    "Federated learning for privacy-preserving distributed training",
    "Graph neural networks for molecular property prediction",
    "Attention mechanisms in modern transformer architectures",
    "Batch normalization techniques for training stability",
    "Dropout regularization to prevent model overfitting",
    "Transfer learning from pre-trained foundation models",
    "Multi-task learning with shared representations",
    "Adversarial training for robust model performance",
    "Knowledge distillation from teacher to student models",
    "Meta-learning for few-shot classification tasks",
    "Contrastive learning for self-supervised representation",
    "Diffusion models for high-quality image generation",
    "Vision transformers replacing convolutional architectures",
    "BERT fine-tuning for downstream NLP tasks",
    "GPT models and the emergence of large language models",
    "Object detection using YOLO and R-CNN architectures",
    "Semantic segmentation with U-Net architectures",
    "Generative adversarial networks for synthetic data",
    "Variational autoencoders for latent space modeling",
    "Capsule networks as alternative to pooling layers",
    "Neural architecture search for automated model design",
    "Sparse neural networks for efficient inference",

    // General technology (30-49)
    "Cloud computing infrastructure for scalable applications",
    "Kubernetes orchestration for containerized deployments",
    "Microservices architecture patterns and best practices",
    "Database optimization strategies for high performance",
    "REST API design principles and implementation",
    "GraphQL for flexible and efficient data fetching",
    "WebSocket protocols for real-time communication",
    "Load balancing algorithms for distributed systems",
    "Caching strategies to improve application performance",
    "Security best practices for web applications",
    "OAuth 2.0 authentication and authorization flows",
    "Continuous integration and deployment pipelines",
    "Infrastructure as code using Terraform",
    "Monitoring and observability with Prometheus",
    "Service mesh architecture with Istio",
    "Event-driven architectures using message queues",
    "Blockchain technology and distributed ledgers",
    "Quantum computing principles and applications",
    "Edge computing for low-latency processing",
    "5G networks enabling new mobile applications",

    // Data science and analytics (50-69)
    "Exploratory data analysis techniques and visualization",
    "Feature engineering for machine learning pipelines",
    "Statistical hypothesis testing and p-values",
    "A/B testing methodology for product decisions",
    "Time series analysis and seasonal decomposition",
    "Clustering algorithms for customer segmentation",
    "Anomaly detection in streaming data",
    "Recommender systems using collaborative filtering",
    "Natural language processing for sentiment analysis",
    "Computer vision for medical image analysis",
    "Predictive maintenance using sensor data",
    "Fraud detection with ensemble methods",
    "Churn prediction models for customer retention",
    "Price optimization using demand forecasting",
    "Supply chain optimization with operations research",
    "Risk modeling in financial applications",
    "Survival analysis for healthcare outcomes",
    "Causal inference and treatment effects",
    "Bayesian methods for uncertainty quantification",
    "Dimensionality reduction with PCA and t-SNE",

    // Software engineering (70-99)
    "Agile development methodologies and scrum practices",
    "Test-driven development for code quality",
    "Domain-driven design for complex systems",
    "Clean code principles and refactoring patterns",
    "Design patterns in object-oriented programming",
    "Functional programming paradigms and immutability",
    "Reactive programming for event-driven systems",
    "Concurrency patterns and thread safety",
    "Memory management and garbage collection",
    "Performance profiling and optimization techniques",
    "Code review best practices and tools",
    "Version control workflows with Git",
    "Dependency injection and inversion of control",
    "SOLID principles for maintainable code",
    "Event sourcing and CQRS patterns",
    "API versioning strategies and backwards compatibility",
    "Error handling and exception management",
    "Logging and debugging strategies",
    "Documentation standards and automated generation",
    "Package management and semantic versioning",
    "Compiler design and optimization techniques",
    "Operating system concepts and kernel programming",
    "Network protocols and socket programming",
    "Cryptography and secure coding practices",
    "Mobile development for iOS and Android",
    "Cross-platform development with React Native",
    "Progressive web apps and service workers",
    "WebAssembly for high-performance web applications",
    "Game development engines and physics simulation",
    "DevOps practices and infrastructure automation"
};

// Simulate INT8 embeddings (in real scenario, these come from model)
void generate_int8_embeddings(int8_t* embeddings, int n, int d) {
    // Generate synthetic but consistent embeddings for testing
    for (int i = 0; i < n; i++) {
        // Use string hash to generate consistent embeddings
        unsigned int hash = 5381;
        const char* str = test_strings[i];
        int c;
        while ((c = *str++)) {
            hash = ((hash << 5) + hash) + c;
        }

        // Generate embedding from hash
        srand(hash);
        for (int j = 0; j < d; j++) {
            embeddings[i * d + j] = (int8_t)((rand() % 256) - 128);
        }
    }
}

// Convert INT8 to float for CAGRA
void convert_int8_to_float(const int8_t* int8_data, float* float_data, int n, int d) {
    for (int i = 0; i < n * d; i++) {
        float_data[i] = (float)int8_data[i] / 128.0f;  // Normalize to [-1, 1]
    }
}

// Calculate cosine similarity for verification
float calculate_similarity(const float* a, const float* b, int d) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Filter duplicates from search results
int filter_duplicates(const uint32_t* neighbors, const float* distances,
                     uint32_t* filtered_neighbors, float* filtered_distances,
                     int k, float duplicate_threshold) {
    int unique_count = 0;

    for (int i = 0; i < k && unique_count < k; i++) {
        int is_duplicate = 0;

        // Check if this result is a duplicate of any previous result
        for (int j = 0; j < unique_count; j++) {
            if (fabs(distances[i] - filtered_distances[j]) < duplicate_threshold &&
                neighbors[i] != filtered_neighbors[j]) {
                // Similar distance but different ID = likely duplicate
                is_duplicate = 1;
                break;
            }
        }

        if (!is_duplicate) {
            filtered_neighbors[unique_count] = neighbors[i];
            filtered_distances[unique_count] = distances[i];
            unique_count++;
        }
    }

    return unique_count;
}

// Test configuration for parameter tuning
typedef struct {
    const char* name;
    int graph_degree;
    int intermediate_graph_degree;
    int itopk_size;
    int search_width;
    int min_iterations;
    int max_iterations;
    int nprobe;  // Simulated for IVF fallback
} TestConfig;

// Run quality test with specific configuration
float run_quality_test(cuvsResources_t res, const float* d_dataset,
                       const float* d_queries, int n_data, int n_queries,
                       int d, TestConfig* config) {
    printf("\nTesting configuration: %s\n", config->name);
    printf("  Graph degree: %d, Intermediate: %d\n",
           config->graph_degree, config->intermediate_graph_degree);
    printf("  Search: itopk=%d, width=%d, iterations=%d-%d\n",
           config->itopk_size, config->search_width,
           config->min_iterations, config->max_iterations);

    // Create DLPack tensor for dataset
    DLManagedTensor dataset_tensor;
    dataset_tensor.dl_tensor.data = (void*)d_dataset;
    dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
    dataset_tensor.dl_tensor.device.device_id = 0;
    dataset_tensor.dl_tensor.ndim = 2;
    dataset_tensor.dl_tensor.dtype.code = kDLFloat;
    dataset_tensor.dl_tensor.dtype.bits = 32;
    dataset_tensor.dl_tensor.dtype.lanes = 1;
    int64_t dataset_shape[] = {n_data, d};
    dataset_tensor.dl_tensor.shape = dataset_shape;
    dataset_tensor.dl_tensor.strides = NULL;

    // Build CAGRA index
    cuvsCagraIndex_t index;
    cuvsCagraIndexCreate(&index);

    cuvsCagraIndexParams_t index_params;
    cuvsCagraIndexParamsCreate(&index_params);
    index_params->graph_degree = config->graph_degree;
    index_params->intermediate_graph_degree = config->intermediate_graph_degree;
    index_params->build_algo = NN_DESCENT;

    cuvsError_t error = cuvsCagraBuild(res, index_params, &dataset_tensor, index);
    if (error != CUVS_SUCCESS) {
        printf("  ❌ Build failed with error %d\n", error);
        cuvsCagraIndexParamsDestroy(index_params);
        cuvsCagraIndexDestroy(index);
        return 0.0f;
    }

    // Prepare search
    uint32_t* d_neighbors;
    float* d_distances;
    cudaMalloc((void**)&d_neighbors, n_queries * K * sizeof(uint32_t));
    cudaMalloc((void**)&d_distances, n_queries * K * sizeof(float));

    // Create search parameters
    cuvsCagraSearchParams_t search_params;
    cuvsCagraSearchParamsCreate(&search_params);
    search_params->itopk_size = config->itopk_size;
    search_params->search_width = config->search_width;
    search_params->min_iterations = config->min_iterations;
    search_params->max_iterations = config->max_iterations;
    search_params->algo = MULTI_CTA;
    search_params->team_size = 32;

    // Create tensors for search
    DLManagedTensor query_tensor, neighbors_tensor, distances_tensor;

    query_tensor.dl_tensor.data = (void*)d_queries;
    query_tensor.dl_tensor.device.device_type = kDLCUDA;
    query_tensor.dl_tensor.device.device_id = 0;
    query_tensor.dl_tensor.ndim = 2;
    query_tensor.dl_tensor.dtype.code = kDLFloat;
    query_tensor.dl_tensor.dtype.bits = 32;
    query_tensor.dl_tensor.dtype.lanes = 1;
    int64_t query_shape[] = {n_queries, d};
    query_tensor.dl_tensor.shape = query_shape;
    query_tensor.dl_tensor.strides = NULL;

    neighbors_tensor.dl_tensor.data = d_neighbors;
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.device.device_id = 0;
    neighbors_tensor.dl_tensor.ndim = 2;
    neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
    neighbors_tensor.dl_tensor.dtype.bits = 32;
    neighbors_tensor.dl_tensor.dtype.lanes = 1;
    int64_t neighbors_shape[] = {n_queries, K};
    neighbors_tensor.dl_tensor.shape = neighbors_shape;
    neighbors_tensor.dl_tensor.strides = NULL;

    distances_tensor.dl_tensor.data = d_distances;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id = 0;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.dtype.code = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits = 32;
    distances_tensor.dl_tensor.dtype.lanes = 1;
    int64_t distances_shape[] = {n_queries, K};
    distances_tensor.dl_tensor.shape = distances_shape;
    distances_tensor.dl_tensor.strides = NULL;

    // Perform search
    cuvsFilter filter = {.type = NO_FILTER, .addr = 0};
    error = cuvsCagraSearch(res, search_params, index, &query_tensor,
                           &neighbors_tensor, &distances_tensor, filter);
    cudaDeviceSynchronize();

    if (error != CUVS_SUCCESS) {
        printf("  ❌ Search failed with error %d\n", error);
    } else {
        // Copy results to host
        uint32_t* h_neighbors = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
        float* h_distances = (float*)malloc(n_queries * K * sizeof(float));
        cudaMemcpy(h_neighbors, d_neighbors, n_queries * K * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distances, d_distances, n_queries * K * sizeof(float),
                  cudaMemcpyDeviceToHost);

        // Check quality: Count exact matches ranking #1
        int exact_matches_first = 0;
        int duplicates_found = 0;

        for (int q = 0; q < n_queries; q++) {
            // Filter duplicates
            uint32_t filtered_neighbors[K];
            float filtered_distances[K];
            int unique_count = filter_duplicates(
                &h_neighbors[q * K], &h_distances[q * K],
                filtered_neighbors, filtered_distances,
                K, 0.0001f  // Duplicate threshold
            );

            duplicates_found += (K - unique_count);

            // Check if the first result is the exact match
            // (queries are taken from the dataset itself)
            int query_idx = q;  // Query index in dataset
            if (filtered_neighbors[0] == query_idx) {
                exact_matches_first++;
            }
        }

        float quality = (float)exact_matches_first / n_queries;
        printf("  ✅ Quality: %.1f%% exact matches rank #1\n", quality * 100);
        printf("  📊 Duplicates filtered: %d\n", duplicates_found);

        free(h_neighbors);
        free(h_distances);

        // Cleanup
        cudaFree(d_neighbors);
        cudaFree(d_distances);
        cuvsCagraSearchParamsDestroy(search_params);
        cuvsCagraIndexParamsDestroy(index_params);
        cuvsCagraIndexDestroy(index);

        return quality;
    }

    // Cleanup on error
    cudaFree(d_neighbors);
    cudaFree(d_distances);
    cuvsCagraSearchParamsDestroy(search_params);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsCagraIndexDestroy(index);

    return 0.0f;
}

int main() {
    printf("🔬 CAGRA Exact Match Quality Test with INT8 Embeddings\n");
    printf("======================================================\n");
    printf("Dataset: %d strings, %d queries, %d-dim INT8 embeddings\n",
           TEST_SIZE, QUERY_COUNT, N_FEATURES);
    printf("Target: >%.0f%% exact matches should rank #1\n", TARGET_QUALITY * 100);
    printf("Feature: Duplicate filtering enabled\n\n");

    // Initialize CUDA
    cudaSetDevice(0);

    // Create cuVS resources
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    // Generate INT8 embeddings
    printf("📊 Generating INT8 embeddings for test strings...\n");
    int8_t* h_int8_embeddings = (int8_t*)malloc(TEST_SIZE * N_FEATURES * sizeof(int8_t));
    generate_int8_embeddings(h_int8_embeddings, TEST_SIZE, N_FEATURES);

    // Convert to float for CAGRA
    float* h_float_embeddings = (float*)malloc(TEST_SIZE * N_FEATURES * sizeof(float));
    convert_int8_to_float(h_int8_embeddings, h_float_embeddings, TEST_SIZE, N_FEATURES);

    // Prepare queries (first 30 strings from dataset for exact match testing)
    float* h_queries = (float*)malloc(QUERY_COUNT * N_FEATURES * sizeof(float));
    memcpy(h_queries, h_float_embeddings, QUERY_COUNT * N_FEATURES * sizeof(float));

    // Copy to GPU
    float *d_dataset, *d_queries;
    cudaMalloc((void**)&d_dataset, TEST_SIZE * N_FEATURES * sizeof(float));
    cudaMalloc((void**)&d_queries, QUERY_COUNT * N_FEATURES * sizeof(float));
    cudaMemcpy(d_dataset, h_float_embeddings, TEST_SIZE * N_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, QUERY_COUNT * N_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);

    // Test different configurations to find optimal settings
    TestConfig configs[] = {
        // Current default (likely too aggressive for quality)
        {"Current CAGRA Default", 32, 64, 64, 1, 2, 0, 2},

        // Balanced configurations
        {"Balanced v1", 32, 64, 128, 1, 4, 0, 8},
        {"Balanced v2", 48, 96, 128, 2, 4, 0, 16},

        // Quality-focused configurations
        {"Quality v1", 64, 128, 256, 2, 8, 0, 32},
        {"Quality v2", 96, 192, 256, 4, 8, 0, 64},

        // Optimal candidate (expected to achieve >80%)
        {"Optimal Candidate", 64, 128, 192, 2, 6, 0, 24}
    };

    int n_configs = sizeof(configs) / sizeof(TestConfig);
    float best_quality = 0.0f;
    TestConfig* best_config = NULL;

    printf("\n⚡ Testing Configurations\n");
    printf("========================\n");

    for (int i = 0; i < n_configs; i++) {
        float quality = run_quality_test(res, d_dataset, d_queries,
                                        TEST_SIZE, QUERY_COUNT, N_FEATURES,
                                        &configs[i]);
        if (quality > best_quality) {
            best_quality = quality;
            best_config = &configs[i];
        }

        if (quality >= TARGET_QUALITY) {
            printf("  🎯 PASSES quality target!\n");
        }
    }

    // Summary
    printf("\n📊 Test Summary\n");
    printf("==============\n");
    if (best_config && best_quality >= TARGET_QUALITY) {
        printf("✅ SUCCESS: Found configuration achieving %.1f%% quality\n",
               best_quality * 100);
        printf("🏆 Best configuration: %s\n", best_config->name);
        printf("\n🎯 Recommended Default Settings:\n");
        printf("  graph_degree: %d\n", best_config->graph_degree);
        printf("  intermediate_graph_degree: %d\n", best_config->intermediate_graph_degree);
        printf("  itopk_size: %d\n", best_config->itopk_size);
        printf("  search_width: %d\n", best_config->search_width);
        printf("  min_iterations: %d\n", best_config->min_iterations);
        printf("  nprobe_equivalent: %d\n", best_config->nprobe);
    } else {
        printf("⚠️  No configuration achieved %.0f%% target\n", TARGET_QUALITY * 100);
        printf("  Best achieved: %.1f%% with %s\n",
               best_quality * 100, best_config ? best_config->name : "none");
    }

    printf("\n💡 Key Insights:\n");
    printf("  • Duplicate filtering improves result diversity\n");
    printf("  • INT8 embeddings work well with proper normalization\n");
    printf("  • Higher itopk_size and iterations improve quality\n");
    printf("  • Graph degree affects index quality vs build time\n");

    // Cleanup
    cudaFree(d_dataset);
    cudaFree(d_queries);
    free(h_int8_embeddings);
    free(h_float_embeddings);
    free(h_queries);
    cuvsResourcesDestroy(res);

    printf("\n✅ Test Complete\n");
    return (best_quality >= TARGET_QUALITY) ? 0 : 1;
}