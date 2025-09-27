#!/bin/bash

# Generate test data for gobed testing
# Creates ai.txt with 240k lines of AI-related content

set -e

echo " Generating test data (ai.txt with 240k lines)..."

# Output file
OUTPUT="ai.txt"

# Topics for generating diverse content
TOPICS=(
    "machine learning algorithms and optimization techniques"
    "deep neural networks and backpropagation methods"
    "natural language processing and transformer models"
    "computer vision and convolutional neural networks"
    "reinforcement learning and policy gradient methods"
    "generative adversarial networks and VAE architectures"
    "transfer learning and fine-tuning strategies"
    "federated learning and privacy-preserving ML"
    "graph neural networks and geometric deep learning"
    "attention mechanisms and self-attention layers"
    "BERT and GPT model architectures"
    "object detection and image segmentation"
    "time series forecasting with RNNs and LSTMs"
    "clustering algorithms and dimensionality reduction"
    "gradient boosting and ensemble methods"
    "support vector machines and kernel methods"
    "Bayesian optimization and hyperparameter tuning"
    "AutoML and neural architecture search"
    "quantum machine learning algorithms"
    "edge AI and model compression techniques"
)

# Technical terms to mix in
TERMS=(
    "tensor" "embedding" "gradient" "optimizer" "loss function"
    "activation" "dropout" "normalization" "regularization" "cross-validation"
    "precision" "recall" "F1-score" "AUC-ROC" "confusion matrix"
    "epoch" "batch size" "learning rate" "momentum" "Adam optimizer"
    "convolution" "pooling" "stride" "padding" "receptive field"
    "attention weights" "query key value" "multi-head attention" "positional encoding"
    "tokenization" "word2vec" "GloVe" "fastText" "sentence embeddings"
    "GPU acceleration" "CUDA kernels" "tensor cores" "mixed precision"
    "distributed training" "data parallelism" "model parallelism" "pipeline parallelism"
    "knowledge distillation" "pruning" "quantization" "neural compression"
)

# Generate content
echo "Generating 240,000 lines of AI-related content..."

# Clear existing file
> "$OUTPUT"

# Progress tracking
TOTAL_LINES=240000
BATCH_SIZE=1000
BATCHES=$((TOTAL_LINES / BATCH_SIZE))

for batch in $(seq 1 $BATCHES); do
    {
        for i in $(seq 1 $BATCH_SIZE); do
            # Calculate global line number
            LINE_NUM=$(((batch - 1) * BATCH_SIZE + i))

            # Pick random topic and terms
            TOPIC=${TOPICS[$((RANDOM % ${#TOPICS[@]}))]}
            TERM1=${TERMS[$((RANDOM % ${#TERMS[@]}))]}
            TERM2=${TERMS[$((RANDOM % ${#TERMS[@]}))]}
            TERM3=${TERMS[$((RANDOM % ${#TERMS[@]}))]}

            # Generate variations of content
            case $((LINE_NUM % 10)) in
                0)
                    echo "Research paper $LINE_NUM: Exploring $TOPIC using $TERM1 and $TERM2 for improved $TERM3 performance in production systems"
                    ;;
                1)
                    echo "Tutorial $LINE_NUM: A comprehensive guide to $TOPIC with practical examples of $TERM1, $TERM2, and $TERM3 implementation"
                    ;;
                2)
                    echo "Blog post $LINE_NUM: Why $TOPIC matters - understanding $TERM1 optimization and $TERM2 techniques for better $TERM3"
                    ;;
                3)
                    echo "Documentation $LINE_NUM: API reference for $TOPIC framework supporting $TERM1 operations with $TERM2 and $TERM3 features"
                    ;;
                4)
                    echo "Case study $LINE_NUM: How we scaled $TOPIC to handle millions of requests using $TERM1 and $TERM2 with optimized $TERM3"
                    ;;
                5)
                    echo "Technical report $LINE_NUM: Benchmarking $TOPIC performance - comparing $TERM1 vs $TERM2 approaches for $TERM3 workloads"
                    ;;
                6)
                    echo "Conference paper $LINE_NUM: Novel approaches to $TOPIC leveraging $TERM1 architectures and $TERM2 for enhanced $TERM3"
                    ;;
                7)
                    echo "Implementation note $LINE_NUM: Best practices for $TOPIC deployment including $TERM1 configuration and $TERM2 tuning for optimal $TERM3"
                    ;;
                8)
                    echo "Review article $LINE_NUM: State-of-the-art in $TOPIC - from classical $TERM1 to modern $TERM2 and emerging $TERM3 techniques"
                    ;;
                9)
                    echo "Experiment log $LINE_NUM: Testing $TOPIC hypothesis with $TERM1 baseline, $TERM2 variations, and $TERM3 ablation studies"
                    ;;
            esac
        done
    } >> "$OUTPUT"

    # Progress indicator
    if [ $((batch % 10)) -eq 0 ]; then
        PROGRESS=$((batch * 100 / BATCHES))
        echo -ne "\rProgress: ${PROGRESS}% (${batch}/${BATCHES} batches)"
    fi
done

echo -e "\n Generated $(wc -l < "$OUTPUT") lines in $OUTPUT"

# Generate smaller test files
echo " Creating smaller test datasets..."

# Small test file (100 lines)
head -100 "$OUTPUT" > testdata/small_test.txt
echo " Created testdata/small_test.txt (100 lines)"

# Medium test file (10k lines)
head -10000 "$OUTPUT" > testdata/medium_test.txt
echo " Created testdata/medium_test.txt (10,000 lines)"

# Large test file (50k lines)
head -50000 "$OUTPUT" > testdata/large_test.txt
echo " Created testdata/large_test.txt (50,000 lines)"

echo ""
echo " Test data summary:"
ls -lh "$OUTPUT" testdata/*.txt 2>/dev/null | awk '{print "  " $9 ": " $5}'