#!/usr/bin/env python3
"""
Comprehensive benchmark for GPU-accelerated filesystem search
Tests performance, scalability, and optimization opportunities
"""

import sys
import os
import time
import torch
import numpy as np
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from gpu_filesystem_search import GPUFilesystemSearch, SearchConfig


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    num_chunks: int
    num_queries: int
    index_time: float
    search_time: float
    qps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_mb: float
    config: Dict


class SearchBenchmark:
    """Comprehensive benchmark suite for GPU search"""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    def generate_synthetic_corpus(self, num_files: int, lines_per_file: int = 100) -> Dict[str, str]:
        """Generate synthetic text corpus for benchmarking"""
        corpus = {}

        # Different document types
        templates = [
            "Technical documentation about {topic} with details on {subtopic}.",
            "Implementation of {topic} algorithm using {subtopic} optimization.",
            "Research paper discussing {topic} and its applications in {subtopic}.",
            "Code review for {topic} module focusing on {subtopic} improvements.",
            "Tutorial on {topic} covering basics of {subtopic}.",
        ]

        topics = ["machine learning", "database", "networking", "security", "optimization",
                 "distributed systems", "compiler", "graphics", "cryptography", "algorithms"]

        subtopics = ["performance", "scalability", "reliability", "efficiency", "accuracy",
                    "latency", "throughput", "memory", "concurrency", "parallelism"]

        for i in range(num_files):
            lines = []
            for j in range(lines_per_file):
                template = templates[j % len(templates)]
                topic = topics[(i + j) % len(topics)]
                subtopic = subtopics[(i * j) % len(subtopics)]
                line = template.format(topic=topic, subtopic=subtopic)
                lines.append(f"Line {j}: {line}")

            filename = f"doc_{i:04d}.txt"
            corpus[filename] = "\n".join(lines)

        return corpus

    def benchmark_indexing(self, searcher: GPUFilesystemSearch, corpus_dir: str) -> float:
        """Benchmark indexing performance"""
        start_time = time.time()
        searcher.index_directory(corpus_dir)
        index_time = time.time() - start_time
        return index_time

    def benchmark_search(self, searcher: GPUFilesystemSearch, queries: List[str], k: int = 10) -> Dict:
        """Benchmark search performance"""
        latencies = []

        # Warmup
        for _ in range(10):
            searcher.search(queries[0], k=k)

        # Actual benchmark
        start_time = time.time()
        for query in queries:
            query_start = time.time()
            results = searcher.search(query, k=k)
            query_time = time.time() - query_start
            latencies.append(query_time * 1000)  # Convert to ms

        total_time = time.time() - start_time

        # Calculate statistics
        latencies.sort()
        return {
            'total_time': total_time,
            'qps': len(queries) / total_time,
            'latency_p50': latencies[len(latencies) // 2],
            'latency_p95': latencies[int(len(latencies) * 0.95)],
            'latency_p99': latencies[int(len(latencies) * 0.99)],
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
        }

    def benchmark_configuration(self, config: SearchConfig, corpus_dir: str,
                               queries: List[str], name: str) -> BenchmarkResult:
        """Benchmark a specific configuration"""
        print(f"\n Benchmarking: {name}")
        print(f"   Config: chunks={config.chunk_size}, ivf={config.ivf_clusters}, "
              f"probe={config.probe_lists}, batch={config.batch_size}")

        # Create searcher
        searcher = GPUFilesystemSearch(config)

        # Benchmark indexing
        index_time = self.benchmark_indexing(searcher, corpus_dir)
        print(f"    Indexed {searcher.num_chunks} chunks in {index_time:.2f}s")

        # Benchmark search
        search_stats = self.benchmark_search(searcher, queries)
        print(f"    Search QPS: {search_stats['qps']:.0f}, "
              f"P50: {search_stats['latency_p50']:.2f}ms")

        # Get memory usage
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            memory_mb = searcher.index_size_mb

        result = BenchmarkResult(
            name=name,
            num_chunks=searcher.num_chunks,
            num_queries=len(queries),
            index_time=index_time,
            search_time=search_stats['total_time'],
            qps=search_stats['qps'],
            latency_p50=search_stats['latency_p50'],
            latency_p95=search_stats['latency_p95'],
            latency_p99=search_stats['latency_p99'],
            memory_mb=memory_mb,
            config=config.__dict__
        )

        self.results.append(result)
        return result

    def run_scalability_test(self, corpus_dir: str):
        """Test scalability with different corpus sizes"""
        print("\n🔬 Running Scalability Test")

        corpus_sizes = [100, 500, 1000, 5000, 10000]
        base_config = SearchConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            chunk_size=256,
            ivf_clusters=256,
            probe_lists=32
        )

        scalability_results = []

        for num_files in corpus_sizes:
            # Generate corpus
            print(f"\n  Testing with {num_files} files...")
            corpus = self.generate_synthetic_corpus(num_files, lines_per_file=50)

            # Write to temp directory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                for filename, content in corpus.items():
                    with open(os.path.join(temp_dir, filename), 'w') as f:
                        f.write(content)

                # Generate queries
                queries = [f"query about topic {i}" for i in range(100)]

                # Benchmark
                result = self.benchmark_configuration(
                    base_config, temp_dir, queries,
                    f"Scalability-{num_files}files"
                )
                scalability_results.append((num_files, result))

        return scalability_results

    def run_parameter_sweep(self, corpus_dir: str, queries: List[str]):
        """Sweep through different parameter configurations"""
        print("\n🔬 Running Parameter Sweep")

        # Parameter ranges to test
        chunk_sizes = [128, 256, 512]
        ivf_clusters = [64, 256, 1024]
        probe_lists = [8, 32, 64]

        for chunk_size in chunk_sizes:
            for ivf in ivf_clusters:
                for probe in probe_lists:
                    config = SearchConfig(
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        chunk_size=chunk_size,
                        ivf_clusters=ivf,
                        probe_lists=probe,
                        batch_size=256
                    )

                    name = f"chunk{chunk_size}_ivf{ivf}_probe{probe}"
                    self.benchmark_configuration(config, corpus_dir, queries, name)

    def run_batch_size_test(self, corpus_dir: str):
        """Test different batch sizes for throughput"""
        print("\n🔬 Running Batch Size Test")

        config = SearchConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            chunk_size=256,
            ivf_clusters=256,
            probe_lists=32
        )

        searcher = GPUFilesystemSearch(config)
        searcher.index_directory(corpus_dir)

        batch_sizes = [1, 10, 50, 100, 500, 1000]
        batch_results = []

        for batch_size in batch_sizes:
            queries = [f"test query {i}" for i in range(batch_size)]

            # Warmup
            for _ in range(5):
                for q in queries[:min(10, len(queries))]:
                    searcher.search(q, k=10)

            # Benchmark
            start_time = time.time()
            for query in queries:
                searcher.search(query, k=10)
            total_time = time.time() - start_time

            qps = len(queries) / total_time
            latency_ms = (total_time / len(queries)) * 1000

            batch_results.append({
                'batch_size': batch_size,
                'qps': qps,
                'latency_ms': latency_ms,
                'total_time': total_time
            })

            print(f"   Batch {batch_size}: {qps:.0f} QPS, {latency_ms:.2f}ms latency")

        return batch_results

    def plot_results(self):
        """Generate visualization of benchmark results"""
        if not self.results:
            print("No results to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # QPS comparison
        names = [r.name for r in self.results]
        qps_values = [r.qps for r in self.results]

        axes[0, 0].bar(range(len(names)), qps_values)
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].set_ylabel('Queries Per Second')
        axes[0, 0].set_title('Search Throughput (QPS)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Latency comparison
        latencies = {
            'P50': [r.latency_p50 for r in self.results],
            'P95': [r.latency_p95 for r in self.results],
            'P99': [r.latency_p99 for r in self.results],
        }

        x = np.arange(len(names))
        width = 0.25

        for i, (label, values) in enumerate(latencies.items()):
            axes[0, 1].bar(x + i * width, values, width, label=label)

        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Search Latency Percentiles')
        axes[0, 1].legend()

        # Index time vs chunks
        axes[1, 0].scatter([r.num_chunks for r in self.results],
                          [r.index_time for r in self.results])
        axes[1, 0].set_xlabel('Number of Chunks')
        axes[1, 0].set_ylabel('Index Time (s)')
        axes[1, 0].set_title('Indexing Performance')

        # Memory usage
        axes[1, 1].bar(range(len(names)), [r.memory_mb for r in self.results])
        axes[1, 1].set_xlabel('Configuration')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].set_title('Memory Usage')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_results.png')
        plt.show()

    def save_results(self):
        """Save benchmark results to JSON"""
        results_dict = []
        for r in self.results:
            results_dict.append({
                'name': r.name,
                'num_chunks': r.num_chunks,
                'num_queries': r.num_queries,
                'index_time': r.index_time,
                'search_time': r.search_time,
                'qps': r.qps,
                'latency_p50': r.latency_p50,
                'latency_p95': r.latency_p95,
                'latency_p99': r.latency_p99,
                'memory_mb': r.memory_mb,
                'config': r.config
            })

        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n📁 Results saved to {self.output_dir}")

    def print_summary(self):
        """Print summary of benchmark results"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Find best configurations
        best_qps = max(self.results, key=lambda r: r.qps)
        best_latency = min(self.results, key=lambda r: r.latency_p50)
        best_memory = min(self.results, key=lambda r: r.memory_mb)

        print(f"\n Best QPS: {best_qps.name}")
        print(f"   - QPS: {best_qps.qps:.0f}")
        print(f"   - P50 Latency: {best_qps.latency_p50:.2f}ms")

        print(f"\n Best Latency: {best_latency.name}")
        print(f"   - P50 Latency: {best_latency.latency_p50:.2f}ms")
        print(f"   - QPS: {best_latency.qps:.0f}")

        print(f"\n Best Memory: {best_memory.name}")
        print(f"   - Memory: {best_memory.memory_mb:.2f}MB")
        print(f"   - QPS: {best_memory.qps:.0f}")

        # Overall statistics
        avg_qps = np.mean([r.qps for r in self.results])
        avg_latency = np.mean([r.latency_p50 for r in self.results])

        print(f"\n Overall Statistics:")
        print(f"   - Average QPS: {avg_qps:.0f}")
        print(f"   - Average P50 Latency: {avg_latency:.2f}ms")
        print(f"   - Configurations tested: {len(self.results)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU filesystem search")
    parser.add_argument("--corpus-dir", type=str, help="Directory to index")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of test queries")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--test", type=str, choices=['all', 'scalability', 'params', 'batch'],
                       default='all', help="Which tests to run")

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = SearchBenchmark(args.output_dir)

    # Generate test corpus if not provided
    if args.corpus_dir:
        corpus_dir = args.corpus_dir
    else:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        corpus = benchmark.generate_synthetic_corpus(100, lines_per_file=100)
        for filename, content in corpus.items():
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write(content)
        corpus_dir = temp_dir

    # Generate test queries
    queries = [f"search for topic {i % 50} and subtopic {i % 20}" for i in range(args.num_queries)]

    # Run benchmarks
    if args.test in ['all', 'scalability']:
        benchmark.run_scalability_test(corpus_dir)

    if args.test in ['all', 'params']:
        benchmark.run_parameter_sweep(corpus_dir, queries[:100])

    if args.test in ['all', 'batch']:
        benchmark.run_batch_size_test(corpus_dir)

    # Generate reports
    benchmark.print_summary()
    benchmark.save_results()
    benchmark.plot_results()


if __name__ == "__main__":
    main()