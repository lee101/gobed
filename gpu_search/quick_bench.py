#!/usr/bin/env python3
"""Quick benchmark of GPU server to see optimization potential"""

import requests
import time
import json

def test_batch_size(batch_size, num_batches=5):
    """Test specific batch size"""
    print(f"Testing batch size: {batch_size}")
    
    # Create test data
    texts = [f"Test text {i} for embedding benchmark" for i in range(batch_size)]
    
    times = []
    
    for i in range(num_batches):
        start = time.perf_counter()
        
        response = requests.post('http://localhost:5000/embed', json={
            'texts': texts
        }, timeout=30)
        
        if response.status_code == 200:
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            result = response.json()
            throughput = result.get('texts_per_sec', batch_size / elapsed)
            print(f"  Batch {i+1}: {elapsed*1000:.1f}ms ({throughput:.0f} texts/sec)")
        else:
            print(f"  Error: {response.status_code}")
            return 0
    
    avg_time = sum(times) / len(times)
    avg_throughput = batch_size / avg_time
    
    print(f"  Average: {avg_time*1000:.1f}ms ({avg_throughput:.0f} texts/sec)")
    return avg_throughput

def main():
    print("🚀 GPU Server Batch Size Benchmark")
    print("=" * 50)
    
    # Test different batch sizes
    batch_sizes = [256, 512, 1024, 2048, 4096]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            throughput = test_batch_size(batch_size, num_batches=3)
            results[batch_size] = throughput
            print()
        except Exception as e:
            print(f"  Failed: {e}")
            results[batch_size] = 0
            print()
    
    # Analysis
    print("📊 BATCH SIZE ANALYSIS")
    print("=" * 50)
    
    baseline = results.get(256, 0)
    
    for batch_size, throughput in results.items():
        if throughput > 0:
            improvement = throughput / baseline if baseline > 0 else 1
            print(f"Batch {batch_size:4d}: {throughput:6.0f} texts/sec ({improvement:.1f}x)")
    
    # Find optimal
    best_batch = max(results.keys(), key=lambda k: results[k])
    best_throughput = results[best_batch]
    
    print(f"\n🎯 OPTIMAL BATCH SIZE: {best_batch}")
    print(f"   Throughput: {best_throughput:.0f} texts/sec")
    if baseline > 0:
        print(f"   Improvement: {best_throughput/baseline:.1f}x over 256 batch")

if __name__ == "__main__":
    main()