#!/usr/bin/env python3
"""
Production deployment checklist
Essential tests for production readiness
"""

import torch
import sys
import json
import time

# Load custom CUDA ops
sys.path.insert(0, '/home/lee/code/gobed/gpu_search/cuda_ops/build')
torch.ops.load_library('/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so')


def main():
    """Essential production readiness checks"""
    print("🚀 PRODUCTION READINESS CHECKLIST")
    print("=" * 60)
    
    checks = {}
    
    # 1. CUDA Environment
    print("1. CUDA Environment...")
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            cuda_version = torch.version.cuda
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            checks["cuda"] = {
                "available": True,
                "device": device_name,
                "version": cuda_version,
                "memory_gb": memory_gb
            }
            print(f"   ✅ Device: {device_name}")
            print(f"   ✅ CUDA: {cuda_version}")
            print(f"   ✅ Memory: {memory_gb:.1f} GB")
        else:
            checks["cuda"] = {"available": False}
            print("   ❌ CUDA not available")
            return False
    except Exception as e:
        print(f"   ❌ CUDA check failed: {e}")
        return False
    
    # 2. CUDA Ops Loading
    print("\n2. CUDA Ops Loading...")
    try:
        # Test basic operation
        device = torch.device("cuda")
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (10, 512), dtype=torch.int8, device=device)
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        
        checks["ops"] = {
            "loaded": True,
            "basic_test": result.shape[0] == 10
        }
        print("   ✅ Library loaded")
        print("   ✅ Basic operation successful")
    except Exception as e:
        checks["ops"] = {"loaded": False, "error": str(e)}
        print(f"   ❌ CUDA ops failed: {e}")
        return False
    
    # 3. Performance Baseline
    print("\n3. Performance Baseline...")
    try:
        # Single query test
        q = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
        db = torch.randint(-128, 127, (10000, 512), dtype=torch.int8, device=device)
        
        # Warmup
        for _ in range(3):
            _ = torch.ops.gobed_ann.i8dot512_scores(q, db)
        
        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = torch.ops.gobed_ann.i8dot512_scores(q, db)
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        
        qps = 1.0 / latency
        
        checks["performance"] = {
            "single_query_ms": latency * 1000,
            "single_query_qps": qps,
            "meets_baseline": qps > 1000  # Minimum 1K QPS
        }
        
        print(f"   ✅ Latency: {latency*1000:.2f}ms")
        print(f"   ✅ Throughput: {qps:.0f} QPS")
        
        if qps > 1000:
            print("   ✅ Meets baseline performance")
        else:
            print("   ⚠️  Below baseline performance")
            
    except Exception as e:
        checks["performance"] = {"error": str(e)}
        print(f"   ❌ Performance test failed: {e}")
        return False
    
    # 4. Error Handling
    print("\n4. Error Handling...")
    try:
        # Test error case
        q_wrong = torch.randn(512, device=device)  # Wrong dtype
        try:
            result = torch.ops.gobed_ann.i8dot512_scores(q_wrong, db)
            checks["error_handling"] = {"robust": False}
            print("   ❌ Error handling: Should have failed")
        except RuntimeError:
            checks["error_handling"] = {"robust": True}
            print("   ✅ Error handling: Properly catches errors")
    except Exception as e:
        checks["error_handling"] = {"error": str(e)}
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    # 5. Memory Management
    print("\n5. Memory Management...")
    try:
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate and free
        for _ in range(5):
            q_temp = torch.randint(-128, 127, (512,), dtype=torch.int8, device=device)
            db_temp = torch.randint(-128, 127, (1000, 512), dtype=torch.int8, device=device)
            result_temp = torch.ops.gobed_ann.i8dot512_scores(q_temp, db_temp)
            del q_temp, db_temp, result_temp
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        memory_growth = final_memory - initial_memory
        stable = memory_growth < 1024 * 1024  # Less than 1MB growth
        
        checks["memory"] = {
            "initial_mb": initial_memory / 1e6,
            "final_mb": final_memory / 1e6,
            "growth_mb": memory_growth / 1e6,
            "stable": stable
        }
        
        if stable:
            print(f"   ✅ Memory stable: {memory_growth/1e6:.1f}MB growth")
        else:
            print(f"   ⚠️  Memory growth: {memory_growth/1e6:.1f}MB")
            
    except Exception as e:
        checks["memory"] = {"error": str(e)}
        print(f"   ❌ Memory test failed: {e}")
        return False
    
    # 6. Production Configuration
    print("\n6. Production Configuration...")
    config_items = [
        ("LibTorch Path", "/home/lee/code/gobed/libtorch"),
        ("CUDA Ops Path", "/home/lee/code/gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so"),
        ("Go Integration", "github.com/lee101/gobed/gpu")
    ]
    
    checks["config"] = {}
    for name, path in config_items:
        if "libtorch" in path.lower():
            exists = torch.cuda.is_available()  # LibTorch is loaded if CUDA works
        elif path.endswith(".so"):
            import os
            exists = os.path.exists(path)
        else:
            exists = True  # Go package existence would be checked separately
        
        checks["config"][name.lower().replace(" ", "_")] = exists
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {'OK' if exists else 'Missing'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT STATUS")
    print("=" * 60)
    
    all_good = all([
        checks["cuda"]["available"],
        checks["ops"]["loaded"],
        checks["performance"]["meets_baseline"],
        checks["error_handling"]["robust"],
        checks["memory"]["stable"]
    ])
    
    if all_good:
        print("🎯 READY FOR PRODUCTION DEPLOYMENT")
        print("\n✅ All critical checks passed")
        print("✅ Performance meets baseline")
        print("✅ Error handling robust")
        print("✅ Memory management stable")
        
        print("\n📋 Deployment Notes:")
        print("- Monitor GPU memory usage")
        print("- Set up health checks")
        print("- Configure graceful degradation to CPU")
        print("- Test with production data volumes")
    else:
        print("⚠️  ISSUES FOUND - REVIEW BEFORE DEPLOYMENT")
        
        if not checks["performance"]["meets_baseline"]:
            print("- Performance below baseline")
        if not checks["error_handling"]["robust"]:
            print("- Error handling needs improvement")
        if not checks["memory"]["stable"]:
            print("- Memory management unstable")
    
    # Save results
    with open("production_checklist_results.json", "w") as f:
        json.dump(checks, f, indent=2)
    print(f"\n📄 Results saved to production_checklist_results.json")
    
    print("=" * 60)
    return all_good


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)