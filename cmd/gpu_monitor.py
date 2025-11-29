#!/usr/bin/env python3
"""
GPU Performance Monitor for LibTorch Demo
Real-time monitoring of GPU usage, memory, and performance
"""

import subprocess
import time
import json
import sys
from datetime import datetime

def get_gpu_stats():
    """Get detailed GPU statistics using nvidia-ml-py"""
    try:
        cmd = [
            "nvidia-smi", 
            "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = [p.strip() for p in line.split(',')]
            return {
                'timestamp': parts[0],
                'name': parts[1],
                'temperature': float(parts[2]) if parts[2] != '[Not Supported]' else 0,
                'gpu_util': float(parts[3]) if parts[3] != '[Not Supported]' else 0,
                'mem_util': float(parts[4]) if parts[4] != '[Not Supported]' else 0,
                'mem_total': float(parts[5]),
                'mem_free': float(parts[6]),
                'mem_used': float(parts[7]),
                'power_draw': float(parts[8]) if parts[8] != '[Not Supported]' else 0,
                'power_limit': float(parts[9]) if parts[9] != '[Not Supported]' else 0
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def get_processes():
    """Get GPU processes"""
    try:
        cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    processes.append({
                        'pid': parts[0],
                        'name': parts[1],
                        'gpu_uuid': parts[2],
                        'memory_mb': float(parts[3])
                    })
            return processes
    except Exception as e:
        print(f"Error getting processes: {e}")
        return []

def monitor_gpu(duration=60, interval=1):
    """Monitor GPU for specified duration"""
    print(" GPU Performance Monitor")
    print("=" * 50)
    print(f"  Monitoring for {duration}s (interval: {interval}s)")
    print(f" Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)
    
    baseline_stats = get_gpu_stats()
    if baseline_stats:
        print(f" GPU: {baseline_stats['name']}")
        print(f" Total Memory: {baseline_stats['mem_total']:.0f} MB")
        print(f"🌡  Baseline Temp: {baseline_stats['temperature']:.0f}°C")
        print(f" Power Limit: {baseline_stats['power_limit']:.0f}W")
    
    print("\n Real-time Monitoring:")
    print(f"{'Time':<8} {'GPU%':<6} {'Mem%':<6} {'Used':<8} {'Temp':<6} {'Power':<7} {'Processes'}")
    print("-" * 70)
    
    max_gpu_util = 0
    max_memory_used = 0
    max_temp = 0
    
    start_time = time.time()
    while time.time() - start_time < duration:
        stats = get_gpu_stats()
        processes = get_processes()
        
        if stats:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Track maximums
            max_gpu_util = max(max_gpu_util, stats['gpu_util'])
            max_memory_used = max(max_memory_used, stats['mem_used'])
            max_temp = max(max_temp, stats['temperature'])
            
            process_count = len(processes)
            
            print(f"{current_time:<8} {stats['gpu_util']:>5.0f}% {stats['mem_util']:>5.0f}% "
                  f"{stats['mem_used']:>7.0f}M {stats['temperature']:>5.0f}°C "
                  f"{stats['power_draw']:>6.0f}W {process_count:>9d}")
            
            # Show process details if there are GPU processes
            if processes:
                for proc in processes:
                    if 'demo' in proc['name'].lower() or 'torch' in proc['name'].lower():
                        print(f"  └─  {proc['name']} (PID: {proc['pid']}) - {proc['memory_mb']:.0f}MB")
        
        time.sleep(interval)
    
    print("\n" + "=" * 50)
    print(" Summary:")
    print(f"    Max GPU Utilization: {max_gpu_util:.0f}%")
    print(f"    Max Memory Used: {max_memory_used:.0f} MB")
    print(f"   🌡  Max Temperature: {max_temp:.0f}°C")
    print("=" * 50)

def run_demo_with_monitoring():
    """Run the interactive demo with GPU monitoring"""
    import threading
    import os
    
    print(" Starting Interactive Demo with GPU Monitoring")
    print("=" * 60)
    
    # Start GPU monitoring in background
    monitor_thread = threading.Thread(target=monitor_gpu, args=(300, 0.5))  # 5 minutes, 0.5s interval
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run the demo
    os.chdir('/home/lee/code/gobed/cmd/interactive_demo')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/lee/code/gobed/libtorch/lib:/usr/local/cuda-12.0/targets/x86_64-linux/lib:/home/lee/code/gobed/gpu'
    
    try:
        subprocess.run(['./interactive_demo'], env=env)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            monitor_gpu(duration)
        elif sys.argv[1] == "demo":
            run_demo_with_monitoring()
    else:
        print("Usage:")
        print("  python3 gpu_monitor.py monitor [duration_seconds]")
        print("  python3 gpu_monitor.py demo")