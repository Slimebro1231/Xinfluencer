#!/usr/bin/env python3
"""
Quick H200 Access Test
Tests basic GPU connectivity and performance on H200
"""

import torch
import time
import numpy as np
import json
from datetime import datetime

def test_basic_gpu_access():
    """Test basic GPU access and properties"""
    print("Testing H200 GPU Access...")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("ERROR: CUDA not available. Please check GPU drivers.")
        return False
    
    # GPU information
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
    
    print(f"GPU Count: {device_count}")
    print(f"Current GPU: {current_device}")
    print(f"GPU Name: {device_name}")
    print(f"Total Memory: {total_memory:.1f} GB")
    
    # Check if it looks like an H200
    if "H200" in device_name or total_memory >= 80:
        print("SUCCESS: H200 or equivalent high-memory GPU detected")
    else:
        print(f"WARNING: GPU detected: {device_name} ({total_memory:.1f}GB)")
        print("   This may not be an H200, but will still work for testing.")
    
    return True

def test_memory_operations():
    """Test GPU memory operations"""
    print("\nTesting GPU Memory Operations...")
    print("=" * 50)
    
    device = torch.device("cuda")
    
    # Test memory allocation
    sizes = [1000, 5000, 10000]
    results = {}
    
    for size in sizes:
        print(f"Testing {size}x{size} matrix operations...")
        
        # Allocate tensors
        start_time = time.time()
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.cuda.synchronize()
        alloc_time = time.time() - start_time
        
        # Matrix multiplication
        start_time = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        # Calculate FLOPS
        flops = 2 * size**3
        gflops = flops / (matmul_time * 1e9)
        
        results[f"size_{size}"] = {
            "allocation_time_ms": alloc_time * 1000,
            "matmul_time_ms": matmul_time * 1000,
            "gflops": gflops,
            "memory_used_gb": (a.numel() + b.numel() + c.numel()) * 4 / 1e9
        }
        
        print(f"  Allocation: {alloc_time*1000:.1f}ms")
        print(f"  MatMul: {matmul_time*1000:.1f}ms")
        print(f"  Performance: {gflops:.1f} GFLOPS")
        
        # Clean up
        del a, b, c
        torch.cuda.empty_cache()
    
    return results

def test_embedding_operations():
    """Test embedding-like operations"""
    print("\nTesting Embedding Operations...")
    print("=" * 50)
    
    device = torch.device("cuda")
    
    # Simulate BGE-large-en embedding dimensions (1024)
    embedding_dim = 1024
    batch_sizes = [1, 4, 8, 16, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        # Create embeddings
        start_time = time.time()
        embeddings = torch.randn(batch_size, embedding_dim, device=device)
        torch.cuda.synchronize()
        alloc_time = time.time() - start_time
        
        # Simulate similarity computation
        start_time = time.time()
        similarities = torch.mm(embeddings, embeddings.T)
        torch.cuda.synchronize()
        sim_time = time.time() - start_time
        
        # Simulate attention-like operations
        start_time = time.time()
        attention_weights = torch.softmax(similarities / np.sqrt(embedding_dim), dim=-1)
        torch.cuda.synchronize()
        attn_time = time.time() - start_time
        
        results[f"batch_{batch_size}"] = {
            "allocation_time_ms": alloc_time * 1000,
            "similarity_time_ms": sim_time * 1000,
            "attention_time_ms": attn_time * 1000,
            "total_time_ms": (alloc_time + sim_time + attn_time) * 1000,
            "throughput": batch_size / (alloc_time + sim_time + attn_time)
        }
        
        print(f"  Total time: {(alloc_time + sim_time + attn_time)*1000:.1f}ms")
        print(f"  Throughput: {batch_size / (alloc_time + sim_time + attn_time):.1f} samples/sec")
        
        # Clean up
        del embeddings, similarities, attention_weights
        torch.cuda.empty_cache()
    
    return results

def test_memory_bandwidth():
    """Test memory bandwidth"""
    print("\nTesting Memory Bandwidth...")
    print("=" * 50)
    
    device = torch.device("cuda")
    
    # Test different data sizes
    sizes_mb = [100, 500, 1000, 2000]  # MB
    results = {}
    
    for size_mb in sizes_mb:
        size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        
        print(f"Testing {size_mb}MB transfer...")
        
        # Host to device
        host_data = torch.randn(size_elements, dtype=torch.float32)
        start_time = time.time()
        device_data = host_data.to(device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start_time
        
        # Device to host
        start_time = time.time()
        host_result = device_data.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start_time
        
        # Calculate bandwidth
        h2d_bandwidth = size_mb / h2d_time  # MB/s
        d2h_bandwidth = size_mb / d2h_time  # MB/s
        
        results[f"size_{size_mb}mb"] = {
            "h2d_time_ms": h2d_time * 1000,
            "d2h_time_ms": d2h_time * 1000,
            "h2d_bandwidth_gbps": h2d_bandwidth / 1000,  # GB/s
            "d2h_bandwidth_gbps": d2h_bandwidth / 1000   # GB/s
        }
        
        print(f"  H2D: {h2d_time*1000:.1f}ms ({h2d_bandwidth/1000:.1f} GB/s)")
        print(f"  D2H: {d2h_time*1000:.1f}ms ({d2h_bandwidth/1000:.1f} GB/s)")
        
        # Clean up
        del host_data, device_data, host_result
        torch.cuda.empty_cache()
    
    return results

def generate_summary_report(memory_results, embedding_results, bandwidth_results):
    """Generate a summary report"""
    print("\nH200 Access Test Summary")
    print("=" * 50)
    
    # Calculate averages
    avg_gflops = np.mean([r["gflops"] for r in memory_results.values()])
    max_throughput = max([r["throughput"] for r in embedding_results.values()])
    avg_h2d_bandwidth = np.mean([r["h2d_bandwidth_gbps"] for r in bandwidth_results.values()])
    avg_d2h_bandwidth = np.mean([r["d2h_bandwidth_gbps"] for r in bandwidth_results.values()])
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpu_name": torch.cuda.get_device_name(),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "avg_gflops": avg_gflops,
        "max_embedding_throughput": max_throughput,
        "avg_h2d_bandwidth_gbps": avg_h2d_bandwidth,
        "avg_d2h_bandwidth_gbps": avg_d2h_bandwidth,
        "memory_results": memory_results,
        "embedding_results": embedding_results,
        "bandwidth_results": bandwidth_results
    }
    
    print(f"GPU: {summary['gpu_name']}")
    print(f"Memory: {summary['total_memory_gb']:.1f} GB")
    print(f"Average GFLOPS: {avg_gflops:.1f}")
    print(f"Max Embedding Throughput: {max_throughput:.1f} samples/sec")
    print(f"Average H2D Bandwidth: {avg_h2d_bandwidth:.1f} GB/s")
    print(f"Average D2H Bandwidth: {avg_d2h_bandwidth:.1f} GB/s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"h200_access_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Performance assessment
    print("\nPerformance Assessment:")
    if avg_gflops > 1000:
        print("SUCCESS: Excellent compute performance")
    elif avg_gflops > 500:
        print("SUCCESS: Good compute performance")
    else:
        print("WARNING: Lower than expected compute performance")
    
    if avg_h2d_bandwidth > 10:
        print("SUCCESS: Excellent memory bandwidth")
    elif avg_h2d_bandwidth > 5:
        print("SUCCESS: Good memory bandwidth")
    else:
        print("WARNING: Lower than expected memory bandwidth")
    
    if max_throughput > 100:
        print("SUCCESS: Excellent embedding throughput")
    elif max_throughput > 50:
        print("SUCCESS: Good embedding throughput")
    else:
        print("WARNING: Lower than expected embedding throughput")
    
    return summary

def main():
    """Main test function"""
    print("H200 Access Test Suite")
    print("=" * 60)
    
    # Test basic access
    if not test_basic_gpu_access():
        return
    
    # Run performance tests
    memory_results = test_memory_operations()
    embedding_results = test_embedding_operations()
    bandwidth_results = test_memory_bandwidth()
    
    # Generate summary
    summary = generate_summary_report(memory_results, embedding_results, bandwidth_results)
    
    print("\nH200 access test completed successfully")
    print("Ready to run full performance suite")

if __name__ == "__main__":
    main() 