#!/usr/bin/env python3
"""
Exploration script for NVIDIA cuVS (CUDA Vector Search) integration.
This script investigates how to use NVIDIA's GPU-accelerated vector search capabilities.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_nvidia_cuvs_availability():
    """Check if NVIDIA cuVS is available and explore capabilities."""
    print("Checking NVIDIA cuVS availability...")
    
    # Check for CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Check CUDA version
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
            
    except ImportError:
        print("PyTorch not available")
        return False
    
    # Check for cuVS-specific libraries
    cuvs_available = False
    
    try:
        # Try to import cuVS (if available)
        import cupy as cp
        print("CuPy available - can be used for GPU vector operations")
        cuvs_available = True
    except ImportError:
        print("CuPy not available")
    
    try:
        # Check for RAPIDS cuML (includes GPU-accelerated similarity search)
        import cuml
        print("RAPIDS cuML available - includes GPU-accelerated similarity search")
        cuvs_available = True
    except ImportError:
        print("RAPIDS cuML not available")
    
    try:
        # Check for FAISS-GPU
        import faiss
        if hasattr(faiss, 'GpuIndexFlatL2'):
            print("FAISS-GPU available - GPU-accelerated similarity search")
            cuvs_available = True
        else:
            print("FAISS available but GPU support not detected")
    except ImportError:
        print("FAISS not available")
    
    return cuvs_available

def explore_gpu_vector_operations():
    """Explore different GPU-accelerated vector operations."""
    print("\nExploring GPU vector operations...")
    
    # Test with CuPy if available
    try:
        import cupy as cp
        
        print("Testing CuPy vector operations...")
        
        # Create test vectors
        n_vectors = 10000
        vector_dim = 768
        
        print(f"Creating {n_vectors} vectors of dimension {vector_dim}...")
        
        # Generate random vectors on GPU
        vectors_gpu = cp.random.randn(n_vectors, vector_dim).astype(cp.float32)
        query_vector = cp.random.randn(vector_dim).astype(cp.float32)
        
        # Normalize vectors
        vectors_gpu = vectors_gpu / cp.linalg.norm(vectors_gpu, axis=1, keepdims=True)
        query_vector = query_vector / cp.linalg.norm(query_vector)
        
        print("Computing cosine similarities...")
        start_time = time.time()
        
        # Compute cosine similarities
        similarities = cp.dot(vectors_gpu, query_vector)
        
        gpu_time = time.time() - start_time
        print(f"GPU computation time: {gpu_time:.4f} seconds")
        
        # Find top-k similar vectors
        k = 10
        top_k_indices = cp.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        
        print(f"Top {k} similarities:")
        for i, (idx, score) in enumerate(zip(top_k_indices.get(), top_k_scores.get())):
            print(f"  {i+1}. Vector {idx}: {score:.4f}")
        
        # Compare with CPU
        print("\nComparing with CPU computation...")
        vectors_cpu = cp.asnumpy(vectors_gpu)
        query_cpu = cp.asnumpy(query_vector)
        
        start_time = time.time()
        similarities_cpu = np.dot(vectors_cpu, query_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CPU computation time: {cpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
    except ImportError:
        print("CuPy not available for GPU vector operations")
    
    # Test with FAISS-GPU if available
    try:
        import faiss
        
        if hasattr(faiss, 'GpuIndexFlatL2'):
            print("\nTesting FAISS-GPU...")
            
            # Create FAISS GPU index
            dimension = 768
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Move to GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # Add vectors
            n_vectors = 10000
            vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            gpu_index.add(vectors)
            
            # Search
            query = np.random.randn(dimension).astype(np.float32)
            query = query / np.linalg.norm(query)
            query = query.reshape(1, -1)
            
            start_time = time.time()
            distances, indices = gpu_index.search(query, 10)
            faiss_time = time.time() - start_time
            
            print(f"FAISS-GPU search time: {faiss_time:.4f} seconds")
            print("Top 10 results:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                print(f"  {i+1}. Vector {idx}: similarity {dist:.4f}")
        
    except ImportError:
        print("FAISS not available")

def explore_qdrant_cuvs_integration():
    """Explore how to integrate cuVS with Qdrant."""
    print("\nExploring Qdrant + cuVS integration...")
    
    # Qdrant supports GPU acceleration through cuVS
    # This would require:
    # 1. Qdrant with GPU support
    # 2. Proper CUDA environment
    # 3. Configuration for GPU-accelerated search
    
    print("Qdrant + cuVS integration would provide:")
    print("- GPU-accelerated similarity search")
    print("- Faster index building")
    print("- Reduced memory usage")
    print("- Better performance for large vector collections")
    
    print("\nTo implement Qdrant + cuVS:")
    print("1. Install Qdrant with GPU support")
    print("2. Configure CUDA environment")
    print("3. Use GPU-optimized distance metrics")
    print("4. Enable GPU acceleration in collection config")

def create_cuvs_requirements():
    """Create requirements file for cuVS integration."""
    requirements = [
        "# NVIDIA cuVS Integration Requirements",
        "# Core GPU libraries",
        "torch>=2.0.0",
        "cupy-cuda11x>=12.0.0",  # Adjust CUDA version as needed",
        "",
        "# Vector search libraries",
        "faiss-gpu>=1.7.0",
        "qdrant-client>=1.3.0",
        "",
        "# RAPIDS ecosystem (optional)",
        "cudf-cu11>=23.0.0",  # Adjust CUDA version",
        "cuml-cu11>=23.0.0",  # Adjust CUDA version",
        "",
        "# Utilities",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        ""
    ]
    
    requirements_file = Path(__file__).parent.parent / "requirements_cuvs.txt"
    with open(requirements_file, "w") as f:
        f.write("\n".join(requirements))
    
    print(f"Created cuVS requirements file: {requirements_file}")

def main():
    """Main exploration function."""
    print("NVIDIA cuVS Integration Exploration")
    print("=" * 50)
    
    # Check availability
    cuvs_available = check_nvidia_cuvs_availability()
    
    if cuvs_available:
        print("\n✅ GPU vector operations available!")
        explore_gpu_vector_operations()
        explore_qdrant_cuvs_integration()
    else:
        print("\n❌ GPU vector operations not available")
        print("To enable cuVS:")
        print("1. Install CUDA toolkit")
        print("2. Install GPU-enabled libraries (CuPy, FAISS-GPU, etc.)")
        print("3. Configure environment for GPU acceleration")
    
    # Create requirements file
    create_cuvs_requirements()
    
    print("\nExploration complete!")

if __name__ == "__main__":
    main() 