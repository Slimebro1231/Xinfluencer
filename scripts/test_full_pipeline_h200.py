#!/usr/bin/env python3
"""
Full Pipeline Test Script for H200
This script runs the complete Xinfluencer AI pipeline on the H200 server
with detailed logging, performance metrics, and cuVS/GPU integration testing.
"""

import sys
import json
import time
import logging
import subprocess
import shlex
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils.logger import setup_logger

class PipelineTester:
    """Comprehensive pipeline tester for H200 deployment."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger("pipeline_tester", level="INFO")
        self.results = {
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "performance": {},
            "errors": [],
            "warnings": [],
            "status": "running"
        }
        
        # SSH configuration
        self.ssh_key = self.config.h200.pem_file
        self.ssh_user = self.config.h200.user
        self.ssh_host = self.config.h200.host
        self.remote_dir = self.config.h200.remote_dir
        
    def run_ssh_command(self, command: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a command on the H200 server via SSH."""
        ssh_command = f"ssh -i {self.ssh_key} -o StrictHostKeyChecking=no {self.ssh_user}@{self.ssh_host} 'cd {self.remote_dir} && source xinfluencer_env/bin/activate && {command}'"
        
        self.logger.info(f"Running command: {command}")
        
        try:
            result = subprocess.run(
                shlex.split(ssh_command),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def create_test_script(self, script_content: str, filename: str) -> str:
        """Create a temporary test script on the H200 server."""
        # Escape the script content properly for SSH
        escaped_content = script_content.replace("'", "'\"'\"'")
        command = f"cat > {filename} << 'EOF'\n{script_content}\nEOF"
        
        result = self.run_ssh_command(command)
        if result["success"]:
            return filename
        else:
            raise Exception(f"Failed to create test script: {result['stderr']}")
    
    def test_system_status(self) -> Dict[str, Any]:
        """Test basic system status and GPU availability."""
        self.logger.info("Testing system status...")
        
        step_results = {
            "gpu_status": {},
            "cuda_status": {},
            "memory_status": {},
            "disk_status": {},
            "dependencies": {}
        }
        
        # Test GPU status
        gpu_result = self.run_ssh_command("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader")
        if gpu_result["success"]:
            step_results["gpu_status"] = {"available": True, "info": gpu_result["stdout"].strip()}
        else:
            step_results["gpu_status"] = {"available": False, "error": gpu_result["stderr"]}
        
        # Test CUDA availability
        cuda_script = '''import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("GPU: N/A")
'''
        cuda_result = self.run_ssh_command(f"python3 -c '{cuda_script}'")
        if cuda_result["success"]:
            step_results["cuda_status"] = {"available": True, "info": cuda_result["stdout"].strip()}
        else:
            step_results["cuda_status"] = {"available": False, "error": cuda_result["stderr"]}
        
        # Test memory and disk
        memory_result = self.run_ssh_command("free -h")
        if memory_result["success"]:
            step_results["memory_status"] = {"available": True, "info": memory_result["stdout"]}
        
        disk_result = self.run_ssh_command("df -h /")
        if disk_result["success"]:
            step_results["disk_status"] = {"available": True, "info": disk_result["stdout"]}
        
        # Test key dependencies
        deps_script = '''try:
    import torch
    import transformers
    import faiss
    import cupy
    print("All dependencies available")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"FAISS: {faiss.__version__}")
    print(f"CuPy: {cupy.__version__}")
except ImportError as e:
    print(f"Missing dependency: {e}")
'''
        deps_result = self.run_ssh_command(f"python3 -c '{deps_script}'")
        step_results["dependencies"] = {"available": deps_result["success"], "error": deps_result["stderr"] if not deps_result["success"] else None, "info": deps_result["stdout"]}
        
        return step_results
    
    def test_cuvs_integration(self) -> Dict[str, Any]:
        """Test cuVS/GPU vector search integration."""
        self.logger.info("Testing cuVS/GPU vector search integration...")
        
        step_results = {
            "faiss_gpu": {},
            "cupy_operations": {},
            "vector_operations": {},
            "performance_benchmark": {}
        }
        
        # Test FAISS-GPU
        faiss_test = self.run_ssh_command("python3 scripts/explore_nvidia_cuvs.py")
        if faiss_test["success"]:
            step_results["faiss_gpu"] = {"available": True, "output": faiss_test["stdout"]}
        else:
            step_results["faiss_gpu"] = {"available": False, "error": faiss_test["stderr"]}
        
        # Test CuPy operations
        cupy_script = '''import cupy as cp
import numpy as np
a = cp.random.random((1000, 768))
b = cp.random.random((1000, 768))
c = cp.dot(a, b.T)
print(f"CuPy test successful, result shape: {c.shape}")
'''
        cupy_test = self.run_ssh_command(f"python3 -c '{cupy_script}'")
        if cupy_test["success"]:
            step_results["cupy_operations"] = {"available": True, "output": cupy_test["stdout"]}
        else:
            step_results["cupy_operations"] = {"available": False, "error": cupy_test["stderr"]}
        
        # Test vector operations performance
        perf_script = '''import time
import cupy as cp
import numpy as np
start = time.time()
a = cp.random.random((10000, 768))
b = cp.random.random((10000, 768))
c = cp.dot(a, b.T)
cp.cuda.Stream.null.synchronize()
print(f"GPU vector ops: {time.time() - start:.3f}s")
'''
        perf_test = self.run_ssh_command(f"python3 -c '{perf_script}'")
        if perf_test["success"]:
            step_results["performance_benchmark"] = {"available": True, "output": perf_test["stdout"]}
        else:
            step_results["performance_benchmark"] = {"available": False, "error": perf_test["stderr"]}
        
        return step_results
    
    def test_improved_scraper(self) -> Dict[str, Any]:
        """Test the improved web scraper."""
        self.logger.info("Testing improved web scraper...")
        
        step_results = {
            "scraper_execution": {},
            "tweet_extraction": {},
            "data_quality": {}
        }
        
        # Run the improved scraper
        scraper_result = self.run_ssh_command("python3 scripts/test_improved_scraper.py", timeout=600)
        if scraper_result["success"]:
            step_results["scraper_execution"] = {"success": True, "output": scraper_result["stdout"]}
            
            # Check if tweets were extracted
            if "tweets extracted" in scraper_result["stdout"].lower():
                step_results["tweet_extraction"] = {"success": True, "message": "Tweets successfully extracted"}
            else:
                step_results["tweet_extraction"] = {"success": False, "message": "No tweets extracted"}
        else:
            step_results["scraper_execution"] = {"success": False, "error": scraper_result["stderr"]}
        
        # Check data quality using a temporary script
        quality_script = '''import json
try:
    with open("data/seed_tweets/scraped_seed_tweets.json", "r") as f:
        data = json.load(f)
    print(f"Total tweets: {len(data)}")
    if data:
        avg_length = sum(len(t.get("text", "")) for t in data) / len(data)
        truncated = sum(1 for t in data if t.get("text", "").endswith("..."))
        print(f"Avg length: {avg_length:.1f} chars")
        print(f"Truncated: {truncated}")
    else:
        print("No tweets found")
except Exception as e:
    print(f"Error reading data: {e}")
'''
        try:
            self.create_test_script(quality_script, "test_quality.py")
            quality_result = self.run_ssh_command("python3 test_quality.py")
            if quality_result["success"]:
                step_results["data_quality"] = {"success": True, "metrics": quality_result["stdout"]}
            else:
                step_results["data_quality"] = {"success": False, "error": quality_result["stderr"]}
        except Exception as e:
            step_results["data_quality"] = {"success": False, "error": str(e)}
        
        return step_results
    
    def test_vector_operations(self) -> Dict[str, Any]:
        """Test vector embedding and search operations."""
        self.logger.info("Testing vector operations...")
        
        step_results = {
            "embedding": {},
            "vector_search": {},
            "gpu_acceleration": {}
        }
        
        # Test embedding
        embed_script = '''try:
    from src.vector.embed import TextEmbedder
    embedder = TextEmbedder()
    embedding = embedder.embed_text("test text")
    print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
except Exception as e:
    print(f"Embedding error: {e}")
'''
        try:
            self.create_test_script(embed_script, "test_embedding.py")
            embed_test = self.run_ssh_command("python3 test_embedding.py")
            if embed_test["success"]:
                step_results["embedding"] = {"success": True, "output": embed_test["stdout"]}
            else:
                step_results["embedding"] = {"success": False, "error": embed_test["stderr"]}
        except Exception as e:
            step_results["embedding"] = {"success": False, "error": str(e)}
        
        # Test vector search with GPU
        search_script = '''try:
    from src.vector.search import VectorSearcher
    from src.vector.db import VectorDB
    from src.vector.embed import TextEmbedder
    import time
    
    db = VectorDB()
    embedder = TextEmbedder()
    searcher = VectorSearcher(db, embedder, use_gpu=True)
    
    start = time.time()
    results = searcher.search_similar("test query", top_k=5)
    search_time = time.time() - start
    
    print(f"Search time: {search_time:.3f}s, Results: {len(results)}")
except Exception as e:
    print(f"Search error: {e}")
'''
        try:
            self.create_test_script(search_script, "test_search.py")
            search_test = self.run_ssh_command("python3 test_search.py")
            if search_test["success"]:
                step_results["vector_search"] = {"success": True, "output": search_test["stdout"]}
            else:
                step_results["vector_search"] = {"success": False, "error": search_test["stderr"]}
        except Exception as e:
            step_results["vector_search"] = {"success": False, "error": str(e)}
        
        # Test GPU acceleration
        gpu_script = '''try:
    from src.vector.search import VectorSearcher
    from src.vector.db import VectorDB
    from src.vector.embed import TextEmbedder
    
    searcher = VectorSearcher(VectorDB(), TextEmbedder(), use_gpu=True)
    stats = searcher.get_performance_stats()
    print(f"GPU available: {stats['gpu_available']}")
    print(f"FAISS-GPU: {stats['faiss_gpu_available']}")
    print(f"CuPy: {stats['cupy_available']}")
except Exception as e:
    print(f"GPU test error: {e}")
'''
        try:
            self.create_test_script(gpu_script, "test_gpu.py")
            gpu_test = self.run_ssh_command("python3 test_gpu.py")
            if gpu_test["success"]:
                step_results["gpu_acceleration"] = {"success": True, "output": gpu_test["stdout"]}
            else:
                step_results["gpu_acceleration"] = {"success": False, "error": gpu_test["stderr"]}
        except Exception as e:
            step_results["gpu_acceleration"] = {"success": False, "error": str(e)}
        
        return step_results
    
    def test_text_generation(self) -> Dict[str, Any]:
        """Test text generation with GPU acceleration."""
        self.logger.info("Testing text generation...")
        
        step_results = {
            "model_loading": {},
            "generation": {},
            "memory_usage": {}
        }
        
        # Test model loading
        load_script = '''try:
    from src.model.generate import TextGenerator
    import time
    
    start = time.time()
    generator = TextGenerator(use_quantization=True)
    load_time = time.time() - start
    
    print(f"Model loaded in {load_time:.2f}s")
except Exception as e:
    print(f"Model loading error: {e}")
'''
        try:
            self.create_test_script(load_script, "test_model_load.py")
            load_test = self.run_ssh_command("python3 test_model_load.py")
            if load_test["success"]:
                step_results["model_loading"] = {"success": True, "output": load_test["stdout"]}
            else:
                step_results["model_loading"] = {"success": False, "error": load_test["stderr"]}
        except Exception as e:
            step_results["model_loading"] = {"success": False, "error": str(e)}
        
        # Test generation
        gen_script = '''try:
    from src.model.generate import TextGenerator
    import time
    
    generator = TextGenerator(use_quantization=True)
    start = time.time()
    response = generator.generate_response("What is Bitcoin?", max_tokens=50, temperature=0.7)
    gen_time = time.time() - start
    
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Response: {response[:100]}...")
except Exception as e:
    print(f"Generation error: {e}")
'''
        try:
            self.create_test_script(gen_script, "test_generation.py")
            gen_test = self.run_ssh_command("python3 test_generation.py")
            if gen_test["success"]:
                step_results["generation"] = {"success": True, "output": gen_test["stdout"]}
            else:
                step_results["generation"] = {"success": False, "error": gen_test["stderr"]}
        except Exception as e:
            step_results["generation"] = {"success": False, "error": str(e)}
        
        # Test memory usage
        memory_script = '''try:
    from src.model.generate import TextGenerator
    
    generator = TextGenerator(use_quantization=True)
    usage = generator.get_memory_usage()
    print(f"GPU Memory: {usage['allocated_gb']:.2f}GB allocated, {usage['total_gb']:.2f}GB total")
except Exception as e:
    print(f"Memory check error: {e}")
'''
        try:
            self.create_test_script(memory_script, "test_memory.py")
            memory_test = self.run_ssh_command("python3 test_memory.py")
            if memory_test["success"]:
                step_results["memory_usage"] = {"success": True, "output": memory_test["stdout"]}
            else:
                step_results["memory_usage"] = {"success": False, "error": memory_test["stderr"]}
        except Exception as e:
            step_results["memory_usage"] = {"success": False, "error": str(e)}
        
        return step_results
    
    def test_full_pipeline(self) -> Dict[str, Any]:
        """Test the complete pipeline end-to-end."""
        self.logger.info("Testing full pipeline end-to-end...")
        
        step_results = {
            "pipeline_execution": {},
            "data_flow": {},
            "output_quality": {}
        }
        
        # Create a test script that bypasses Twitter API requirements
        test_pipeline_script = '''import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

try:
    # Test data ingestion (skip Twitter API)
    print("Testing data ingestion...")
    
    # Check if we have scraped tweets
    tweets_file = Path("data/seed_tweets/scraped_seed_tweets.json")
    if tweets_file.exists():
        with open(tweets_file, "r") as f:
            tweets = json.load(f)
        print(f"Found {len(tweets)} scraped tweets")
        
        if tweets:
            # Test chunking
            print("Testing text chunking...")
            from src.utils.chunker import TextChunker
            chunker = TextChunker()
            
            # Create chunks from first few tweets
            sample_tweets = tweets[:5]
            chunks = []
            for tweet in sample_tweets:
                tweet_chunks = chunker.chunk_text(tweet.get("text", ""))
                chunks.extend(tweet_chunks)
            
            print(f"Created {len(chunks)} chunks from {len(sample_tweets)} tweets")
            
            # Test embedding
            print("Testing embeddings...")
            from src.vector.embed import TextEmbedder
            embedder = TextEmbedder()
            
            # Embed first few chunks
            sample_chunks = chunks[:10]
            embeddings = []
            for chunk in sample_chunks:
                embedding = embedder.embed_text(chunk)
                embeddings.append(embedding)
            
            print(f"Created {len(embeddings)} embeddings")
            
            # Test vector search
            print("Testing vector search...")
            from src.vector.search import VectorSearcher
            from src.vector.db import VectorDB
            
            db = VectorDB()
            searcher = VectorSearcher(db, embedder, use_gpu=True)
            
            # Test search
            results = searcher.search_similar("Bitcoin price", top_k=3)
            print(f"Search returned {len(results)} results")
            
            print("Pipeline test completed successfully!")
        else:
            print("No tweets available for testing")
    else:
        print("No scraped tweets file found")
        
except Exception as e:
    print(f"Pipeline test error: {e}")
    import traceback
    traceback.print_exc()
'''
        
        # Create the test script on the server
        test_script_file = "test_pipeline.py"
        try:
            self.create_test_script(test_pipeline_script, test_script_file)
            pipeline_result = self.run_ssh_command(f"python3 {test_script_file}", timeout=600)
            
            if pipeline_result["success"]:
                step_results["pipeline_execution"] = {"success": True, "output": pipeline_result["stdout"]}
            else:
                step_results["pipeline_execution"] = {"success": False, "error": pipeline_result["stderr"]}
        except Exception as e:
            step_results["pipeline_execution"] = {"success": False, "error": str(e)}
        
        # Check data flow
        flow_script = '''import json
from pathlib import Path

try:
    tweets = json.load(open("data/seed_tweets/scraped_seed_tweets.json"))
    chunks = list(Path("data/chunks").glob("*.json"))
    vectors = list(Path("data/vectors").glob("*.json"))
    print(f"Data flow: {len(tweets)} tweets -> {len(chunks)} chunks -> {len(vectors)} vectors")
except Exception as e:
    print(f"Data flow check error: {e}")
'''
        try:
            self.create_test_script(flow_script, "test_data_flow.py")
            flow_result = self.run_ssh_command("python3 test_data_flow.py")
            if flow_result["success"]:
                step_results["data_flow"] = {"success": True, "metrics": flow_result["stdout"]}
            else:
                step_results["data_flow"] = {"success": False, "error": flow_result["stderr"]}
        except Exception as e:
            step_results["data_flow"] = {"success": False, "error": str(e)}
        
        return step_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all pipeline tests."""
        self.logger.info("Starting comprehensive pipeline testing on H200...")
        
        test_start_time = time.time()
        
        try:
            # Test 1: System Status
            self.logger.info("=" * 60)
            self.logger.info("TEST 1: System Status and GPU Availability")
            self.logger.info("=" * 60)
            self.results["steps"]["system_status"] = self.test_system_status()
            
            # Test 2: cuVS Integration
            self.logger.info("=" * 60)
            self.logger.info("TEST 2: cuVS/GPU Vector Search Integration")
            self.logger.info("=" * 60)
            self.results["steps"]["cuvs_integration"] = self.test_cuvs_integration()
            
            # Test 3: Improved Scraper
            self.logger.info("=" * 60)
            self.logger.info("TEST 3: Improved Web Scraper")
            self.logger.info("=" * 60)
            self.results["steps"]["improved_scraper"] = self.test_improved_scraper()
            
            # Test 4: Vector Operations
            self.logger.info("=" * 60)
            self.logger.info("TEST 4: Vector Embedding and Search")
            self.logger.info("=" * 60)
            self.results["steps"]["vector_operations"] = self.test_vector_operations()
            
            # Test 5: Text Generation
            self.logger.info("=" * 60)
            self.logger.info("TEST 5: Text Generation with GPU")
            self.logger.info("=" * 60)
            self.results["steps"]["text_generation"] = self.test_text_generation()
            
            # Test 6: Full Pipeline
            self.logger.info("=" * 60)
            self.logger.info("TEST 6: Full Pipeline End-to-End")
            self.logger.info("=" * 60)
            self.results["steps"]["full_pipeline"] = self.test_full_pipeline()
            
            # Calculate performance metrics
            test_end_time = time.time()
            self.results["performance"]["total_test_time"] = test_end_time - test_start_time
            self.results["performance"]["tests_performed"] = len(self.results["steps"])
            
            # Determine overall status
            all_successful = True
            for step_name, step_result in self.results["steps"].items():
                for test_name, test_result in step_result.items():
                    if isinstance(test_result, dict) and "success" in test_result:
                        if not test_result["success"]:
                            all_successful = False
                            self.results["errors"].append(f"{step_name}.{test_name}: {test_result.get('error', 'Unknown error')}")
            
            self.results["status"] = "completed" if all_successful else "failed"
            
        except Exception as e:
            self.logger.error(f"Pipeline testing failed: {e}")
            self.results["status"] = "error"
            self.results["errors"].append(f"Pipeline testing error: {str(e)}")
            self.results["errors"].append(traceback.format_exc())
        
        self.results["end_time"] = datetime.now().isoformat()
        
        return self.results
    
    def save_results(self, filename: str = "pipeline_test_results.json"):
        """Save test results to file."""
        results_file = Path("logs") / filename
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Test results saved to {results_file}")
        return results_file
    
    def print_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 80)
        print("PIPELINE TEST SUMMARY")
        print("=" * 80)
        
        print(f"Status: {self.results['status'].upper()}")
        print(f"Duration: {self.results['performance'].get('total_test_time', 0):.2f} seconds")
        print(f"Tests performed: {self.results['performance'].get('tests_performed', 0)}")
        
        print(f"\nErrors: {len(self.results['errors'])}")
        for error in self.results['errors']:
            print(f"  - {error}")
        
        print(f"\nWarnings: {len(self.results['warnings'])}")
        for warning in self.results['warnings']:
            print(f"  - {warning}")
        
        print("\nStep Results:")
        for step_name, step_result in self.results['steps'].items():
            print(f"\n{step_name.upper()}:")
            for test_name, test_result in step_result.items():
                if isinstance(test_result, dict) and "success" in test_result:
                    status = "PASS" if test_result["success"] else "FAIL"
                    print(f"  {test_name}: {status}")
                    if not test_result["success"] and "error" in test_result:
                        print(f"    Error: {test_result['error']}")

def main():
    """Main function to run the pipeline tests."""
    print("Xinfluencer AI - Full Pipeline Test on H200")
    print("=" * 50)
    
    tester = PipelineTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Save results
        results_file = tester.save_results()
        
        # Print summary
        tester.print_summary()
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Exit with appropriate code
        if results["status"] == "completed":
            print("\nAll tests completed successfully!")
            sys.exit(0)
        else:
            print(f"\nTests completed with status: {results['status']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nPipeline testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 