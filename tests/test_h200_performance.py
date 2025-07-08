#!/usr/bin/env python3
"""
H200 Performance Testing Suite for Xinfluencer AI

This module tests the performance of key components on the H200 GPU:
- Embedding generation with BGE-large-en
- Vector database operations (Qdrant + cuVS)
- LoRA fine-tuning performance
- Self-RAG inference pipeline
- Twitter API ingestion speed
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from data.ingest import TwitterIngester
from data.filter import DataFilter
from data.chunk import TextChunker
from vector.embed import EmbeddingGenerator
from vector.db import VectorDatabase
from model.lora import LoRATrainer
from model.selfrag import SelfRAGPipeline
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

class H200PerformanceTester:
    """Comprehensive performance testing for H200 deployment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test data
        self.test_texts = [
            "Bitcoin is showing strong momentum as institutional adoption increases.",
            "Ethereum's transition to proof-of-stake has been successful.",
            "DeFi protocols are revolutionizing traditional finance.",
            "NFTs continue to gain mainstream acceptance.",
            "Layer 2 solutions are scaling blockchain networks effectively."
        ] * 20  # 100 total test texts
        
        logger.info(f"Initialized H200 tester on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def test_gpu_setup(self) -> Dict:
        """Test basic GPU setup and memory"""
        logger.info("Testing GPU setup...")
        start_time = time.time()
        
        results = {
            "gpu_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            "free_memory_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
        }
        
        # Test basic tensor operations
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).to(self.device)
            y = torch.randn(1000, 1000).to(self.device)
            
            # Matrix multiplication test
            start = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            matmul_time = time.time() - start
            
            results["matmul_time_ms"] = matmul_time * 1000
            results["matmul_gflops"] = (2 * 1000**3) / (matmul_time * 1e9)
        
        total_time = time.time() - start_time
        results["setup_time_ms"] = total_time * 1000
        
        logger.info(f"GPU setup test completed in {total_time:.3f}s")
        return results
    
    def test_embedding_generation(self) -> Dict:
        """Test BGE-large-en embedding generation performance"""
        logger.info("Testing embedding generation...")
        start_time = time.time()
        
        try:
            embedder = EmbeddingGenerator(self.config)
            
            # Warm up
            _ = embedder.generate_embeddings(["Test text"])
            
            # Benchmark
            batch_sizes = [1, 4, 8, 16, 32]
            results = {}
            
            for batch_size in batch_sizes:
                batch_texts = self.test_texts[:batch_size]
                
                start = time.time()
                embeddings = embedder.generate_embeddings(batch_texts)
                torch.cuda.synchronize()
                batch_time = time.time() - start
                
                results[f"batch_{batch_size}_time_ms"] = batch_time * 1000
                results[f"batch_{batch_size}_throughput"] = batch_size / batch_time
                results[f"batch_{batch_size}_embedding_dim"] = len(embeddings[0])
            
            total_time = time.time() - start_time
            results["total_time_ms"] = total_time * 1000
            
            logger.info(f"Embedding generation test completed in {total_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Embedding generation test failed: {e}")
            return {"error": str(e)}
    
    def test_vector_database(self) -> Dict:
        """Test Qdrant vector database operations"""
        logger.info("Testing vector database operations...")
        start_time = time.time()
        
        try:
            vdb = VectorDatabase(self.config)
            
            # Generate test embeddings
            embedder = EmbeddingGenerator(self.config)
            embeddings = embedder.generate_embeddings(self.test_texts[:10])
            
            # Test insertion
            start = time.time()
            vdb.insert_vectors(embeddings, self.test_texts[:10])
            insert_time = time.time() - start
            
            # Test search
            query_embedding = embeddings[0]
            start = time.time()
            results = vdb.search_similar(query_embedding, k=5)
            search_time = time.time() - start
            
            # Test batch operations
            batch_embeddings = embeddings[:5]
            start = time.time()
            batch_results = vdb.batch_search(batch_embeddings, k=3)
            batch_search_time = time.time() - start
            
            results_dict = {
                "insert_time_ms": insert_time * 1000,
                "search_time_ms": search_time * 1000,
                "batch_search_time_ms": batch_search_time * 1000,
                "vectors_inserted": len(embeddings),
                "search_results_count": len(results),
                "batch_search_results_count": len(batch_results)
            }
            
            total_time = time.time() - start_time
            results_dict["total_time_ms"] = total_time * 1000
            
            logger.info(f"Vector database test completed in {total_time:.3f}s")
            return results_dict
            
        except Exception as e:
            logger.error(f"Vector database test failed: {e}")
            return {"error": str(e)}
    
    def test_lora_training(self) -> Dict:
        """Test LoRA fine-tuning performance"""
        logger.info("Testing LoRA training...")
        start_time = time.time()
        
        try:
            trainer = LoRATrainer(self.config)
            
            # Create dummy training data
            training_texts = self.test_texts[:50]
            training_embeddings = EmbeddingGenerator(self.config).generate_embeddings(training_texts)
            
            # Test training setup
            start = time.time()
            model = trainer.setup_model()
            setup_time = time.time() - start
            
            # Test one training step
            start = time.time()
            loss = trainer.training_step(model, training_embeddings[:4], training_texts[:4])
            step_time = time.time() - start
            
            # Test memory usage during training
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0) / 1e9
            
            start = time.time()
            for _ in range(5):  # 5 training steps
                loss = trainer.training_step(model, training_embeddings[:4], training_texts[:4])
            torch.cuda.synchronize()
            training_time = time.time() - start
            
            final_memory = torch.cuda.memory_allocated(0) / 1e9
            
            results = {
                "model_setup_time_ms": setup_time * 1000,
                "single_step_time_ms": step_time * 1000,
                "training_time_ms": training_time * 1000,
                "steps_per_second": 5 / training_time,
                "initial_memory_gb": initial_memory,
                "final_memory_gb": final_memory,
                "memory_increase_gb": final_memory - initial_memory,
                "loss_value": float(loss) if loss is not None else 0.0
            }
            
            total_time = time.time() - start_time
            results["total_time_ms"] = total_time * 1000
            
            logger.info(f"LoRA training test completed in {total_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"LoRA training test failed: {e}")
            return {"error": str(e)}
    
    def test_self_rag_pipeline(self) -> Dict:
        """Test Self-RAG inference pipeline"""
        logger.info("Testing Self-RAG pipeline...")
        start_time = time.time()
        
        try:
            rag_pipeline = SelfRAGPipeline(self.config)
            
            # Test query processing
            query = "What are the latest trends in cryptocurrency?"
            
            start = time.time()
            response = rag_pipeline.generate_response(query)
            generation_time = time.time() - start
            
            # Test with different query lengths
            queries = [
                "Bitcoin price",
                "What are the latest trends in cryptocurrency and DeFi protocols?",
                "Explain the impact of institutional adoption on cryptocurrency markets and how it affects retail investors"
            ]
            
            query_times = []
            for q in queries:
                start = time.time()
                _ = rag_pipeline.generate_response(q)
                query_times.append(time.time() - start)
            
            results = {
                "single_query_time_ms": generation_time * 1000,
                "avg_query_time_ms": np.mean(query_times) * 1000,
                "min_query_time_ms": np.min(query_times) * 1000,
                "max_query_time_ms": np.max(query_times) * 1000,
                "response_length": len(response) if response else 0,
                "queries_processed": len(queries)
            }
            
            total_time = time.time() - start_time
            results["total_time_ms"] = total_time * 1000
            
            logger.info(f"Self-RAG pipeline test completed in {total_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Self-RAG pipeline test failed: {e}")
            return {"error": str(e)}
    
    def test_twitter_ingestion(self) -> Dict:
        """Test Twitter API ingestion speed"""
        logger.info("Testing Twitter API ingestion...")
        start_time = time.time()
        
        try:
            ingester = TwitterIngester(self.config)
            
            # Test with a small number of accounts
            test_accounts = ["elonmusk", "VitalikButerin"]  # Small test set
            
            start = time.time()
            tweets = ingester.fetch_tweets_from_accounts(test_accounts, max_tweets_per_account=10)
            fetch_time = time.time() - start
            
            # Test filtering
            filter_obj = DataFilter(self.config)
            start = time.time()
            filtered_tweets = filter_obj.filter_tweets(tweets)
            filter_time = time.time() - start
            
            # Test chunking
            chunker = TextChunker(self.config)
            start = time.time()
            chunks = chunker.chunk_texts([tweet.text for tweet in filtered_tweets])
            chunk_time = time.time() - start
            
            results = {
                "fetch_time_ms": fetch_time * 1000,
                "filter_time_ms": filter_time * 1000,
                "chunk_time_ms": chunk_time * 1000,
                "tweets_fetched": len(tweets),
                "tweets_filtered": len(filtered_tweets),
                "chunks_created": len(chunks),
                "throughput_tweets_per_second": len(tweets) / fetch_time if fetch_time > 0 else 0
            }
            
            total_time = time.time() - start_time
            results["total_time_ms"] = total_time * 1000
            
            logger.info(f"Twitter ingestion test completed in {total_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Twitter ingestion test failed: {e}")
            return {"error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """Run all performance tests and return comprehensive results"""
        logger.info("Starting comprehensive H200 performance testing...")
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "device_info": self.test_gpu_setup(),
            "embedding_performance": self.test_embedding_generation(),
            "vector_db_performance": self.test_vector_database(),
            "lora_training_performance": self.test_lora_training(),
            "self_rag_performance": self.test_self_rag_pipeline(),
            "twitter_ingestion_performance": self.test_twitter_ingestion()
        }
        
        # Calculate summary metrics
        summary = self._calculate_summary(all_results)
        all_results["summary"] = summary
        
        # Save results
        self._save_results(all_results)
        
        logger.info("All performance tests completed!")
        return all_results
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary metrics from all test results"""
        summary = {
            "total_test_time": 0,
            "gpu_utilization": "Available" if results["device_info"]["gpu_available"] else "Not Available",
            "memory_usage_gb": results["device_info"].get("total_memory_gb", 0),
            "embedding_throughput": results["embedding_performance"].get("batch_16_throughput", 0),
            "vector_db_operations": "Success" if "error" not in results["vector_db_performance"] else "Failed",
            "lora_training_speed": results["lora_training_performance"].get("steps_per_second", 0),
            "rag_response_time_ms": results["self_rag_performance"].get("avg_query_time_ms", 0),
            "twitter_ingestion_rate": results["twitter_ingestion_performance"].get("throughput_tweets_per_second", 0)
        }
        
        # Calculate total test time
        for test_name, test_results in results.items():
            if isinstance(test_results, dict) and "total_time_ms" in test_results:
                summary["total_test_time"] += test_results["total_time_ms"]
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"h200_performance_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Also save a summary report
        summary_file = f"h200_performance_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("H200 Performance Test Summary\n")
            f.write("=" * 40 + "\n\n")
            
            summary = results["summary"]
            f.write(f"Test Date: {results['timestamp']}\n")
            f.write(f"GPU: {summary['gpu_utilization']}\n")
            f.write(f"Memory: {summary['memory_usage_gb']:.1f} GB\n")
            f.write(f"Embedding Throughput: {summary['embedding_throughput']:.1f} texts/sec\n")
            f.write(f"Vector DB: {summary['vector_db_operations']}\n")
            f.write(f"LoRA Training: {summary['lora_training_speed']:.1f} steps/sec\n")
            f.write(f"RAG Response Time: {summary['rag_response_time_ms']:.1f} ms\n")
            f.write(f"Twitter Ingestion: {summary['twitter_ingestion_rate']:.1f} tweets/sec\n")
            f.write(f"Total Test Time: {summary['total_test_time']:.1f} ms\n")
        
        logger.info(f"Summary saved to {summary_file}")


def main():
    """Main function to run H200 performance tests"""
    # Load configuration
    config = Config()
    
    # Initialize tester
    tester = H200PerformanceTester(config)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*60)
    print("H200 PERFORMANCE TEST SUMMARY")
    print("="*60)
    
    summary = results["summary"]
    print(f"GPU: {summary['gpu_utilization']}")
    print(f"Memory: {summary['memory_usage_gb']:.1f} GB")
    print(f"Embedding Throughput: {summary['embedding_throughput']:.1f} texts/sec")
    print(f"Vector DB: {summary['vector_db_operations']}")
    print(f"LoRA Training: {summary['lora_training_speed']:.1f} steps/sec")
    print(f"RAG Response Time: {summary['rag_response_time_ms']:.1f} ms")
    print(f"Twitter Ingestion: {summary['twitter_ingestion_rate']:.1f} tweets/sec")
    print(f"Total Test Time: {summary['total_test_time']:.1f} ms")
    print("="*60)


if __name__ == "__main__":
    main() 