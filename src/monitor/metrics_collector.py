"""Metrics collector for Xinfluencer AI system."""

import logging
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import requests
import psutil
import subprocess
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ScrapingMetrics:
    """Metrics for data scraping (Loop 1)."""
    posts_scraped: int = 0
    posts_filtered: Dict[str, int] = None
    posts_chunked: int = 0
    quality_score: float = 0.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.posts_filtered is None:
            self.posts_filtered = {
                'language': 0,
                'toxicity': 0,
                'bot': 0,
                'perplexity': 0,
                'engagement': 0
            }

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance (Loop 2)."""
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    context_precision: float = 0.0
    latency_p95: float = 0.0
    vector_count: int = 0
    index_rebuilds: int = 0
    retrieval_time: float = 0.0

@dataclass
class GenerationMetrics:
    """Metrics for text generation (Loop 3)."""
    faithfulness_score: float = 0.0
    hallucination_rate: float = 0.0
    self_rag_passes: int = 0
    avg_tokens: float = 0.0
    lora_updates: int = 0
    generation_time: float = 0.0

@dataclass
class BehavioralMetrics:
    """Metrics for behavioral feedback (Loop 4)."""
    reward_score: float = 0.0
    engagement_rate: float = 0.0
    ppo_updates: int = 0
    human_reviews: int = 0
    ai_reviews: int = 0
    posts_published: int = 0

class MetricsCollector:
    """Centralized metrics collector for the Xinfluencer AI system."""
    
    def __init__(self, dashboard_url: str = "http://localhost:8000"):
        self.dashboard_url = dashboard_url
        self.scraping_metrics = ScrapingMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.behavioral_metrics = BehavioralMetrics()
        
        # Performance tracking
        self.performance_timers = {}
        self.error_counts = defaultdict(int)
        
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_timers[operation] = duration
            logger.debug(f"{operation} took {duration:.3f}s")
    
    def record_scraping_metrics(self, 
                              posts_scraped: int = 0,
                              posts_filtered: Optional[Dict[str, int]] = None,
                              posts_chunked: int = 0,
                              quality_score: float = 0.0):
        """Record scraping metrics."""
        if posts_filtered:
            for filter_type, count in posts_filtered.items():
                self.scraping_metrics.posts_filtered[filter_type] += count
        
        self.scraping_metrics.posts_scraped += posts_scraped
        self.scraping_metrics.posts_chunked += posts_chunked
        self.scraping_metrics.quality_score = quality_score
        
        logger.info(f"Scraping metrics: {posts_scraped} scraped, {posts_chunked} chunked, quality: {quality_score:.3f}")
    
    def record_retrieval_metrics(self,
                               precision_at_5: float = 0.0,
                               recall_at_10: float = 0.0,
                               context_precision: float = 0.0,
                               latency_p95: float = 0.0,
                               vector_count: int = 0):
        """Record retrieval metrics."""
        self.retrieval_metrics.precision_at_5 = precision_at_5
        self.retrieval_metrics.recall_at_10 = recall_at_10
        self.retrieval_metrics.context_precision = context_precision
        self.retrieval_metrics.latency_p95 = latency_p95
        self.retrieval_metrics.vector_count = vector_count
        
        logger.info(f"Retrieval metrics: P@5={precision_at_5:.3f}, R@10={recall_at_10:.3f}, latency={latency_p95:.3f}s")
    
    def record_generation_metrics(self,
                                faithfulness_score: float = 0.0,
                                hallucination_rate: float = 0.0,
                                self_rag_passes: int = 0,
                                avg_tokens: float = 0.0):
        """Record generation metrics."""
        self.generation_metrics.faithfulness_score = faithfulness_score
        self.generation_metrics.hallucination_rate = hallucination_rate
        self.generation_metrics.self_rag_passes += self_rag_passes
        self.generation_metrics.avg_tokens = avg_tokens
        
        logger.info(f"Generation metrics: faithfulness={faithfulness_score:.3f}, hallucination={hallucination_rate:.3f}")
    
    def record_behavioral_metrics(self,
                                reward_score: float = 0.0,
                                engagement_rate: float = 0.0,
                                posts_published: int = 0,
                                human_reviews: int = 0,
                                ai_reviews: int = 0):
        """Record behavioral metrics."""
        self.behavioral_metrics.reward_score = reward_score
        self.behavioral_metrics.engagement_rate = engagement_rate
        self.behavioral_metrics.posts_published += posts_published
        self.behavioral_metrics.human_reviews += human_reviews
        self.behavioral_metrics.ai_reviews += ai_reviews
        
        logger.info(f"Behavioral metrics: reward={reward_score:.3f}, engagement={engagement_rate:.3f}")
    
    def record_error(self, error_type: str, error_message: str = ""):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        logger.error(f"Error recorded: {error_type} - {error_message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            # CPU and memory
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU info
            gpu_memory_gb = 0.0
            gpu_utilization = 0.0
            
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(',')
                    if len(gpu_info) >= 2:
                        gpu_memory_mb = float(gpu_info[0])
                        gpu_utilization = float(gpu_info[1])
                        gpu_memory_gb = gpu_memory_mb / 1024
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
            
            return {
                'gpu_memory_gb': gpu_memory_gb,
                'gpu_utilization': gpu_utilization,
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'gpu_memory_gb': 0.0,
                'gpu_utilization': 0.0,
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0
            }
    
    async def send_metrics_to_dashboard(self):
        """Send all collected metrics to the monitoring dashboard."""
        try:
            metrics_data = {
                'quality': {
                    'language_filtered': self.scraping_metrics.posts_filtered['language'],
                    'toxicity_filtered': self.scraping_metrics.posts_filtered['toxicity'],
                    'bot_filtered': self.scraping_metrics.posts_filtered['bot'],
                    'perplexity_filtered': self.scraping_metrics.posts_filtered['perplexity'],
                    'engagement_filtered': self.scraping_metrics.posts_filtered['engagement'],
                    'total_processed': self.scraping_metrics.posts_scraped,
                    'quality_score': self.scraping_metrics.quality_score
                },
                'retrieval': {
                    'precision_at_5': self.retrieval_metrics.precision_at_5,
                    'recall_at_10': self.retrieval_metrics.recall_at_10,
                    'context_precision': self.retrieval_metrics.context_precision,
                    'latency_p95': self.retrieval_metrics.latency_p95,
                    'vector_count': self.retrieval_metrics.vector_count
                },
                'generation': {
                    'faithfulness_score': self.generation_metrics.faithfulness_score,
                    'hallucination_rate': self.generation_metrics.hallucination_rate,
                    'self_rag_passes': self.generation_metrics.self_rag_passes,
                    'avg_tokens': self.generation_metrics.avg_tokens
                },
                'behavioral': {
                    'reward_score': self.behavioral_metrics.reward_score,
                    'engagement_rate': self.behavioral_metrics.engagement_rate,
                    'ppo_updates': self.behavioral_metrics.ppo_updates,
                    'human_reviews': self.behavioral_metrics.human_reviews,
                    'ai_reviews': self.behavioral_metrics.ai_reviews
                },
                'system': self.get_system_health()
            }
            
            response = requests.post(
                f"{self.dashboard_url}/api/metrics/update",
                json=metrics_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug("Metrics sent to dashboard successfully")
            else:
                logger.warning(f"Failed to send metrics to dashboard: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending metrics to dashboard: {e}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary report of all metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'scraping': asdict(self.scraping_metrics),
            'retrieval': asdict(self.retrieval_metrics),
            'generation': asdict(self.generation_metrics),
            'behavioral': asdict(self.behavioral_metrics),
            'system': self.get_system_health(),
            'performance': dict(self.performance_timers),
            'errors': dict(self.error_counts)
        }
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.scraping_metrics = ScrapingMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.behavioral_metrics = BehavioralMetrics()
        self.performance_timers.clear()
        self.error_counts.clear()
        logger.info("All metrics reset")

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector

# Convenience functions for easy metric recording
def record_scraping_metrics(**kwargs):
    """Record scraping metrics using the global collector."""
    metrics_collector.record_scraping_metrics(**kwargs)

def record_retrieval_metrics(**kwargs):
    """Record retrieval metrics using the global collector."""
    metrics_collector.record_retrieval_metrics(**kwargs)

def record_generation_metrics(**kwargs):
    """Record generation metrics using the global collector."""
    metrics_collector.record_generation_metrics(**kwargs)

def record_behavioral_metrics(**kwargs):
    """Record behavioral metrics using the global collector."""
    metrics_collector.record_behavioral_metrics(**kwargs)

def record_error(error_type: str, error_message: str = ""):
    """Record an error using the global collector."""
    metrics_collector.record_error(error_type, error_message)

@contextmanager
def timer(operation: str):
    """Context manager for timing operations using the global collector."""
    with metrics_collector.timer(operation):
        yield 