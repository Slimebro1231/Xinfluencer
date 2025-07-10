"""Comprehensive monitoring dashboard for Xinfluencer AI system based on flow.md architecture."""

import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import uvicorn
import numpy as np
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)

# ============================================================================
# PROMETHEUS METRICS - GOLDEN SIGNALS
# ============================================================================

# Data Quality Metrics (Loop 1)
POSTS_SCRAPED = Counter('posts_scraped_total', 'Total number of posts scraped', ['source', 'kol'])
POSTS_FILTERED = Counter('posts_filtered_total', 'Posts filtered by quality gates', ['filter_type'])
POSTS_CHUNKED = Counter('posts_chunked_total', 'Posts successfully chunked and embedded')
QUALITY_SCORE = Histogram('quality_score', 'Quality score distribution', ['metric'])

# Retrieval Metrics (Loop 2)
RETRIEVAL_REQUESTS = Counter('retrieval_requests_total', 'Total retrieval requests')
RETRIEVAL_LATENCY = Histogram('retrieval_latency_seconds', 'Retrieval latency in seconds')
RETRIEVAL_PRECISION = Histogram('retrieval_precision_at_k', 'Precision@k for retrieval', ['k'])
RETRIEVAL_RECALL = Histogram('retrieval_recall_at_k', 'Recall@k for retrieval', ['k'])
CONTEXT_PRECISION = Histogram('context_precision', 'RAGAS context precision score')
VECTOR_DB_SIZE = Gauge('vector_db_size', 'Number of vectors in database')

# Generation Metrics (Loop 3)
GENERATION_REQUESTS = Counter('generation_requests_total', 'Total generation requests')
GENERATION_LATENCY = Histogram('generation_latency_seconds', 'Generation latency in seconds')
GENERATION_TOKENS = Histogram('generation_tokens', 'Number of tokens generated')
SELF_RAG_PASSES = Counter('self_rag_passes_total', 'Self-RAG critique passes')
HALLUCINATION_RATE = Gauge('hallucination_rate', 'Rate of detected hallucinations')
FAITHFULNESS_SCORE = Histogram('faithfulness_score', 'RAGAS faithfulness score')

# Behavioral Metrics (Loop 4)
POSTS_PUBLISHED = Counter('posts_published_total', 'Total posts published')
ENGAGEMENT_METRICS = Counter('engagement_metrics_total', 'Engagement metrics', ['metric_type'])
REWARD_SCORE = Histogram('reward_score', 'Reward model scores')
PPO_UPDATES = Counter('ppo_updates_total', 'PPO policy updates')
AI_PEER_REVIEWS = Counter('ai_peer_reviews_total', 'AI peer review scores', ['score_range'])

# System Health Metrics
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage in bytes')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
MODEL_LOAD_TIME = Histogram('model_load_time_seconds', 'Model loading time')
ERROR_RATE = Counter('error_rate_total', 'Error occurrences', ['error_type'])

# ============================================================================
# DATA STRUCTURES FOR MONITORING
# ============================================================================

@dataclass
class QualityMetrics:
    """Data quality metrics from Loop 1."""
    language_filtered: int = 0
    toxicity_filtered: int = 0
    bot_filtered: int = 0
    perplexity_filtered: int = 0
    engagement_filtered: int = 0
    total_processed: int = 0
    quality_score: float = 0.0

@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics from Loop 2."""
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    context_precision: float = 0.0
    latency_p95: float = 0.0
    vector_count: int = 0
    index_rebuilds: int = 0

@dataclass
class GenerationMetrics:
    """Generation performance metrics from Loop 3."""
    faithfulness_score: float = 0.0
    hallucination_rate: float = 0.0
    self_rag_passes: int = 0
    avg_tokens: float = 0.0
    lora_updates: int = 0

@dataclass
class BehavioralMetrics:
    """Behavioral feedback metrics from Loop 4."""
    reward_score: float = 0.0
    engagement_rate: float = 0.0
    ppo_updates: int = 0
    human_reviews: int = 0
    ai_reviews: int = 0

@dataclass
class SystemHealth:
    """System health metrics."""
    gpu_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0

class MonitoringDashboard:
    """Comprehensive monitoring dashboard for Xinfluencer AI."""
    
    def __init__(self):
        self.app = FastAPI(title="Xinfluencer AI Monitoring Dashboard")
        self.setup_routes()
        self.setup_websockets()
        
        # Metrics storage
        self.quality_metrics = QualityMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.behavioral_metrics = BehavioralMetrics()
        self.system_health = SystemHealth()
        
        # Historical data for trend analysis
        self.historical_data = {
            'quality': deque(maxlen=1000),
            'retrieval': deque(maxlen=1000),
            'generation': deque(maxlen=1000),
            'behavioral': deque(maxlen=1000),
            'system': deque(maxlen=1000)
        }
        
        # Alert thresholds from flow.md
        self.alert_thresholds = {
            'retrieval_precision_drop': 0.05,  # 5% WoW drop
            'faithfulness_min': 0.90,
            'hallucination_max': 0.10,
            'latency_p95_max': 2.0,  # 2 seconds
            'adapter_count_max': 8
        }
        
        # Active alerts
        self.active_alerts = []
        
        # Background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitor_thread.start()
    
    def setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_dashboard_html()
        
        @self.app.get("/metrics")
        async def metrics():
            return generate_latest(REGISTRY)
        
        @self.app.get("/api/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": asdict(self.system_health)
            }
        
        @self.app.get("/api/metrics/quality")
        async def get_quality_metrics():
            return asdict(self.quality_metrics)
        
        @self.app.get("/api/metrics/retrieval")
        async def get_retrieval_metrics():
            return asdict(self.retrieval_metrics)
        
        @self.app.get("/api/metrics/generation")
        async def get_generation_metrics():
            return asdict(self.generation_metrics)
        
        @self.app.get("/api/metrics/behavioral")
        async def get_behavioral_metrics():
            return asdict(self.behavioral_metrics)
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            return {
                "active_alerts": self.active_alerts,
                "thresholds": self.alert_thresholds
            }
        
        @self.app.get("/api/historical/{metric_type}")
        async def get_historical_data(metric_type: str):
            if metric_type in self.historical_data:
                return list(self.historical_data[metric_type])
            raise HTTPException(status_code=404, detail="Metric type not found")
        
        @self.app.post("/api/metrics/update")
        async def update_metrics(metrics_data: Dict[str, Any]):
            """Update metrics from external sources."""
            await self._update_metrics(metrics_data)
            return {"status": "updated"}
    
    def setup_websockets(self):
        """Set up WebSocket for real-time updates."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    # Send real-time metrics every 5 seconds
                    await asyncio.sleep(5)
                    data = {
                        "timestamp": datetime.now().isoformat(),
                        "quality": asdict(self.quality_metrics),
                        "retrieval": asdict(self.retrieval_metrics),
                        "generation": asdict(self.generation_metrics),
                        "behavioral": asdict(self.behavioral_metrics),
                        "system": asdict(self.system_health),
                        "alerts": self.active_alerts
                    }
                    await websocket.send_text(json.dumps(data))
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Xinfluencer AI Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
                .metric-value { font-size: 24px; font-weight: bold; }
                .metric-label { color: #666; margin-bottom: 10px; }
                .alert { background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }
                .success { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
                .warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
            </style>
        </head>
        <body>
            <h1>Xinfluencer AI Monitoring Dashboard</h1>
            <div id="alerts"></div>
            <div class="dashboard">
                <div class="metric-card">
                    <div class="metric-label">Data Quality (Loop 1)</div>
                    <div class="metric-value" id="quality-score">-</div>
                    <canvas id="qualityChart"></canvas>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Retrieval Precision@5</div>
                    <div class="metric-value" id="retrieval-precision">-</div>
                    <canvas id="retrievalChart"></canvas>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Generation Faithfulness</div>
                    <div class="metric-value" id="faithfulness-score">-</div>
                    <canvas id="generationChart"></canvas>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Behavioral Reward</div>
                    <div class="metric-value" id="reward-score">-</div>
                    <canvas id="behavioralChart"></canvas>
                </div>
                <div class="metric-card">
                    <div class="metric-label">System Health</div>
                    <div class="metric-value" id="gpu-usage">-</div>
                    <canvas id="systemChart"></canvas>
                </div>
            </div>
            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    document.getElementById('quality-score').textContent = 
                        (data.quality.quality_score * 100).toFixed(1) + '%';
                    document.getElementById('retrieval-precision').textContent = 
                        (data.retrieval.precision_at_5 * 100).toFixed(1) + '%';
                    document.getElementById('faithfulness-score').textContent = 
                        (data.generation.faithfulness_score * 100).toFixed(1) + '%';
                    document.getElementById('reward-score').textContent = 
                        data.behavioral.reward_score.toFixed(2);
                    document.getElementById('gpu-usage').textContent = 
                        data.system.gpu_utilization.toFixed(1) + '%';
                    
                    updateAlerts(data.alerts);
                }
                
                function updateAlerts(alerts) {
                    const alertsDiv = document.getElementById('alerts');
                    alertsDiv.innerHTML = '';
                    alerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert ' + alert.severity;
                        alertDiv.textContent = alert.message;
                        alertsDiv.appendChild(alertDiv);
                    });
                }
            </script>
        </body>
        </html>
        """
    
    async def _update_metrics(self, metrics_data: Dict[str, Any]):
        """Update metrics from external sources."""
        try:
            # Update quality metrics
            if 'quality' in metrics_data:
                quality = metrics_data['quality']
                self.quality_metrics.language_filtered = quality.get('language_filtered', 0)
                self.quality_metrics.toxicity_filtered = quality.get('toxicity_filtered', 0)
                self.quality_metrics.bot_filtered = quality.get('bot_filtered', 0)
                self.quality_metrics.perplexity_filtered = quality.get('perplexity_filtered', 0)
                self.quality_metrics.engagement_filtered = quality.get('engagement_filtered', 0)
                self.quality_metrics.total_processed = quality.get('total_processed', 0)
                self.quality_metrics.quality_score = quality.get('quality_score', 0.0)
                
                # Update Prometheus metrics
                POSTS_FILTERED.labels(filter_type='language').inc(quality.get('language_filtered', 0))
                POSTS_FILTERED.labels(filter_type='toxicity').inc(quality.get('toxicity_filtered', 0))
                POSTS_FILTERED.labels(filter_type='bot').inc(quality.get('bot_filtered', 0))
                POSTS_FILTERED.labels(filter_type='perplexity').inc(quality.get('perplexity_filtered', 0))
                POSTS_FILTERED.labels(filter_type='engagement').inc(quality.get('engagement_filtered', 0))
                QUALITY_SCORE.labels(metric='overall').observe(quality.get('quality_score', 0.0))
            
            # Update retrieval metrics
            if 'retrieval' in metrics_data:
                retrieval = metrics_data['retrieval']
                self.retrieval_metrics.precision_at_5 = retrieval.get('precision_at_5', 0.0)
                self.retrieval_metrics.recall_at_10 = retrieval.get('recall_at_10', 0.0)
                self.retrieval_metrics.context_precision = retrieval.get('context_precision', 0.0)
                self.retrieval_metrics.latency_p95 = retrieval.get('latency_p95', 0.0)
                self.retrieval_metrics.vector_count = retrieval.get('vector_count', 0)
                
                # Update Prometheus metrics
                RETRIEVAL_PRECISION.labels(k='5').observe(retrieval.get('precision_at_5', 0.0))
                RETRIEVAL_RECALL.labels(k='10').observe(retrieval.get('recall_at_10', 0.0))
                CONTEXT_PRECISION.observe(retrieval.get('context_precision', 0.0))
                RETRIEVAL_LATENCY.observe(retrieval.get('latency_p95', 0.0))
                VECTOR_DB_SIZE.set(retrieval.get('vector_count', 0))
            
            # Update generation metrics
            if 'generation' in metrics_data:
                generation = metrics_data['generation']
                self.generation_metrics.faithfulness_score = generation.get('faithfulness_score', 0.0)
                self.generation_metrics.hallucination_rate = generation.get('hallucination_rate', 0.0)
                self.generation_metrics.self_rag_passes = generation.get('self_rag_passes', 0)
                self.generation_metrics.avg_tokens = generation.get('avg_tokens', 0.0)
                
                # Update Prometheus metrics
                FAITHFULNESS_SCORE.observe(generation.get('faithfulness_score', 0.0))
                HALLUCINATION_RATE.set(generation.get('hallucination_rate', 0.0))
                SELF_RAG_PASSES.inc(generation.get('self_rag_passes', 0))
                GENERATION_TOKENS.observe(generation.get('avg_tokens', 0.0))
            
            # Update behavioral metrics
            if 'behavioral' in metrics_data:
                behavioral = metrics_data['behavioral']
                self.behavioral_metrics.reward_score = behavioral.get('reward_score', 0.0)
                self.behavioral_metrics.engagement_rate = behavioral.get('engagement_rate', 0.0)
                self.behavioral_metrics.ppo_updates = behavioral.get('ppo_updates', 0)
                self.behavioral_metrics.human_reviews = behavioral.get('human_reviews', 0)
                self.behavioral_metrics.ai_reviews = behavioral.get('ai_reviews', 0)
                
                # Update Prometheus metrics
                REWARD_SCORE.observe(behavioral.get('reward_score', 0.0))
                ENGAGEMENT_METRICS.labels(metric_type='rate').inc(behavioral.get('engagement_rate', 0.0))
                PPO_UPDATES.inc(behavioral.get('ppo_updates', 0))
            
            # Update system health
            if 'system' in metrics_data:
                system = metrics_data['system']
                self.system_health.gpu_memory_gb = system.get('gpu_memory_gb', 0.0)
                self.system_health.gpu_utilization = system.get('gpu_utilization', 0.0)
                self.system_health.cpu_usage = system.get('cpu_usage', 0.0)
                self.system_health.memory_usage = system.get('memory_usage', 0.0)
                self.system_health.disk_usage = system.get('disk_usage', 0.0)
                
                # Update Prometheus metrics
                GPU_MEMORY_USAGE.set(system.get('gpu_memory_gb', 0.0) * 1e9)  # Convert to bytes
                GPU_UTILIZATION.set(system.get('gpu_utilization', 0.0))
            
            # Store historical data
            timestamp = datetime.now()
            self.historical_data['quality'].append({
                'timestamp': timestamp.isoformat(),
                'data': asdict(self.quality_metrics)
            })
            self.historical_data['retrieval'].append({
                'timestamp': timestamp.isoformat(),
                'data': asdict(self.retrieval_metrics)
            })
            self.historical_data['generation'].append({
                'timestamp': timestamp.isoformat(),
                'data': asdict(self.generation_metrics)
            })
            self.historical_data['behavioral'].append({
                'timestamp': timestamp.isoformat(),
                'data': asdict(self.behavioral_metrics)
            })
            self.historical_data['system'].append({
                'timestamp': timestamp.isoformat(),
                'data': asdict(self.system_health)
            })
            
            # Check for alerts
            await self._check_alerts()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            ERROR_RATE.labels(error_type='metrics_update').inc()
    
    async def _check_alerts(self):
        """Check for alert conditions based on thresholds."""
        alerts = []
        
        # Check retrieval precision drop
        if self.retrieval_metrics.precision_at_5 < self.alert_thresholds['retrieval_precision_drop']:
            alerts.append({
                'severity': 'warning',
                'message': f'Retrieval precision@5 below threshold: {self.retrieval_metrics.precision_at_5:.3f}'
            })
        
        # Check faithfulness
        if self.generation_metrics.faithfulness_score < self.alert_thresholds['faithfulness_min']:
            alerts.append({
                'severity': 'alert',
                'message': f'Faithfulness score below threshold: {self.generation_metrics.faithfulness_score:.3f}'
            })
        
        # Check hallucination rate
        if self.generation_metrics.hallucination_rate > self.alert_thresholds['hallucination_max']:
            alerts.append({
                'severity': 'alert',
                'message': f'Hallucination rate above threshold: {self.generation_metrics.hallucination_rate:.3f}'
            })
        
        # Check latency
        if self.retrieval_metrics.latency_p95 > self.alert_thresholds['latency_p95_max']:
            alerts.append({
                'severity': 'warning',
                'message': f'P95 latency above threshold: {self.retrieval_metrics.latency_p95:.2f}s'
            })
        
        # Check GPU memory
        if self.system_health.gpu_memory_gb > 20:  # 20GB threshold for H200
            alerts.append({
                'severity': 'warning',
                'message': f'GPU memory usage high: {self.system_health.gpu_memory_gb:.1f}GB'
            })
        
        self.active_alerts = alerts
    
    def _background_monitoring(self):
        """Background monitoring thread."""
        while self.monitoring_active:
            try:
                # Simulate system health monitoring
                import psutil
                import subprocess
                
                # Get GPU info using nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_info = result.stdout.strip().split(',')
                        if len(gpu_info) >= 2:
                            gpu_memory_mb = float(gpu_info[0])
                            gpu_util = float(gpu_info[1])
                            
                            self.system_health.gpu_memory_gb = gpu_memory_mb / 1024
                            self.system_health.gpu_utilization = gpu_util
                            
                            GPU_MEMORY_USAGE.set(gpu_memory_mb * 1024 * 1024)  # Convert to bytes
                            GPU_UTILIZATION.set(gpu_util)
                except Exception as e:
                    logger.warning(f"Could not get GPU info: {e}")
                
                # Get system info
                self.system_health.cpu_usage = psutil.cpu_percent()
                self.system_health.memory_usage = psutil.virtual_memory().percent
                self.system_health.disk_usage = psutil.disk_usage('/').percent
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the monitoring dashboard."""
        uvicorn.run(self.app, host=host, port=port)

# Global dashboard instance
dashboard = MonitoringDashboard()

def get_dashboard() -> MonitoringDashboard:
    """Get the global dashboard instance."""
    return dashboard

def update_quality_metrics(metrics: Dict[str, Any]):
    """Update quality metrics from external sources."""
    asyncio.create_task(dashboard._update_metrics({'quality': metrics}))

def update_retrieval_metrics(metrics: Dict[str, Any]):
    """Update retrieval metrics from external sources."""
    asyncio.create_task(dashboard._update_metrics({'retrieval': metrics}))

def update_generation_metrics(metrics: Dict[str, Any]):
    """Update generation metrics from external sources."""
    asyncio.create_task(dashboard._update_metrics({'generation': metrics}))

def update_behavioral_metrics(metrics: Dict[str, Any]):
    """Update behavioral metrics from external sources."""
    asyncio.create_task(dashboard._update_metrics({'behavioral': metrics}))

def update_system_health(metrics: Dict[str, Any]):
    """Update system health metrics from external sources."""
    asyncio.create_task(dashboard._update_metrics({'system': metrics}))

if __name__ == "__main__":
    dashboard.run() 