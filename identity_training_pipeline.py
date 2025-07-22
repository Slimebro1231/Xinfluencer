#!/usr/bin/env python3
"""
Identity Training Pipeline for Xinfluencer AI
Uses retrieved tweets to train crypto bot identity and style
Stores ALL retrieved posts since we're paying for them
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model.lora import LoRAFineTuner
from model.generate import TextGenerator
from utils.logger import setup_logger
from vector.embed import TextEmbedder
from vector.db import VectorDB

logger = logging.getLogger(__name__)

@dataclass
class TrainingData:
    """Structured training data from tweets."""
    tweet_id: str
    text: str
    author: str
    engagement_score: float
    crypto_relevance: float
    quality_score: float
    source: str  # 'collection', 'failed_api', 'manual'
    timestamp: datetime
    metadata: Dict[str, Any]

class PostStorage:
    """Store ALL retrieved posts with efficient indexing."""
    
    def __init__(self, storage_dir: str = "data/all_posts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite for efficient querying
        self.db_path = self.storage_dir / "posts.db"
        self._init_database()
        
        logger.info(f"Post storage initialized at {self.storage_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for posts."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            author TEXT,
            engagement_score REAL,
            crypto_relevance REAL,
            quality_score REAL,
            source TEXT,
            timestamp TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes separately
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_author ON posts(author)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON posts(quality_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crypto_relevance ON posts(crypto_relevance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON posts(source)")
        
        conn.commit()
        conn.close()
    
    def store_posts(self, posts: List[TrainingData]) -> int:
        """Store posts in database."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        for post in posts:
            try:
                cursor.execute("""
                INSERT OR REPLACE INTO posts 
                (id, text, author, engagement_score, crypto_relevance, quality_score, source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post.tweet_id,
                    post.text,
                    post.author,
                    post.engagement_score,
                    post.crypto_relevance,
                    post.quality_score,
                    post.source,
                    post.timestamp.isoformat(),
                    json.dumps(post.metadata)
                ))
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store post {post.tweet_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {stored_count}/{len(posts)} posts")
        return stored_count
    
    def get_training_posts(self, min_quality: float = 0.7, 
                          min_crypto_relevance: float = 0.8,
                          limit: int = 1000) -> List[TrainingData]:
        """Get high-quality posts for training."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, text, author, engagement_score, crypto_relevance, quality_score, source, timestamp, metadata
        FROM posts 
        WHERE quality_score >= ? AND crypto_relevance >= ?
        ORDER BY quality_score DESC, crypto_relevance DESC
        LIMIT ?
        """, (min_quality, min_crypto_relevance, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        training_data = []
        for row in results:
            training_data.append(TrainingData(
                tweet_id=row[0],
                text=row[1],
                author=row[2],
                engagement_score=row[3],
                crypto_relevance=row[4],
                quality_score=row[5],
                source=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                metadata=json.loads(row[8]) if row[8] else {}
            ))
        
        return training_data
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total posts
        cursor.execute("SELECT COUNT(*) FROM posts")
        total_posts = cursor.fetchone()[0]
        
        # Posts by source
        cursor.execute("SELECT source, COUNT(*) FROM posts GROUP BY source")
        by_source = dict(cursor.fetchall())
        
        # Posts by author (top 10)
        cursor.execute("SELECT author, COUNT(*) FROM posts GROUP BY author ORDER BY COUNT(*) DESC LIMIT 10")
        top_authors = dict(cursor.fetchall())
        
        # Quality distribution
        cursor.execute("SELECT AVG(quality_score), MIN(quality_score), MAX(quality_score) FROM posts")
        quality_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_posts": total_posts,
            "by_source": by_source,
            "top_authors": top_authors,
            "quality_stats": {
                "avg": quality_stats[0],
                "min": quality_stats[1],
                "max": quality_stats[2]
            }
        }

class CryptoIdentityAnalyzer:
    """Analyze crypto content for identity training."""
    
    def __init__(self):
        self.crypto_keywords = {
            'high_value': ['protocol', 'defi', 'ethereum', 'bitcoin', 'rwa', 'yield', 'liquidity', 'staking'],
            'medium_value': ['crypto', 'blockchain', 'token', 'dapp', 'smart contract', 'dao'],
            'technical': ['consensus', 'validator', 'merkle', 'hash', 'nonce', 'gas', 'evm'],
            'institutional': ['regulation', 'compliance', 'custody', 'institutional', 'etf', 'sec']
        }
        
        self.quality_indicators = {
            'positive': ['innovation', 'development', 'adoption', 'utility', 'research', 'analysis'],
            'negative': ['pump', 'moon', 'lambo', 'shilling', 'wen', 'diamond hands']
        }
    
    def analyze_crypto_relevance(self, text: str) -> float:
        """Calculate crypto relevance score (0-1)."""
        text_lower = text.lower()
        
        score = 0.0
        total_possible = 0
        
        for category, keywords in self.crypto_keywords.items():
            category_weight = {'high_value': 0.4, 'medium_value': 0.3, 'technical': 0.2, 'institutional': 0.1}[category]
            
            for keyword in keywords:
                total_possible += category_weight
                if keyword in text_lower:
                    score += category_weight
        
        return min(score / total_possible if total_possible > 0 else 0, 1.0)
    
    def analyze_content_quality(self, text: str, engagement_metrics: Dict = None) -> float:
        """Calculate content quality score (0-1)."""
        text_lower = text.lower()
        
        # Base quality score
        quality_score = 0.5
        
        # Length check (not too short, not too long)
        length_score = 0.8 if 50 <= len(text) <= 280 else 0.3
        quality_score += length_score * 0.2
        
        # Positive indicators
        positive_count = sum(1 for indicator in self.quality_indicators['positive'] if indicator in text_lower)
        quality_score += min(positive_count * 0.1, 0.3)
        
        # Negative indicators (penalize)
        negative_count = sum(1 for indicator in self.quality_indicators['negative'] if indicator in text_lower)
        quality_score -= min(negative_count * 0.15, 0.4)
        
        # Engagement boost
        if engagement_metrics:
            likes = engagement_metrics.get('like_count', 0)
            retweets = engagement_metrics.get('retweet_count', 0)
            total_engagement = likes + retweets * 2  # Weight retweets more
            
            if total_engagement > 100:
                quality_score += 0.2
            elif total_engagement > 20:
                quality_score += 0.1
        
        return max(0, min(quality_score, 1.0))
    
    def extract_identity_features(self, posts: List[TrainingData]) -> Dict[str, Any]:
        """Extract identity features from high-quality posts."""
        features = {
            'writing_style': {},
            'topic_preferences': {},
            'sentiment_patterns': {},
            'engagement_patterns': {}
        }
        
        # Analyze writing style
        avg_length = sum(len(post.text) for post in posts) / len(posts)
        
        # Count technical vs casual language
        technical_count = 0
        casual_count = 0
        
        for post in posts:
            text_lower = post.text.lower()
            
            # Technical indicators
            if any(word in text_lower for word in self.crypto_keywords['technical']):
                technical_count += 1
            
            # Casual indicators  
            if any(word in text_lower for word in ['btw', 'imo', 'tbh', 'ngl']):
                casual_count += 1
        
        features['writing_style'] = {
            'avg_length': avg_length,
            'technical_ratio': technical_count / len(posts),
            'casual_ratio': casual_count / len(posts),
            'preferred_length': 'medium' if 100 <= avg_length <= 200 else 'short' if avg_length < 100 else 'long'
        }
        
        # Topic preferences
        topic_counts = {}
        for category, keywords in self.crypto_keywords.items():
            topic_counts[category] = sum(
                1 for post in posts 
                if any(keyword in post.text.lower() for keyword in keywords)
            )
        
        features['topic_preferences'] = topic_counts
        
        return features

class IdentityTrainingPipeline:
    """Main identity training pipeline."""
    
    def __init__(self, storage_dir: str = "data/all_posts"):
        self.storage = PostStorage(storage_dir)
        self.analyzer = CryptoIdentityAnalyzer()
        self.lora_trainer = None
        self.embedder = TextEmbedder()
        self.vector_db = VectorDB()
        
        # Training configuration
        self.training_config = {
            'min_quality_threshold': 0.7,
            'min_crypto_relevance': 0.8,
            'max_training_examples': 1000,
            'identity_weight': 1.5,  # Weight identity examples higher
            'lora_epochs': 3,
            'batch_size': 4
        }
        
        logger.info("Identity training pipeline initialized")
    
    def ingest_collection_data(self, collection_dir: str = "data/safe_collection") -> int:
        """Ingest data from bulletproof collection system."""
        collection_path = Path(collection_dir)
        
        if not collection_path.exists():
            logger.warning(f"Collection directory {collection_dir} not found")
            return 0
        
        training_data = []
        
        # Process all collection files
        for json_file in collection_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                tweets = data.get('tweets', [])
                logger.info(f"Processing {len(tweets)} tweets from {json_file.name}")
                
                for tweet in tweets:
                    # Extract engagement metrics
                    engagement_metrics = tweet.get('public_metrics', {})
                    engagement_score = (
                        engagement_metrics.get('like_count', 0) +
                        engagement_metrics.get('retweet_count', 0) * 2 +
                        engagement_metrics.get('reply_count', 0)
                    ) / 100.0  # Normalize
                    
                    # Analyze content
                    text = tweet.get('text', '')
                    crypto_relevance = self.analyzer.analyze_crypto_relevance(text)
                    quality_score = self.analyzer.analyze_content_quality(text, engagement_metrics)
                    
                    training_data.append(TrainingData(
                        tweet_id=tweet.get('id', f"unknown_{len(training_data)}"),
                        text=text,
                        author=tweet.get('author_username', 'unknown'),
                        engagement_score=min(engagement_score, 1.0),
                        crypto_relevance=crypto_relevance,
                        quality_score=quality_score,
                        source='collection',
                        timestamp=datetime.now(),
                        metadata={
                            'collection_file': json_file.name,
                            'public_metrics': engagement_metrics,
                            'context_annotations': tweet.get('context_annotations', [])
                        }
                    ))
                
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
        
        # Store all data (even low quality - we paid for it!)
        stored_count = self.storage.store_posts(training_data)
        
        logger.info(f"Ingested {stored_count} posts from collection data")
        return stored_count
    
    def ingest_failed_api_posts(self, failed_dir: str = "data/failed_collection") -> int:
        """Ingest failed API collection attempts - we paid for these too!"""
        # This would ingest posts from failed API calls
        # Since we're charged per post retrieved, not per successful call
        # Implementation depends on how failed calls are logged
        logger.info("Checking for failed API collection data...")
        return 0
    
    def prepare_identity_training_data(self) -> List[Dict]:
        """Prepare training data for identity learning."""
        # Get high-quality posts
        high_quality_posts = self.storage.get_training_posts(
            min_quality=self.training_config['min_quality_threshold'],
            min_crypto_relevance=self.training_config['min_crypto_relevance'],
            limit=self.training_config['max_training_examples']
        )
        
        logger.info(f"Selected {len(high_quality_posts)} posts for identity training")
        
        # Extract identity features
        identity_features = self.analyzer.extract_identity_features(high_quality_posts)
        
        # Create training examples
        training_examples = []
        
        for post in high_quality_posts:
            # Create identity training prompt
            prompt = f"Write a crypto analysis tweet in the style of expert crypto educators"
            response = post.text
            
            # Weight by quality and crypto relevance
            weight = post.quality_score * post.crypto_relevance * self.training_config['identity_weight']
            
            training_examples.append({
                'query': prompt,
                'response': response,
                'approved': True,  # High quality posts are pre-approved
                'weight': weight,
                'author': post.author,
                'metadata': post.metadata
            })
        
        # Save identity features for reference
        features_file = Path("data/training") / "identity_features.json"
        features_file.parent.mkdir(exist_ok=True)
        
        with open(features_file, 'w') as f:
            json.dump({
                'features': identity_features,
                'training_stats': {
                    'total_examples': len(training_examples),
                    'avg_quality': sum(ex['weight'] for ex in training_examples) / len(training_examples),
                    'date_created': datetime.now().isoformat()
                }
            }, f, indent=2)
        
        logger.info(f"Prepared {len(training_examples)} identity training examples")
        logger.info(f"Identity features saved to {features_file}")
        
        return training_examples
    
    def train_identity_model(self, output_dir: str = "lora_checkpoints/identity") -> str:
        """Train the bot with crypto identity using LoRA."""
        logger.info("Starting identity training...")
        
        # Initialize LoRA trainer
        self.lora_trainer = LoRAFineTuner()
        
        # Prepare training data
        training_examples = self.prepare_identity_training_data()
        
        if not training_examples:
            logger.error("No training examples available")
            return None
        
        # Prepare for LoRA training
        lora_training_data = self.lora_trainer.prepare_training_data(training_examples)
        
        # Train with identity focus
        adapter_path = self.lora_trainer.fine_tune(
            lora_training_data,
            output_dir=output_dir
        )
        
        logger.info(f"Identity training completed. Adapter saved to: {adapter_path}")
        
        # Update vector database with identity examples
        self._update_vector_db_with_identity(training_examples)
        
        return adapter_path
    
    def _update_vector_db_with_identity(self, training_examples: List[Dict]):
        """Update vector database with identity training examples."""
        logger.info("Updating vector database with identity examples...")
        
        # Prepare chunks for embedding
        chunks = []
        for i, example in enumerate(training_examples):
            chunk = {
                'text': example['response'],
                'metadata': {
                    'type': 'identity_training',
                    'author': example['author'],
                    'weight': example['weight'],
                    'chunk_id': f"identity_{i}"
                }
            }
            chunks.append(chunk)
        
        # Generate embeddings
        embedded_chunks = self.embedder.embed_chunks(chunks[:100])  # Limit for now
        
        # Store in vector DB
        try:
            self.vector_db.create_collection('identity_training')
            self.vector_db.upsert_chunks(embedded_chunks, collection_name='identity_training')
            logger.info(f"Stored {len(embedded_chunks)} identity chunks in vector DB")
        except Exception as e:
            logger.error(f"Failed to update vector DB: {e}")
    
    def run_full_identity_training(self) -> Dict[str, Any]:
        """Run complete identity training pipeline."""
        logger.info("🤖 Starting FULL Identity Training Pipeline")
        
        start_time = time.time()
        results = {
            'start_time': datetime.now().isoformat(),
            'ingestion_stats': {},
            'training_stats': {},
            'storage_stats': {},
            'adapter_path': None
        }
        
        # Step 1: Ingest all available data
        logger.info("📥 Step 1: Ingesting collected data...")
        collection_count = self.ingest_collection_data()
        failed_count = self.ingest_failed_api_posts()
        
        results['ingestion_stats'] = {
            'collection_posts': collection_count,
            'failed_posts': failed_count,
            'total_ingested': collection_count + failed_count
        }
        
        # Step 2: Train identity model
        logger.info("🧠 Step 2: Training identity model...")
        adapter_path = self.train_identity_model()
        results['adapter_path'] = adapter_path
        
        # Step 3: Get final statistics
        logger.info("📊 Step 3: Generating statistics...")
        storage_stats = self.storage.get_storage_stats()
        results['storage_stats'] = storage_stats
        
        # Calculate training duration
        training_duration = time.time() - start_time
        results['training_duration_minutes'] = training_duration / 60
        results['end_time'] = datetime.now().isoformat()
        
        # Save results
        results_file = Path("data/training") / f"identity_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"🎉 Identity training completed in {training_duration/60:.1f} minutes")
        logger.info(f"📄 Results saved to: {results_file}")
        
        return results

def main():
    """Run identity training pipeline."""
    # Setup logging
    logger = setup_logger("identity_training", level="INFO", log_file="logs/identity_training.log")
    
    # Initialize pipeline
    pipeline = IdentityTrainingPipeline()
    
    # Run full training
    results = pipeline.run_full_identity_training()
    
    # Print summary
    print("\n🎯 IDENTITY TRAINING SUMMARY")
    print("=" * 50)
    print(f"📥 Posts ingested: {results['ingestion_stats']['total_ingested']}")
    print(f"💾 Total posts stored: {results['storage_stats']['total_posts']}")
    print(f"⏱️ Training time: {results['training_duration_minutes']:.1f} minutes")
    print(f"🧠 Model adapter: {results['adapter_path']}")
    print(f"📊 Top authors: {list(results['storage_stats']['top_authors'].keys())[:5]}")

if __name__ == "__main__":
    main() 