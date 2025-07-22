"""
Enhanced Data Collection Pipeline
Integrates safe collection features with existing data collection infrastructure
"""

import logging
import json
import time
import sqlite3
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict

from .data_collection_pipeline import DataCollectionPipeline
from .x_api_client import XAPIClient, TweetData
from .crypto_analyzer import CryptoIdentityAnalyzer
from ..config import Config

logger = logging.getLogger(__name__)


class APIUsageSafeguard:
    """Enhanced API usage safeguard integrated with existing infrastructure."""
    
    def __init__(self, usage_file: str = "data/api_usage_log.json"):
        self.usage_file = Path(usage_file)
        self.usage_file.parent.mkdir(exist_ok=True)
        
        # Post-focused limits
        self.limits = {
            "posts_per_hour": 1000,
            "posts_per_day": 10000,
            "api_calls_per_15min": {
                "search": 450,  # Buffer from 500 limit
                "user_lookup": 90   # Buffer from 100 limit
            }
        }
        
        logger.info("API usage safeguard initialized")
    
    def check_post_limits(self) -> Dict[str, Any]:
        """Check if collection is allowed based on post limits."""
        current_usage = self.get_usage_summary()
        
        can_collect = (
            current_usage["posts_last_hour"] < self.limits["posts_per_hour"] and
            current_usage["posts_last_day"] < self.limits["posts_per_day"]
        )
        
        return {
            "can_collect": can_collect,
            "posts_last_hour": current_usage["posts_last_hour"],
            "posts_last_day": current_usage["posts_last_day"],
            "hourly_limit": self.limits["posts_per_hour"],
            "daily_limit": self.limits["posts_per_day"]
        }
    
    def record_collection(self, posts_collected: int, api_calls_made: Dict[str, int], success: bool = True):
        """Record a collection session."""
        session_record = {
            "timestamp": datetime.now().isoformat(),
            "posts_collected": posts_collected,
            "api_calls_made": api_calls_made,
            "success": success
        }
        
        # Load existing records
        records = []
        if self.usage_file.exists():
            try:
                with open(self.usage_file) as f:
                    records = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load usage log: {e}")
        
        # Add new record
        records.append(session_record)
        
        # Keep only last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        records = [r for r in records if datetime.fromisoformat(r["timestamp"]) > cutoff]
        
        # Save updated records
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(records, f, indent=2)
        except Exception as e:
                logger.error(f"Failed to save usage log: {e}")
    
    def get_usage_summary(self) -> Dict[str, int]:
        """Get usage summary for recent periods."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        posts_last_hour = 0
        posts_last_day = 0
        
        if self.usage_file.exists():
            try:
                with open(self.usage_file) as f:
                    records = json.load(f)
                
                for record in records:
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    posts = record.get("posts_collected", 0)
                    
                    if timestamp > hour_ago:
                        posts_last_hour += posts
                    if timestamp > day_ago:
                        posts_last_day += posts
                        
            except Exception as e:
                logger.error(f"Failed to read usage log: {e}")
        
        return {
            "posts_last_hour": posts_last_hour,
            "posts_last_day": posts_last_day
        }


class TrainingDataStorage:
    """Unified training data storage integrated with existing data structure."""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.training_dir = self.base_dir / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for training data
        self.db_path = self.training_dir / "posts.db"
        self._init_database()
        
        logger.info(f"Training data storage initialized at {self.training_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for training posts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_posts (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            author TEXT,
            engagement_score REAL,
            crypto_relevance REAL,
            quality_score REAL,
            source TEXT,
            collection_session TEXT,
            timestamp TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_author ON training_posts(author)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON training_posts(quality_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crypto_relevance ON training_posts(crypto_relevance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON training_posts(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON training_posts(collection_session)")
        
        conn.commit()
        conn.close()
    
    def store_collection_for_training(self, collected_data: Dict[str, Any], session_id: str):
        """Store collected data for training purposes."""
        analyzer = CryptoIdentityAnalyzer()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        
        # Process tweets from collection
        all_tweets = []
        
        # Handle different collection formats
        if "tweets" in collected_data:
            # Direct tweet list
            all_tweets = collected_data["tweets"]
        elif "kol_data" in collected_data:
            # KOL-based collection
            for username, tweets in collected_data["kol_data"].items():
                all_tweets.extend(tweets)
        
        for tweet in all_tweets:
            try:
                # Extract data
                if isinstance(tweet, dict):
                    tweet_id = tweet.get("id", f"unknown_{stored_count}")
                    text = tweet.get("text", "")
                    author = tweet.get("author_username", tweet.get("username", "unknown"))
                    public_metrics = tweet.get("public_metrics", {})
                else:
                    # Handle TweetData objects
                    tweet_id = tweet.id
                    text = tweet.text
                    author = tweet.username
                    public_metrics = tweet.public_metrics
                
                # Calculate scores
                engagement_score = (
                    public_metrics.get("like_count", 0) +
                    public_metrics.get("retweet_count", 0) * 2 +
                    public_metrics.get("reply_count", 0)
                ) / 100.0
                
                crypto_relevance = analyzer.analyze_crypto_relevance(text)
                quality_score = analyzer.analyze_content_quality(text, public_metrics)
                
                # Store in database
                cursor.execute("""
                INSERT OR REPLACE INTO training_posts 
                (id, text, author, engagement_score, crypto_relevance, quality_score, 
                 source, collection_session, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tweet_id, text, author, min(engagement_score, 1.0),
                    crypto_relevance, quality_score, "enhanced_collection",
                    session_id, datetime.now().isoformat(),
                    json.dumps(public_metrics)
                ))
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store tweet {tweet_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {stored_count} posts for training in session {session_id}")
        return stored_count
    
    def get_training_data_stats(self) -> Dict[str, Any]:
        """Get statistics about stored training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total posts
            cursor.execute("SELECT COUNT(*) FROM training_posts")
            total_posts = cursor.fetchone()[0]
            
            # High quality posts
            cursor.execute("SELECT COUNT(*) FROM training_posts WHERE quality_score > 0.7")
            high_quality = cursor.fetchone()[0]
            
            # High crypto relevance
            cursor.execute("SELECT COUNT(*) FROM training_posts WHERE crypto_relevance > 0.8")
            high_crypto = cursor.fetchone()[0]
            
            # Top authors
            cursor.execute("SELECT author, COUNT(*) FROM training_posts GROUP BY author ORDER BY COUNT(*) DESC LIMIT 5")
            top_authors = dict(cursor.fetchall())
            
            # Recent sessions
            cursor.execute("SELECT collection_session, COUNT(*) FROM training_posts GROUP BY collection_session ORDER BY created_at DESC LIMIT 5")
            recent_sessions = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_posts": total_posts,
                "high_quality_posts": high_quality,
                "high_crypto_relevance": high_crypto,
                "top_authors": top_authors,
                "recent_sessions": recent_sessions
            }
            
        except Exception as e:
            conn.close()
            logger.error(f"Failed to get training data stats: {e}")
            return {"error": str(e)}


class EnhancedDataCollectionPipeline(DataCollectionPipeline):
    """Enhanced data collection pipeline with safe collection and training integration."""
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        
        # Add safeguard and training storage
        self.safeguard = APIUsageSafeguard()
        self.training_storage = TrainingDataStorage()
        
        logger.info("Enhanced data collection pipeline initialized")
    
    def safe_collect_crypto_content(self, target_posts: int = 500, 
                                  save_for_training: bool = True) -> Dict[str, Any]:
        """
        Safely collect crypto content with training integration.
        Combines the existing comprehensive collection with safe collection features.
        """
        # Check limits before starting
        limits_check = self.safeguard.check_post_limits()
        if not limits_check["can_collect"]:
            logger.warning("Collection not allowed due to limits")
            return {
                "success": False,
                "error": "Post limits exceeded",
                "limits": limits_check
            }
        
        logger.info(f"Starting safe collection (target: {target_posts} posts)")
        start_time = datetime.now()
        session_id = start_time.strftime("%Y%m%d_%H%M%S")
        
        # Use existing comprehensive collection with optimized parameters
        collection_results = self.run_comprehensive_collection(
            kol_usernames=self.config.crypto_kols[:10],  # Top 10 KOLs
            collect_trending=True,
            collect_high_engagement=True
        )
        
        # Calculate actual posts collected
        total_posts = 0
        if collection_results["kol_data"]:
            for tweets in collection_results["kol_data"].values():
                total_posts += len(tweets)
        total_posts += len(collection_results["trending_tweets"])
        total_posts += len(collection_results["high_engagement_tweets"])
        
        # Record usage
        api_calls_made = collection_results["collection_stats"].get("api_calls_made", 0)
        self.safeguard.record_collection(
            posts_collected=total_posts,
            api_calls_made={"search": api_calls_made, "user_lookup": 0},
            success=total_posts > 0
        )
        
        # Store for training if requested
        training_posts_stored = 0
        if save_for_training and total_posts > 0:
            training_posts_stored = self.training_storage.store_collection_for_training(
                collection_results, session_id
            )
        
        # Enhanced results
        enhanced_results = {
            **collection_results,
            "session_id": session_id,
            "total_posts_collected": total_posts,
            "training_posts_stored": training_posts_stored,
            "collection_efficiency": total_posts / max(api_calls_made, 1),
            "safeguard_status": self.safeguard.get_usage_summary(),
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }
        
        logger.info(f"Safe collection completed: {total_posts} posts, {training_posts_stored} stored for training")
        return enhanced_results 