"""Engagement tracking system for monitoring tweet performance over time."""

import logging
import sqlite3
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from .x_api_client import XAPIClient, TweetData

logger = logging.getLogger(__name__)


@dataclass
class EngagementSnapshot:
    """Snapshot of engagement metrics at a specific time."""
    tweet_id: str
    timestamp: datetime
    likes: int
    retweets: int
    replies: int
    quotes: int
    impressions: Optional[int] = None
    engagement_rate: Optional[float] = None
    velocity: Optional[float] = None  # Engagement per hour


@dataclass
class EngagementTrend:
    """Engagement trend analysis for a tweet."""
    tweet_id: str
    total_snapshots: int
    first_snapshot: datetime
    last_snapshot: datetime
    peak_engagement: int
    current_engagement: int
    average_velocity: float
    is_trending_up: bool
    engagement_patterns: Dict[str, Any]


class EngagementTracker:
    """Track and analyze tweet engagement over time."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize engagement tracker."""
        self.db_path = db_path or "data/engagement/engagement_tracker.db"
        self.x_api = XAPIClient()
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db = self._init_database()
        
        # Tracking configuration
        self.tracking_config = {
            "snapshot_interval_minutes": 30,  # Take snapshot every 30 minutes
            "max_tracking_days": 7,  # Track for maximum 7 days
            "min_engagement_threshold": 5,  # Minimum engagement to continue tracking
            "velocity_window_hours": 4  # Calculate velocity over 4-hour windows
        }
        
        # Active tracking threads
        self.tracking_threads = {}
        self.stop_tracking = threading.Event()
        
        logger.info("Engagement tracker initialized")
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for engagement tracking."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engagement_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                likes INTEGER NOT NULL,
                retweets INTEGER NOT NULL,
                replies INTEGER NOT NULL,
                quotes INTEGER NOT NULL,
                impressions INTEGER,
                engagement_rate REAL,
                velocity REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tweet_id, timestamp)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tracking_metadata (
                tweet_id TEXT PRIMARY KEY,
                author_username TEXT,
                tweet_text TEXT,
                tweet_created_at TEXT,
                tracking_started_at TEXT,
                tracking_ended_at TEXT,
                is_active BOOLEAN DEFAULT 1,
                total_snapshots INTEGER DEFAULT 0,
                peak_engagement INTEGER DEFAULT 0,
                source_collection TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engagement_analysis (
                tweet_id TEXT PRIMARY KEY,
                analysis_timestamp TEXT,
                trend_direction TEXT,
                average_velocity REAL,
                engagement_patterns TEXT,
                peak_time TEXT,
                decline_rate REAL,
                total_engagement INTEGER,
                analysis_data TEXT
            )
        """)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_tweet_id ON engagement_snapshots(tweet_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON engagement_snapshots(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_active ON tracking_metadata(is_active)")
        
        conn.commit()
        return conn
    
    def start_tracking_tweet(self, tweet_id: str, 
                           author_username: str = None,
                           tweet_text: str = None,
                           source_collection: str = "manual") -> bool:
        """
        Start tracking engagement for a specific tweet.
        
        Args:
            tweet_id: ID of the tweet to track
            author_username: Username of tweet author
            tweet_text: Text content of the tweet
            source_collection: Source of the tweet (e.g., "kol_collection", "trending")
            
        Returns:
            True if tracking started successfully
        """
        if tweet_id in self.tracking_threads:
            logger.warning(f"Tweet {tweet_id} is already being tracked")
            return False
        
        try:
            # Get initial tweet data if not provided
            if not author_username or not tweet_text:
                tweet_data = self._get_tweet_details(tweet_id)
                if tweet_data:
                    author_username = tweet_data.get('author_username', 'unknown')
                    tweet_text = tweet_data.get('text', '')
            
            # Record in metadata table
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tracking_metadata 
                (tweet_id, author_username, tweet_text, tweet_created_at, 
                 tracking_started_at, is_active, source_collection)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet_id, 
                author_username or 'unknown',
                tweet_text or '',
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                True,
                source_collection
            ))
            self.db.commit()
            
            # Take initial snapshot
            self._take_engagement_snapshot(tweet_id)
            
            # Start tracking thread
            thread = threading.Thread(
                target=self._track_tweet_continuously,
                args=(tweet_id,),
                daemon=True
            )
            thread.start()
            self.tracking_threads[tweet_id] = thread
            
            logger.info(f"Started tracking tweet {tweet_id} from @{author_username}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting tracking for tweet {tweet_id}: {e}")
            return False
    
    def start_tracking_batch(self, tweet_data_list: List[TweetData]) -> Dict[str, bool]:
        """
        Start tracking multiple tweets in batch.
        
        Args:
            tweet_data_list: List of TweetData objects to track
            
        Returns:
            Dictionary mapping tweet IDs to tracking start success
        """
        results = {}
        
        for tweet_data in tweet_data_list:
            success = self.start_tracking_tweet(
                tweet_id=tweet_data.id,
                author_username=tweet_data.author_username,
                tweet_text=tweet_data.text,
                source_collection="batch_tracking"
            )
            results[tweet_data.id] = success
            
            # Small delay between starting tracking threads
            time.sleep(0.5)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Started tracking {successful}/{len(tweet_data_list)} tweets")
        
        return results
    
    def stop_tracking_tweet(self, tweet_id: str) -> bool:
        """
        Stop tracking a specific tweet.
        
        Args:
            tweet_id: ID of the tweet to stop tracking
            
        Returns:
            True if tracking stopped successfully
        """
        try:
            # Update metadata
            cursor = self.db.cursor()
            cursor.execute("""
                UPDATE tracking_metadata 
                SET is_active = 0, tracking_ended_at = ?
                WHERE tweet_id = ?
            """, (datetime.utcnow().isoformat(), tweet_id))
            self.db.commit()
            
            # Remove from active threads
            if tweet_id in self.tracking_threads:
                del self.tracking_threads[tweet_id]
            
            logger.info(f"Stopped tracking tweet {tweet_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping tracking for tweet {tweet_id}: {e}")
            return False
    
    def _track_tweet_continuously(self, tweet_id: str):
        """Continuously track a tweet's engagement (runs in separate thread)."""
        interval_seconds = self.tracking_config["snapshot_interval_minutes"] * 60
        max_tracking_seconds = self.tracking_config["max_tracking_days"] * 24 * 3600
        
        start_time = time.time()
        last_snapshot_time = 0
        
        while not self.stop_tracking.is_set():
            current_time = time.time()
            
            # Check if we've exceeded maximum tracking time
            if current_time - start_time > max_tracking_seconds:
                logger.info(f"Maximum tracking time reached for tweet {tweet_id}")
                self.stop_tracking_tweet(tweet_id)
                break
            
            # Take snapshot if interval has passed
            if current_time - last_snapshot_time >= interval_seconds:
                try:
                    snapshot = self._take_engagement_snapshot(tweet_id)
                    if snapshot:
                        last_snapshot_time = current_time
                        
                        # Check if engagement is too low to continue tracking
                        total_engagement = (
                            snapshot.likes + snapshot.retweets + 
                            snapshot.replies + snapshot.quotes
                        )
                        
                        if total_engagement < self.tracking_config["min_engagement_threshold"]:
                            # Check if this is consistent low engagement
                            recent_snapshots = self._get_recent_snapshots(tweet_id, hours=2)
                            if len(recent_snapshots) >= 3 and all(
                                s.likes + s.retweets + s.replies + s.quotes < 
                                self.tracking_config["min_engagement_threshold"] 
                                for s in recent_snapshots
                            ):
                                logger.info(f"Low engagement detected for tweet {tweet_id}, stopping tracking")
                                self.stop_tracking_tweet(tweet_id)
                                break
                    
                except Exception as e:
                    logger.error(f"Error taking snapshot for tweet {tweet_id}: {e}")
            
            # Sleep for a short interval before checking again
            time.sleep(60)  # Check every minute
    
    def _take_engagement_snapshot(self, tweet_id: str) -> Optional[EngagementSnapshot]:
        """Take a snapshot of current engagement metrics."""
        try:
            # Get current metrics from X API
            engagement_data = self.x_api.track_engagement_metrics([tweet_id])
            
            if not engagement_data or tweet_id not in engagement_data:
                logger.warning(f"No engagement data retrieved for tweet {tweet_id}")
                return None
            
            metrics = engagement_data[tweet_id]["metrics"]
            timestamp = datetime.utcnow()
            
            # Create snapshot
            snapshot = EngagementSnapshot(
                tweet_id=tweet_id,
                timestamp=timestamp,
                likes=metrics.get('like_count', 0),
                retweets=metrics.get('retweet_count', 0),
                replies=metrics.get('reply_count', 0),
                quotes=metrics.get('quote_count', 0),
                impressions=metrics.get('impression_count')
            )
            
            # Calculate engagement rate and velocity
            total_engagement = snapshot.likes + snapshot.retweets + snapshot.replies + snapshot.quotes
            
            if snapshot.impressions and snapshot.impressions > 0:
                snapshot.engagement_rate = (total_engagement / snapshot.impressions) * 100
            
            # Calculate velocity (engagement per hour)
            previous_snapshot = self._get_latest_snapshot(tweet_id)
            if previous_snapshot:
                time_diff_hours = (timestamp - previous_snapshot.timestamp).total_seconds() / 3600
                if time_diff_hours > 0:
                    prev_total = (previous_snapshot.likes + previous_snapshot.retweets + 
                                previous_snapshot.replies + previous_snapshot.quotes)
                    engagement_diff = total_engagement - prev_total
                    snapshot.velocity = engagement_diff / time_diff_hours
            
            # Store in database
            self._store_snapshot(snapshot)
            
            # Update metadata
            cursor = self.db.cursor()
            cursor.execute("""
                UPDATE tracking_metadata 
                SET total_snapshots = total_snapshots + 1,
                    peak_engagement = MAX(peak_engagement, ?)
                WHERE tweet_id = ?
            """, (total_engagement, tweet_id))
            self.db.commit()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking engagement snapshot for tweet {tweet_id}: {e}")
            return None
    
    def _store_snapshot(self, snapshot: EngagementSnapshot):
        """Store engagement snapshot in database."""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO engagement_snapshots
            (tweet_id, timestamp, likes, retweets, replies, quotes, 
             impressions, engagement_rate, velocity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.tweet_id,
            snapshot.timestamp.isoformat(),
            snapshot.likes,
            snapshot.retweets,
            snapshot.replies,
            snapshot.quotes,
            snapshot.impressions,
            snapshot.engagement_rate,
            snapshot.velocity
        ))
        self.db.commit()
    
    def _get_latest_snapshot(self, tweet_id: str) -> Optional[EngagementSnapshot]:
        """Get the latest engagement snapshot for a tweet."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT * FROM engagement_snapshots 
            WHERE tweet_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (tweet_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return EngagementSnapshot(
            tweet_id=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            likes=row[3],
            retweets=row[4],
            replies=row[5],
            quotes=row[6],
            impressions=row[7],
            engagement_rate=row[8],
            velocity=row[9]
        )
    
    def _get_recent_snapshots(self, tweet_id: str, hours: int = 24) -> List[EngagementSnapshot]:
        """Get recent engagement snapshots for a tweet."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT * FROM engagement_snapshots 
            WHERE tweet_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (tweet_id, cutoff_time.isoformat()))
        
        snapshots = []
        for row in cursor.fetchall():
            snapshot = EngagementSnapshot(
                tweet_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                likes=row[3],
                retweets=row[4],
                replies=row[5],
                quotes=row[6],
                impressions=row[7],
                engagement_rate=row[8],
                velocity=row[9]
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def analyze_tweet_engagement(self, tweet_id: str) -> Optional[EngagementTrend]:
        """Analyze engagement trends for a specific tweet."""
        snapshots = self._get_recent_snapshots(tweet_id, hours=24*7)  # Last 7 days
        
        if len(snapshots) < 2:
            logger.warning(f"Insufficient data to analyze tweet {tweet_id}")
            return None
        
        # Calculate metrics
        engagements = []
        velocities = []
        timestamps = []
        
        for snapshot in snapshots:
            total_engagement = (snapshot.likes + snapshot.retweets + 
                              snapshot.replies + snapshot.quotes)
            engagements.append(total_engagement)
            if snapshot.velocity is not None:
                velocities.append(snapshot.velocity)
            timestamps.append(snapshot.timestamp)
        
        # Trend analysis
        peak_engagement = max(engagements)
        current_engagement = engagements[-1]
        average_velocity = np.mean(velocities) if velocities else 0
        
        # Determine trend direction (comparing recent vs earlier periods)
        if len(engagements) >= 6:
            recent_avg = np.mean(engagements[-3:])
            earlier_avg = np.mean(engagements[:3])
            is_trending_up = recent_avg > earlier_avg
        else:
            is_trending_up = engagements[-1] > engagements[0]
        
        # Engagement patterns analysis
        patterns = self._analyze_engagement_patterns(snapshots)
        
        trend = EngagementTrend(
            tweet_id=tweet_id,
            total_snapshots=len(snapshots),
            first_snapshot=snapshots[0].timestamp,
            last_snapshot=snapshots[-1].timestamp,
            peak_engagement=peak_engagement,
            current_engagement=current_engagement,
            average_velocity=average_velocity,
            is_trending_up=is_trending_up,
            engagement_patterns=patterns
        )
        
        # Store analysis
        self._store_engagement_analysis(trend)
        
        return trend
    
    def _analyze_engagement_patterns(self, snapshots: List[EngagementSnapshot]) -> Dict[str, Any]:
        """Analyze engagement patterns from snapshots."""
        if len(snapshots) < 3:
            return {}
        
        patterns = {}
        
        # Time-based patterns
        hour_engagement = {}
        for snapshot in snapshots:
            hour = snapshot.timestamp.hour
            total_eng = snapshot.likes + snapshot.retweets + snapshot.replies + snapshot.quotes
            if hour not in hour_engagement:
                hour_engagement[hour] = []
            hour_engagement[hour].append(total_eng)
        
        # Peak hours
        avg_by_hour = {h: np.mean(engs) for h, engs in hour_engagement.items() if engs}
        if avg_by_hour:
            peak_hour = max(avg_by_hour.keys(), key=lambda h: avg_by_hour[h])
            patterns['peak_hour'] = peak_hour
            patterns['peak_hour_avg_engagement'] = avg_by_hour[peak_hour]
        
        # Engagement acceleration/deceleration
        engagements = [s.likes + s.retweets + s.replies + s.quotes for s in snapshots]
        if len(engagements) >= 3:
            # Calculate second derivative to find acceleration
            velocities = np.diff(engagements)
            accelerations = np.diff(velocities)
            patterns['avg_acceleration'] = np.mean(accelerations)
            patterns['is_accelerating'] = np.mean(accelerations[-3:]) > 0 if len(accelerations) >= 3 else False
        
        # Engagement distribution by type
        total_likes = sum(s.likes for s in snapshots)
        total_retweets = sum(s.retweets for s in snapshots)
        total_replies = sum(s.replies for s in snapshots)
        total_quotes = sum(s.quotes for s in snapshots)
        total_all = total_likes + total_retweets + total_replies + total_quotes
        
        if total_all > 0:
            patterns['engagement_distribution'] = {
                'likes_pct': (total_likes / total_all) * 100,
                'retweets_pct': (total_retweets / total_all) * 100,
                'replies_pct': (total_replies / total_all) * 100,
                'quotes_pct': (total_quotes / total_all) * 100
            }
        
        return patterns
    
    def _store_engagement_analysis(self, trend: EngagementTrend):
        """Store engagement analysis in database."""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO engagement_analysis
            (tweet_id, analysis_timestamp, trend_direction, average_velocity,
             engagement_patterns, peak_time, total_engagement, analysis_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trend.tweet_id,
            datetime.utcnow().isoformat(),
            "up" if trend.is_trending_up else "down",
            trend.average_velocity,
            json.dumps(trend.engagement_patterns),
            trend.first_snapshot.isoformat(),
            trend.peak_engagement,
            json.dumps(asdict(trend), default=str)
        ))
        self.db.commit()
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """Get current tracking status and statistics."""
        cursor = self.db.cursor()
        
        # Active tracking count
        cursor.execute("SELECT COUNT(*) FROM tracking_metadata WHERE is_active = 1")
        active_count = cursor.fetchone()[0]
        
        # Total tracked tweets
        cursor.execute("SELECT COUNT(*) FROM tracking_metadata")
        total_count = cursor.fetchone()[0]
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM engagement_snapshots 
            WHERE timestamp >= datetime('now', '-1 hour')
        """)
        recent_snapshots = cursor.fetchone()[0]
        
        return {
            "active_tracking": active_count,
            "total_tracked": total_count,
            "recent_snapshots_1h": recent_snapshots,
            "active_threads": len(self.tracking_threads),
            "tracking_config": self.tracking_config
        }
    
    def get_top_performing_tweets(self, limit: int = 10, 
                                 time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Get top performing tweets by engagement."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT 
                m.tweet_id,
                m.author_username,
                m.tweet_text,
                m.peak_engagement,
                m.total_snapshots,
                MAX(s.likes + s.retweets + s.replies + s.quotes) as max_total_engagement,
                AVG(s.velocity) as avg_velocity
            FROM tracking_metadata m
            JOIN engagement_snapshots s ON m.tweet_id = s.tweet_id
            WHERE s.timestamp >= ?
            GROUP BY m.tweet_id
            ORDER BY max_total_engagement DESC
            LIMIT ?
        """, (cutoff_time.isoformat(), limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "tweet_id": row[0],
                "author_username": row[1],
                "tweet_text": row[2][:100] + "..." if len(row[2]) > 100 else row[2],
                "peak_engagement": row[3],
                "total_snapshots": row[4],
                "max_total_engagement": row[5],
                "avg_velocity": row[6] or 0
            })
        
        return results
    
    def _get_tweet_details(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get tweet details from X API."""
        try:
            response = self.x_api.client.get_tweet(
                id=tweet_id,
                tweet_fields=['author_id', 'created_at', 'text'],
                expansions=['author_id'],
                user_fields=['username']
            )
            
            if response and response.data:
                tweet = response.data
                author_username = "unknown"
                
                if hasattr(response, 'includes') and response.includes:
                    if 'users' in response.includes:
                        author = response.includes['users'][0]
                        author_username = author.username
                
                return {
                    'text': tweet.text,
                    'author_username': author_username,
                    'created_at': tweet.created_at.isoformat() if tweet.created_at else None
                }
            
        except Exception as e:
            logger.error(f"Error getting tweet details for {tweet_id}: {e}")
        
        return None
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old tracking data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        cursor = self.db.cursor()
        
        # Delete old snapshots
        cursor.execute("""
            DELETE FROM engagement_snapshots 
            WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))
        
        # Delete old analysis
        cursor.execute("""
            DELETE FROM engagement_analysis 
            WHERE analysis_timestamp < ?
        """, (cutoff_date.isoformat(),))
        
        deleted_snapshots = cursor.rowcount
        self.db.commit()
        
        logger.info(f"Cleaned up {deleted_snapshots} old engagement records")
    
    def stop_all_tracking(self):
        """Stop all active tracking."""
        self.stop_tracking.set()
        
        # Update all active metadata
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE tracking_metadata 
            SET is_active = 0, tracking_ended_at = ?
            WHERE is_active = 1
        """, (datetime.utcnow().isoformat(),))
        self.db.commit()
        
        # Clear threads
        self.tracking_threads.clear()
        
        logger.info("Stopped all engagement tracking") 