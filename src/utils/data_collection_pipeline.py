"""Data collection pipeline for efficient tweet retrieval and processing."""

import logging
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .x_api_client import XAPIClient, TweetData
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config_manager

logger = logging.getLogger(__name__)


class DataCollectionPipeline:
    """Efficient data collection pipeline for X API integration."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize data collection pipeline."""
        self.config = config or config_manager
        self.x_api = XAPIClient()
        
        # Set up data storage
        self.data_dir = Path("data/collected")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection statistics
        self.collection_stats = {
            "tweets_collected": 0,
            "kols_processed": 0,
            "api_calls_made": 0,
            "errors": 0,
            "start_time": None,
            "last_collection": None
        }
        
        logger.info("Data collection pipeline initialized")
    
    def collect_tweets(self, 
                      kol_usernames: List[str] = None,
                      tweets_per_kol: int = 50,
                      max_total_tweets: int = 500,
                      crypto_keywords_only: bool = True,
                      min_engagement: int = 0,
                      save_to_file: bool = True) -> Dict[str, Any]:
        """
        Unified tweet collection method from focused KOLs.
        
        Args:
            kol_usernames: List of KOL usernames to collect from (defaults to focused KOLs)
            tweets_per_kol: Number of tweets to collect per KOL
            max_total_tweets: Maximum total tweets to collect
            crypto_keywords_only: Whether to filter for crypto-related tweets only
            min_engagement: Minimum engagement threshold (likes + retweets + replies)
            save_to_file: Whether to save collected data to files
            
        Returns:
            Dictionary with collected tweets and metadata
        """
        # Get focused KOL list if not specified
        if kol_usernames is None:
            from src.config.config_manager import config_manager
            kol_usernames = config_manager.get_crypto_kols("primary")
        
        logger.info(f"Starting unified tweet collection from {len(kol_usernames)} focused KOLs")
        logger.info(f"Settings: {tweets_per_kol} tweets/KOL, max {max_total_tweets} total, crypto_only={crypto_keywords_only}, min_engagement={min_engagement}")
        
        self.collection_stats["start_time"] = datetime.utcnow()
        
        # Test API connection first
        connection_test = self.x_api.test_connection()
        if not connection_test["connected"]:
            logger.error("X API not connected. Aborting collection.")
            return {"kol_data": {}, "total_tweets": 0, "collection_stats": self.collection_stats}
        
        logger.info(f"API capabilities: {connection_test['capabilities']}")
        
        # Batch get user IDs to optimize API usage
        logger.info("Optimizing user ID lookups...")
        user_ids = self.x_api.batch_get_user_ids(kol_usernames)
        logger.info(f"Resolved {len(user_ids)}/{len(kol_usernames)} user IDs from cache/API")
        
        collected_data = {}
        total_tweets_collected = 0
        
        # Process each KOL
        for i, username in enumerate(kol_usernames, 1):
            if username not in user_ids:
                logger.warning(f"Skipping @{username} - user ID not found")
                continue
                
            if total_tweets_collected >= max_total_tweets:
                logger.info(f"Reached max total tweets ({max_total_tweets}). Stopping collection.")
                break
                
            logger.info(f"Processing KOL {i}/{len(kol_usernames)}: @{username}")
            
            try:
                # Determine collection method based on settings
                if crypto_keywords_only:
                    # Use search method for crypto-specific tweets
                    tweets = self._collect_crypto_tweets_from_kol(username, tweets_per_kol, min_engagement)
                else:
                    # Use direct user tweets method for all tweets
                    tweets = self.x_api.get_user_tweets(
                        username=username,
                        max_results=tweets_per_kol,
                        exclude_replies=True,
                        exclude_retweets=True
                    )
                    # Apply engagement filter if needed
                    if min_engagement > 0:
                        tweets = self._filter_by_engagement(tweets, min_engagement)
                
                if tweets:
                    collected_data[username] = tweets
                    self.collection_stats["tweets_collected"] += len(tweets)
                    total_tweets_collected += len(tweets)
                    logger.info(f"Collected {len(tweets)} tweets from @{username}")
                else:
                    logger.warning(f"No tweets collected from @{username}")
                
                self.collection_stats["api_calls_made"] += 1
                self.collection_stats["kols_processed"] += 1
                
                # Rate limiting delay
                if i < len(kol_usernames) and total_tweets_collected < max_total_tweets:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error collecting from @{username}: {e}")
                self.collection_stats["errors"] += 1
                continue
        
        self.collection_stats["last_collection"] = datetime.utcnow()
        
        # Save to file if requested
        if save_to_file and collected_data:
            self._save_collection_data(collected_data, "unified_collection")
        
        logger.info(f"Unified collection completed. Total tweets: {total_tweets_collected}, Stats: {self.collection_stats}")
        
        return {
            "kol_data": collected_data,
            "total_tweets": total_tweets_collected,
            "kols_processed": len(collected_data),
            "collection_stats": self.collection_stats.copy()
        }
    
    def _collect_crypto_tweets_from_kol(self, username: str, max_tweets: int, min_engagement: int = 0) -> List[TweetData]:
        """Helper method to collect crypto tweets from a specific KOL."""
        from src.config.config_manager import config_manager
        keywords = config_manager.get_search_queries("trending_crypto")
        
        all_tweets = []
        
        for keyword in keywords:
            try:
                # Search ONLY from this specific KOL
                query = f"from:{username} ({keyword}) -is:retweet -is:reply lang:en"
                
                tweets = self.x_api.search_recent_tweets(
                    query=query,
                    max_results=max(10, min(max_tweets // len(keywords), 20))
                )
                
                if tweets:
                    all_tweets.extend(tweets)
                
                self.collection_stats["api_calls_made"] += 1
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error searching for @{username}: {e}")
                self.collection_stats["errors"] += 1
                continue
        
        # Remove duplicates and apply engagement filter
        unique_tweets = {}
        for tweet in all_tweets:
            unique_tweets[tweet.id] = tweet
        
        final_tweets = list(unique_tweets.values())
        
        # Apply engagement filter if needed
        if min_engagement > 0:
            final_tweets = self._filter_by_engagement(final_tweets, min_engagement)
        
        return final_tweets
    
    def _filter_by_engagement(self, tweets: List[TweetData], min_engagement: int) -> List[TweetData]:
        """Helper method to filter tweets by engagement threshold."""
        filtered_tweets = []
        
        for tweet in tweets:
            metrics = tweet.public_metrics
            total_engagement = (
                metrics.get('like_count', 0) + 
                metrics.get('retweet_count', 0) + 
                metrics.get('reply_count', 0)
            )
            
            if total_engagement >= min_engagement:
                filtered_tweets.append(tweet)
        
        return filtered_tweets
    

    
    def run_comprehensive_collection(self, 
                                   kol_usernames: List[str] = None,
                                   max_total_tweets: int = 500,
                                   crypto_keywords_only: bool = True,
                                   min_engagement: int = 0) -> Dict[str, Any]:
        """
        Run unified data collection from focused KOLs.
        
        Args:
            kol_usernames: List of KOL usernames (uses default focused KOLs if None)
            max_total_tweets: Maximum total tweets to collect
            crypto_keywords_only: Whether to filter for crypto-related tweets only
            min_engagement: Minimum engagement threshold
            
        Returns:
            Dictionary with collected data and statistics
        """
        logger.info("Starting unified comprehensive data collection")
        
        # Use the unified collection method
        results = self.collect_tweets(
            kol_usernames=kol_usernames,
            tweets_per_kol=50,
            max_total_tweets=max_total_tweets,
            crypto_keywords_only=crypto_keywords_only,
            min_engagement=min_engagement,
            save_to_file=True
        )
        
        # Add timestamp and additional metadata
        results["timestamp"] = datetime.utcnow().isoformat()
        results["collection_type"] = "unified_comprehensive"
        
        logger.info("Unified comprehensive data collection completed")
        logger.info(f"Final stats: {results['collection_stats']}")
        
        return results
    
    def get_kol_performance_analysis(self, usernames: List[str] = None) -> Dict[str, Any]:
        """
        Analyze KOL performance using the X API client.
        
        Args:
            usernames: List of usernames to analyze (uses default if None)
            
        Returns:
            Dictionary with KOL performance analysis
        """
        if usernames is None:
            from src.config.config_manager import config_manager
            usernames = config_manager.get_crypto_kols("primary")[:5]  # Analyze top 5
        
        logger.info(f"Running KOL performance analysis for {len(usernames)} accounts")
        
        analysis = self.x_api.get_kol_analysis(usernames)
        
        # Save analysis results
        if analysis:
            self._save_collection_data(
                {"kol_analysis": analysis}, 
                "kol_performance_analysis"
            )
        
        return analysis
    
    def track_tweet_engagement_over_time(self, tweet_ids: List[str]) -> Dict[str, Any]:
        """
        Track engagement metrics for specific tweets over time.
        
        Args:
            tweet_ids: List of tweet IDs to track
            
        Returns:
            Dictionary with engagement tracking data
        """
        logger.info(f"Tracking engagement for {len(tweet_ids)} tweets")
        
        engagement_data = self.x_api.track_engagement_metrics(tweet_ids)
        
        # Save tracking data
        if engagement_data:
            self._save_collection_data(
                {"engagement_tracking": engagement_data},
                "engagement_tracking"
            )
        
        return engagement_data
    
    def _save_collection_data(self, data: Dict[str, Any], collection_type: str):
        """Save collected data to JSON files."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{collection_type}_{timestamp}.json"
            filepath = self.data_dir / filename
            
            # Convert TweetData objects to dictionaries for JSON serialization
            serializable_data = self._prepare_for_serialization(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved collection data to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving collection data: {e}")
    
    def _prepare_for_serialization(self, data: Any) -> Any:
        """Prepare data for JSON serialization by converting objects to dicts."""
        if isinstance(data, TweetData):
            return asdict(data)
        elif isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        stats = self.collection_stats.copy()
        
        # Add rate limit information
        rate_limit_status = self.x_api.get_rate_limit_status()
        stats["rate_limits"] = rate_limit_status
        
        # Add API connection status
        connection_status = self.x_api.test_connection()
        stats["api_status"] = connection_status
        
        return stats
    
    def reset_statistics(self):
        """Reset collection statistics."""
        self.collection_stats = {
            "tweets_collected": 0,
            "kols_processed": 0,
            "api_calls_made": 0,
            "errors": 0,
            "start_time": None,
            "last_collection": None
        }
        logger.info("Collection statistics reset") 