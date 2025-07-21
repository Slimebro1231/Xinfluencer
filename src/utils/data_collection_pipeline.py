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
from config import Config

logger = logging.getLogger(__name__)


class DataCollectionPipeline:
    """Efficient data collection pipeline for X API integration."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize data collection pipeline."""
        self.config = config or Config()
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
    
    def collect_kol_data(self, kol_usernames: List[str], 
                        tweets_per_kol: int = 50,
                        save_to_file: bool = True) -> Dict[str, List[TweetData]]:
        """
        Collect recent tweets from specified KOLs.
        
        Args:
            kol_usernames: List of KOL usernames to collect from
            tweets_per_kol: Number of recent tweets to collect per KOL
            save_to_file: Whether to save collected data to files
            
        Returns:
            Dictionary mapping usernames to their collected tweets
        """
        logger.info(f"Starting KOL data collection for {len(kol_usernames)} accounts")
        self.collection_stats["start_time"] = datetime.utcnow()
        
        collected_data = {}
        
        # Test API connection first
        connection_test = self.x_api.test_connection()
        if not connection_test["connected"]:
            logger.error("X API not connected. Aborting collection.")
            return collected_data
        
        logger.info(f"API capabilities: {connection_test['capabilities']}")
        
        # Process each KOL
        for i, username in enumerate(kol_usernames, 1):
            logger.info(f"Processing KOL {i}/{len(kol_usernames)}: @{username}")
            
            try:
                # Get user tweets
                tweets = self.x_api.get_user_tweets(
                    username=username,
                    max_results=tweets_per_kol,
                    exclude_replies=True,
                    exclude_retweets=True
                )
                
                if tweets:
                    collected_data[username] = tweets
                    self.collection_stats["tweets_collected"] += len(tweets)
                    logger.info(f"Collected {len(tweets)} tweets from @{username}")
                else:
                    logger.warning(f"No tweets collected from @{username}")
                
                self.collection_stats["api_calls_made"] += 1
                self.collection_stats["kols_processed"] += 1
                
                # Rate limiting delay (be respectful)
                if i < len(kol_usernames):  # Don't delay after last KOL
                    time.sleep(1.5)  # Small delay between requests
                    
            except Exception as e:
                logger.error(f"Error collecting from @{username}: {e}")
                self.collection_stats["errors"] += 1
                continue
        
        self.collection_stats["last_collection"] = datetime.utcnow()
        
        # Save to file if requested
        if save_to_file and collected_data:
            self._save_collection_data(collected_data, "kol_collection")
        
        logger.info(f"KOL collection completed. Stats: {self.collection_stats}")
        return collected_data
    
    def collect_trending_crypto_tweets(self, max_tweets: int = 100,
                                     keywords: List[str] = None,
                                     save_to_file: bool = True) -> List[TweetData]:
        """
        Collect trending crypto-related tweets.
        
        Args:
            max_tweets: Maximum number of tweets to collect
            keywords: Specific crypto keywords to search for
            save_to_file: Whether to save collected data to files
            
        Returns:
            List of collected crypto tweets
        """
        if keywords is None:
            keywords = [
                "crypto", "bitcoin", "ethereum", "DeFi", "RWA", 
                "blockchain", "altcoin", "NFT", "web3"
            ]
        
        logger.info(f"Collecting trending crypto tweets with keywords: {keywords}")
        
        all_tweets = []
        
        # Search for each keyword
        for keyword in keywords:
            try:
                # Build search query
                query = f"{keyword} -is:retweet -is:reply lang:en"
                
                logger.info(f"Searching for: {query}")
                
                tweets = self.x_api.search_recent_tweets(
                    query=query,
                    max_results=min(max_tweets // len(keywords), 100)
                )
                
                if tweets:
                    all_tweets.extend(tweets)
                    logger.info(f"Found {len(tweets)} tweets for '{keyword}'")
                
                self.collection_stats["api_calls_made"] += 1
                
                # Small delay between searches
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching for '{keyword}': {e}")
                self.collection_stats["errors"] += 1
                continue
        
        # Remove duplicates by tweet ID
        unique_tweets = {}
        for tweet in all_tweets:
            unique_tweets[tweet.id] = tweet
        
        final_tweets = list(unique_tweets.values())
        
        self.collection_stats["tweets_collected"] += len(final_tweets)
        
        # Save to file if requested
        if save_to_file and final_tweets:
            self._save_collection_data({"trending": final_tweets}, "crypto_trending")
        
        logger.info(f"Collected {len(final_tweets)} unique trending crypto tweets")
        return final_tweets
    
    def collect_high_engagement_tweets(self, min_engagement: int = 50,
                                     max_tweets: int = 100,
                                     save_to_file: bool = True) -> List[TweetData]:
        """
        Collect high-engagement crypto tweets.
        
        Args:
            min_engagement: Minimum engagement threshold (likes + retweets)
            max_tweets: Maximum number of tweets to collect
            save_to_file: Whether to save collected data to files
            
        Returns:
            List of high-engagement tweets
        """
        logger.info(f"Collecting high-engagement crypto tweets (min_engagement={min_engagement})")
        
        # Search for crypto tweets with engagement filters
        query = "crypto OR bitcoin OR ethereum min_faves:20 min_retweets:5 -is:retweet lang:en"
        
        try:
            tweets = self.x_api.search_recent_tweets(
                query=query,
                max_results=max_tweets
            )
            
            # Filter by engagement threshold
            high_engagement_tweets = []
            for tweet in tweets:
                metrics = tweet.public_metrics
                total_engagement = (
                    metrics.get('like_count', 0) + 
                    metrics.get('retweet_count', 0) + 
                    metrics.get('reply_count', 0)
                )
                
                if total_engagement >= min_engagement:
                    high_engagement_tweets.append(tweet)
            
            self.collection_stats["tweets_collected"] += len(high_engagement_tweets)
            self.collection_stats["api_calls_made"] += 1
            
            # Save to file if requested
            if save_to_file and high_engagement_tweets:
                self._save_collection_data(
                    {"high_engagement": high_engagement_tweets}, 
                    "high_engagement"
                )
            
            logger.info(f"Collected {len(high_engagement_tweets)} high-engagement tweets")
            return high_engagement_tweets
            
        except Exception as e:
            logger.error(f"Error collecting high-engagement tweets: {e}")
            self.collection_stats["errors"] += 1
            return []
    
    def run_comprehensive_collection(self, 
                                   kol_usernames: List[str] = None,
                                   collect_trending: bool = True,
                                   collect_high_engagement: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive data collection across multiple sources.
        
        Args:
            kol_usernames: List of KOL usernames (uses default if None)
            collect_trending: Whether to collect trending tweets
            collect_high_engagement: Whether to collect high-engagement tweets
            
        Returns:
            Dictionary with all collected data and statistics
        """
        logger.info("Starting comprehensive data collection")
        
        # Use default KOL list if none provided
        if kol_usernames is None:
            kol_usernames = self.config.crypto_kols[:10]  # Limit to first 10 for testing
        
        collection_results = {
            "kol_data": {},
            "trending_tweets": [],
            "high_engagement_tweets": [],
            "collection_stats": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 1. Collect KOL data
        if kol_usernames:
            logger.info("Phase 1: Collecting KOL data")
            collection_results["kol_data"] = self.collect_kol_data(
                kol_usernames=kol_usernames,
                tweets_per_kol=20  # Reduced for comprehensive collection
            )
        
        # 2. Collect trending tweets
        if collect_trending:
            logger.info("Phase 2: Collecting trending crypto tweets")
            collection_results["trending_tweets"] = self.collect_trending_crypto_tweets(
                max_tweets=50
            )
        
        # 3. Collect high-engagement tweets
        if collect_high_engagement:
            logger.info("Phase 3: Collecting high-engagement tweets")
            collection_results["high_engagement_tweets"] = self.collect_high_engagement_tweets(
                min_engagement=30,
                max_tweets=30
            )
        
        # Update final statistics
        collection_results["collection_stats"] = self.collection_stats.copy()
        
        # Save comprehensive results
        self._save_collection_data(collection_results, "comprehensive_collection")
        
        logger.info("Comprehensive data collection completed")
        logger.info(f"Final stats: {self.collection_stats}")
        
        return collection_results
    
    def get_kol_performance_analysis(self, usernames: List[str] = None) -> Dict[str, Any]:
        """
        Analyze KOL performance using the X API client.
        
        Args:
            usernames: List of usernames to analyze (uses default if None)
            
        Returns:
            Dictionary with KOL performance analysis
        """
        if usernames is None:
            usernames = self.config.crypto_kols[:5]  # Analyze top 5
        
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