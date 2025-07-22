#!/usr/bin/env python3
"""
Historic Tweet Collector for High-Quality A/B Testing Data.
Collects tweets from ~1 week ago with significant engagement.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.x_api_client import XAPIClient, TweetData
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricTweetCollector:
    """Collect historic tweets with meaningful engagement for A/B testing."""
    
    def __init__(self):
        self.config = Config()
        self.x_api = XAPIClient()
        self.collected_tweets = []
        
        # Primary focus KOL list (from focused_kol_list.py)
        self.primary_kols = [
            "VitalikButerin",      # Ethereum founder
            "haydenzadams",        # Uniswap founder  
            "stani_kulechov",      # Aave founder
            "AndreCronjeTech",     # DeFi architect
            "rleshner",            # Compound founder
            "bantg",               # Yearn core developer
            "centrifuge",          # RWA tokenization
            "MakerDAO",            # DAI and RWA
            "chainlink",           # Oracle infrastructure
            "MessariCrypto",       # Research
            "DeFiPulse",           # Analytics
            "defiprime",           # Education
            "evan_van_ness",       # Ethereum weekly
            "sassal0x",            # Technical analysis
            "tokenbrice"           # DeFi educator
        ]
        
        # Engagement thresholds
        self.min_likes = 10
        self.min_engagement_rate = 0.001  # 0.1% minimum engagement rate
        
    def calculate_engagement_rate(self, tweet: TweetData, follower_count: int) -> float:
        """Calculate engagement rate normalized by follower count."""
        try:
            metrics = tweet.public_metrics
            total_engagement = (
                metrics.get('like_count', 0) +
                metrics.get('retweet_count', 0) * 3 +  # Retweets worth 3x
                metrics.get('reply_count', 0) * 2 +    # Replies worth 2x
                metrics.get('quote_count', 0) * 2      # Quotes worth 2x
            )
            
            engagement_rate = total_engagement / max(follower_count, 1)
            return engagement_rate
            
        except Exception as e:
            logger.warning(f"Error calculating engagement rate: {e}")
            return 0.0
    
    def is_high_quality_tweet(self, tweet: TweetData, author_followers: int) -> bool:
        """Filter for high-quality tweets suitable for A/B testing."""
        try:
            metrics = tweet.public_metrics
            likes = metrics.get('like_count', 0)
            
            # Must have minimum likes
            if likes < self.min_likes:
                return False
            
            # Must have minimum engagement rate
            engagement_rate = self.calculate_engagement_rate(tweet, author_followers)
            if engagement_rate < self.min_engagement_rate:
                return False
            
            # Must be original content (not just retweets)
            if tweet.text.startswith('RT @'):
                return False
            
            # Must be crypto/DeFi relevant
            crypto_keywords = [
                'bitcoin', 'ethereum', 'defi', 'crypto', 'blockchain', 
                'rwa', 'token', 'protocol', 'dapp', 'yield', 'liquidity'
            ]
            text_lower = tweet.text.lower()
            if not any(keyword in text_lower for keyword in crypto_keywords):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error filtering tweet quality: {e}")
            return False
    
    def collect_historic_tweets_for_kol(self, username: str, target_count: int = 20) -> List[TweetData]:
        """Collect historic tweets from a specific KOL."""
        logger.info(f"Collecting historic tweets from @{username}")
        
        try:
            # Get user info for follower count
            user_info = self.x_api.get_user_by_username(username)
            if not user_info:
                logger.warning(f"Could not get user info for @{username}")
                return []
            
            follower_count = user_info.get('public_metrics', {}).get('followers_count', 1000)
            
            # Calculate date range (1 week ago, with some buffer)
            end_time = datetime.utcnow() - timedelta(days=5)  # 5 days ago
            start_time = end_time - timedelta(days=7)  # 12 days ago to 5 days ago
            
            # Get tweets from that period
            all_tweets = self.x_api.get_user_tweets(
                username=username,
                max_results=100,  # Get more to filter from
                exclude_replies=True,
                exclude_retweets=True,
                start_time=start_time,
                end_time=end_time
            )
            
            if not all_tweets:
                logger.warning(f"No tweets found for @{username} in date range")
                return []
            
            # Filter for high-quality tweets
            quality_tweets = []
            for tweet in all_tweets:
                if self.is_high_quality_tweet(tweet, follower_count):
                    quality_tweets.append(tweet)
                    
                    # Add engagement rate for later sorting
                    tweet.engagement_rate = self.calculate_engagement_rate(tweet, follower_count)
            
            # Sort by engagement rate and take top tweets
            quality_tweets.sort(key=lambda t: getattr(t, 'engagement_rate', 0), reverse=True)
            selected_tweets = quality_tweets[:target_count]
            
            logger.info(f"@{username}: {len(selected_tweets)}/{len(all_tweets)} tweets selected (follower_count: {follower_count:,})")
            
            return selected_tweets
            
        except Exception as e:
            logger.error(f"Error collecting from @{username}: {e}")
            return []
    
    def collect_historic_dataset(self, target_total: int = 500) -> Dict:
        """Collect historic tweets from all primary KOLs."""
        logger.info(f"Starting historic tweet collection (target: {target_total} tweets)")
        
        # Test API connection
        connection_test = self.x_api.test_connection()
        if not connection_test["connected"]:
            logger.error("X API not connected. Aborting collection.")
            return {"error": "API not connected"}
        
        # Calculate tweets per KOL
        tweets_per_kol = max(1, target_total // len(self.primary_kols))
        logger.info(f"Target: {tweets_per_kol} tweets per KOL from {len(self.primary_kols)} accounts")
        
        collection_results = {
            "strategy": "historic_high_engagement",
            "target_total": target_total,
            "tweets_per_kol": tweets_per_kol,
            "date_range": {
                "start": (datetime.utcnow() - timedelta(days=12)).isoformat(),
                "end": (datetime.utcnow() - timedelta(days=5)).isoformat()
            },
            "quality_filters": {
                "min_likes": self.min_likes,
                "min_engagement_rate": self.min_engagement_rate,
                "crypto_keywords_required": True,
                "original_content_only": True
            },
            "collected_data": {},
            "collection_stats": {
                "start_time": datetime.utcnow().isoformat(),
                "kols_processed": 0,
                "tweets_collected": 0,
                "api_calls_made": 0,
                "high_quality_tweets": 0
            }
        }
        
        # Collect from each KOL
        for i, username in enumerate(self.primary_kols, 1):
            logger.info(f"Processing KOL {i}/{len(self.primary_kols)}: @{username}")
            
            tweets = self.collect_historic_tweets_for_kol(username, tweets_per_kol)
            
            if tweets:
                collection_results["collected_data"][username] = [
                    {
                        "id": tweet.id,
                        "text": tweet.text,
                        "author_username": tweet.author_username,
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                        "public_metrics": tweet.public_metrics,
                        "engagement_rate": getattr(tweet, 'engagement_rate', 0),
                        "context_annotations": getattr(tweet, 'context_annotations', [])
                    } for tweet in tweets
                ]
                
                collection_results["collection_stats"]["tweets_collected"] += len(tweets)
                collection_results["collection_stats"]["high_quality_tweets"] += len(tweets)
            
            collection_results["collection_stats"]["kols_processed"] += 1
            collection_results["collection_stats"]["api_calls_made"] += 2  # User info + tweets
            
            # Rate limiting (be respectful)
            if i < len(self.primary_kols):
                time.sleep(1.0)  # 1 second between KOLs
        
        collection_results["collection_stats"]["end_time"] = datetime.utcnow().isoformat()
        collection_results["collection_stats"]["duration_seconds"] = (
            datetime.fromisoformat(collection_results["collection_stats"]["end_time"]) - 
            datetime.fromisoformat(collection_results["collection_stats"]["start_time"])
        ).total_seconds()
        
        logger.info(f"Historic collection completed: {collection_results['collection_stats']['tweets_collected']} tweets")
        return collection_results
    
    def save_historic_dataset(self, collection_results: Dict, output_dir: str = "data/historic_tweets"):
        """Save historic tweet dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"historic_tweets_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(collection_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Historic dataset saved to: {filepath}")
        return str(filepath)

def main():
    """Run historic tweet collection."""
    collector = HistoricTweetCollector()
    
    print("🕰️  Historic Tweet Collector for A/B Testing")
    print("=" * 50)
    print("Strategy: High-engagement tweets from 5-12 days ago")
    print("Filters: 10+ likes, 0.1%+ engagement rate, crypto-relevant")
    print(f"Target: 500 tweets from {len(collector.primary_kols)} primary KOLs\n")
    
    # Collect historic dataset
    results = collector.collect_historic_dataset(target_total=500)
    
    if "error" in results:
        print(f"❌ Collection failed: {results['error']}")
        return
    
    # Save dataset
    filepath = collector.save_historic_dataset(results)
    
    # Print summary
    stats = results["collection_stats"]
    print(f"\n🎯 Historic Collection Results:")
    print(f"📊 Total tweets collected: {stats['tweets_collected']}")
    print(f"👥 KOLs processed: {stats['kols_processed']}")
    print(f"🔥 High-quality tweets: {stats['high_quality_tweets']}")
    print(f"⚡ API calls made: {stats['api_calls_made']}")
    print(f"⏱️  Duration: {stats['duration_seconds']:.1f}s")
    print(f"💾 Saved to: {filepath}")
    
    # Show top performers
    print(f"\n⭐ Top Performing KOLs:")
    kol_counts = [(kol, len(tweets)) for kol, tweets in results["collected_data"].items()]
    kol_counts.sort(key=lambda x: x[1], reverse=True)
    
    for kol, count in kol_counts[:5]:
        print(f"  @{kol}: {count} high-quality tweets")
    
    print(f"\n✅ Ready for improved A/B testing with historic engagement data!")

if __name__ == "__main__":
    main() 