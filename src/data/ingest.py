"""Tweet ingestion from KOL accounts using Twitter API."""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import tweepy
from ..config import get_config

logger = logging.getLogger(__name__)

class TwitterIngester:
    """Twitter API client for fetching KOL tweets."""
    
    def __init__(self):
        """Initialize Twitter API client."""
        self.config = get_config()
        
        # Initialize Twitter API v2 client
        self.client = tweepy.Client(
            bearer_token=self.config.twitter.bearer_token,
            consumer_key=self.config.twitter.api_key,
            consumer_secret=self.config.twitter.api_secret,
            access_token=self.config.twitter.access_token,
            access_token_secret=self.config.twitter.access_token_secret,
            wait_on_rate_limit=True
        )
        
        logger.info("Twitter API client initialized")
    
    def fetch_user_id(self, username: str) -> Optional[str]:
        """Get user ID from username."""
        try:
            user = self.client.get_user(username=username)
            if user.data:
                return str(user.data.id)
            return None
        except Exception as e:
            logger.warning(f"Could not fetch user ID for {username}: {e}")
            return None
    
    def fetch_user_tweets(self, user_id: str, max_results: int = 100) -> List[Dict]:
        """Fetch recent tweets from a user."""
        try:
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'lang', 'context_annotations'],
                exclude=['retweets', 'replies']
            )
            
            if not tweets.data:
                return []
            
            tweet_list = []
            for tweet in tweets.data:
                tweet_dict = {
                    "id": str(tweet.id),
                    "user": user_id,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "lang": tweet.lang,
                    "public_metrics": {
                        "like_count": tweet.public_metrics.get('like_count', 0),
                        "retweet_count": tweet.public_metrics.get('retweet_count', 0),
                        "reply_count": tweet.public_metrics.get('reply_count', 0),
                        "quote_count": tweet.public_metrics.get('quote_count', 0)
                    }
                }
                tweet_list.append(tweet_dict)
            
            logger.info(f"Fetched {len(tweet_list)} tweets from user {user_id}")
            return tweet_list
            
        except Exception as e:
            logger.error(f"Error fetching tweets for user {user_id}: {e}")
            return []
    
    def fetch_kol_tweets(self, max_tweets_per_kol: Optional[int] = None) -> List[Dict]:
        """Fetch tweets from all KOL accounts."""
        if max_tweets_per_kol is None:
            max_tweets_per_kol = self.config.pipeline.max_tweets_per_kol
        
        all_tweets = []
        successful_kols = 0
        
        for username in self.config.crypto_kols:
            logger.info(f"Fetching tweets from @{username}")
            
            # Get user ID
            user_id = self.fetch_user_id(username)
            if not user_id:
                logger.warning(f"Could not get user ID for @{username}, skipping")
                continue
            
            # Fetch tweets
            tweets = self.fetch_user_tweets(user_id, max_tweets_per_kol)
            if tweets:
                all_tweets.extend(tweets)
                successful_kols += 1
                logger.info(f"Successfully fetched {len(tweets)} tweets from @{username}")
            else:
                logger.warning(f"No tweets fetched from @{username}")
        
        logger.info(f"Total tweets fetched: {len(all_tweets)} from {successful_kols} KOLs")
        return all_tweets

def fetch_tweets(max_tweets: int = 50) -> List[Dict]:
    """
    Fetch recent tweets from crypto KOL accounts.
    
    Args:
        max_tweets: Maximum number of tweets to fetch per KOL
        
    Returns:
        List of tweet dictionaries
    """
    try:
        ingester = TwitterIngester()
        return ingester.fetch_kol_tweets(max_tweets)
    except Exception as e:
        logger.error(f"Error initializing Twitter ingester: {e}")
        logger.info("Falling back to mock data")
        return _fetch_mock_tweets(max_tweets)

def _fetch_mock_tweets(max_tweets: int = 50) -> List[Dict]:
    """Fallback mock data when Twitter API is not available."""
    logger.info("Using mock tweet data")
    
    # Sample KOL usernames for demo purposes
    crypto_kols = [
        "elonmusk", "VitalikButerin", "novogratz", "CryptoCobain",
        "CryptoBullish", "TheCryptoDog", "CryptoKaleo", "KoroushAK"
    ]
    
    mock_tweets = []
    base_time = datetime.now()
    
    for i, username in enumerate(crypto_kols[:8]):
        for j in range(min(3, max_tweets // len(crypto_kols))):
            tweet_time = base_time - timedelta(hours=i*2 + j)
            mock_tweets.append({
                "id": f"{username}_{j}",
                "user": username,
                "text": f"Sample crypto tweet {j+1} from {username}. #Bitcoin #Crypto #DeFi",
                "created_at": tweet_time.isoformat(),
                "lang": "en",
                "public_metrics": {
                    "like_count": 100 + i*j*10,
                    "retweet_count": 20 + i*j*2,
                    "reply_count": 5 + i*j,
                    "quote_count": 2 + i*j
                }
            })
    
    logger.info(f"Generated {len(mock_tweets)} mock tweets")
    return mock_tweets[:max_tweets] 