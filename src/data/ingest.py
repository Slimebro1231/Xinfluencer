"""Tweet ingestion from KOL accounts."""

import logging
from typing import List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Sample KOL usernames for demo purposes
CRYPTO_KOLS = [
    "elonmusk", "VitalikButerin", "novogratz", "CryptoCobain",
    "CryptoBullish", "TheCryptoDog", "CryptoKaleo", "KoroushAK"
]

def fetch_tweets(max_tweets: int = 50) -> List[Dict]:
    """
    Fetch recent tweets from crypto KOL accounts.
    
    Args:
        max_tweets: Maximum number of tweets to fetch
        
    Returns:
        List of tweet dictionaries
    """
    logger.info(f"Fetching tweets from {len(CRYPTO_KOLS)} KOL accounts")
    
    # Mock data for demo - in production this would use Twitter API
    mock_tweets = []
    base_time = datetime.now()
    
    for i, username in enumerate(CRYPTO_KOLS[:8]):  # Limit for demo
        for j in range(min(3, max_tweets // len(CRYPTO_KOLS))):
            tweet_time = base_time - timedelta(hours=i*2 + j)
            mock_tweets.append({
                "id": f"{username}_{j}",
                "user": username,
                "text": f"Sample crypto tweet {j+1} from {username}. #Bitcoin #Crypto #DeFi",
                "created_at": tweet_time.isoformat(),
                "public_metrics": {
                    "like_count": 100 + i*j*10,
                    "retweet_count": 20 + i*j*2,
                    "reply_count": 5 + i*j
                }
            })
    
    logger.info(f"Retrieved {len(mock_tweets)} tweets")
    return mock_tweets[:max_tweets] 