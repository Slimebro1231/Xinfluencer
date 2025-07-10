"""Web scraper for crypto KOL posts using ntscraper and snscrape fallback."""

import logging
import asyncio
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class ScrapedPost:
    """Data structure for scraped posts."""
    id: str
    username: str
    text: str
    created_at: datetime
    likes: int
    retweets: int
    replies: int
    quotes: int
    url: str
    is_retweet: bool = False
    is_reply: bool = False

class CryptoKOLScraper:
    """Scraper for crypto KOL posts using ntscraper and snscrape fallback."""
    
    def __init__(self, use_snscrape_fallback: bool = True):
        """Initialize the scraper."""
        self.config = get_config()
        self.use_snscrape_fallback = use_snscrape_fallback
        self._setup_scrapers()
    
    def _setup_scrapers(self):
        """Set up the scraping sessions."""
        try:
            from ntscraper import Nitter
            self.ntscraper = Nitter
            logger.info("ntscraper (Nitter) initialized successfully")
        except ImportError:
            logger.error("ntscraper not installed. Please install with: pip install ntscraper")
            self.ntscraper = None
        
        if self.use_snscrape_fallback:
            try:
                import snscrape.modules.twitter as sntwitter
                self.snscrape = sntwitter
                logger.info("snscrape fallback initialized successfully")
            except ImportError:
                logger.warning("snscrape not installed. Fallback disabled.")
                self.snscrape = None
    
    def scrape_user_posts(self, username: str, max_posts: int = 50) -> List[ScrapedPost]:
        """Scrape posts from a specific user."""
        try:
            logger.info(f"Scraping posts from @{username}")
            
            # Try ntscraper first
            if self.ntscraper:
                posts = self._scrape_with_ntscraper(username, max_posts)
                if posts:
                    logger.info(f"Successfully scraped {len(posts)} posts from @{username} using ntscraper")
                    return posts
            
            # Fallback to snscrape
            if self.snscrape and self.use_snscrape_fallback:
                posts = self._scrape_with_snscrape(username, max_posts)
                if posts:
                    logger.info(f"Successfully scraped {len(posts)} posts from @{username} using snscrape")
                    return posts
            
            logger.warning(f"No posts found for @{username}")
            return []
            
        except Exception as e:
            logger.error(f"Error scraping posts from @{username}: {e}")
            return []
    
    def _scrape_with_ntscraper(self, username: str, max_posts: int) -> List[ScrapedPost]:
        """Scrape using ntscraper."""
        try:
            scraper = self.ntscraper()
            
            # Use get_tweets with mode='user' to get user posts
            tweets_data = scraper.get_tweets(
                terms=username,
                mode='user',
                number=max_posts
            )
            
            if not tweets_data:
                return []
            
            posts = []
            for tweet in tweets_data:
                try:
                    # Parse tweet data
                    post_obj = ScrapedPost(
                        id=str(tweet.get('id', '')),
                        username=username,
                        text=tweet.get('text', ''),
                        created_at=datetime.fromisoformat(tweet.get('date', datetime.now().isoformat())),
                        likes=tweet.get('likes', 0),
                        retweets=tweet.get('retweets', 0),
                        replies=tweet.get('replies', 0),
                        quotes=tweet.get('quotes', 0),
                        url=tweet.get('url', ''),
                        is_retweet=tweet.get('is_retweet', False),
                        is_reply=tweet.get('is_reply', False)
                    )
                    posts.append(post_obj)
                except Exception as e:
                    logger.warning(f"Error parsing tweet from @{username}: {e}")
                    continue
            
            return posts
            
        except Exception as e:
            logger.error(f"ntscraper failed for @{username}: {e}")
            return []
    
    def _scrape_with_snscrape(self, username: str, max_posts: int) -> List[ScrapedPost]:
        """Scrape using snscrape fallback."""
        try:
            posts = []
            count = 0
            
            # Use snscrape to get user tweets
            for tweet in self.snscrape.TwitterUserScraper(username).get_items():
                if count >= max_posts:
                    break
                
                try:
                    post_obj = ScrapedPost(
                        id=str(tweet.id),
                        username=username,
                        text=tweet.rawContent,
                        created_at=tweet.date,
                        likes=tweet.likeCount,
                        retweets=tweet.retweetCount,
                        replies=tweet.replyCount,
                        quotes=tweet.quoteCount,
                        url=f"https://twitter.com/{username}/status/{tweet.id}",
                        is_retweet=hasattr(tweet, 'retweetedTweet') and tweet.retweetedTweet is not None,
                        is_reply=hasattr(tweet, 'inReplyToTweetId') and tweet.inReplyToTweetId is not None
                    )
                    posts.append(post_obj)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error parsing snscrape tweet from @{username}: {e}")
                    continue
            
            return posts
            
        except Exception as e:
            logger.error(f"snscrape failed for @{username}: {e}")
            return []
    
    def scrape_kol_posts(self, max_posts_per_kol: Optional[int] = None) -> List[ScrapedPost]:
        """Scrape posts from all KOL accounts."""
        if max_posts_per_kol is None:
            max_posts_per_kol = self.config.pipeline.max_tweets_per_kol
        
        all_posts = []
        successful_kols = 0
        failed_attempts = 0
        max_failed_attempts = 3  # Fail fast after 3 failed attempts
        
        # Limit to first 10 KOLs to avoid long delays
        kols_to_try = self.config.crypto_kols[:10]
        
        # Add random delay to avoid rate limiting
        for i, username in enumerate(kols_to_try):
            logger.info(f"Scraping posts from @{username} ({i+1}/{len(kols_to_try)})")
            
            try:
                posts = self.scrape_user_posts(username, max_posts_per_kol)
                if posts:
                    all_posts.extend(posts)
                    successful_kols += 1
                    logger.info(f"Successfully scraped {len(posts)} posts from @{username}")
                    failed_attempts = 0  # Reset failed attempts on success
                else:
                    logger.warning(f"No posts scraped from @{username}")
                    failed_attempts += 1
                
                # Random delay between requests (1-3 seconds)
                time.sleep(random.uniform(1, 3))
                
                # Fail fast if too many consecutive failures
                if failed_attempts >= max_failed_attempts:
                    logger.warning(f"Too many consecutive failures ({failed_attempts}), falling back to mock data")
                    break
                
            except Exception as e:
                logger.error(f"Error scraping from @{username}: {e}")
                failed_attempts += 1
                continue
        
        logger.info(f"Total posts scraped: {len(all_posts)} from {successful_kols} KOLs")
        return all_posts
    
    def scrape_trending_topics(self, max_posts: int = 100) -> List[ScrapedPost]:
        """Scrape posts from trending crypto topics."""
        try:
            logger.info("Scraping trending crypto topics")
            
            # Define trending crypto hashtags
            trending_hashtags = [
                "#Bitcoin", "#Ethereum", "#Crypto", "#DeFi", "#NFT", 
                "#Web3", "#Blockchain", "#Altcoin", "#Trading", "#RWA"
            ]
            
            all_posts = []
            for hashtag in trending_hashtags[:5]:  # Limit to 5 hashtags
                try:
                    logger.info(f"Scraping posts with {hashtag}")
                    
                    # Try ntscraper first
                    if self.ntscraper:
                        scraper = self.ntscraper()
                        posts_data = scraper.get_tweets(
                            terms=hashtag,
                            mode='hashtag',
                            number=max_posts // 5
                        )
                        
                        if posts_data:
                            for tweet in posts_data:
                                try:
                                    post_obj = ScrapedPost(
                                        id=str(tweet.get('id', '')),
                                        username=tweet.get('user', {}).get('username', ''),
                                        text=tweet.get('text', ''),
                                        created_at=datetime.fromisoformat(tweet.get('date', datetime.now().isoformat())),
                                        likes=tweet.get('likes', 0),
                                        retweets=tweet.get('retweets', 0),
                                        replies=tweet.get('replies', 0),
                                        quotes=tweet.get('quotes', 0),
                                        url=tweet.get('url', ''),
                                        is_retweet=tweet.get('is_retweet', False),
                                        is_reply=tweet.get('is_reply', False)
                                    )
                                    all_posts.append(post_obj)
                                except Exception as e:
                                    logger.warning(f"Error parsing trending tweet: {e}")
                                    continue
                    
                    # Random delay
                    time.sleep(random.uniform(2, 4))
                    
                except Exception as e:
                    logger.error(f"Error scraping {hashtag}: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(all_posts)} trending posts")
            return all_posts
            
        except Exception as e:
            logger.error(f"Error scraping trending topics: {e}")
            return []

def scrape_posts(max_posts: int = 50, include_trending: bool = True) -> List[Dict]:
    """
    Scrape posts from crypto KOL accounts and trending topics.
    
    Args:
        max_posts: Maximum number of posts to scrape per KOL
        include_trending: Whether to include trending topic posts
        
    Returns:
        List of post dictionaries
    """
    try:
        scraper = CryptoKOLScraper(use_snscrape_fallback=True)
        
        # Scrape KOL posts
        kol_posts = scraper.scrape_kol_posts(max_posts)
        
        # Scrape trending posts if requested
        trending_posts = []
        if include_trending:
            trending_posts = scraper.scrape_trending_topics(max_posts // 2)
        
        # Combine all posts
        all_posts = kol_posts + trending_posts
        
        # If no posts were scraped, use mock data
        if not all_posts:
            logger.warning("No posts scraped, using mock data")
            return _fetch_mock_posts(max_posts)
        
        # Convert to dictionaries
        posts_dict = []
        for post in all_posts:
            posts_dict.append({
                'id': post.id,
                'user': post.username,
                'text': post.text,
                'created_at': post.created_at.isoformat(),
                'public_metrics': {
                    'like_count': post.likes,
                    'retweet_count': post.retweets,
                    'reply_count': post.replies,
                    'quote_count': post.quotes
                },
                'url': post.url,
                'is_retweet': post.is_retweet,
                'is_reply': post.is_reply
            })
        
        logger.info(f"Total posts scraped: {len(posts_dict)}")
        return posts_dict
        
    except Exception as e:
        logger.error(f"Error in scrape_posts: {e}")
        # Return mock data as fallback
        return _fetch_mock_posts(max_posts)

def _fetch_mock_posts(max_posts: int = 50) -> List[Dict]:
    """Generate mock posts for testing when scraping fails."""
    logger.warning("Using mock data due to scraping failure")
    
    mock_posts = []
    crypto_topics = [
        "Bitcoin", "Ethereum", "DeFi", "NFTs", "Web3", "Blockchain", 
        "Cryptocurrency", "Trading", "Altcoins", "RWA"
    ]
    
    for i in range(min(max_posts, 20)):
        topic = random.choice(crypto_topics)
        mock_posts.append({
            'id': f'mock_{i}',
            'user': f'crypto_user_{i}',
            'text': f'Mock post about {topic} and its impact on the market. #{topic} #Crypto',
            'created_at': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
            'public_metrics': {
                'like_count': random.randint(10, 1000),
                'retweet_count': random.randint(5, 500),
                'reply_count': random.randint(1, 100),
                'quote_count': random.randint(0, 50)
            },
            'url': f'https://twitter.com/crypto_user_{i}/status/mock_{i}',
            'is_retweet': False,
            'is_reply': False
        })
    
    return mock_posts 