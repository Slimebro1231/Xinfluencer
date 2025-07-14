"""Web scraper for crypto KOL posts using Twikit."""

import logging
import asyncio
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

from twikit import Client

from src.config import get_config

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
    """
    Scraper for crypto KOL posts using the Twikit library.
    
    NOTE: This scraper requires Twitter credentials (username, email, password)
    to be set in the environment variables:
    - TWITTER_USERNAME
    - TWITTER_EMAIL
    - TWITTER_PASSWORD
    """

    def __init__(self):
        """Initialize the scraper."""
        self.config = get_config()
        self.client = Client()
        self._login()

    def _login(self):
        """Logs in to Twitter using credentials from config."""
        try:
            # NOTE: You need to add these to your .env file or environment
            username = self.config.twitter.username
            email = self.config.twitter.email
            password = self.config.twitter.password
            
            if not all([username, email, password]):
                raise ValueError("Twitter credentials not found in config.")

            self.client.login(
                auth_info_1=username,
                auth_info_2=email,
                password=password
            )
            logger.info("Successfully logged in to Twitter.")
        except Exception as e:
            logger.error(f"Failed to login to Twitter: {e}")
            raise

    def scrape_user_posts(self, username: str, max_posts: int = 50) -> List[ScrapedPost]:
        """Scrape posts from a specific user."""
        try:
            logger.info(f"Scraping posts for @{username} using Twikit")
            user = self.client.get_user_by_screen_name(username)
            tweets = user.get_tweets('Tweets', count=max_posts)

            posts = []
            for tweet in tweets:
                posts.append(
                    ScrapedPost(
                        id=tweet.id,
                        username=username,
                        text=tweet.text,
                        created_at=tweet.created_at,
                        likes=tweet.favorite_count,
                        retweets=tweet.retweet_count,
                        replies=tweet.reply_count,
                        quotes=tweet.quote_count,
                        url=f'https://twitter.com/{username}/status/{tweet.id}',
                        is_retweet=tweet.retweeted,
                        is_reply=tweet.in_reply_to_status_id is not None
                    )
                )
            logger.info(f"Successfully scraped {len(posts)} posts from @{username} using Twikit.")
            return posts

        except Exception as e:
            logger.error(f"Error scraping posts from @{username} with Twikit: {e}")
            return []

    def scrape_kol_posts(self, max_posts_per_kol: Optional[int] = None) -> List[ScrapedPost]:
        """Scrape posts from all KOL accounts."""
        if max_posts_per_kol is None:
            max_posts_per_kol = self.config.pipeline.max_tweets_per_kol

        all_posts = []
        for username in self.config.crypto_kols:
            posts = self.scrape_user_posts(username, max_posts=max_posts_per_kol)
            all_posts.extend(posts)
            # Add a small delay between scraping different users
            time.sleep(random.uniform(1, 3))
            
        logger.info(f"Total posts scraped: {len(all_posts)}")
        return all_posts

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    scraper = CryptoKOLScraper()
    
    # Scrape a single user
    # posts = scraper.scrape_user_posts('elonmusk', max_posts=10)
    # for post in posts:
    #     print(post)
        
    # Scrape all KOLs
    all_kol_posts = scraper.scrape_kol_posts(max_posts_per_kol=5)
    print(f"Scraped {len(all_kol_posts)} posts from all KOLs.") 