"""X API Client for data collection, engagement tracking, and evaluation."""

import logging
import time
import os
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import tweepy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from collections import defaultdict
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Rate limit information for API endpoints."""
    limit: int
    remaining: int
    reset_time: float
    window_seconds: int = 900  # 15 minutes default


@dataclass
class TweetData:
    """Structured tweet data for evaluation."""
    id: str
    text: str
    author_id: str
    author_username: str
    created_at: datetime
    public_metrics: Dict[str, int]
    context_annotations: List[Dict] = None
    referenced_tweets: List[Dict] = None
    non_public_metrics: Dict[str, int] = None
    source_api: str = "twitter_api_v2"


class RateLimitManager:
    """Manage rate limits across multiple API endpoints."""
    
    def __init__(self):
        """Initialize rate limit manager."""
        self.limits = {}
        self.lock = threading.Lock()
        
        # Define API endpoint limits based on X API v2 Basic plan
        self.endpoint_limits = {
            "search_recent": {"limit": 500, "window": 900},  # 500 per 15 min
            "user_lookup": {"limit": 100, "window": 900},    # 100 per 15 min  
            "tweet_lookup": {"limit": 300, "window": 900},   # 300 per 15 min
            "user_tweets": {"limit": 100, "window": 900},    # 100 per 15 min
        }
    
    def can_make_request(self, endpoint: str) -> bool:
        """Check if we can make a request to the endpoint."""
        with self.lock:
            now = time.time()
            
            if endpoint not in self.limits:
                self.limits[endpoint] = {
                    "requests": [],
                    "window": self.endpoint_limits.get(endpoint, {}).get("window", 900),
                    "limit": self.endpoint_limits.get(endpoint, {}).get("limit", 100)
                }
            
            endpoint_data = self.limits[endpoint]
            window_start = now - endpoint_data["window"]
            
            # Remove old requests outside the window
            endpoint_data["requests"] = [
                req_time for req_time in endpoint_data["requests"]
                if req_time > window_start
            ]
            
            # Check if we can make another request
            return len(endpoint_data["requests"]) < endpoint_data["limit"]
    
    def record_request(self, endpoint: str):
        """Record a request to the endpoint."""
        with self.lock:
            now = time.time()
            if endpoint not in self.limits:
                self.can_make_request(endpoint)  # Initialize if needed
            
            self.limits[endpoint]["requests"].append(now)
    
    def wait_for_reset(self, endpoint: str) -> float:
        """Calculate wait time until next request is allowed."""
        with self.lock:
            if endpoint not in self.limits:
                return 0
            
            endpoint_data = self.limits[endpoint]
            if not endpoint_data["requests"]:
                return 0
            
            oldest_request = min(endpoint_data["requests"])
            wait_time = endpoint_data["window"] - (time.time() - oldest_request)
            return max(0, wait_time)
    
    def get_status(self) -> Dict[str, Dict]:
        """Get current rate limit status for all endpoints."""
        with self.lock:
            status = {}
            now = time.time()
            
            for endpoint, data in self.limits.items():
                window_start = now - data["window"]
                recent_requests = [
                    req for req in data["requests"] if req > window_start
                ]
                
                status[endpoint] = {
                    "limit": data["limit"],
                    "used": len(recent_requests),
                    "remaining": data["limit"] - len(recent_requests),
                    "reset_in_seconds": self.wait_for_reset(endpoint)
                }
            
            return status


class XAPIClient:
    """Comprehensive X API client for data collection and engagement tracking."""
    
    def __init__(self, bearer_token: Optional[str] = None, 
                 consumer_key: Optional[str] = None,
                 consumer_secret: Optional[str] = None,
                 access_token: Optional[str] = None,
                 access_token_secret: Optional[str] = None):
        """Initialize X API client with authentication."""
        
        # Get credentials from environment if not provided
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.consumer_key = consumer_key or os.getenv('TWITTER_API_KEY')
        self.consumer_secret = consumer_secret or os.getenv('TWITTER_API_SECRET')
        self.access_token = access_token or os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = access_token_secret or os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Initialize rate limiting
        self.rate_limiter = RateLimitManager()
        
        # Initialize clients
        self.client = None
        self.api = None
        self._setup_clients()
        
        # Add user ID cache to avoid redundant lookups
        self.user_id_cache = {}
        self.cached_tweets = set()  # Track cached tweet IDs
        
        # Initialize local data cache
        self.cache_db = self._init_cache_db()
        
        # Load existing user cache
        self._load_user_cache()
        
        logger.info("X API Client initialized")
    
    def _setup_clients(self):
        """Set up Tweepy clients for different authentication methods."""
        try:
            # Set up v2 client with OAuth 2.0 (preferred)
            if (self.consumer_key and self.consumer_secret and 
                self.access_token and self.access_token_secret):
                
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret,
                    wait_on_rate_limit=False  # Patch: do not sleep on rate limit, raise error
                )
                logger.info("Initialized v2 client with full OAuth")
                
            elif self.bearer_token:
                # Bearer token only (app-only auth)
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    wait_on_rate_limit=False  # Patch: do not sleep on rate limit, raise error
                )
                logger.info("Initialized v2 client with bearer token")
            
            # Test connection
            if self.client:
                try:
                    me = self.client.get_me()
                    if me and hasattr(me, 'data') and me.data:
                        logger.info(f"Authenticated as: @{me.data.username}")
                    else:
                        logger.info("App-only authentication successful")
                except Exception as e:
                    logger.warning(f"Authentication test failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to set up X API clients: {e}")
            self.client = None
    
    def _init_cache_db(self) -> sqlite3.Connection:
        """Initialize SQLite cache database for storing tweet data."""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = cache_dir / "x_api_cache.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables for caching
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT PRIMARY KEY,
                author_username TEXT,
                text TEXT,
                created_at TEXT,
                public_metrics TEXT,
                non_public_metrics TEXT,
                context_annotations TEXT,
                cached_at TEXT,
                source_api TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engagement_history (
                tweet_id TEXT,
                timestamp TEXT,
                likes INTEGER,
                retweets INTEGER,
                replies INTEGER,
                quotes INTEGER,
                impressions INTEGER,
                PRIMARY KEY (tweet_id, timestamp)
            )
        """)
        
        # Add user ID cache table to avoid redundant API calls
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_cache (
                username TEXT PRIMARY KEY,
                user_id TEXT,
                cached_at TEXT,
                user_data TEXT
            )
        """)
        
        conn.commit()
        
        return conn
    
    def _load_user_cache(self):
        """Load cached user IDs from database."""
        try:
            cursor = self.cache_db.cursor()
            cursor.execute("SELECT username, user_id FROM user_cache")
            for username, user_id in cursor.fetchall():
                self.user_id_cache[username] = user_id
            logger.info(f"Loaded {len(self.user_id_cache)} cached user IDs")
        except Exception as e:
            logger.error(f"Error loading user cache: {e}")
    
    def _get_user_id(self, username: str) -> Optional[str]:
        """Get user ID with caching to avoid redundant API calls."""
        # Check cache first
        if username in self.user_id_cache:
            return self.user_id_cache[username]
        
        # Check rate limits for user lookup
        if not self.rate_limiter.can_make_request("user_lookup"):
            wait_time = self.rate_limiter.wait_for_reset("user_lookup")
            logger.warning(f"Rate limit reached for user_lookup. Waiting {wait_time:.1f}s")
            time.sleep(wait_time + 1)
        
        try:
            self.rate_limiter.record_request("user_lookup")
            user = self.client.get_user(username=username)
            
            if user and user.data:
                user_id = user.data.id
                # Cache the result
                self.user_id_cache[username] = user_id
                
                # Store in database
                cursor = self.cache_db.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_cache 
                    (username, user_id, cached_at, user_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    username, 
                    user_id, 
                    datetime.utcnow().isoformat(),
                    json.dumps({"verified": getattr(user.data, 'verified', False)})
                ))
                self.cache_db.commit()
                
                return user_id
            else:
                logger.warning(f"User not found: {username}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting user ID for {username}: {e}")
            return None
    
    def batch_get_user_ids(self, usernames: List[str]) -> Dict[str, str]:
        """Get multiple user IDs efficiently, using cache where possible."""
        user_ids = {}
        uncached_usernames = []
        
        # Check cache first
        for username in usernames:
            if username in self.user_id_cache:
                user_ids[username] = self.user_id_cache[username]
            else:
                uncached_usernames.append(username)
        
        if not uncached_usernames:
            return user_ids
        
        # Batch lookup for uncached users (up to 100 at a time)
        for i in range(0, len(uncached_usernames), 100):
            batch = uncached_usernames[i:i+100]
            
            if not self.rate_limiter.can_make_request("user_lookup"):
                wait_time = self.rate_limiter.wait_for_reset("user_lookup")
                logger.warning(f"Rate limit reached for user_lookup. Waiting {wait_time:.1f}s")
                time.sleep(wait_time + 1)
            
            try:
                self.rate_limiter.record_request("user_lookup")
                response = self.client.get_users(usernames=batch)
                
                if response and response.data:
                    for user in response.data:
                        username = user.username
                        user_id = user.id
                        
                        # Cache the result
                        self.user_id_cache[username] = user_id
                        user_ids[username] = user_id
                        
                        # Store in database
                        cursor = self.cache_db.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO user_cache 
                            (username, user_id, cached_at, user_data)
                            VALUES (?, ?, ?, ?)
                        """, (
                            username, 
                            user_id, 
                            datetime.utcnow().isoformat(),
                            json.dumps({"verified": getattr(user, 'verified', False)})
                        ))
                    self.cache_db.commit()
                    
            except Exception as e:
                logger.error(f"Error in batch user lookup: {e}")
        
        return user_ids
    
    def _check_cached_tweets(self, query_params: Dict) -> List[TweetData]:
        """Check if we have cached tweets that match the query."""
        try:
            cursor = self.cache_db.cursor()
            
            # For now, simple cache check based on recency (last 1 hour)
            recent_cutoff = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            
            cursor.execute("""
                SELECT id, author_username, text, created_at, public_metrics, 
                       context_annotations, source_api
                FROM tweets 
                WHERE cached_at > ? 
                ORDER BY cached_at DESC 
                LIMIT 50
            """, (recent_cutoff,))
            
            cached_tweets = []
            for row in cursor.fetchall():
                tweet_data = TweetData(
                    id=row[0],
                    author_id="cached",  # We don't store author_id in old schema
                    author_username=row[1],
                    text=row[2],
                    created_at=datetime.fromisoformat(row[3].replace('Z', '')),
                    public_metrics=json.loads(row[4]) if row[4] else {},
                    context_annotations=json.loads(row[5]) if row[5] else [],
                    source_api=row[6]
                )
                cached_tweets.append(tweet_data)
            
            return cached_tweets
            
        except Exception as e:
            logger.error(f"Error checking cached tweets: {e}")
            return []
    
    def search_recent_tweets(self, query: str, max_results: int = 10,
                           tweet_fields: List[str] = None,
                           user_fields: List[str] = None,
                           expansions: List[str] = None) -> List[TweetData]:
        """Search for recent tweets with comprehensive data."""
        if not self.client:
            logger.error("X API client not initialized")
            return []
        
        endpoint = "search_recent"
        
        # Check rate limits
        if not self.rate_limiter.can_make_request(endpoint):
            wait_time = self.rate_limiter.wait_for_reset(endpoint)
            logger.warning(f"Rate limit reached for {endpoint}. Waiting {wait_time:.1f}s")
            time.sleep(wait_time + 1)
        
        # Default fields for comprehensive data collection
        if tweet_fields is None:
            tweet_fields = [
                'id', 'text', 'created_at', 'author_id', 'public_metrics',
                'context_annotations', 'referenced_tweets', 'lang'
            ]
        
        if user_fields is None:
            user_fields = ['id', 'username', 'name', 'verified', 'public_metrics']
        
        if expansions is None:
            expansions = ['author_id', 'referenced_tweets.id']
        
        try:
            self.rate_limiter.record_request(endpoint)
            
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit is 100
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions
            )
            
            if not response or not response.data:
                logger.info(f"No tweets found for query: {query}")
                return []
            
            # Parse response into TweetData objects
            tweets = []
            users_dict = {}
            
            # Build user lookup dict
            if hasattr(response, 'includes') and response.includes:
                if 'users' in response.includes:
                    for user in response.includes['users']:
                        users_dict[user.id] = user
            
            for tweet in response.data:
                # Get author info
                author = users_dict.get(tweet.author_id)
                author_username = author.username if author else "unknown"
                
                # Extract metrics
                public_metrics = tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                
                # Extract context annotations
                context_annotations = []
                if hasattr(tweet, 'context_annotations') and tweet.context_annotations:
                    context_annotations = [
                        {
                            'domain': ann.domain.name if hasattr(ann, 'domain') and ann.domain else None,
                            'entity': ann.entity.name if hasattr(ann, 'entity') and ann.entity else None
                        }
                        for ann in tweet.context_annotations
                    ]
                
                tweet_data = TweetData(
                    id=tweet.id,
                    text=tweet.text,
                    author_id=tweet.author_id,
                    author_username=author_username,
                    created_at=tweet.created_at,
                    public_metrics=public_metrics,
                    context_annotations=context_annotations,
                    source_api="twitter_api_v2"
                )
                
                tweets.append(tweet_data)
                
                # Cache the tweet
                self._cache_tweet(tweet_data)
            
            logger.info(f"Retrieved {len(tweets)} tweets for query: {query}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def get_user_tweets(self, username: str, max_results: int = 10,
                       exclude_replies: bool = False,
                       exclude_retweets: bool = False) -> List[TweetData]:
        """Get recent tweets from a specific user using robust X API v2 with pagination and diagnostics."""
        if not self.client:
            logger.error("X API client not initialized")
            return []
        endpoint = "user_tweets"
        try:
            user_id = self._get_user_id(username)
            logger.info(f"Resolved @{username} to user_id: {user_id}")
            if not user_id:
                logger.warning(f"User not found: {username}")
                return []
            exclusions = []
            if exclude_replies:
                exclusions.append('replies')
            if exclude_retweets:
                exclusions.append('retweets')
            tweets = []
            pagination_token = None
            total_fetched = 0
            while total_fetched < max_results:
                batch_size = min(100, max_results - total_fetched)
                if batch_size < 5:
                    logger.info(f"Skipping final batch for @{username} (batch_size={batch_size} < 5, X API v2 minimum)")
                    break
                logger.info(f"Requesting batch: user_id={user_id}, batch_size={batch_size}, pagination_token={pagination_token}")
                response = self.client.get_users_tweets(
                    id=user_id,
                    max_results=batch_size,
                    tweet_fields=[
                        'id', 'text', 'created_at', 'author_id', 'public_metrics',
                        'context_annotations', 'referenced_tweets', 'lang'
                    ],
                    exclude=exclusions if exclusions else None,
                    pagination_token=pagination_token
                )
                logger.info(f"Raw API response: {response}")
                if not response or not response.data:
                    logger.info(f"No more tweets found for user: {username}")
                    break
                for tweet in response.data:
                    public_metrics = tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                    context_annotations = []
                    if hasattr(tweet, 'context_annotations') and tweet.context_annotations:
                        context_annotations = [
                            {
                                'domain': ann.domain.name if hasattr(ann, 'domain') and ann.domain else None,
                                'entity': ann.entity.name if hasattr(ann, 'entity') and ann.entity else None
                            }
                            for ann in tweet.context_annotations
                        ]
                    tweet_data = TweetData(
                        id=tweet.id,
                        text=tweet.text,
                        author_id=tweet.author_id,
                        author_username=username,
                        created_at=tweet.created_at,
                        public_metrics=public_metrics,
                        context_annotations=context_annotations,
                        source_api="twitter_api_v2"
                    )
                    tweets.append(tweet_data)
                    self._cache_tweet(tweet_data)
                total_fetched += len(response.data)
                pagination_token = response.meta.get('next_token') if hasattr(response, 'meta') and response.meta else None
                if not pagination_token:
                    break
            logger.info(f"Retrieved {len(tweets)} tweets from @{username} (with pagination)")
            return tweets
        except Exception as e:
            logger.error(f"Error getting user tweets for {username}: {e}")
            return []
    
    def track_engagement_metrics(self, tweet_ids: List[str]) -> Dict[str, Dict]:
        """Track engagement metrics for specific tweets over time."""
        if not self.client:
            logger.error("X API client not initialized")
            return {}
        
        endpoint = "tweet_lookup"
        
        try:
            # Check rate limits
            if not self.rate_limiter.can_make_request(endpoint):
                wait_time = self.rate_limiter.wait_for_reset(endpoint)
                logger.warning(f"Rate limit reached for {endpoint}. Waiting {wait_time:.1f}s")
                time.sleep(wait_time + 1)
            
            self.rate_limiter.record_request(endpoint)
            
            # Get current metrics for tweets
            response = self.client.get_tweets(
                ids=tweet_ids[:100],  # API limit
                tweet_fields=[
                    'id', 'public_metrics', 'non_public_metrics', 'created_at'
                ]
            )
            
            if not response or not response.data:
                logger.warning("No tweet data retrieved for engagement tracking")
                return {}
            
            engagement_data = {}
            timestamp = datetime.utcnow().isoformat()
            
            for tweet in response.data:
                metrics = {}
                
                # Public metrics (always available)
                if hasattr(tweet, 'public_metrics'):
                    metrics.update(tweet.public_metrics)
                
                # Non-public metrics (requires user context auth)
                if hasattr(tweet, 'non_public_metrics'):
                    metrics.update(tweet.non_public_metrics)
                
                engagement_data[tweet.id] = {
                    'timestamp': timestamp,
                    'metrics': metrics,
                    'created_at': tweet.created_at.isoformat() if tweet.created_at else None
                }
                
                # Store in engagement history
                self._store_engagement_history(tweet.id, metrics)
            
            logger.info(f"Tracked engagement for {len(engagement_data)} tweets")
            return engagement_data
            
        except Exception as e:
            logger.error(f"Error tracking engagement metrics: {e}")
            return {}
    
    def get_kol_analysis(self, usernames: List[str], 
                        days_back: int = 7) -> Dict[str, Dict]:
        """Analyze KOL performance and engagement patterns."""
        analysis = {}
        
        for username in usernames:
            try:
                # Get recent tweets
                tweets = self.get_user_tweets(
                    username=username,
                    max_results=50,
                    exclude_replies=True,
                    exclude_retweets=True
                )
                
                if not tweets:
                    continue
                
                # Calculate metrics
                total_tweets = len(tweets)
                total_engagement = 0
                topic_engagement = defaultdict(int)
                
                for tweet in tweets:
                    metrics = tweet.public_metrics
                    engagement = (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) +
                        metrics.get('reply_count', 0) +
                        metrics.get('quote_count', 0)
                    )
                    total_engagement += engagement
                    
                    # Analyze crypto-related content
                    if tweet.context_annotations:
                        for annotation in tweet.context_annotations:
                            if annotation.get('domain') and 'crypto' in annotation['domain'].lower():
                                topic_engagement['crypto'] += engagement
                            elif annotation.get('entity') and any(term in annotation['entity'].lower() 
                                                                for term in ['bitcoin', 'ethereum', 'crypto']):
                                topic_engagement['crypto'] += engagement
                
                avg_engagement = total_engagement / total_tweets if total_tweets > 0 else 0
                
                analysis[username] = {
                    'total_tweets': total_tweets,
                    'total_engagement': total_engagement,
                    'avg_engagement_per_tweet': avg_engagement,
                    'crypto_engagement': topic_engagement.get('crypto', 0),
                    'crypto_engagement_ratio': (
                        topic_engagement.get('crypto', 0) / total_engagement 
                        if total_engagement > 0 else 0
                    ),
                    'analysis_date': datetime.utcnow().isoformat()
                }
                
                logger.info(f"Analyzed @{username}: {total_tweets} tweets, {avg_engagement:.1f} avg engagement")
                
            except Exception as e:
                logger.error(f"Error analyzing KOL {username}: {e}")
                continue
        
        return analysis
    
    def _cache_tweet(self, tweet_data: TweetData):
        """Cache tweet data to local database."""
        try:
            cursor = self.cache_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tweets 
                (id, author_username, text, created_at, public_metrics, 
                 context_annotations, cached_at, source_api)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet_data.id,
                tweet_data.author_username,
                tweet_data.text,
                tweet_data.created_at.isoformat(),
                json.dumps(tweet_data.public_metrics),
                json.dumps(tweet_data.context_annotations) if tweet_data.context_annotations else None,
                datetime.utcnow().isoformat(),
                tweet_data.source_api
            ))
            self.cache_db.commit()
        except Exception as e:
            logger.error(f"Error caching tweet {tweet_data.id}: {e}")
    
    def _store_engagement_history(self, tweet_id: str, metrics: Dict):
        """Store engagement metrics in history table."""
        try:
            cursor = self.cache_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO engagement_history
                (tweet_id, timestamp, likes, retweets, replies, quotes, impressions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet_id,
                datetime.utcnow().isoformat(),
                metrics.get('like_count', 0),
                metrics.get('retweet_count', 0),
                metrics.get('reply_count', 0),
                metrics.get('quote_count', 0),
                metrics.get('impression_count', 0)  # Only available with user context auth
            ))
            self.cache_db.commit()
        except Exception as e:
            logger.error(f"Error storing engagement history for {tweet_id}: {e}")
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for all endpoints."""
        status = self.rate_limiter.get_status()
        
        # Add API connection status
        status['api_connected'] = self.client is not None
        status['auth_type'] = 'full_oauth' if (
            self.consumer_key and self.access_token
        ) else 'bearer_token' if self.bearer_token else 'none'
        
        return status
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return capabilities."""
        if not self.client:
            return {
                'connected': False,
                'error': 'No API client initialized',
                'capabilities': []
            }
        
        capabilities = []
        errors = []
        
        try:
            # Test basic search
            test_search = self.client.search_recent_tweets(
                query="crypto", max_results=1
            )
            if test_search:
                capabilities.append('search_tweets')
        except Exception as e:
            errors.append(f"Search test failed: {e}")
        
        try:
            # Test user lookup
            me = self.client.get_me()
            if me:
                capabilities.append('user_lookup')
                capabilities.append('authenticated_user')
        except Exception as e:
            errors.append(f"User lookup test failed: {e}")
        
        return {
            'connected': len(capabilities) > 0,
            'capabilities': capabilities,
            'errors': errors,
            'rate_limits': self.get_rate_limit_status()
        } 