"""Twitter service for posting tweets with multiple authentication methods."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import tweepy

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


class TwitterService:
    """Twitter service that supports both OAuth 1.0a and OAuth 2.0 authentication."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Twitter service with configuration."""
        self.config = config or Config()
        self.client = None
        self.api = None
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Set up Twitter authentication based on available credentials."""
        twitter_config = self.config.twitter
        
        # Try OAuth 2.0 first (newer, preferred method)
        if self._can_use_oauth2():
            logger.info("Setting up Twitter OAuth 2.0 authentication...")
            self._setup_oauth2()
        
        # Fallback to OAuth 1.0a if OAuth 2.0 not available
        elif self._can_use_oauth1():
            logger.info("Setting up Twitter OAuth 1.0a authentication...")
            self._setup_oauth1()
        
        else:
            logger.warning("No valid Twitter credentials found")
    
    def _can_use_oauth2(self) -> bool:
        """Check if OAuth 2.0 credentials are available."""
        twitter_config = self.config.twitter
        
        # For posting, we need either:
        # 1. Bearer token (for app-only auth, limited functionality)
        # 2. Client ID + Client Secret (for full OAuth 2.0)
        return bool(
            twitter_config.bearer_token or 
            (twitter_config.client_id and twitter_config.client_secret)
        )
    
    def _can_use_oauth1(self) -> bool:
        """Check if OAuth 1.0a credentials are available."""
        twitter_config = self.config.twitter
        return bool(
            twitter_config.api_key and 
            twitter_config.api_secret and 
            twitter_config.access_token and 
            twitter_config.access_token_secret
        )
    
    def _setup_oauth2(self):
        """Set up OAuth 2.0 authentication."""
        twitter_config = self.config.twitter
        
        try:
            # For posting tweets, we need the full OAuth 2.0 flow with user context
            if (twitter_config.client_id and twitter_config.client_secret and 
                twitter_config.api_key and twitter_config.api_secret and
                twitter_config.access_token and twitter_config.access_token_secret):
                
                logger.info("Using OAuth 2.0 with full user context")
                self.client = tweepy.Client(
                    consumer_key=twitter_config.api_key,
                    consumer_secret=twitter_config.api_secret,
                    access_token=twitter_config.access_token,
                    access_token_secret=twitter_config.access_token_secret
                )
            
            elif twitter_config.bearer_token:
                # Bearer token authentication (app-only, limited functionality)
                logger.info("Using Bearer token authentication (read-only)")
                self.client = tweepy.Client(bearer_token=twitter_config.bearer_token)
            
            elif twitter_config.client_id and twitter_config.client_secret:
                # OAuth 2.0 with client credentials only (limited functionality)
                logger.info("Using OAuth 2.0 client credentials (limited functionality)")
                self.client = tweepy.Client(
                    client_id=twitter_config.client_id,
                    client_secret=twitter_config.client_secret
                )
            
            # Test the connection
            if self.client:
                try:
                    # Try to get user info to verify authentication
                    user = self.client.get_me()
                    if user and user.data:
                        logger.info(f"Successfully authenticated as: @{user.data.username}")
                    else:
                        logger.info("OAuth 2.0 authentication successful (app-only)")
                except Exception as e:
                    logger.warning(f"OAuth 2.0 auth verification failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to set up OAuth 2.0 authentication: {e}")
            self.client = None
    
    def _setup_oauth1(self):
        """Set up OAuth 1.0a authentication."""
        twitter_config = self.config.twitter
        
        try:
            # OAuth 1.0a authentication
            auth = tweepy.OAuth1UserHandler(
                twitter_config.api_key,
                twitter_config.api_secret,
                twitter_config.access_token,
                twitter_config.access_token_secret
            )
            
            # Create both v1.1 API and v2 Client
            self.api = tweepy.API(auth)
            self.client = tweepy.Client(
                consumer_key=twitter_config.api_key,
                consumer_secret=twitter_config.api_secret,
                access_token=twitter_config.access_token,
                access_token_secret=twitter_config.access_token_secret
            )
            
            # Test the connection
            try:
                user = self.api.verify_credentials()
                if user:
                    logger.info(f"Successfully authenticated as: @{user.screen_name}")
            except Exception as e:
                logger.warning(f"OAuth 1.0a auth verification failed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to set up OAuth 1.0a authentication: {e}")
            self.api = None
            self.client = None
    
    def post_tweet(self, text: str) -> Dict[str, Any]:
        """
        Post a tweet using the best available authentication method.
        
        Args:
            text: Tweet content (max 280 characters)
            
        Returns:
            Dictionary with posting result
        """
        if len(text) > 280:
            logger.warning(f"Tweet text too long ({len(text)} chars), truncating...")
            text = text[:277] + "..."
        
        # Try Twitter API v2 first
        if self.client:
            try:
                logger.info(f"Posting tweet via Twitter API v2: '{text[:50]}...'")
                response = self.client.create_tweet(text=text)
                
                if response and response.data:
                    tweet_id = response.data.get('id')
                    logger.info(f"Successfully posted tweet via v2 API. Tweet ID: {tweet_id}")
                    return {
                        "success": True,
                        "tweet_id": tweet_id,
                        "method": "v2_api",
                        "text": text
                    }
                else:
                    raise Exception("No response data from v2 API")
                    
            except Exception as e:
                logger.error(f"Failed to post via Twitter v2 API: {e}")
        
        # Fallback to Twitter API v1.1
        if self.api:
            try:
                logger.info(f"Posting tweet via Twitter API v1.1: '{text[:50]}...'")
                status = self.api.update_status(text)
                
                if status:
                    tweet_id = status.id_str
                    logger.info(f"Successfully posted tweet via v1.1 API. Tweet ID: {tweet_id}")
                    return {
                        "success": True,
                        "tweet_id": tweet_id,
                        "method": "v1_api",
                        "text": text
                    }
                else:
                    raise Exception("No status returned from v1.1 API")
                    
            except Exception as e:
                logger.error(f"Failed to post via Twitter v1.1 API: {e}")
        
        # No authentication methods available
        logger.error("No valid Twitter authentication available for posting")
        return {
            "success": False,
            "error": "No valid Twitter authentication available",
            "text": text
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test Twitter API connection and return status.
        
        Returns:
            Dictionary with connection test results
        """
        results = {
            "oauth2_available": self.client is not None,
            "oauth1_available": self.api is not None,
            "can_post": False,
            "user_info": None,
            "errors": []
        }
        
        # Test OAuth 2.0 connection
        if self.client:
            try:
                user = self.client.get_me()
                if user and user.data:
                    results["user_info"] = {
                        "username": user.data.username,
                        "name": user.data.name,
                        "id": user.data.id
                    }
                    results["can_post"] = True
                    logger.info("OAuth 2.0 connection test successful")
                else:
                    results["can_post"] = True  # App-only auth can still post
                    logger.info("OAuth 2.0 app-only connection successful")
            except Exception as e:
                results["errors"].append(f"OAuth 2.0 test failed: {e}")
                logger.error(f"OAuth 2.0 connection test failed: {e}")
        
        # Test OAuth 1.0a connection
        if self.api and not results["can_post"]:
            try:
                user = self.api.verify_credentials()
                if user:
                    results["user_info"] = {
                        "username": user.screen_name,
                        "name": user.name,
                        "id": str(user.id)
                    }
                    results["can_post"] = True
                    logger.info("OAuth 1.0a connection test successful")
            except Exception as e:
                results["errors"].append(f"OAuth 1.0a test failed: {e}")
                logger.error(f"OAuth 1.0a connection test failed: {e}")
        
        return results
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """Get current authentication status and available methods."""
        twitter_config = self.config.twitter
        
        return {
            "oauth2_configured": self._can_use_oauth2(),
            "oauth1_configured": self._can_use_oauth1(),
            "active_client": self.client is not None,
            "active_api": self.api is not None,
            "credentials_available": {
                "bearer_token": bool(twitter_config.bearer_token),
                "client_id": bool(twitter_config.client_id),
                "client_secret": bool(twitter_config.client_secret),
                "api_key": bool(twitter_config.api_key),
                "api_secret": bool(twitter_config.api_secret),
                "access_token": bool(twitter_config.access_token),
                "access_token_secret": bool(twitter_config.access_token_secret)
            }
        } 