#!/usr/bin/env python3
"""
Twitter API Access Test
Tests the Twitter API credentials and basic functionality
"""

import os
import sys
import tweepy
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config

def test_twitter_api_access():
    """Test Twitter API access with credentials from .env"""
    print("Testing Twitter API Access...")
    print("=" * 50)
    
    try:
        # Load configuration
        config = Config()
        
        # Test API credentials
        auth = tweepy.OAuthHandler(
            config.twitter_api_key,
            config.twitter_api_secret
        )
        auth.set_access_token(
            config.twitter_access_token,
            config.twitter_access_token_secret
        )
        
        # Create API object
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Test API connection
        print("Testing API connection...")
        user = api.verify_credentials()
        print(f"SUCCESS: Connected as @{user.screen_name}")
        print(f"   User ID: {user.id}")
        print(f"   Followers: {user.followers_count}")
        
        # Test rate limits
        print("\nChecking rate limits...")
        rate_limit_status = api.rate_limit_status()
        
        # Check relevant endpoints
        endpoints = {
            "users/show": "User lookup",
            "statuses/user_timeline": "User timeline",
            "search/tweets": "Search tweets"
        }
        
        for endpoint, description in endpoints.items():
            if endpoint in rate_limit_status['resources']:
                remaining = rate_limit_status['resources'][endpoint]['remaining']
                limit = rate_limit_status['resources'][endpoint]['limit']
                reset_time = rate_limit_status['resources'][endpoint]['reset']
                
                print(f"  {description}: {remaining}/{limit} remaining")
                print(f"    Reset time: {datetime.fromtimestamp(reset_time)}")
        
        return True, api
        
    except tweepy.TweepError as e:
        print(f"ERROR: Twitter API Error: {e}")
        return False, None
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False, None

def test_user_lookup(api):
    """Test looking up specific users"""
    print("\nTesting User Lookup...")
    print("=" * 50)
    
    # Test with some known crypto influencers
    test_users = [
        "VitalikButerin",  # Ethereum founder
        "cz_binance",      # Binance CEO
        "elonmusk"         # Tesla CEO (for testing)
    ]
    
    results = {}
    
    for username in test_users:
        try:
            print(f"Looking up @{username}...")
            user = api.get_user(screen_name=username)
            
            user_info = {
                "id": user.id,
                "screen_name": user.screen_name,
                "name": user.name,
                "followers_count": user.followers_count,
                "friends_count": user.friends_count,
                "statuses_count": user.statuses_count,
                "created_at": user.created_at.isoformat(),
                "verified": user.verified,
                "protected": user.protected
            }
            
            results[username] = user_info
            
            print(f"  SUCCESS: Found {user.name} (@{user.screen_name})")
            print(f"     Followers: {user.followers_count:,}")
            print(f"     Tweets: {user.statuses_count:,}")
            print(f"     Verified: {user.verified}")
            
        except tweepy.TweepError as e:
            print(f"  ERROR: Error looking up @{username}: {e}")
            results[username] = {"error": str(e)}
    
    return results

def test_timeline_fetch(api, max_tweets=5):
    """Test fetching user timelines"""
    print(f"\nTesting Timeline Fetch (max {max_tweets} tweets)...")
    print("=" * 50)
    
    # Test with a single user first
    test_user = "VitalikButerin"
    
    try:
        print(f"Fetching timeline for @{test_user}...")
        tweets = api.user_timeline(
            screen_name=test_user,
            count=max_tweets,
            tweet_mode="extended"
        )
        
        results = []
        for i, tweet in enumerate(tweets, 1):
            tweet_info = {
                "id": tweet.id,
                "created_at": tweet.created_at.isoformat(),
                "text": tweet.full_text[:100] + "..." if len(tweet.full_text) > 100 else tweet.full_text,
                "retweet_count": tweet.retweet_count,
                "favorite_count": tweet.favorite_count,
                "lang": tweet.lang
            }
            
            results.append(tweet_info)
            
            print(f"  {i}. {tweet.created_at.strftime('%Y-%m-%d %H:%M')}")
            print(f"     {tweet_info['text']}")
            print(f"     RT: {tweet.retweet_count}, ❤️: {tweet.favorite_count}")
            print()
        
        return results
        
    except tweepy.TweepError as e:
        print(f"ERROR: Error fetching timeline: {e}")
        return []

def test_search_functionality(api):
    """Test search functionality"""
    print("\nTesting Search Functionality...")
    print("=" * 50)
    
    search_queries = [
        "bitcoin",
        "ethereum",
        "cryptocurrency"
    ]
    
    results = {}
    
    for query in search_queries:
        try:
            print(f"Searching for '{query}'...")
            tweets = api.search_tweets(
                q=query,
                count=5,
                tweet_mode="extended",
                lang="en"
            )
            
            query_results = []
            for tweet in tweets:
                tweet_info = {
                    "id": tweet.id,
                    "user": tweet.user.screen_name,
                    "created_at": tweet.created_at.isoformat(),
                    "text": tweet.full_text[:100] + "..." if len(tweet.full_text) > 100 else tweet.full_text,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count
                }
                query_results.append(tweet_info)
            
            results[query] = query_results
            
            print(f"  SUCCESS: Found {len(tweets)} tweets")
            for tweet in tweets[:2]:  # Show first 2
                print(f"    @{tweet.user.screen_name}: {tweet.full_text[:50]}...")
            
        except tweepy.TweepError as e:
            print(f"  ERROR: Error searching for '{query}': {e}")
            results[query] = {"error": str(e)}
    
    return results

def generate_test_report(api_success, user_results, timeline_results, search_results):
    """Generate a comprehensive test report"""
    print("\nTwitter API Test Summary")
    print("=" * 50)
    
    timestamp = datetime.now().isoformat()
    
    summary = {
        "timestamp": timestamp,
        "api_access": api_success,
        "user_lookup_results": user_results,
        "timeline_results": timeline_results,
        "search_results": search_results
    }
    
    # Calculate success rates
    successful_users = sum(1 for user_data in user_results.values() if "error" not in user_data)
    total_users = len(user_results)
    
    print(f"API Access: {'SUCCESS' if api_success else 'FAILED'}")
    print(f"User Lookup: {successful_users}/{total_users} successful")
    print(f"Timeline Fetch: {'SUCCESS' if timeline_results else 'FAILED'}")
    print(f"Search: {'SUCCESS' if search_results else 'FAILED'}")
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"twitter_api_test_{timestamp_str}.json"
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Recommendations
    print("\nRecommendations:")
    if api_success:
        print("SUCCESS: Twitter API access is working correctly")
        print("SUCCESS: Ready to proceed with data ingestion")
    else:
        print("ERROR: Check Twitter API credentials in .env file")
        print("ERROR: Verify API permissions and rate limits")
    
    return summary

def main():
    """Main test function"""
    print("Twitter API Access Test Suite")
    print("=" * 60)
    
    # Test basic API access
    api_success, api = test_twitter_api_access()
    
    if not api_success:
        print("\nERROR: Cannot proceed without API access")
        return
    
    # Test user lookup
    user_results = test_user_lookup(api)
    
    # Test timeline fetch
    timeline_results = test_timeline_fetch(api)
    
    # Test search functionality
    search_results = test_search_functionality(api)
    
    # Generate report
    summary = generate_test_report(api_success, user_results, timeline_results, search_results)
    
    print("\nTwitter API test completed")

if __name__ == "__main__":
    main() 