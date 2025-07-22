#!/usr/bin/env python3
"""
Safe Twitter Collection Script for Basic Plan
Only uses endpoints that work with Basic plan limitations
FOCUSES ON POST COLLECTION EFFICIENCY
STORES ALL POSTS FOR IDENTITY TRAINING
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from api_safeguard import TwitterAPISafeguard

class SafeTwitterCollector:
    """Safe Twitter collector that respects Basic plan limits with post-focused approach."""
    
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not self.bearer_token:
            raise ValueError("TWITTER_BEARER_TOKEN not found in environment")
        
        self.headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        
        self.safeguard = TwitterAPISafeguard()
        self.data_dir = Path("data/safe_collection")
        self.data_dir.mkdir(exist_ok=True)
        
        # Identity training integration
        self.training_storage_dir = Path("data/training_posts")
        self.training_storage_dir.mkdir(exist_ok=True)
        
        # Track API calls for this session
        self.session_api_calls = {"search": 0, "user_lookup": 0}
        
        # Store ALL posts for training (even failed attempts)
        self.all_posts_collected = []
        self.failed_requests = []
    
    def safe_search_tweets(self, query: str, max_results: int = 100) -> list:
        """Safely search for tweets using the search endpoint."""
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': query,
            'max_results': min(max_results, 100),  # Basic plan max
            'tweet.fields': 'id,text,created_at,public_metrics,context_annotations,author_id',
            'expansions': 'author_id',
            'user.fields': 'username,public_metrics'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.session_api_calls["search"] += 1
            
            # Store request details for training (even if failed)
            request_record = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'max_results': max_results,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])
                users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
                
                # Add username to each tweet
                for tweet in tweets:
                    user = users.get(tweet['author_id'], {})
                    tweet['author_username'] = user.get('username', 'unknown')
                    
                    # Store for identity training
                    self.all_posts_collected.append({
                        'tweet': tweet,
                        'query_used': query,
                        'collection_time': datetime.now().isoformat(),
                        'api_source': 'search'
                    })
                
                request_record['tweets_returned'] = len(tweets)
                print(f"✅ Successfully collected {len(tweets)} tweets for query: {query}")
                return tweets
            
            elif response.status_code == 429:
                print("❌ Rate limited! Stopping collection.")
                request_record['error'] = 'rate_limited'
                self.failed_requests.append(request_record)
                return []
            
            else:
                error_msg = response.text[:200]
                print(f"❌ API Error {response.status_code}: {error_msg}")
                request_record['error'] = error_msg
                self.failed_requests.append(request_record)
                return []
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Request failed: {error_msg}")
            self.failed_requests.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'error': error_msg,
                'success': False
            })
            return []
    
    def save_training_data(self):
        """Save all collected posts for identity training."""
        if not self.all_posts_collected and not self.failed_requests:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save successful collections
        if self.all_posts_collected:
            training_file = self.training_storage_dir / f"training_posts_{timestamp}.json"
            training_data = {
                'collection_session': {
                    'timestamp': datetime.now().isoformat(),
                    'total_posts': len(self.all_posts_collected),
                    'api_calls_made': self.session_api_calls,
                    'session_type': 'safe_collection'
                },
                'posts': self.all_posts_collected
            }
            
            with open(training_file, 'w') as f:
                json.dump(training_data, f, indent=2, default=str)
            
            print(f"💾 Saved {len(self.all_posts_collected)} posts for training: {training_file}")
        
        # Save failed requests (we paid for these API calls!)
        if self.failed_requests:
            failed_file = self.training_storage_dir / f"failed_requests_{timestamp}.json"
            failed_data = {
                'session_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_failed_requests': len(self.failed_requests),
                    'note': 'These API calls were charged but failed - store for analysis'
                },
                'failed_requests': self.failed_requests
            }
            
            with open(failed_file, 'w') as f:
                json.dump(failed_data, f, indent=2, default=str)
            
            print(f"⚠️ Saved {len(self.failed_requests)} failed requests: {failed_file}")
    
    def collect_crypto_content(self, target_posts: int = 500, save_file: bool = True) -> dict:
        """Collect crypto content with post-based limits."""
        
        # Check if we can collect before starting
        limits_check = self.safeguard.check_post_limits()
        if not limits_check["can_collect"]:
            print("❌ Collection not allowed due to limits")
            return {"tweets": [], "stats": {"error": "limits_exceeded"}}
        
        # Adaptive query strategy based on target
        if target_posts <= 250:
            queries_and_limits = [
                ("bitcoin OR BTC lang:en -is:retweet", 50),
                ("ethereum OR ETH lang:en -is:retweet", 50),
                ("crypto OR cryptocurrency lang:en -is:retweet", 50),
                ("DeFi OR yield lang:en -is:retweet", 50),
                ("NFT OR web3 lang:en -is:retweet", 50)
            ]
        else:
            # For larger collections, use more targeted queries
            queries_and_limits = [
                ("bitcoin OR BTC lang:en -is:retweet", 100),
                ("ethereum OR ETH lang:en -is:retweet", 100),
                ("crypto OR cryptocurrency lang:en -is:retweet", 100),
                ("DeFi OR yield farming lang:en -is:retweet", 100),
                ("NFT OR web3 OR metaverse lang:en -is:retweet", 100),
                ("altcoin OR altcoins lang:en -is:retweet", 50),
                ("blockchain OR crypto lang:en -is:retweet", 50)
            ]
        
        all_tweets = []
        collection_stats = {
            "target_posts": target_posts,
            "total_tweets": 0,
            "queries_processed": 0,
            "queries_failed": 0,
            "start_time": datetime.now().isoformat(),
            "api_calls_made": {"search": 0, "user_lookup": 0},
            "training_data_collected": 0,
            "failed_api_calls": 0
        }
        
        print(f"🔍 Starting crypto collection (target: {target_posts} posts)")
        current_usage = self.safeguard.get_usage_summary()
        print(f"📊 Current usage: {current_usage['posts_last_hour']}/h, {current_usage['posts_last_day']}/day")
        
        posts_collected = 0
        
        for i, (query, max_results) in enumerate(queries_and_limits, 1):
            # Stop if we've hit our target
            if posts_collected >= target_posts:
                print(f"🎯 Target reached: {posts_collected} posts collected")
                break
                
            print(f"\n📊 Query {i}/{len(queries_and_limits)}: {query}")
            
            # Adjust max_results based on remaining target
            remaining_needed = target_posts - posts_collected
            actual_max = min(max_results, remaining_needed, 100)  # Never exceed 100
            
            tweets = self.safe_search_tweets(query, max_results=actual_max)
            
            if tweets:
                all_tweets.extend(tweets)
                posts_collected += len(tweets)
                collection_stats["queries_processed"] += 1
                collection_stats["total_tweets"] += len(tweets)
                
                # Rate limiting - be respectful
                if i < len(queries_and_limits) and posts_collected < target_posts:
                    print("⏱️ Waiting 2 seconds between queries...")
                    time.sleep(2)
            else:
                collection_stats["queries_failed"] += 1
                collection_stats["failed_api_calls"] += 1
                # Continue with other queries even if one fails
        
        collection_stats["end_time"] = datetime.now().isoformat()
        collection_stats["api_calls_made"] = self.session_api_calls.copy()
        collection_stats["training_data_collected"] = len(self.all_posts_collected)
        collection_stats["failed_api_calls"] = len(self.failed_requests)
        
        # Remove duplicates
        unique_tweets = {}
        for tweet in all_tweets:
            unique_tweets[tweet['id']] = tweet
        
        final_tweets = list(unique_tweets.values())
        collection_stats["unique_tweets"] = len(final_tweets)
        collection_stats["duplicates_removed"] = len(all_tweets) - len(final_tweets)
        
        # Record this collection in safeguard
        self.safeguard.record_collection(
            posts_collected=len(final_tweets),
            api_calls_made=self.session_api_calls,
            success=len(final_tweets) > 0
        )
        
        result = {
            "tweets": final_tweets,
            "stats": collection_stats,
            "safeguard_status": self.safeguard.get_usage_summary(),
            "training_data": {
                "posts_for_training": len(self.all_posts_collected),
                "failed_requests": len(self.failed_requests),
                "total_api_cost": self.session_api_calls["search"]  # Each call costs money
            }
        }
        
        # Save collection file
        if save_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.data_dir / f"crypto_collection_{len(final_tweets)}posts_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"💾 Saved collection to: {filename}")
        
        # ALWAYS save training data - we paid for every API call
        self.save_training_data()
        
        print(f"\n📊 Collection Summary:")
        print(f"  🎯 Target: {target_posts} posts")
        print(f"  ✅ Collected: {collection_stats['unique_tweets']} unique posts")
        print(f"  🔄 Duplicates removed: {collection_stats['duplicates_removed']}")
        print(f"  📞 API calls: {sum(self.session_api_calls.values())}")
        print(f"  ⏱️ Efficiency: {collection_stats['unique_tweets'] / sum(self.session_api_calls.values()):.1f} posts/call")
        print(f"  📈 Updated usage: {result['safeguard_status']['posts_last_hour']}/h")
        print(f"  🤖 Training data: {len(self.all_posts_collected)} posts + {len(self.failed_requests)} failed requests")
        print(f"  💰 API cost: {self.session_api_calls['search']} search calls (charged per post retrieved)")
        
        return result

if __name__ == "__main__":
    try:
        collector = SafeTwitterCollector()
        
        # Allow user to specify target posts
        import sys
        target = int(sys.argv[1]) if len(sys.argv) > 1 else 300
        
        result = collector.collect_crypto_content(target_posts=target)
        
        if result["tweets"]:
            posts_count = len(result['tweets'])
            efficiency = posts_count / sum(result['stats']['api_calls_made'].values())
            training_posts = result['training_data']['posts_for_training']
            
            print(f"\n🎉 Success! Collected {posts_count} crypto tweets!")
            print(f"📊 Efficiency: {efficiency:.1f} posts per API call")
            print(f"🤖 Training ready: {training_posts} posts stored for identity training")
            print(f"📝 Sample: {result['tweets'][0]['text'][:100]}...")
        else:
            print("\n❌ No tweets collected. Check limits and API credentials.")
            print(f"🤖 Training data: {result['training_data']['failed_requests']} failed requests stored")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Make sure TWITTER_BEARER_TOKEN is set in your environment")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 