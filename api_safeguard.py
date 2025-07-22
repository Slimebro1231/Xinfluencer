#!/usr/bin/env python3
"""
API Safeguard System for Twitter Basic Plan
Prevents hanging processes and API quota waste
FOCUSES ON POSTS COLLECTED, NOT JUST API CALLS
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

class TwitterAPISafeguard:
    """Safeguard system to prevent API quota waste with post-based limits."""
    
    def __init__(self):
        self.usage_file = Path("api_usage_log.json")
        
        # Updated limits focusing on posts collected
        self.limits = {
            "posts_per_hour": 1000,      # Max posts to collect per hour
            "posts_per_day": 10000,      # Max posts to collect per day
            "api_calls_per_15min": {     # Still track API calls for safety
                "search": 450,           # Leave 50 buffer from 500 limit
                "user_lookup": 90        # Leave 10 buffer from 100 limit
            }
        }
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - API_SAFEGUARD - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('api_safeguard.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_post_limits(self) -> dict:
        """Check if we're within post collection limits."""
        usage = self.load_usage()
        current_time = datetime.now()
        
        # Check hourly post limit
        hour_ago = current_time - timedelta(hours=1)
        posts_last_hour = sum(
            collection.get("posts_collected", 0)
            for collection in usage.get("collections", [])
            if datetime.fromisoformat(collection["timestamp"]) > hour_ago
        )
        
        # Check daily post limit  
        day_ago = current_time - timedelta(days=1)
        posts_last_day = sum(
            collection.get("posts_collected", 0)
            for collection in usage.get("collections", [])
            if datetime.fromisoformat(collection["timestamp"]) > day_ago
        )
        
        # Check API rate limits
        api_limits_ok = self._check_api_rate_limits()
        
        limits_status = {
            "posts_last_hour": posts_last_hour,
            "posts_last_day": posts_last_day,
            "hourly_limit_ok": posts_last_hour < self.limits["posts_per_hour"],
            "daily_limit_ok": posts_last_day < self.limits["posts_per_day"],
            "api_limits_ok": api_limits_ok,
            "can_collect": True
        }
        
        # Determine if collection can proceed
        if not limits_status["hourly_limit_ok"]:
            self.logger.warning(f"Hourly post limit reached: {posts_last_hour}/{self.limits['posts_per_hour']}")
            limits_status["can_collect"] = False
            
        if not limits_status["daily_limit_ok"]:
            self.logger.warning(f"Daily post limit reached: {posts_last_day}/{self.limits['posts_per_day']}")
            limits_status["can_collect"] = False
            
        if not limits_status["api_limits_ok"]["all_ok"]:
            self.logger.warning("API rate limits reached")
            limits_status["can_collect"] = False
        
        if limits_status["can_collect"]:
            self.logger.info(f"Collection OK - Posts: {posts_last_hour}/h, {posts_last_day}/day")
            
        return limits_status
    
    def _check_api_rate_limits(self) -> dict:
        """Check API rate limits."""
        usage = self.load_usage()
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=15)
        
        api_status = {}
        all_ok = True
        
        for endpoint_type, limit in self.limits["api_calls_per_15min"].items():
            recent_calls = [
                call for call in usage.get("api_calls", {}).get(endpoint_type, [])
                if datetime.fromisoformat(call["timestamp"]) > window_start
            ]
            
            current_count = len(recent_calls)
            endpoint_ok = current_count < limit
            
            api_status[endpoint_type] = {
                "current": current_count,
                "limit": limit,
                "ok": endpoint_ok
            }
            
            if not endpoint_ok:
                all_ok = False
        
        api_status["all_ok"] = all_ok
        return api_status
    
    def record_collection(self, posts_collected: int, api_calls_made: dict, success: bool = True):
        """Record a collection session with posts collected and API calls made."""
        usage = self.load_usage()
        
        # Initialize structure if needed
        if "collections" not in usage:
            usage["collections"] = []
        if "api_calls" not in usage:
            usage["api_calls"] = {}
        
        # Record collection session
        collection_record = {
            "timestamp": datetime.now().isoformat(),
            "posts_collected": posts_collected,
            "api_calls_made": api_calls_made,
            "success": success
        }
        
        usage["collections"].append(collection_record)
        
        # Record individual API calls
        for endpoint_type, count in api_calls_made.items():
            if endpoint_type not in usage["api_calls"]:
                usage["api_calls"][endpoint_type] = []
            
            for _ in range(count):
                usage["api_calls"][endpoint_type].append({
                    "timestamp": datetime.now().isoformat(),
                    "success": success
                })
        
        # Clean old data (keep last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        usage["collections"] = [
            c for c in usage["collections"]
            if datetime.fromisoformat(c["timestamp"]) > cutoff
        ]
        
        for endpoint_type in usage["api_calls"]:
            usage["api_calls"][endpoint_type] = [
                call for call in usage["api_calls"][endpoint_type]
                if datetime.fromisoformat(call["timestamp"]) > cutoff
            ]
        
        self.save_usage(usage)
        
        if success:
            self.logger.info(f"Collection recorded: {posts_collected} posts, {sum(api_calls_made.values())} API calls")
        else:
            self.logger.error(f"Failed collection recorded: {api_calls_made}")
    
    def load_usage(self) -> dict:
        """Load usage data from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading usage file: {e}")
        return {"collections": [], "api_calls": {}}
    
    def save_usage(self, usage: dict):
        """Save usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(usage, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving usage file: {e}")
    
    def get_usage_summary(self) -> dict:
        """Get current usage summary focused on posts."""
        usage = self.load_usage()
        current_time = datetime.now()
        
        # Calculate post collection stats
        hour_ago = current_time - timedelta(hours=1)
        day_ago = current_time - timedelta(days=1)
        
        posts_last_hour = sum(
            c.get("posts_collected", 0) for c in usage.get("collections", [])
            if datetime.fromisoformat(c["timestamp"]) > hour_ago
        )
        
        posts_last_day = sum(
            c.get("posts_collected", 0) for c in usage.get("collections", [])
            if datetime.fromisoformat(c["timestamp"]) > day_ago
        )
        
        # Recent collections
        recent_collections = [
            c for c in usage.get("collections", [])
            if datetime.fromisoformat(c["timestamp"]) > hour_ago
        ]
        
        return {
            "posts_last_hour": posts_last_hour,
            "posts_last_day": posts_last_day,
            "hourly_limit": self.limits["posts_per_hour"],
            "daily_limit": self.limits["posts_per_day"],
            "recent_collections": len(recent_collections),
            "total_collections_today": len([
                c for c in usage.get("collections", [])
                if datetime.fromisoformat(c["timestamp"]) > day_ago
            ]),
            "api_status": self._check_api_rate_limits()
        }

# Usage example
if __name__ == "__main__":
    safeguard = TwitterAPISafeguard()
    
    print("🛡️ Twitter API Safeguard Status (Post-Focused)")
    print("=" * 50)
    
    # Check if we can collect
    limits = safeguard.check_post_limits()
    
    print(f"📊 Current Status:")
    print(f"  Posts last hour: {limits['posts_last_hour']}/{safeguard.limits['posts_per_hour']}")
    print(f"  Posts last day: {limits['posts_last_day']}/{safeguard.limits['posts_per_day']}")
    print(f"  Collection allowed: {'✅ YES' if limits['can_collect'] else '❌ NO'}")
    
    print(f"\n🔧 API Rate Limits:")
    for endpoint, status in limits["api_limits_ok"].items():
        if endpoint != "all_ok":
            print(f"  {endpoint}: {status['current']}/{status['limit']} {'✅' if status['ok'] else '❌'}")
    
    summary = safeguard.get_usage_summary()
    print(f"\n📈 Recent Activity:")
    print(f"  Collections last hour: {summary['recent_collections']}")
    print(f"  Total collections today: {summary['total_collections_today']}") 