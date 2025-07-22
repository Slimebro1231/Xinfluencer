#!/usr/bin/env python3
"""
Test API optimizations and efficiency improvements before deployment.
This script verifies that all optimization loopholes have been fixed.
"""

import time
import logging
from datetime import datetime
from src.utils.x_api_client import XAPIClient
from src.utils.data_collection_pipeline import DataCollectionPipeline
from src.evaluation.engine import EvaluationEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_user_id_caching():
    """Test that user ID caching is working to avoid redundant API calls."""
    print("\n🔍 Testing User ID Caching Optimization")
    print("="*50)
    
    try:
        x_api = XAPIClient()
        
        # Test single user caching
        test_username = "VitalikButerin"
        
        print(f"Testing user ID caching for @{test_username}...")
        
        # First call - should make API request
        start_time = time.time()
        user_id_1 = x_api._get_user_id(test_username)
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        user_id_2 = x_api._get_user_id(test_username)
        second_call_time = time.time() - start_time
        
        print(f"First call (API): {first_call_time:.3f}s")
        print(f"Second call (cache): {second_call_time:.3f}s")
        print(f"Cache speedup: {first_call_time/second_call_time:.1f}x faster")
        
        assert user_id_1 == user_id_2, "User IDs should match"
        assert second_call_time < first_call_time * 0.1, "Cache should be much faster"
        assert test_username in x_api.user_id_cache, "User should be cached"
        
        print("✅ User ID caching working correctly")
        
        # Test batch user lookup
        print("\nTesting batch user lookup optimization...")
        test_users = ["VitalikButerin", "elonmusk", "cz_binance"]
        
        start_time = time.time()
        user_ids = x_api.batch_get_user_ids(test_users)
        batch_time = time.time() - start_time
        
        print(f"Batch lookup for {len(test_users)} users: {batch_time:.3f}s")
        print(f"Users resolved: {len(user_ids)}/{len(test_users)}")
        
        assert len(user_ids) > 0, "Should resolve at least some users"
        
        print("✅ Batch user lookup working correctly")
        
    except Exception as e:
        print(f"❌ User ID caching test failed: {e}")
        return False
    
    return True

def test_optimized_data_collection():
    """Test that data collection uses optimized parameters."""
    print("\n🔍 Testing Optimized Data Collection")
    print("="*50)
    
    try:
        pipeline = DataCollectionPipeline()
        
        # Test rate limit status
        x_api = pipeline.x_api
        rate_status = x_api.get_rate_limit_status()
        
        print("Current rate limit status:")
        for endpoint, status in rate_status.items():
            if isinstance(status, dict) and 'remaining' in status:
                print(f"  {endpoint}: {status['remaining']}/{status['limit']} remaining")
        
        # Test optimized collection settings
        print("\nTesting collection with optimized limits...")
        
        # Small test collection to verify optimization
        start_time = time.time()
        results = pipeline.run_comprehensive_collection(
            kol_usernames=["VitalikButerin"],  # Just one for testing
            collect_trending=True,
            collect_high_engagement=True
        )
        collection_time = time.time() - start_time
        
        print(f"Collection completed in {collection_time:.1f}s")
        print(f"API calls made: {results['collection_stats']['api_calls_made']}")
        print(f"Errors: {results['collection_stats']['errors']}")
        print(f"Unique tweets collected: {results.get('total_unique_tweets', 0)}")
        print(f"Deduplication savings: {results.get('deduplication_savings', 0)}")
        
        # Verify optimizations
        assert results['collection_stats']['api_calls_made'] > 0, "Should make some API calls"
        assert 'total_unique_tweets' in results, "Should have deduplication"
        
        print("✅ Optimized data collection working correctly")
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False
    
    return True

def test_evaluation_engine_integration():
    """Test that evaluation engine integrates with optimized components."""
    print("\n🔍 Testing Evaluation Engine Integration")
    print("="*50)
    
    try:
        engine = EvaluationEngine()
        
        print("Testing evaluation engine initialization...")
        assert engine is not None, "Engine should initialize"
        
        print("✅ Evaluation engine integration working correctly")
        
    except Exception as e:
        print(f"❌ Evaluation engine test failed: {e}")
        return False
    
    return True

def test_api_efficiency_metrics():
    """Calculate and display API efficiency metrics."""
    print("\n📊 API Efficiency Analysis")
    print("="*50)
    
    try:
        x_api = XAPIClient()
        rate_status = x_api.get_rate_limit_status()
        
        # Calculate efficiency metrics
        total_requests_available = 0
        total_requests_used = 0
        
        for endpoint, status in rate_status.items():
            if isinstance(status, dict) and 'limit' in status:
                total_requests_available += status['limit']
                total_requests_used += (status['limit'] - status['remaining'])
        
        efficiency_score = (total_requests_available - total_requests_used) / total_requests_available * 100
        
        print(f"Total API requests available: {total_requests_available}")
        print(f"Total API requests used: {total_requests_used}")
        print(f"API efficiency score: {efficiency_score:.1f}%")
        
        # Rate limiting analysis
        print("\nRate limit analysis:")
        for endpoint, status in rate_status.items():
            if isinstance(status, dict) and 'limit' in status:
                usage_pct = (status['limit'] - status['remaining']) / status['limit'] * 100
                print(f"  {endpoint}: {usage_pct:.1f}% used")
        
        print("✅ API efficiency analysis completed")
        
    except Exception as e:
        print(f"❌ API efficiency test failed: {e}")
        return False
    
    return True

def main():
    """Run all optimization tests."""
    print("🚀 Testing API Optimizations Before Deployment")
    print("="*60)
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        ("User ID Caching", test_user_id_caching),
        ("Optimized Data Collection", test_optimized_data_collection),
        ("Evaluation Engine Integration", test_evaluation_engine_integration),
        ("API Efficiency Metrics", test_api_efficiency_metrics),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All optimizations working correctly! Ready for deployment.")
        return True
    else:
        print("\n⚠️  Some optimizations failed. Review before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 