#!/usr/bin/env python3
"""
Focused crypto/RWA KOL collection - eliminates noise, maximizes signal.
Only collects from accounts that post primarily crypto/DeFi/RWA content.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from src.utils.data_collection_pipeline import DataCollectionPipeline
from src.utils.x_api_client import XAPIClient

# FOCUSED CRYPTO/RWA KOL LIST (Quality over quantity)
CRYPTO_FOCUSED_KOLS = [
    # Core Ethereum/DeFi Builders (Highest Signal)
    "VitalikButerin",      # Ethereum founder - technical insights
    "haydenzadams",        # Uniswap founder - DeFi innovation
    "stani_kulechov",      # Aave founder - lending protocols
    
    # DeFi Protocol Architects
    "AndreCronjeTech",     # DeFi architect (YFI, Fantom)
    "rleshner",            # Compound founder
    "bantg",               # Yearn core developer
    
    # RWA/Institutional Crypto
    "centrifuge",          # RWA tokenization protocol
    "MakerDAO",            # DAI and RWA integration
    "chainlink",           # Oracle infrastructure for RWA
    
    # Research/Educational (Crypto-focused)
    "MessariCrypto",       # Institutional research
    "DeFiPulse",           # DeFi analytics
    "defiprime",           # DeFi education
    
    # Technical Educators (No price noise)
    "evan_van_ness",       # Ethereum weekly newsletter
    "sassal0x",            # Technical analysis/education
    "tokenbrice"           # DeFi educator
]

def check_api_readiness():
    """Check if API is ready for collection."""
    print("🔧 Checking API readiness...")
    
    try:
        x_api = XAPIClient()
        status = x_api.get_rate_limit_status()
        
        total_remaining = 0
        for endpoint, info in status.items():
            if isinstance(info, dict) and 'remaining' in info:
                remaining = info['remaining']
                total_remaining += remaining
                print(f"  {endpoint}: {remaining}/{info['limit']} remaining")
        
        if total_remaining < 20:
            print(f"  ⚠️  Low quota ({total_remaining} requests) - consider waiting")
            return False
        else:
            print(f"  ✅ Ready ({total_remaining} requests available)")
            return True
            
    except Exception as e:
        print(f"  ❌ API check failed: {e}")
        return False

def collect_conservative_sample():
    """Collect a small, high-quality sample from top crypto KOLs."""
    print("🎯 Conservative Collection Strategy")
    print("=" * 50)
    
    # Check API first
    if not check_api_readiness():
        print("\n⏳ Rate limits too low. Please wait and try again later.")
        return False
    
    # Top priority accounts (most crypto-focused)
    priority_kols = CRYPTO_FOCUSED_KOLS[:5]  # Start with just 5
    
    print(f"\n📋 Priority KOLs ({len(priority_kols)} accounts):")
    for i, kol in enumerate(priority_kols, 1):
        print(f"  {i}. @{kol}")
    
    # Conservative collection
    output_dir = Path("data/manual_review")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = DataCollectionPipeline()
    
    print(f"\n🚀 Starting conservative collection (3 tweets per KOL)...")
    start_time = time.time()
    
    try:
        collected_data = pipeline.collect_kol_data(
            kol_usernames=priority_kols,
            tweets_per_kol=3,  # Very conservative
            save_to_file=False
        )
        
        collection_time = time.time() - start_time
        total_tweets = sum(len(tweets) for tweets in collected_data.values())
        
        print(f"\n📊 Results:")
        print(f"  ⏱️  Time: {collection_time:.1f}s")
        print(f"  👥 KOLs: {len(collected_data)}/{len(priority_kols)}")
        print(f"  📝 Tweets: {total_tweets}")
        print(f"  🔧 API calls: {pipeline.collection_stats['api_calls_made']}")
        
        # Analyze content quality
        analysis = analyze_crypto_content(collected_data)
        
        # Save results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"focused_crypto_sample_{timestamp}.json"
        
        save_results = {
            "strategy": "focused_crypto_conservative",
            "timestamp": datetime.utcnow().isoformat(),
            "collection_stats": pipeline.collection_stats,
            "content_analysis": analysis,
            "collected_data": collected_data
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Create readable summary
        create_crypto_review_summary(save_results, output_dir / f"crypto_review_{timestamp}.md")
        
        print(f"\n💾 Files saved:")
        print(f"  📄 Data: {results_file}")
        print(f"  📋 Review: crypto_review_{timestamp}.md")
        
        print(f"\n🎉 Conservative collection complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        return False

def analyze_crypto_content(collected_data):
    """Analyze the crypto-relevance of collected content."""
    analysis = {}
    
    crypto_keywords = [
        'defi', 'dapp', 'protocol', 'token', 'crypto', 'blockchain', 
        'ethereum', 'rwa', 'yield', 'liquidity', 'smart contract',
        'tokenization', 'dao', 'governance', 'staking', 'lending'
    ]
    
    for username, tweets in collected_data.items():
        if not tweets:
            continue
            
        crypto_count = 0
        total_engagement = 0
        avg_length = 0
        
        for tweet in tweets:
            # Check crypto relevance
            text_lower = tweet.text.lower()
            if any(keyword in text_lower for keyword in crypto_keywords):
                crypto_count += 1
            
            # Calculate engagement
            metrics = tweet.public_metrics
            engagement = sum(metrics.get(k, 0) for k in ['like_count', 'retweet_count', 'reply_count', 'quote_count'])
            total_engagement += engagement
            avg_length += len(tweet.text)
        
        analysis[username] = {
            "tweets_collected": len(tweets),
            "crypto_relevance_ratio": crypto_count / len(tweets) if tweets else 0,
            "avg_engagement": total_engagement / len(tweets) if tweets else 0,
            "avg_tweet_length": avg_length / len(tweets) if tweets else 0,
            "quality_score": (crypto_count / len(tweets)) * min(total_engagement / len(tweets) / 100, 1) if tweets else 0
        }
    
    return analysis

def create_crypto_review_summary(results, output_file):
    """Create a summary focused on crypto content quality."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Focused Crypto KOL Collection - Quality Review\n\n")
        
        f.write(f"**Strategy**: Conservative, crypto-focused sampling\n")
        f.write(f"**Date**: {results['timestamp']}\n")
        f.write(f"**API Calls**: {results['collection_stats']['api_calls_made']}\n")
        f.write(f"**Total Tweets**: {results['collection_stats']['tweets_collected']}\n\n")
        
        f.write("## Content Quality Analysis\n\n")
        
        # Sort KOLs by quality score
        analysis = results['content_analysis']
        sorted_kols = sorted(analysis.items(), key=lambda x: x[1]['quality_score'], reverse=True)
        
        f.write("| Rank | KOL | Crypto % | Avg Engagement | Quality Score |\n")
        f.write("|------|-----|----------|---------------|---------------|\n")
        
        for i, (username, data) in enumerate(sorted_kols, 1):
            f.write(f"| {i} | @{username} | {data['crypto_relevance_ratio']:.1%} | {data['avg_engagement']:.1f} | {data['quality_score']:.3f} |\n")
        
        f.write("\n## Sample High-Quality Tweets\n\n")
        
        # Show best tweets from each KOL
        for username, tweets in results['collected_data'].items():
            if not tweets:
                continue
                
            f.write(f"### @{username}\n")
            
            # Find best crypto tweet
            crypto_tweets = []
            for tweet in tweets:
                text_lower = tweet.text.lower()
                if any(keyword in text_lower for keyword in ['defi', 'protocol', 'ethereum', 'rwa', 'crypto']):
                    metrics = tweet.public_metrics
                    engagement = sum(metrics.get(k, 0) for k in ['like_count', 'retweet_count', 'reply_count', 'quote_count'])
                    crypto_tweets.append((tweet, engagement))
            
            if crypto_tweets:
                best_tweet, engagement = max(crypto_tweets, key=lambda x: x[1])
                f.write(f"**Best Tweet** ({engagement} engagement):\n")
                f.write(f"```\n{best_tweet.text}\n```\n")
            
            f.write("\n")
        
        f.write("## Next Steps\n\n")
        f.write("- [ ] Review sample quality for crypto relevance\n")
        f.write("- [ ] Identify top-performing KOLs for expanded collection\n")
        f.write("- [ ] Validate technical content vs price speculation\n")
        f.write("- [ ] Proceed with full collection if quality approved\n")

def main():
    print("🔍 Focused Crypto/RWA KOL Collection")
    print("=" * 50)
    print("Strategy: Quality over quantity, crypto-focused only")
    print("Objective: Eliminate noise, maximize learning signal\n")
    
    success = collect_conservative_sample()
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 