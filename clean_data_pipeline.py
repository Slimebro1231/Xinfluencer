#!/usr/bin/env python3
"""
Clean Data Pipeline - Organize Real Twitter API Data
Separates real API data from fake/scraped data and sets up proper training structure
"""

import os
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

class DataCleaner:
    def __init__(self):
        self.real_data_sources = [
            "data/safe_collection/safe_crypto_collection_20250721_183136.json"
        ]
        self.fake_data_patterns = [
            "data/scraped/",  # All scraped data
            "data/seed_tweets/",  # Seed/mock data
            "test_scraper_results.json"  # Test data
        ]
        self.output_dir = "data/training_ready"
        self.db_path = f"{self.output_dir}/real_posts.db"
        
    def analyze_data_sources(self):
        """Analyze all data sources to identify real vs fake"""
        print("🔍 Data Source Analysis")
        print("=" * 50)
        
        real_count = 0
        fake_count = 0
        
        print("\n✅ REAL API DATA:")
        for source in self.real_data_sources:
            if os.path.exists(source):
                try:
                    with open(source, 'r') as f:
                        data = json.load(f)
                    tweets = data.get('tweets', [])
                    real_count += len(tweets)
                    print(f"   {source}: {len(tweets)} real tweets")
                    
                    # Check for API authenticity markers
                    if tweets:
                        sample = tweets[0]
                        markers = []
                        if 'public_metrics' in sample:
                            markers.append("public_metrics")
                        if 'edit_history_tweet_ids' in sample:
                            markers.append("edit_history")
                        if 'context_annotations' in sample:
                            markers.append("context_annotations")
                        print(f"     API markers: {', '.join(markers) if markers else 'basic'}")
                except Exception as e:
                    print(f"   ❌ Error reading {source}: {e}")
            else:
                print(f"   ❌ Not found: {source}")
        
        print(f"\n❌ FAKE/SCRAPED DATA (to ignore):")
        for pattern in self.fake_data_patterns:
            if os.path.exists(pattern):
                if os.path.isdir(pattern):
                    files = list(Path(pattern).glob("*.json"))
                    for file in files:
                        try:
                            with open(file, 'r') as f:
                                data = json.load(f)
                            count = len(data) if isinstance(data, list) else len(data.get('tweets', []))
                            fake_count += count
                            print(f"   {file}: {count} fake/scraped tweets")
                        except:
                            print(f"   {file}: unreadable")
                else:
                    try:
                        with open(pattern, 'r') as f:
                            data = json.load(f)
                        count = len(data) if isinstance(data, list) else len(data.get('tweets', []))
                        fake_count += count
                        print(f"   {pattern}: {count} fake tweets")
                    except:
                        if os.path.exists(pattern):
                            print(f"   {pattern}: exists but unreadable")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Real API tweets: {real_count}")
        print(f"   Fake/scraped tweets: {fake_count}")
        print(f"   Quality ratio: {real_count}/{real_count + fake_count} real")
        
        return real_count, fake_count
    
    def setup_clean_database(self):
        """Create a clean database for real training data only"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Remove old database if exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE real_posts (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                author_username TEXT,
                author_id TEXT,
                created_at TEXT,
                likes_count INTEGER DEFAULT 0,
                retweets_count INTEGER DEFAULT 0,
                replies_count INTEGER DEFAULT 0,
                bookmarks_count INTEGER DEFAULT 0,
                impressions_count INTEGER DEFAULT 0,
                has_context_annotations BOOLEAN DEFAULT FALSE,
                crypto_relevance REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.0,
                is_retweet BOOLEAN DEFAULT FALSE,
                data_source TEXT,
                processed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX idx_crypto_relevance ON real_posts(crypto_relevance)')
        cursor.execute('CREATE INDEX idx_quality_score ON real_posts(quality_score)')
        cursor.execute('CREATE INDEX idx_created_at ON real_posts(created_at)')
        
        conn.commit()
        conn.close()
        print(f"✅ Clean database created: {self.db_path}")
    
    def analyze_crypto_relevance(self, text):
        """Analyze crypto relevance of a tweet"""
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain', 
            'defi', 'rwa', 'tokenization', 'smart contract', 'nft', 'dao', 
            'yield', 'staking', 'real world assets', 'liquidity', 'protocol', 
            'dapp', 'web3', 'layer 2', 'lightning network', 'solana', 'sol',
            'cardano', 'ada', 'polygon', 'matic', 'chainlink', 'link'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in crypto_keywords if keyword in text_lower)
        return min(matches / 5.0, 1.0)
    
    def analyze_quality(self, tweet_data):
        """Analyze tweet quality based on engagement and content"""
        text = tweet_data.get('text', '')
        metrics = tweet_data.get('public_metrics', {})
        
        base_score = 0.3
        
        # Length factor (not too short, not too long)
        if 30 <= len(text) <= 280:
            base_score += 0.2
        
        # Engagement factor
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)
        
        total_engagement = likes + (retweets * 2) + replies
        if total_engagement > 50:
            base_score += 0.3
        elif total_engagement > 10:
            base_score += 0.1
        
        # Content quality indicators
        quality_words = [
            'analysis', 'insight', 'trend', 'update', 'important', 
            'breakthrough', 'innovation', 'adoption', 'development'
        ]
        if any(word in text.lower() for word in quality_words):
            base_score += 0.1
        
        # Penalize retweets without additional content
        if text.startswith('RT @') and len(text) < 100:
            base_score *= 0.5
        
        # Bonus for context annotations (indicates Twitter classified it as crypto)
        if 'context_annotations' in tweet_data:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def process_real_data(self):
        """Process and store only real Twitter API data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total_processed = 0
        high_quality_count = 0
        
        for source in self.real_data_sources:
            if not os.path.exists(source):
                continue
                
            print(f"📥 Processing {source}...")
            
            try:
                with open(source, 'r') as f:
                    data = json.load(f)
                
                tweets = data.get('tweets', [])
                for tweet in tweets:
                    # Extract data
                    tweet_id = tweet.get('id')
                    text = tweet.get('text', '')
                    author_username = tweet.get('author_username', 'unknown')
                    author_id = tweet.get('author_id', '')
                    created_at = tweet.get('created_at', '')
                    
                    metrics = tweet.get('public_metrics', {})
                    likes = metrics.get('like_count', 0)
                    retweets = metrics.get('retweet_count', 0)
                    replies = metrics.get('reply_count', 0)
                    bookmarks = metrics.get('bookmark_count', 0)
                    impressions = metrics.get('impression_count', 0)
                    
                    has_context = 'context_annotations' in tweet
                    is_retweet = text.startswith('RT @')
                    
                    # Analysis
                    crypto_relevance = self.analyze_crypto_relevance(text)
                    quality_score = self.analyze_quality(tweet)
                    
                    # Store
                    cursor.execute('''
                        INSERT OR REPLACE INTO real_posts 
                        (id, text, author_username, author_id, created_at, 
                         likes_count, retweets_count, replies_count, bookmarks_count, impressions_count,
                         has_context_annotations, crypto_relevance, quality_score, is_retweet, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        tweet_id, text, author_username, author_id, created_at,
                        likes, retweets, replies, bookmarks, impressions,
                        has_context, crypto_relevance, quality_score, is_retweet, source
                    ))
                    
                    total_processed += 1
                    if crypto_relevance >= 0.3 and quality_score >= 0.5:
                        high_quality_count += 1
                
                print(f"   ✅ Processed {len(tweets)} tweets from {source}")
                
            except Exception as e:
                print(f"   ❌ Error processing {source}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total real tweets processed: {total_processed}")
        print(f"   High-quality training candidates: {high_quality_count}")
        print(f"   Quality ratio: {high_quality_count/total_processed:.1%}")
        
        return total_processed, high_quality_count
    
    def create_training_dataset(self):
        """Create clean training dataset from real data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get high-quality, crypto-relevant posts
        cursor.execute('''
            SELECT id, text, author_username, crypto_relevance, quality_score, 
                   likes_count, retweets_count, created_at
            FROM real_posts 
            WHERE crypto_relevance >= 0.3 
              AND quality_score >= 0.5
              AND NOT is_retweet
            ORDER BY (crypto_relevance + quality_score) DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("❌ No high-quality training data found")
            return None
        
        # Create training examples
        training_data = []
        for row in results:
            tweet_id, text, author, crypto_rel, quality, likes, retweets, created_at = row
            
            # Create instruction-tuning format
            training_example = {
                "instruction": "You are a crypto and RWA expert. Analyze this tweet and provide insightful commentary:",
                "input": text[:200],  # Truncate for training
                "output": f"This tweet demonstrates {crypto_rel:.1%} crypto relevance. {text}",
                "metadata": {
                    "tweet_id": tweet_id,
                    "author": author,
                    "crypto_relevance": crypto_rel,
                    "quality_score": quality,
                    "engagement": {"likes": likes, "retweets": retweets},
                    "created_at": created_at
                }
            }
            training_data.append(training_example)
        
        # Save training dataset
        dataset_path = f"{self.output_dir}/crypto_training_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"✅ Training dataset created: {dataset_path}")
        print(f"   Training examples: {len(training_data)}")
        
        return dataset_path
    
    def generate_cleanup_report(self):
        """Generate a summary report of the cleanup"""
        report = f"""
# Data Cleanup Report
Generated: {datetime.now().isoformat()}

## Summary
- Real API data identified and processed
- Fake/scraped data catalogued (not used for training)
- Clean database created with quality scoring
- Training dataset prepared

## File Structure
```
{self.output_dir}/
├── real_posts.db           # Clean SQLite database
├── crypto_training_dataset.json  # Ready for LoRA training
└── cleanup_report.md       # This report
```

## Next Steps
1. Transfer clean data to H200 server
2. Set up LoRA training with Llama-3.1-8B-Instruct
3. Train identity model on real crypto data
4. Test trained vs base model performance
"""
        
        report_path = f"{self.output_dir}/cleanup_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Cleanup report: {report_path}")

def main():
    print("🧹 Data Cleanup Pipeline")
    print("=" * 50)
    print("Separating real Twitter API data from fake/scraped data")
    print()
    
    cleaner = DataCleaner()
    
    # Step 1: Analyze current mess
    real_count, fake_count = cleaner.analyze_data_sources()
    
    if real_count == 0:
        print("❌ No real API data found. Cannot proceed.")
        return
    
    print(f"\n🧹 Cleaning up data structure...")
    
    # Step 2: Setup clean database
    cleaner.setup_clean_database()
    
    # Step 3: Process real data only
    processed, high_quality = cleaner.process_real_data()
    
    # Step 4: Create training dataset
    dataset_path = cleaner.create_training_dataset()
    
    # Step 5: Generate report
    cleaner.generate_cleanup_report()
    
    print(f"\n✅ Data cleanup complete!")
    print(f"📊 {processed} real tweets processed")
    print(f"🎯 {high_quality} high-quality training examples")
    print(f"📚 Training dataset: {dataset_path}")
    print(f"\n🚀 Ready for H200 deployment and training!")

if __name__ == "__main__":
    main() 