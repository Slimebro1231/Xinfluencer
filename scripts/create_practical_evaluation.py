#!/usr/bin/env python3
"""
Practical A/B evaluation system that works with current tweet data.
Balances quality requirements with data availability.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TweetComparison:
    """A/B comparison between two tweets."""
    tweet_a: Dict
    tweet_b: Dict
    winner: str  # 'A' or 'B'
    confidence: float  # 1.0-5.0 based on engagement difference
    engagement_ratio: float
    comparison_type: str

class PracticalEvaluationCreator:
    """Create A/B evaluation data from available tweets with practical quality filters."""
    
    def __init__(self):
        self.data_dir = Path("data")
        
        # Practical filtering (works with fresh tweets)
        self.min_likes = 2  # Lower threshold for fresh tweets
        self.min_total_engagement = 3  # At least some engagement
        
    def calculate_engagement_score(self, tweet: Dict) -> float:
        """Calculate weighted engagement score."""
        try:
            metrics = tweet.get('public_metrics', {})
            likes = metrics.get('like_count', 0)
            retweets = metrics.get('retweet_count', 0)
            replies = metrics.get('reply_count', 0)
            quotes = metrics.get('quote_count', 0)
            
            # Weighted engagement score
            total_engagement = (likes * 1.0) + (retweets * 3.0) + (replies * 2.0) + (quotes * 2.5)
            
            # For fresh tweets, consider recency boost
            created_at = tweet.get('created_at', '')
            if created_at:
                try:
                    from dateutil import parser
                    tweet_date = parser.parse(created_at)
                    now = datetime.now(tweet_date.tzinfo)
                    age_hours = (now - tweet_date).total_seconds() / 3600
                    
                    # Boost for very fresh tweets (under 24 hours)
                    if age_hours < 24:
                        recency_boost = 1.5
                    elif age_hours < 72:  # Under 3 days
                        recency_boost = 1.2
                    else:
                        recency_boost = 1.0
                    
                    total_engagement *= recency_boost
                except:
                    pass  # If date parsing fails, use original score
            
            return total_engagement
            
        except Exception as e:
            logger.warning(f"Error calculating engagement: {e}")
            return 0.0
    
    def is_crypto_relevant(self, tweet: Dict) -> bool:
        """Check if tweet is crypto/DeFi relevant."""
        text = tweet.get('text', '').lower()
        
        crypto_keywords = [
            'bitcoin', 'ethereum', 'defi', 'crypto', 'blockchain', 
            'rwa', 'token', 'protocol', 'dapp', 'yield', 'liquidity',
            'btc', 'eth', 'dao', 'nft', 'web3', 'staking', 'trading',
            'market', 'price', 'analysis', 'investment'
        ]
        
        return any(keyword in text for keyword in crypto_keywords)
    
    def is_quality_content(self, tweet: Dict) -> bool:
        """Check for content quality indicators."""
        text = tweet.get('text', '')
        
        # Minimum length
        if len(text) < 30:
            return False
        
        # Must be crypto relevant
        if not self.is_crypto_relevant(tweet):
            return False
        
        # Filter out pure spam/speculation
        spam_indicators = ['follow for follow', 'rt for rt', 'dm me', 'check bio']
        if any(spam in text.lower() for spam in spam_indicators):
            return False
        
        # Must have some engagement OR be educational
        metrics = tweet.get('public_metrics', {})
        total_engagement = sum([
            metrics.get('like_count', 0),
            metrics.get('retweet_count', 0),
            metrics.get('reply_count', 0),
            metrics.get('quote_count', 0)
        ])
        
        # Educational keywords boost
        educational_words = ['how', 'what', 'why', 'learn', 'understand', 'explain', 'guide']
        is_educational = any(word in text.lower() for word in educational_words)
        
        return total_engagement >= self.min_total_engagement or is_educational
    
    def find_comparable_pairs(self, tweets: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Find meaningful tweet pairs for comparison."""
        pairs = []
        
        # Filter for quality content
        quality_tweets = [t for t in tweets if self.is_quality_content(t)]
        logger.info(f"Filtered to {len(quality_tweets)}/{len(tweets)} quality tweets")
        
        if len(quality_tweets) < 4:
            logger.warning("Not enough quality tweets for comparison")
            return pairs
        
        # Sort by engagement
        sorted_tweets = sorted(quality_tweets, key=self.calculate_engagement_score, reverse=True)
        
        # Same author comparisons
        by_author = {}
        for tweet in sorted_tweets:
            author = tweet.get('author_username', 'unknown')
            if author not in by_author:
                by_author[author] = []
            by_author[author].append(tweet)
        
        # Compare within authors (most reliable)
        for author, author_tweets in by_author.items():
            if len(author_tweets) >= 2:
                # Compare best with others
                for i in range(min(2, len(author_tweets))):
                    for j in range(i+1, min(i+3, len(author_tweets))):
                        tweet_a = author_tweets[i]
                        tweet_b = author_tweets[j]
                        
                        score_a = self.calculate_engagement_score(tweet_a)
                        score_b = self.calculate_engagement_score(tweet_b)
                        
                        # Ensure meaningful difference (even small is OK for fresh tweets)
                        if max(score_a, score_b) >= 1.5 * min(score_a, score_b):
                            pairs.append((tweet_a, tweet_b))
        
        # Cross-author comparisons by topic
        topics = {
            'bitcoin': ['bitcoin', 'btc'],
            'ethereum': ['ethereum', 'eth', 'defi'],
            'trading': ['trade', 'trading', 'buy', 'sell', 'market'],
            'analysis': ['analysis', 'chart', 'technical', 'price']
        }
        
        for topic, keywords in topics.items():
            topic_tweets = []
            for tweet in sorted_tweets:
                text = tweet.get('text', '').lower()
                if any(keyword in text for keyword in keywords):
                    topic_tweets.append(tweet)
            
            if len(topic_tweets) >= 4:
                # Compare top performers across authors
                for i in range(0, min(4, len(topic_tweets)), 2):
                    if i+1 < len(topic_tweets):
                        tweet_a = topic_tweets[i]
                        tweet_b = topic_tweets[i+1]
                        
                        # Different authors
                        if tweet_a.get('author_username') != tweet_b.get('author_username'):
                            pairs.append((tweet_a, tweet_b))
        
        logger.info(f"Found {len(pairs)} comparison pairs")
        return pairs
    
    def create_comparison(self, tweet_a: Dict, tweet_b: Dict) -> TweetComparison:
        """Create comparison with practical confidence scoring."""
        score_a = self.calculate_engagement_score(tweet_a)
        score_b = self.calculate_engagement_score(tweet_b)
        
        # Determine winner
        if score_a > score_b:
            winner = 'A'
            ratio = score_a / max(score_b, 0.1)
        else:
            winner = 'B'
            ratio = score_b / max(score_a, 0.1)
        
        # Practical confidence calculation
        ratio_diff = abs(score_a - score_b) / max(max(score_a, score_b), 0.1)
        
        # Factor in content quality indicators
        text_a = tweet_a.get('text', '')
        text_b = tweet_b.get('text', '')
        
        quality_boost_a = 0
        quality_boost_b = 0
        
        educational_words = ['how', 'what', 'why', 'learn', 'explain']
        if any(word in text_a.lower() for word in educational_words):
            quality_boost_a += 0.5
        if any(word in text_b.lower() for word in educational_words):
            quality_boost_b += 0.5
        
        # Adjust scores with quality boosts
        adjusted_score_a = score_a + quality_boost_a
        adjusted_score_b = score_b + quality_boost_b
        
        confidence = min(2.0 + (ratio_diff * 2) + max(quality_boost_a, quality_boost_b), 5.0)
        
        # Comparison type
        if tweet_a.get('author_username') == tweet_b.get('author_username'):
            comp_type = 'same_author'
        else:
            comp_type = 'cross_author'
        
        return TweetComparison(
            tweet_a=tweet_a,
            tweet_b=tweet_b,
            winner=winner,
            confidence=confidence,
            engagement_ratio=ratio,
            comparison_type=comp_type
        )
    
    def load_available_tweets(self) -> List[Dict]:
        """Load all available tweets from collection files."""
        tweets = []
        
        # Load from all available sources
        sources = [
            "safe_collection",
            "collected", 
            "historic_tweets"
        ]
        
        for source_dir in sources:
            dir_path = self.data_dir / source_dir
            if dir_path.exists():
                for json_file in dir_path.glob("*.json"):
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                            
                            if isinstance(data, list):
                                tweets.extend(data)
                            elif isinstance(data, dict):
                                if 'collected_data' in data:
                                    for author_tweets in data['collected_data'].values():
                                        tweets.extend(author_tweets)
                                elif 'tweets' in data:
                                    tweets.extend(data['tweets'])
                        
                        logger.info(f"Loaded tweets from {source_dir}/{json_file.name}")
                    except Exception as e:
                        logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Total tweets loaded: {len(tweets)}")
        return tweets
    
    def generate_evaluation_data(self) -> List[TweetComparison]:
        """Generate practical A/B evaluation data."""
        logger.info("Generating practical A/B evaluation data...")
        
        tweets = self.load_available_tweets()
        
        if len(tweets) < 10:
            logger.error("Not enough tweets for evaluation")
            return []
        
        pairs = self.find_comparable_pairs(tweets)
        
        comparisons = []
        for tweet_a, tweet_b in pairs:
            comparison = self.create_comparison(tweet_a, tweet_b)
            
            # Include all reasonable quality comparisons
            if comparison.confidence >= 2.5:
                comparisons.append(comparison)
        
        # Sort by confidence
        comparisons.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Generated {len(comparisons)} practical comparisons")
        return comparisons[:75]  # Top 75 comparisons
    
    def save_evaluation_data(self, comparisons: List[TweetComparison]) -> str:
        """Save evaluation data."""
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        eval_data = []
        for comp in comparisons:
            eval_record = {
                'prompt': f"Generate a crypto tweet about: {self.extract_topic(comp.tweet_a)}",
                'baseline_response': comp.tweet_a.get('text', ''),
                'experimental_response': comp.tweet_b.get('text', ''),
                'human_preference': comp.winner,
                'human_confidence': min(int(comp.confidence), 5),
                'human_reasoning': f"Practical engagement comparison. Ratio: {comp.engagement_ratio:.2f}",
                'ai_evaluation': {
                    'baseline': {'overall_score': self.calculate_engagement_score(comp.tweet_a)},
                    'experimental': {'overall_score': self.calculate_engagement_score(comp.tweet_b)},
                    'improvement': self.calculate_engagement_score(comp.tweet_b) - self.calculate_engagement_score(comp.tweet_a)
                },
                'engagement_metrics': {
                    'tweet_a_metrics': comp.tweet_a.get('public_metrics', {}),
                    'tweet_b_metrics': comp.tweet_b.get('public_metrics', {}),
                    'engagement_ratio': comp.engagement_ratio,
                    'comparison_type': comp.comparison_type
                },
                'timestamp': datetime.now().isoformat()
            }
            eval_data.append(eval_record)
        
        output_file = output_dir / "practical_engagement_eval.json"
        
        with open(output_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        logger.info(f"Saved {len(eval_data)} evaluation records to {output_file}")
        return str(output_file)
    
    def extract_topic(self, tweet: Dict) -> str:
        """Extract topic for prompt generation."""
        text = tweet.get('text', '').lower()
        
        if any(word in text for word in ['bitcoin', 'btc']):
            return "Bitcoin market analysis and trends"
        elif any(word in text for word in ['ethereum', 'eth', 'defi']):
            return "Ethereum and DeFi ecosystem developments"
        elif any(word in text for word in ['trading', 'trade', 'buy', 'sell']):
            return "cryptocurrency trading strategies and tips"
        elif any(word in text for word in ['rwa', 'real world', 'tokeniz']):
            return "Real World Assets and tokenization"
        elif any(word in text for word in ['market', 'price', 'chart']):
            return "crypto market analysis and price movements"
        else:
            return "general cryptocurrency insights and education"

def main():
    """Generate practical A/B evaluation data."""
    creator = PracticalEvaluationCreator()
    
    print("🔄 Practical A/B Evaluation Generator")
    print("=" * 45)
    print("Strategy: Work with available tweet data while maintaining quality")
    print("Filters: 2+ likes OR educational content, crypto-relevant")
    
    # Generate comparisons
    comparisons = creator.generate_evaluation_data()
    
    if not comparisons:
        print("❌ No evaluation data generated.")
        return
    
    # Save data
    output_file = creator.save_evaluation_data(comparisons)
    
    # Summary
    print(f"\n🎯 Practical A/B Evaluation Results:")
    print(f"📊 Total comparisons: {len(comparisons)}")
    print(f"💾 Saved to: {output_file}")
    
    # Distribution
    by_type = {}
    high_confidence = 0
    total_confidence = 0
    
    for comp in comparisons:
        by_type[comp.comparison_type] = by_type.get(comp.comparison_type, 0) + 1
        total_confidence += comp.confidence
        if comp.confidence >= 4.0:
            high_confidence += 1
    
    print(f"\n📈 Comparison Types:")
    for comp_type, count in by_type.items():
        print(f"  {comp_type}: {count}")
    
    print(f"\n⭐ Quality Metrics:")
    print(f"  Average Confidence: {total_confidence / len(comparisons):.1f}/5.0")
    print(f"  High Confidence (4.0+): {high_confidence}")
    
    print(f"\n✅ Practical A/B testing data ready for evaluation!")

if __name__ == "__main__":
    main() 