#!/usr/bin/env python3
"""
Create A/B evaluation data from historic tweets using engagement metrics.
Uses real Twitter performance with proper engagement filtering.
"""

import json
import sqlite3
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
    comparison_type: str  # 'same_author', 'same_topic', 'similar_content'

class EnhancedEngagementEvaluator:
    """Create evaluation data from high-quality historic tweet engagement metrics."""
    
    def __init__(self):
        self.comparisons = []
        self.data_dir = Path("data")
        
        # Enhanced filtering thresholds
        self.min_likes = 10  # Minimum likes for A/B testing
        self.min_engagement_rate = 0.001  # 0.1% minimum engagement rate
        self.min_follower_count = 1000  # Minimum followers for reliable normalization
        
    def calculate_engagement_score(self, tweet: Dict) -> float:
        """Calculate normalized engagement score for a tweet."""
        try:
            metrics = tweet.get('public_metrics', {})
            likes = metrics.get('like_count', 0)
            retweets = metrics.get('retweet_count', 0)
            replies = metrics.get('reply_count', 0)
            quotes = metrics.get('quote_count', 0)
            
            # Weighted engagement score
            engagement = (likes * 1.0) + (retweets * 3.0) + (replies * 2.0) + (quotes * 2.5)
            
            # Get follower count if available (for normalization)
            author_metrics = tweet.get('author_public_metrics', {})
            followers = author_metrics.get('followers_count', 
                                         tweet.get('follower_count', 1000))  # Fallback
            
            # Normalize by follower count (engagement rate)
            engagement_rate = engagement / max(followers, 1) * 100
            
            return min(engagement_rate, 10.0)  # Cap at 10% engagement rate
            
        except Exception as e:
            logger.warning(f"Error calculating engagement: {e}")
            return 0.0
    
    def is_suitable_for_ab_testing(self, tweet: Dict) -> bool:
        """Check if tweet is suitable for A/B testing (high engagement, historic)."""
        try:
            metrics = tweet.get('public_metrics', {})
            likes = metrics.get('like_count', 0)
            
            # Must have minimum likes
            if likes < self.min_likes:
                return False
            
            # Check if it's historic data (has engagement_rate field from collector)
            if 'engagement_rate' in tweet:
                engagement_rate = tweet['engagement_rate']
                if engagement_rate < self.min_engagement_rate:
                    return False
            
            # Must be crypto/DeFi relevant
            crypto_keywords = [
                'bitcoin', 'ethereum', 'defi', 'crypto', 'blockchain', 
                'rwa', 'token', 'protocol', 'dapp', 'yield', 'liquidity',
                'btc', 'eth', 'dao', 'nft', 'web3', 'staking'
            ]
            text_lower = tweet.get('text', '').lower()
            if not any(keyword in text_lower for keyword in crypto_keywords):
                return False
            
            # Filter out pure price speculation
            speculation_words = ['moon', 'pump', 'dump', 'wen lambo', 'to the moon']
            if any(word in text_lower for word in speculation_words):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking A/B suitability: {e}")
            return False
    
    def find_comparable_tweets(self, tweets: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Find pairs of tweets that can be meaningfully compared."""
        pairs = []
        
        # Filter for A/B testing quality first
        ab_suitable_tweets = [t for t in tweets if self.is_suitable_for_ab_testing(t)]
        logger.info(f"Filtered to {len(ab_suitable_tweets)}/{len(tweets)} A/B suitable tweets")
        
        if len(ab_suitable_tweets) < 4:
            logger.warning("Not enough high-quality tweets for A/B testing")
            return pairs
        
        # Group by author for same-author comparisons
        by_author = {}
        for tweet in ab_suitable_tweets:
            author = tweet.get('author_username', 'unknown')
            if author not in by_author:
                by_author[author] = []
            by_author[author].append(tweet)
        
        # Same author comparisons (most reliable)
        for author, author_tweets in by_author.items():
            if len(author_tweets) >= 2:
                # Sort by engagement for meaningful comparisons
                sorted_tweets = sorted(author_tweets, 
                                     key=self.calculate_engagement_score, 
                                     reverse=True)
                
                # Compare high performers with medium performers
                for i in range(min(3, len(sorted_tweets))):
                    for j in range(i+1, min(i+3, len(sorted_tweets))):
                        tweet_a = sorted_tweets[i]
                        tweet_b = sorted_tweets[j]
                        
                        # Ensure meaningful engagement difference
                        score_a = self.calculate_engagement_score(tweet_a)
                        score_b = self.calculate_engagement_score(tweet_b)
                        ratio = max(score_a, score_b) / max(min(score_a, score_b), 0.001)
                        
                        if ratio >= 2.0:  # At least 2x difference
                            pairs.append((tweet_a, tweet_b))
        
        # Topic-based comparisons from different authors
        crypto_topics = {
            'bitcoin': ['bitcoin', 'btc'],
            'ethereum': ['ethereum', 'eth', 'defi'],
            'rwa': ['rwa', 'real world', 'tokeniz'],
            'regulation': ['regulat', 'policy', 'sec', 'law']
        }
        
        for topic, keywords in crypto_topics.items():
            topic_tweets = []
            for tweet in ab_suitable_tweets:
                text = tweet.get('text', '').lower()
                if any(keyword in text for keyword in keywords):
                    topic_tweets.append(tweet)
            
            if len(topic_tweets) >= 4:
                # Sort by engagement and compare across authors
                sorted_topic = sorted(topic_tweets, 
                                    key=self.calculate_engagement_score, 
                                    reverse=True)
                
                # Compare top tweets from different authors
                for i in range(0, min(6, len(sorted_topic)), 2):
                    if i+1 < len(sorted_topic):
                        tweet_a = sorted_topic[i]
                        tweet_b = sorted_topic[i+1]
                        
                        # Different authors
                        if tweet_a.get('author_username') != tweet_b.get('author_username'):
                            score_a = self.calculate_engagement_score(tweet_a)
                            score_b = self.calculate_engagement_score(tweet_b)
                            ratio = max(score_a, score_b) / max(min(score_a, score_b), 0.001)
                            
                            if ratio >= 1.5:  # At least 50% difference for cross-author
                                pairs.append((tweet_a, tweet_b))
        
        logger.info(f"Found {len(pairs)} comparable tweet pairs")
        return pairs
    
    def create_comparison(self, tweet_a: Dict, tweet_b: Dict) -> TweetComparison:
        """Create A/B comparison from two tweets."""
        score_a = self.calculate_engagement_score(tweet_a)
        score_b = self.calculate_engagement_score(tweet_b)
        
        # Determine winner
        if score_a > score_b:
            winner = 'A'
            ratio = score_a / max(score_b, 0.001)
        else:
            winner = 'B' 
            ratio = score_b / max(score_a, 0.001)
        
        # Calculate confidence based on engagement difference and sample size
        ratio_diff = abs(score_a - score_b) / max(max(score_a, score_b), 0.001)
        
        # Factor in absolute engagement levels (higher engagement = more confidence)
        avg_engagement = (score_a + score_b) / 2
        engagement_boost = min(avg_engagement / 0.5, 2.0)  # Up to 2x boost for high engagement
        
        confidence = min(1.0 + (ratio_diff * 3) + engagement_boost, 5.0)  # Scale to 1-5
        
        # Determine comparison type
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
    
    def load_tweets_from_files(self) -> List[Dict]:
        """Load tweets from collection files, prioritizing historic data."""
        tweets = []
        
        # Load historic tweets first (highest quality)
        historic_dir = self.data_dir / "historic_tweets"
        if historic_dir.exists():
            for json_file in historic_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'collected_data' in data:
                            for author_tweets in data['collected_data'].values():
                                tweets.extend(author_tweets)
                    
                    logger.info(f"Loaded historic tweets from {json_file.name}")
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        
        # Load from safe_collection (secondary)
        safe_dir = self.data_dir / "safe_collection"
        if safe_dir.exists():
            for json_file in safe_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            tweets.extend(data)
                        elif isinstance(data, dict) and 'tweets' in data:
                            tweets.extend(data['tweets'])
                        elif isinstance(data, dict) and 'collected_data' in data:
                            for author_tweets in data['collected_data'].values():
                                tweets.extend(author_tweets)
                    
                    logger.info(f"Loaded tweets from {json_file.name}")
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        
        # Load from collected (tertiary)
        collected_dir = self.data_dir / "collected"
        if collected_dir.exists():
            for json_file in collected_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            tweets.extend(data)
                    logger.info(f"Loaded tweets from {json_file.name}")
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Total tweets loaded: {len(tweets)}")
        return tweets
    
    def generate_evaluation_data(self) -> List[TweetComparison]:
        """Generate A/B evaluation data from existing tweets."""
        logger.info("Generating enhanced engagement-based evaluation data...")
        
        # Load all tweets
        tweets = self.load_tweets_from_files()
        
        if len(tweets) < 10:
            logger.error("Not enough tweets for evaluation generation")
            return []
        
        # Find comparable pairs
        pairs = self.find_comparable_tweets(tweets)
        
        # Create comparisons
        comparisons = []
        for tweet_a, tweet_b in pairs:
            comparison = self.create_comparison(tweet_a, tweet_b)
            
            # Only include high-confidence comparisons
            if comparison.confidence >= 3.0:  # Increased threshold
                comparisons.append(comparison)
        
        logger.info(f"Generated {len(comparisons)} high-confidence comparisons")
        
        # Sort by confidence (most confident comparisons first)
        comparisons.sort(key=lambda x: x.confidence, reverse=True)
        
        return comparisons[:50]  # Top 50 most confident comparisons
    
    def save_evaluation_data(self, comparisons: List[TweetComparison], 
                           output_file: str = "evaluation_results/enhanced_engagement_eval.json"):
        """Save evaluation data in format compatible with evaluation engine."""
        
        Path(output_file).parent.mkdir(exist_ok=True)
        
        eval_data = []
        for comp in comparisons:
            eval_record = {
                'prompt': f"Generate a crypto-focused tweet about: {self.extract_topic(comp.tweet_a)}",
                'baseline_response': comp.tweet_a.get('text', ''),
                'experimental_response': comp.tweet_b.get('text', ''),
                'human_preference': comp.winner,
                'human_confidence': min(int(comp.confidence), 5),
                'human_reasoning': f"Based on historic engagement metrics. Ratio: {comp.engagement_ratio:.2f}, Type: {comp.comparison_type}",
                'ai_evaluation': {
                    'baseline': {'overall_score': self.calculate_engagement_score(comp.tweet_a)},
                    'experimental': {'overall_score': self.calculate_engagement_score(comp.tweet_b)},
                    'improvement': self.calculate_engagement_score(comp.tweet_b) - self.calculate_engagement_score(comp.tweet_a)
                },
                'engagement_metrics': {
                    'tweet_a_metrics': comp.tweet_a.get('public_metrics', {}),
                    'tweet_b_metrics': comp.tweet_b.get('public_metrics', {}),
                    'engagement_ratio': comp.engagement_ratio,
                    'comparison_type': comp.comparison_type,
                    'confidence_level': comp.confidence
                },
                'timestamp': datetime.now().isoformat()
            }
            eval_data.append(eval_record)
        
        with open(output_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        logger.info(f"Saved {len(eval_data)} evaluation records to {output_file}")
        return output_file
    
    def extract_topic(self, tweet: Dict) -> str:
        """Extract main topic from tweet for prompt generation."""
        text = tweet.get('text', '')
        
        # Enhanced topic extraction
        if any(word in text.lower() for word in ['bitcoin', 'btc']):
            return "Bitcoin adoption and market analysis"
        elif any(word in text.lower() for word in ['ethereum', 'eth', 'defi']):
            return "Ethereum ecosystem and DeFi innovation"
        elif any(word in text.lower() for word in ['rwa', 'real world', 'tokeniz']):
            return "Real World Assets and tokenization opportunities"
        elif any(word in text.lower() for word in ['regulat', 'policy', 'law', 'sec']):
            return "crypto regulation and compliance developments"
        elif any(word in text.lower() for word in ['dao', 'governance', 'voting']):
            return "DAO governance and decentralized decision making"
        elif any(word in text.lower() for word in ['staking', 'yield', 'liquidity']):
            return "DeFi yield strategies and staking opportunities"
        else:
            return "cryptocurrency market trends and technical analysis"

def main():
    """Generate enhanced engagement-based evaluation data."""
    evaluator = EnhancedEngagementEvaluator()
    
    # Generate comparisons
    comparisons = evaluator.generate_evaluation_data()
    
    if not comparisons:
        print("No evaluation data generated. Check tweet data availability.")
        print("Try running: python scripts/collect_historic_tweets.py first")
        return
    
    # Save in evaluation format
    output_file = evaluator.save_evaluation_data(comparisons)
    
    # Print summary
    print(f"\n🎯 Enhanced Engagement-Based Evaluation Data Generated!")
    print(f"📊 High-confidence comparisons: {len(comparisons)}")
    print(f"💾 Saved to: {output_file}")
    
    # Show distribution
    by_type = {}
    by_confidence = {'high': 0, 'medium': 0}
    total_confidence = 0
    
    for comp in comparisons:
        by_type[comp.comparison_type] = by_type.get(comp.comparison_type, 0) + 1
        total_confidence += comp.confidence
        
        if comp.confidence >= 4.0:
            by_confidence['high'] += 1
        else:
            by_confidence['medium'] += 1
    
    print(f"\n📈 Comparison Types:")
    for comp_type, count in by_type.items():
        print(f"  {comp_type}: {count}")
    
    print(f"\n⭐ Quality Metrics:")
    print(f"  Average Confidence: {total_confidence / len(comparisons):.1f}/5.0")
    print(f"  High Confidence (4.0+): {by_confidence['high']}")
    print(f"  Medium Confidence (3.0-4.0): {by_confidence['medium']}")
    
    print(f"\n✅ Ready for high-quality A/B testing with engagement-filtered data!")

if __name__ == "__main__":
    main() 