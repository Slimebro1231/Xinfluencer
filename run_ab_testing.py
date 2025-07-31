#!/usr/bin/env python3
"""Run A/B testing with adjusted thresholds for current data quality."""

import json
import sqlite3
from src.evaluation.engagement_evaluator import EngagementEvaluator
from src.evaluation.ab_testing import ABTestingEngine

def run_ab_testing_with_adjusted_thresholds():
    """Run A/B testing with lower thresholds suitable for current data."""
    
    # Load tweets directly from database with lower thresholds
    conn = sqlite3.connect("data/training/posts.db")
    cursor = conn.cursor()
    
    # Get tweets with lower quality thresholds
    cursor.execute("""
        SELECT id, text, author, engagement_score, crypto_relevance, quality_score, metadata
        FROM training_posts 
        WHERE quality_score > 0.2 AND engagement_score > 0.1
        ORDER BY quality_score DESC, engagement_score DESC
        LIMIT 200
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} tweets meeting lower quality thresholds")
    
    if len(rows) < 10:
        print("Not enough tweets for A/B testing")
        return
    
    # Convert to tweet format expected by evaluator
    tweets = []
    for row in rows:
        tweet_id, text, author, engagement_score, crypto_relevance, quality_score, metadata = row
        
        # Parse metadata
        try:
            meta = json.loads(metadata) if metadata else {}
        except:
            meta = {}
        
        tweet = {
            'id': tweet_id,
            'text': text,
            'author_username': author,
            'author': author,
            'public_metrics': meta.get('public_metrics', {}),
            'engagement_score': engagement_score,
            'crypto_relevance': crypto_relevance,
            'quality_score': quality_score,
            'metadata': meta
        }
        tweets.append(tweet)
    
    print(f"Prepared {len(tweets)} tweets for A/B testing")
    
    # Run A/B testing with adjusted thresholds
    evaluator = EngagementEvaluator()
    
    # Temporarily adjust thresholds in the evaluator
    evaluator.min_likes = 1  # Lower from 10
    evaluator.min_engagement_rate = 0.0001  # Lower from 0.001
    
    # Find comparable pairs
    pairs = evaluator.find_comparable_tweets(tweets)
    print(f"Found {len(pairs)} comparable tweet pairs")
    
    if len(pairs) < 4:
        print("Not enough comparable pairs for A/B testing")
        return
    
    # Create comparisons with lower confidence threshold
    comparisons = []
    for tweet_a, tweet_b in pairs:
        comparison = evaluator.create_comparison(tweet_a, tweet_b)
        
        # Lower confidence threshold from 3.0 to 1.5
        if comparison.confidence >= 1.5:
            comparisons.append(comparison)
    
    print(f"Generated {len(comparisons)} comparisons with confidence >= 1.5")
    
    if comparisons:
        # Save evaluation data
        evaluator.save_evaluation_data(comparisons, "evaluation_results/ab_test_results.json")
        print(f"Saved {len(comparisons)} A/B test comparisons to evaluation_results/ab_test_results.json")
        
        # Show some statistics
        confidence_scores = [c.confidence for c in comparisons]
        engagement_ratios = [c.engagement_ratio for c in comparisons]
        
        print(f"\nA/B Testing Results:")
        print(f"  Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
        print(f"  Average engagement ratio: {sum(engagement_ratios)/len(engagement_ratios):.2f}")
        print(f"  Max confidence: {max(confidence_scores):.2f}")
        print(f"  Max engagement ratio: {max(engagement_ratios):.2f}")
        
        # Show top comparisons
        top_comparisons = sorted(comparisons, key=lambda x: x.confidence, reverse=True)[:5]
        print(f"\nTop 5 comparisons by confidence:")
        for i, comp in enumerate(top_comparisons, 1):
            print(f"  {i}. Confidence: {comp.confidence:.2f}, Ratio: {comp.engagement_ratio:.2f}, Type: {comp.comparison_type}")
    
    return comparisons

if __name__ == "__main__":
    run_ab_testing_with_adjusted_thresholds() 