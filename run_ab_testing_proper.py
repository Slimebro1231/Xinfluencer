#!/usr/bin/env python3
"""Proper A/B testing script using existing codebase structure."""

import json
import sqlite3
import os
from pathlib import Path
from src.evaluation.engagement_evaluator import EngagementEvaluator
from src.evaluation.ab_testing import ABTestingEngine

def run_proper_ab_testing():
    """Run A/B testing using the existing EngagementEvaluator."""
    
    print("Starting proper A/B testing with existing codebase...")
    
    # Initialize evaluator using existing infrastructure
    evaluator = EngagementEvaluator()
    
    # Load tweets directly from database
    conn = sqlite3.connect("data/training/posts.db")
    cursor = conn.cursor()
    
    # Get all tweets with their scores
    cursor.execute("""
        SELECT id, text, author, engagement_score, crypto_relevance, quality_score, metadata
        FROM training_posts 
        WHERE quality_score > 0.2 AND engagement_score > 0.1
        ORDER BY quality_score DESC, engagement_score DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} tweets meeting quality thresholds")
    
    if len(rows) < 10:
        print("Not enough tweets for A/B testing")
        return None
    
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
    
    # Temporarily adjust thresholds for current data quality
    original_min_likes = evaluator.min_likes
    original_min_engagement_rate = evaluator.min_engagement_rate
    
    evaluator.min_likes = 1  # Lower from 10
    evaluator.min_engagement_rate = 0.0001  # Lower from 0.001
    
    # Find comparable pairs using existing method
    pairs = evaluator.find_comparable_tweets(tweets)
    print(f"Found {len(pairs)} comparable tweet pairs")
    
    # Restore original thresholds
    evaluator.min_likes = original_min_likes
    evaluator.min_engagement_rate = original_min_engagement_rate
    
    if len(pairs) < 4:
        print("Not enough comparable pairs for A/B testing")
        return None
    
    # Create comparisons with lower confidence threshold
    comparisons = []
    for tweet_a, tweet_b in pairs:
        comparison = evaluator.create_comparison(tweet_a, tweet_b)
        
        # Lower confidence threshold from 3.0 to 1.5 for current data
        if comparison.confidence >= 1.5:
            comparisons.append(comparison)
    
    print(f"Generated {len(comparisons)} comparisons with confidence >= 1.5")
    
    if comparisons:
        # Save evaluation data using existing method
        output_file = "evaluation_results/proper_ab_test_results.json"
        evaluator.save_evaluation_data(comparisons, output_file)
        print(f"Saved {len(comparisons)} A/B test comparisons to {output_file}")
        
        # Show detailed statistics
        confidence_scores = [c.confidence for c in comparisons]
        engagement_ratios = [c.engagement_ratio for c in comparisons]
        
        print(f"\nA/B Testing Results:")
        print(f"  Total comparisons: {len(comparisons)}")
        print(f"  Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
        print(f"  Average engagement ratio: {sum(engagement_ratios)/len(engagement_ratios):.2f}")
        print(f"  Max confidence: {max(confidence_scores):.2f}")
        print(f"  Max engagement ratio: {max(engagement_ratios):.2f}")
        
        # Show top comparisons
        top_comparisons = sorted(comparisons, key=lambda x: x.confidence, reverse=True)[:5]
        print(f"\nTop 5 comparisons by confidence:")
        for i, comp in enumerate(top_comparisons, 1):
            print(f"  {i}. Confidence: {comp.confidence:.2f}, Ratio: {comp.engagement_ratio:.2f}, Type: {comp.comparison_type}")
        
        # Run statistical A/B testing using existing ABTestingEngine
        print(f"\nRunning statistical A/B testing...")
        ab_engine = ABTestingEngine()
        
        # Split tweets into two groups for statistical testing
        mid_point = len(tweets) // 2
        group_a = tweets[:mid_point]
        group_b = tweets[mid_point:]
        
        # Run quality A/B test
        quality_results = ab_engine.run_quality_ab_test(group_a, group_b, 'quality_score')
        print(f"Quality A/B test results: {quality_results}")
        
        # Run engagement A/B test
        engagement_results = ab_engine.run_engagement_ab_test(group_a, group_b)
        print(f"Engagement A/B test results: {engagement_results}")
        
        return comparisons
    
    return None

if __name__ == "__main__":
    run_proper_ab_testing() 