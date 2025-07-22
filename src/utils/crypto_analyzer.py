"""
Crypto Content Analyzer
Analyzes crypto relevance and content quality for training data selection
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class CryptoIdentityAnalyzer:
    """Analyze crypto content for identity training."""
    
    def __init__(self):
        self.crypto_keywords = {
            'high_value': ['protocol', 'defi', 'ethereum', 'bitcoin', 'rwa', 'yield', 'liquidity', 'staking'],
            'medium_value': ['crypto', 'blockchain', 'token', 'dapp', 'smart contract', 'dao'],
            'technical': ['consensus', 'validator', 'merkle', 'hash', 'nonce', 'gas', 'evm'],
            'institutional': ['regulation', 'compliance', 'custody', 'institutional', 'etf', 'sec']
        }
        
        self.quality_indicators = {
            'positive': ['innovation', 'development', 'adoption', 'utility', 'research', 'analysis'],
            'negative': ['pump', 'moon', 'lambo', 'shilling', 'wen', 'diamond hands']
        }
    
    def analyze_crypto_relevance(self, text: str) -> float:
        """Calculate crypto relevance score (0-1)."""
        text_lower = text.lower()
        
        score = 0.0
        total_possible = 0
        
        for category, keywords in self.crypto_keywords.items():
            category_weight = {'high_value': 0.4, 'medium_value': 0.3, 'technical': 0.2, 'institutional': 0.1}[category]
            
            for keyword in keywords:
                total_possible += category_weight
                if keyword in text_lower:
                    score += category_weight
        
        return min(score / total_possible if total_possible > 0 else 0, 1.0)
    
    def analyze_content_quality(self, text: str, engagement_metrics: Dict = None) -> float:
        """Calculate content quality score (0-1)."""
        text_lower = text.lower()
        
        # Base quality score
        quality_score = 0.5
        
        # Length check (not too short, not too long)
        length_score = 0.8 if 50 <= len(text) <= 280 else 0.3
        quality_score += length_score * 0.2
        
        # Positive indicators
        positive_count = sum(1 for indicator in self.quality_indicators['positive'] if indicator in text_lower)
        quality_score += min(positive_count * 0.1, 0.3)
        
        # Negative indicators (penalize)
        negative_count = sum(1 for indicator in self.quality_indicators['negative'] if indicator in text_lower)
        quality_score -= min(negative_count * 0.15, 0.4)
        
        # Engagement boost
        if engagement_metrics:
            likes = engagement_metrics.get('like_count', 0)
            retweets = engagement_metrics.get('retweet_count', 0)
            total_engagement = likes + retweets * 2  # Weight retweets more
            
            if total_engagement > 100:
                quality_score += 0.2
            elif total_engagement > 20:
                quality_score += 0.1
        
        return max(0, min(quality_score, 1.0)) 