"""Quality gate filters for tweet processing."""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class QualityGate:
    """Multi-layer quality filtering for tweets."""
    
    def __init__(self):
        self.toxicity_threshold = 0.8
        self.bot_score_threshold = 0.9
        self.min_length = 10
        self.max_length = 500
    
    def filter(self, tweets: List[Dict]) -> List[Dict]:
        """
        Apply quality filters to tweets.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            Filtered list of tweets
        """
        logger.info(f"Applying quality filters to {len(tweets)} tweets")
        
        filtered = []
        for tweet in tweets:
            if self._passes_filters(tweet):
                filtered.append(tweet)
        
        filter_rate = (len(tweets) - len(filtered)) / len(tweets) * 100
        logger.info(f"Filtered out {filter_rate:.1f}% of tweets")
        
        return filtered
    
    def _passes_filters(self, tweet: Dict) -> bool:
        """Check if tweet passes all quality filters."""
        text = tweet.get("text", "")
        
        # Language filter (simple English check)
        if not self._is_english(text):
            return False
        
        # Length filter
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Bot detection (mock implementation)
        if self._is_bot_content(text):
            return False
        
        # Toxicity filter (mock implementation)
        if self._is_toxic(text):
            return False
        
        # Engagement threshold
        metrics = tweet.get("public_metrics", {})
        if metrics.get("like_count", 0) < 10:  # Minimum engagement
            return False
        
        return True
    
    def _is_english(self, text: str) -> bool:
        """Simple English language detection."""
        # Basic check for English characters
        english_ratio = len(re.findall(r'[a-zA-Z\s]', text)) / len(text) if text else 0
        return english_ratio > 0.7
    
    def _is_bot_content(self, text: str) -> bool:
        """Detect bot-like content patterns."""
        # Simple bot detection heuristics
        bot_patterns = [
            r'follow.*back',
            r'click.*link',
            r'visit.*site',
            r'check.*bio'
        ]
        
        for pattern in bot_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_toxic(self, text: str) -> bool:
        """Mock toxicity detection."""
        # In production, this would use Google Perspective API
        toxic_words = ['hate', 'scam', 'fraud', 'stupid']
        text_lower = text.lower()
        
        toxic_count = sum(1 for word in toxic_words if word in text_lower)
        return toxic_count > 0 