"""Basic tests for Xinfluencer AI pipeline."""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.ingest import fetch_tweets
from data.filter import QualityGate
from data.chunk import chunk_text

def test_fetch_tweets():
    """Test tweet fetching functionality."""
    tweets = fetch_tweets(max_tweets=10)
    assert len(tweets) <= 10
    assert all('id' in tweet for tweet in tweets)
    assert all('text' in tweet for tweet in tweets)

def test_quality_gate():
    """Test quality filtering."""
    # Mock tweets for testing
    mock_tweets = [
        {
            "id": "1", 
            "text": "This is a good crypto tweet about Bitcoin",
            "public_metrics": {"like_count": 50}
        },
        {
            "id": "2", 
            "text": "Short",  # Too short
            "public_metrics": {"like_count": 5}
        },
        {
            "id": "3", 
            "text": "This tweet contains hate speech and scam",
            "public_metrics": {"like_count": 100}
        }
    ]
    
    gate = QualityGate()
    filtered = gate.filter(mock_tweets)
    
    # Should filter out the short tweet and toxic tweet
    assert len(filtered) == 1
    assert filtered[0]["id"] == "1"

def test_text_chunking():
    """Test text chunking functionality."""
    short_text = "Short text"
    chunks = chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    long_text = "This is a much longer text " * 20  # ~540 characters
    chunks = chunk_text(long_text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + some buffer

if __name__ == "__main__":
    pytest.main([__file__]) 