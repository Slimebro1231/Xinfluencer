#!/usr/bin/env python3
"""
Test only the vector pipeline on H200 with real tweet data.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector.embed import TextEmbedder
from vector.db import VectorDB
from vector.search import VectorSearcher

def load_tweet_data():
    """Load tweet data from JSON file."""
    data_path = Path("data/seed_tweets/scraped_seed_tweets.json")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return []
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} tweets from {data_path}")
    return data

def prepare_chunks(tweets):
    """Convert tweets to chunks for vector database."""
    chunks = []
    
    for tweet in tweets:
        # Create chunks from tweet text
        text = tweet.get('text', '')
        if not text or len(text.strip()) < 10:
            continue
            
        chunk = {
            "text": text,
            "tweet_id": tweet.get('id', f"tweet_{len(chunks)}"),
            "timestamp": tweet.get('timestamp', time.time()),
            "metadata": {
                "author": tweet.get('author', 'unknown'),
                "likes": tweet.get('likes', 0),
                "retweets": tweet.get('retweets', 0),
                "replies": tweet.get('replies', 0)
            }
        }
        chunks.append(chunk)
    
    print(f"Prepared {len(chunks)} chunks from tweets")
    return chunks

def test_vector_pipeline():
    """Test the complete vector pipeline."""
    print("=== Testing Vector Pipeline ===")
    
    # Load tweet data
    tweets = load_tweet_data()
    if not tweets:
        print("No tweet data available")
        return False
    
    # Prepare chunks
    chunks = prepare_chunks(tweets)
    if not chunks:
        print("No valid chunks prepared")
        return False
    
    # Initialize components
    print("Initializing vector components...")
    embedder = TextEmbedder()
    db = VectorDB()
    
    # Create collection with correct dimension
    dimension = embedder.get_sentence_embedding_dimension()
    print(f"Creating collection with dimension: {dimension}")
    db.create_collection(vector_size=dimension)
    
    # Generate embeddings and store
    print("Generating embeddings...")
    start_time = time.time()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    embed_time = time.time() - start_time
    print(f"Generated embeddings in {embed_time:.2f} seconds")
    
    # Store in vector database
    print("Storing in vector database...")
    start_time = time.time()
    db.upsert_chunks(chunks_with_embeddings)
    store_time = time.time() - start_time
    print(f"Stored in database in {store_time:.2f} seconds")
    
    # Test search
    print("Testing vector search...")
    searcher = VectorSearcher(db, embedder)
    
    test_queries = [
        "Bitcoin price",
        "Ethereum smart contracts", 
        "DeFi protocols",
        "Crypto regulations",
        "Blockchain technology"
    ]
    
    for query in test_queries:
        start_time = time.time()
        results = searcher.search_similar(query, top_k=3)
        search_time = time.time() - start_time
        
        print(f"\nQuery: '{query}'")
        print(f"Search time: {search_time:.3f}s")
        print(f"Results: {len(results)} found")
        
        for i, result in enumerate(results[:2]):  # Show top 2
            print(f"  {i+1}. Score: {result.get('score', 0):.3f}")
            # Handle different result formats
            if 'text' in result:
                text = result['text']
                author = result.get('metadata', {}).get('author', 'unknown')
            elif 'payload' in result:
                text = result['payload'].get('text', 'No text')
                author = result['payload'].get('metadata', {}).get('author', 'unknown')
            else:
                text = str(result)
                author = 'unknown'
            
            print(f"     Text: {text[:100]}...")
            print(f"     Author: {author}")
    
    return True

def main():
    """Run vector pipeline test."""
    print("Starting H200 Vector Pipeline Test")
    print("=" * 50)
    
    # Test vector pipeline
    vector_success = test_vector_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Vector Pipeline: {'PASS' if vector_success else 'FAIL'}")
    
    if vector_success:
        print("\nVector pipeline test passed! BAAI embeddings are working correctly.")
    else:
        print("\nVector pipeline test failed. Check the output above for details.")

if __name__ == "__main__":
    main() 