#!/usr/bin/env python3
"""
Test script for improved web scraper with better tweet extraction.
This script tests the enhanced scraper against the current scraped data.
"""

import sys
import json
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.web_scraper import search_for_tweets
from src.config import Config

def test_improved_scraper():
    """Test the improved scraper with a few KOLs."""
    config = Config()
    
    # Test with a subset of KOLs
    test_kols = config.crypto_kols[:3]  # Test first 3 KOLs
    
    print("Testing improved web scraper...")
    print(f"Testing with KOLs: {test_kols}")
    print("=" * 50)
    
    all_results = []
    
    for kol in test_kols:
        print(f"\nTesting scraper for @{kol}...")
        
        # Test with different search queries
        queries = [
            f"tweets from @{kol}",
            f"@{kol} recent tweets",
            f"{kol} crypto tweets"
        ]
        
        kol_results = []
        
        for query in queries:
            print(f"  Query: '{query}'")
            try:
                results = search_for_tweets(query, max_results=3)
                kol_results.extend(results)
                print(f"    Found {len(results)} results")
                time.sleep(2)  # Be respectful to search engines
            except Exception as e:
                print(f"    Error: {e}")
        
        # Remove duplicates based on text similarity
        unique_results = []
        seen_texts = set()
        
        for result in kol_results:
            text = result['text'].lower()
            # Simple deduplication
            if not any(text in seen or seen in text for seen in seen_texts):
                unique_results.append(result)
                seen_texts.add(text)
        
        print(f"  Unique results for @{kol}: {len(unique_results)}")
        
        # Show sample results
        for i, result in enumerate(unique_results[:2]):
            print(f"    Sample {i+1}: {result['text'][:100]}...")
        
        all_results.extend(unique_results)
    
    print(f"\nTotal unique results: {len(all_results)}")
    
    # Save test results
    output_file = Path(__file__).parent.parent / "data" / "test_scraper_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"Test results saved to: {output_file}")
    
    # Compare with existing scraped data
    existing_file = Path(__file__).parent.parent / "data" / "seed_tweets" / "scraped_seed_tweets.json"
    if existing_file.exists():
        print("\nComparing with existing scraped data...")
        with open(existing_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        
        print(f"Existing data: {len(existing_data)} tweets")
        print(f"New test data: {len(all_results)} tweets")
        
        # Analyze text lengths
        if existing_data:
            existing_lengths = [len(tweet['text']) for tweet in existing_data]
            print(f"Existing avg text length: {sum(existing_lengths)/len(existing_lengths):.1f} chars")
        else:
            print("Existing avg text length: No existing data")
        
        if all_results:
            new_lengths = [len(tweet['text']) for tweet in all_results]
            print(f"New avg text length: {sum(new_lengths)/len(new_lengths):.1f} chars")
        else:
            print("New avg text length: No results to analyze")
        
        # Check for truncation
        if existing_data:
            existing_truncated = sum(1 for text in existing_data if text['text'].endswith('...'))
            print(f"Existing truncated tweets: {existing_truncated}")
        else:
            print("Existing truncated tweets: No existing data")
        
        if all_results:
            new_truncated = sum(1 for text in all_results if text['text'].endswith('...'))
            print(f"New truncated tweets: {new_truncated}")
        else:
            print("New truncated tweets: No results to analyze")

if __name__ == "__main__":
    test_improved_scraper() 