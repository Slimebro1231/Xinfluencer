
import os
import sys
import json
import time
from pathlib import Path
import random
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils.web_scraper import search_for_tweets


def scrape_tweets_for_kol(kol_username: str, num_tweets: int = 5):
    """
    Fetches tweets for a KOL using a web search.
    """
    print(f"Web scraping for recent tweets from {kol_username}...")
    try:
        # Use web search to get tweet-like text
        search_results = search_for_tweets(f"tweets from @{kol_username}", max_results=num_tweets)
        
        # The search_for_tweets function already returns data in the desired format.
        # We can add extra processing here if needed in the future.
        processed_tweets = search_results

        print(f"Successfully scraped {len(processed_tweets)} tweets for {kol_username}.")
        return processed_tweets

    except Exception as e:
        print(f"An error occurred while scraping for {kol_username}: {e}")
        return []

def main():
    """Main function to scrape tweets for all KOLs and save them."""
    config = Config()
    all_tweets = []
    
    # Use a smaller subset for testing if needed, e.g., config.crypto_kols[:3]
    for kol in config.crypto_kols:
        tweets = scrape_tweets_for_kol(kol, num_tweets=5)
        if tweets:
            all_tweets.extend(tweets)
        # Add a small delay to be respectful to the search engine
        time.sleep(random.uniform(1, 3))

    # Define the output directory and file
    output_dir = Path(__file__).parent.parent / "data" / "seed_tweets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scraped_seed_tweets.json"

    # Save the tweets to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, indent=4, ensure_ascii=False)

    print(f"\nTotal of {len(all_tweets)} tweets scraped.")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main() 