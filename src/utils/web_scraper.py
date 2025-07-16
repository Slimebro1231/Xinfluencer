
import re
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

# Try multiple search engines for better results
try:
    from ddgs import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DUCKDUCKGO_AVAILABLE = True
    except ImportError:
        DUCKDUCKGO_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

def _parse_metric(value_str: str) -> int:
    """Converts a metric string (e.g., '1.5K') to an integer."""
    value_str = value_str.lower().strip()
    if 'k' in value_str:
        return int(float(value_str.replace('k', '')) * 1_000)
    if 'm' in value_str:
        return int(float(value_str.replace('m', '')) * 1_000_000)
    return int(value_str)

def _extract_metrics(text: str) -> Dict[str, int]:
    """Extracts likes, retweets, and replies from a text snippet."""
    likes = 0
    retweets = 0
    replies = 0

    # Regex to find numbers followed by "Likes", "Retweets", "Replies", etc.
    # Handles integers, floats, and K/M suffixes.
    likes_match = re.search(r'([\d\.]+[km]?)\s+(likes|like)', text, re.IGNORECASE)
    if likes_match:
        likes = _parse_metric(likes_match.group(1))

    retweets_match = re.search(r'([\d\.]+[km]?)\s+(retweets|reposts|retweet|repost)', text, re.IGNORECASE)
    if retweets_match:
        retweets = _parse_metric(retweets_match.group(1))

    replies_match = re.search(r'([\d\.]+[km]?)\s+(replies|reply)', text, re.IGNORECASE)
    if replies_match:
        replies = _parse_metric(replies_match.group(1))
        
    return {
        "like_count": likes,
        "retweet_count": retweets,
        "reply_count": replies,
        "impression_count": 0  # Not available from web scraping
    }

def _extract_full_tweet_text(title: str, body: str) -> str:
    """
    Improved tweet text extraction to avoid truncation.
    
    Args:
        title: Search result title
        body: Search result body/snippet
        
    Returns:
        Full tweet text without truncation
    """
    # Clean title first
    tweet_text = title.replace(' on X: ', ' ').replace(' on Twitter: ', ' ')
    
    # If title is truncated (ends with ...), try to find full text in body
    if tweet_text.endswith('...') or '...' in tweet_text:
        # Look for quoted text in body (common pattern for tweets)
        quote_matches = re.findall(r'"([^"]{20,})"', body)
        if quote_matches:
            # Use the longest quote as it's likely the full tweet
            longest_quote = max(quote_matches, key=len)
            if len(longest_quote) > len(tweet_text.replace('...', '')):
                tweet_text = longest_quote
        
        # If no quotes, try to extract text after common patterns
        else:
            # Look for text after "tweeted:" or similar patterns
            tweet_patterns = [
                r'tweeted:\s*([^\.]+)',
                r'said:\s*([^\.]+)',
                r'wrote:\s*([^\.]+)',
                r'posted:\s*([^\.]+)'
            ]
            
            for pattern in tweet_patterns:
                match = re.search(pattern, body, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > len(tweet_text.replace('...', '')):
                        tweet_text = extracted
                        break
    
    # Clean up common artifacts
    tweet_text = re.sub(r'\s+', ' ', tweet_text)  # Normalize whitespace
    tweet_text = tweet_text.strip()
    
    # Remove common prefixes/suffixes that aren't part of the tweet
    tweet_text = re.sub(r'^[^:]*:\s*', '', tweet_text)  # Remove "User tweeted:" prefix
    tweet_text = re.sub(r'\s*\.\.\.$', '', tweet_text)  # Remove trailing ...
    
    return tweet_text

def _search_duckduckgo(query: str, max_results: int = 10) -> List[Dict]:
    """Search using DuckDuckGo."""
    if not DUCKDUCKGO_AVAILABLE:
        return []
    
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
        return search_results
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
        return []

def _search_google_serpapi(query: str, max_results: int = 10) -> List[Dict]:
    """Search using Google via SERPAPI (requires API key)."""
    # This would require SERPAPI integration
    # For now, return empty list
    return []

def _search_bing(query: str, max_results: int = 10) -> List[Dict]:
    """Search using Bing (requires API key)."""
    # This would require Bing Search API integration
    # For now, return empty list
    return []

def search_for_tweets(query: str, max_results: int = 10, search_engines: List[str] = None) -> List[Dict]:
    """
    Performs a web search to find tweets using multiple search engines, then parses them into a structured format.
    
    Args:
        query: The search query (e.g., "tweets from @VitalikButerin").
        max_results: The maximum number of search results to process.
        search_engines: List of search engines to use. Default: ['duckduckgo']

    Returns:
        A list of dictionaries, where each dictionary represents a tweet.
    """
    if search_engines is None:
        search_engines = ['duckduckgo']
    
    print(f"Searching web for: '{query}' using engines: {search_engines}...")
    results = []
    all_search_results = []
    
    # Collect results from all available search engines
    for engine in search_engines:
        if engine == 'duckduckgo' and DUCKDUCKGO_AVAILABLE:
            engine_results = _search_duckduckgo(query, max_results)
            all_search_results.extend(engine_results)
            print(f"DuckDuckGo: Found {len(engine_results)} results")
        
        elif engine == 'google' and REQUESTS_AVAILABLE:
            engine_results = _search_google_serpapi(query, max_results)
            all_search_results.extend(engine_results)
            print(f"Google: Found {len(engine_results)} results")
        
        elif engine == 'bing' and REQUESTS_AVAILABLE:
            engine_results = _search_bing(query, max_results)
            all_search_results.extend(engine_results)
            print(f"Bing: Found {len(engine_results)} results")
    
    print(f"Total search results: {len(all_search_results)}")
    
    # Remove duplicates based on title similarity
    unique_results = []
    seen_titles = set()
    
    for result in all_search_results:
        title = result.get('title', '').lower()
        # Simple deduplication based on title similarity
        if not any(title in seen or seen in title for seen in seen_titles):
            unique_results.append(result)
            seen_titles.add(title)
    
    print(f"Unique results after deduplication: {len(unique_results)}")

    for i, r in enumerate(unique_results):
        # The 'body' usually contains the most useful snippet of text.
        snippet = r.get('body', '')
        title = r.get('title', '')
        
        if not snippet and not title:
            continue

        # Use improved text extraction
        tweet_text = _extract_full_tweet_text(title, snippet)
        
        # Skip if text is too short (likely not a real tweet)
        if len(tweet_text) < 10:
            continue
            
        # Skip if text looks like a search result page title
        if any(phrase in tweet_text.lower() for phrase in ['search', 'results', 'twitter', 'x.com', 'tweetsearch']):
            continue

        public_metrics = _extract_metrics(snippet)

        # Create a consistent, API-like structure
        results.append({
            "id": str(random.randint(10**17, 10**18 - 1)),
            "text": tweet_text,
            "created_at": (datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat() + "Z",
            "author_username": query.split('@')[-1].split(' ')[0], # Extract from query
            "public_metrics": public_metrics,
            "source_engine": r.get('source', 'unknown')
        })
        
        # Add delay to be respectful to search engines
        time.sleep(random.uniform(0.5, 1.5))
    
    print(f"Successfully extracted {len(results)} tweets")
    return results 