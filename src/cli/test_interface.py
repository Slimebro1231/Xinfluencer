#!/usr/bin/env python3
"""
Xinfluencer AI Test Interface
Professional CLI for testing collection, training, and model interaction
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(title):
    """Print a clean header."""
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")

def print_status(message, status="INFO"):
    """Print status message."""
    prefix = {"INFO": "[INFO]", "ERROR": "[ERROR]", "SUCCESS": "[OK]"}
    print(f"{prefix.get(status, '[INFO]')} {message}")

def check_collection_status():
    """Check status of collected posts."""
    print_header("COLLECTION STATUS")
    
    safe_dir = Path("data/safe_collection")
    training_dir = Path("data/training_posts") 
    db_path = Path("data/all_posts/posts.db")
    
    if safe_dir.exists():
        files = list(safe_dir.glob("*.json"))
        print_status(f"Collection files: {len(files)}")
        if files:
            latest = max(files, key=lambda x: x.stat().st_mtime)
            print_status(f"Latest: {latest.name}")
    else:
        print_status("No collection directory found")
    
    if training_dir.exists():
        files = list(training_dir.glob("*.json"))
        print_status(f"Training files: {len(files)}")
    else:
        print_status("No training directory found")
    
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM posts")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM posts WHERE quality_score > 0.7")
            high_quality = cursor.fetchone()[0]
            print_status(f"Database: {total} total posts, {high_quality} high quality")
            conn.close()
        except Exception as e:
            print_status(f"Database error: {e}", "ERROR")
    else:
        print_status("No database found")

def test_collection(target_posts=50):
    """Test the collection system."""
    print_header(f"TESTING COLLECTION ({target_posts} posts)")
    
    try:
        # Check if we're in the right environment
        import requests
        print_status("Dependencies OK")
        
        # Import and test collection
        from safe_collection_script import SafeTwitterCollector
        
        collector = SafeTwitterCollector()
        result = collector.collect_crypto_content(target_posts=target_posts)
        
        if result["tweets"]:
            posts_count = len(result['tweets'])
            api_calls = sum(result['stats']['api_calls_made'].values())
            efficiency = posts_count / api_calls if api_calls > 0 else 0
            training_data = result['training_data']['posts_for_training']
            
            print_status(f"Collection successful: {posts_count} posts", "SUCCESS")
            print_status(f"Efficiency: {efficiency:.1f} posts per API call")
            print_status(f"Training data stored: {training_data} posts")
        else:
            print_status("No posts collected", "ERROR")
            
    except ImportError as e:
        print_status(f"Missing dependency: {e}", "ERROR")
        print_status("Run: pip install requests")
    except Exception as e:
        print_status(f"Collection test failed: {e}", "ERROR")

def test_analysis():
    """Test the analysis system."""
    print_header("TESTING ANALYSIS")
    
    try:
        from identity_training_pipeline import CryptoIdentityAnalyzer
        
        analyzer = CryptoIdentityAnalyzer()
        print_status("Analyzer initialized", "SUCCESS")
        
        # Test crypto relevance
        test_cases = [
            ("Bitcoin DeFi protocols with yield farming", "crypto"),
            ("Good morning everyone, great day", "generic"),
            ("Ethereum smart contracts for RWA tokenization", "technical")
        ]
        
        print_status("Testing content analysis:")
        for text, category in test_cases:
            crypto_score = analyzer.analyze_crypto_relevance(text)
            quality_score = analyzer.analyze_content_quality(text)
            print(f"  {category}: Crypto={crypto_score:.2f}, Quality={quality_score:.2f}")
            print(f"    Text: {text[:50]}...")
        
        print_status("Analysis system working", "SUCCESS")
        
    except ImportError as e:
        print_status(f"Missing dependency: {e}", "ERROR")
    except Exception as e:
        print_status(f"Analysis test failed: {e}", "ERROR")

def test_model_simple():
    """Test model with simple generation."""
    print_header("TESTING MODEL")
    
    try:
        from model.generate import TextGenerator
        
        print_status("Loading model...")
        generator = TextGenerator()
        
        prompt = "What is decentralized finance?"
        print_status(f"Prompt: {prompt}")
        
        response = generator.generate_response(prompt, max_new_tokens=100)
        print_status("Response generated", "SUCCESS")
        print(f"Output: {response[:200]}...")
        
    except ImportError as e:
        print_status(f"Missing dependency: {e}", "ERROR")
        print_status("Model dependencies not available locally")
    except Exception as e:
        print_status(f"Model test failed: {e}", "ERROR")

def test_model_with_context():
    """Test model with context from database."""
    print_header("TESTING MODEL WITH CONTEXT")
    
    db_path = Path("data/all_posts/posts.db")
    if not db_path.exists():
        print_status("No training database found", "ERROR")
        print_status("Run collection first to generate training data")
        return
    
    try:
        from model.generate import TextGenerator
        
        # Get context from database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM posts WHERE quality_score > 0.7 LIMIT 3")
        context_posts = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not context_posts:
            print_status("No high-quality posts found for context", "ERROR")
            return
        
        context = "\n".join(context_posts[:2])
        
        print_status("Loading model...")
        generator = TextGenerator()
        
        query = "Analyze current crypto trends"
        print_status(f"Query: {query}")
        print_status(f"Using context from {len(context_posts)} posts")
        
        response = generator.generate_with_context(query, context, max_new_tokens=100)
        print_status("Context-aware response generated", "SUCCESS")
        print(f"Output: {response[:200]}...")
        
    except ImportError as e:
        print_status(f"Missing dependency: {e}", "ERROR")
    except Exception as e:
        print_status(f"Context test failed: {e}", "ERROR")

def query_database():
    """Interactive database queries."""
    print_header("DATABASE QUERIES")
    
    db_path = Path("data/all_posts/posts.db")
    if not db_path.exists():
        print_status("No database found", "ERROR")
        return
    
    queries = {
        "1": ("Total posts", "SELECT COUNT(*) FROM posts"),
        "2": ("High quality posts", "SELECT COUNT(*) FROM posts WHERE quality_score > 0.7"),
        "3": ("Top authors", "SELECT author, COUNT(*) FROM posts GROUP BY author ORDER BY COUNT(*) DESC LIMIT 5"),
        "4": ("Best posts", "SELECT author, quality_score, text FROM posts ORDER BY quality_score DESC LIMIT 3"),
        "5": ("Recent posts", "SELECT author, text FROM posts ORDER BY created_at DESC LIMIT 3")
    }
    
    print("Available queries:")
    for key, (desc, _) in queries.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect query (1-5): ").strip()
    
    if choice in queries:
        desc, query = queries[choice]
        print(f"\n{desc}:")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            for row in results:
                if len(row) == 1:
                    print(f"  {row[0]}")
                elif len(row) == 2:
                    print(f"  {row[0]}: {row[1]}")
                else:
                    print(f"  {row[0]} (score: {row[1]:.2f})")
                    print(f"    {row[2][:80]}...")
                    
        except Exception as e:
            print_status(f"Query failed: {e}", "ERROR")

def show_menu():
    """Show the main menu."""
    print_header("XINFLUENCER AI TEST INTERFACE")
    
    menu = {
        "1": ("Check collection status", check_collection_status),
        "2": ("Test collection (50 posts)", lambda: test_collection(50)),
        "3": ("Test analysis system", test_analysis), 
        "4": ("Test model (simple)", test_model_simple),
        "5": ("Test model with context", test_model_with_context),
        "6": ("Database queries", query_database),
        "7": ("Quick collection (20 posts)", lambda: test_collection(20)),
        "q": ("Quit", None)
    }
    
    for key, (desc, _) in menu.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect option: ").strip().lower()
    
    if choice == "q":
        return False
    elif choice in menu and menu[choice][1]:
        try:
            menu[choice][1]()
        except KeyboardInterrupt:
            print_status("Operation cancelled")
        except Exception as e:
            print_status(f"Error: {e}", "ERROR")
        
        input("\nPress Enter to continue...")
    
    return True

def main():
    """Main CLI loop."""
    print("Xinfluencer AI Test Interface")
    print("Professional testing interface for collection, training, and model interaction")
    
    # Environment check
    if not os.getenv('TWITTER_BEARER_TOKEN'):
        print_status("Warning: TWITTER_BEARER_TOKEN not found in environment")
    
    try:
        while show_menu():
            pass
    except KeyboardInterrupt:
        print_status("Goodbye")

if __name__ == "__main__":
    main() 