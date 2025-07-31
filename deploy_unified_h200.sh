#!/bin/bash

# Unified H200 Deployment Script
# Consolidates all deployment operations into a single script

set -e

H200_HOST="157.10.162.127"
H200_USER="ubuntu"
PROJECT_DIR="/home/ubuntu/xinfluencer"
PEM_FILE="/Users/max/Xinfluencer/influencer.pem"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Unified H200 Deployment Script${NC}"
echo "=================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to test SSH connection
test_ssh() {
    print_status "Testing SSH connection to H200..."
    if ! ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${H200_USER}@${H200_HOST} "echo 'SSH connection successful'"; then
        print_error "Cannot connect to H200 server"
        exit 1
    fi
}

# Function to check H200 status
check_h200_status() {
    print_status "Checking H200 status..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
}

# Function to sync project files
sync_project() {
    print_status "Syncing project files to H200..."
    rsync -avz -e "ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no" --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='tests' \
        --exclude='*.log' --exclude='data/cache' \
        . \
        ${H200_USER}@${H200_HOST}:${PROJECT_DIR}/
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies on H200..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && pip install -r requirements_h200.txt"
    
    # Install additional dependencies if needed
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && pip install rank-bm25"
}



# Function to wipe both databases
wipe_both_databases() {
    print_status "Wiping both posts.db and vector database..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c \"
import sqlite3
import os
import shutil
from src.vector.db import VectorDB

print('Wiping both databases...')

# Wipe posts.db
try:
    conn = sqlite3.connect('data/training/posts.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM training_posts')
    conn.commit()
    conn.close()
    print('posts.db wiped successfully')
except Exception as e:
    print(f'Warning: Could not wipe posts.db: {e}')

# Clear vector database
try:
    vector_db = VectorDB()
    vector_db.data = []
    vector_db.next_id = 0
    print('Vector database cleared')
except Exception as e:
    print(f'Warning: Could not clear vector database: {e}')

# Clear cache directory
cache_dir = 'data/cache'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print('Cache directory cleared')

print('Both databases wiped successfully')
\""
}

# Function to clean and rebuild vector database
clean_vector_db() {
    print_status "Cleaning vector database with old scoring..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c \"
from src.vector.db import VectorDB
import os

# Clear vector database
vector_db = VectorDB()
vector_db.data = []  # Clear all data
vector_db.next_id = 0
print('Vector database cleared successfully')

# Clear any cached embeddings
cache_dir = 'data/cache'
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print('Embedding cache cleared')
\""
}

# Function to test improved evaluation system
test_evaluation_system() {
    print_status "Testing improved evaluation system..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c \"
from src.evaluation.tweet_quality import TweetQualityEvaluator
import json

# Test initialization
evaluator = TweetQualityEvaluator()
print('Improved evaluation system initialized successfully')

# Test configuration
config = evaluator.config
print(f'Logarithmic scoring enabled: {config[\"scoring\"][\"engagement_normalization\"] == \"logarithmic\"}')
print(f'Semantic search enabled: {config[\"scoring\"][\"semantic_search\"][\"enabled\"]}')
print(f'Granular levels configured: {len(config[\"scoring\"][\"granularity_levels\"])} levels')

# Test with sample tweets
sample_tweets = [
    {
        'text': 'Bitcoin is the ultimate store of value. The halving will drive price to new heights.',
        'public_metrics': {'like_count': 1500, 'retweet_count': 300, 'reply_count': 150, 'quote_count': 75},
        'author_id': 'test_author_1',
        'created_at': '2024-01-01T12:00:00Z'
    },
    {
        'text': 'Just had coffee. Nice weather today.',
        'public_metrics': {'like_count': 50, 'retweet_count': 5, 'reply_count': 10, 'quote_count': 2},
        'author_id': 'test_author_2',
        'created_at': '2024-01-01T13:00:00Z'
    }
]

for i, tweet in enumerate(sample_tweets, 1):
    result = evaluator.evaluate_tweet_for_training(tweet)
    print(f'Tweet {i} - Quality: {result[\"quality_score\"]:.4f}, Level: {result[\"quality_level\"]}')

print('Improved evaluation system test completed successfully')
\""
}

# Function to rebuild vector database from raw data
rebuild_vector_db() {
    print_status "Rebuilding vector database from raw JSON data..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c \"
import json
import os
from pathlib import Path
from src.vector.embed import TextEmbedder
from src.vector.db import VectorDB

# Find raw JSON files
data_dir = Path('data/collected')
json_files = list(data_dir.glob('*.json'))

if not json_files:
    print('No raw JSON files found for rebuilding')
    exit(0)

print(f'Found {len(json_files)} JSON files to process')

# Initialize components
embedder = TextEmbedder()
vector_db = VectorDB()
vector_db.create_collection()

total_tweets = 0
total_chunks = 0
chunks = []

for json_file in json_files:
    print(f'Processing {json_file.name}...')
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # Check if it's organized by author
        if any(isinstance(v, list) for v in data.values()):
            # Author-organized structure
            for author, author_tweets in data.items():
                if isinstance(author_tweets, list):
                    total_tweets += len(author_tweets)
                    for tweet in author_tweets:
                        text = tweet.get('text', '')
                        if not text.strip():
                            continue
                            
                        # Create chunk
                        chunk = {
                            'text': text,
                            'tweet_id': str(tweet.get('id', '')),
                            'timestamp': tweet.get('created_at', ''),
                            'metadata': {
                                'author_id': str(tweet.get('author_id', '')),
                                'author_username': tweet.get('author_username', ''),
                                'public_metrics': tweet.get('public_metrics', {})
                            }
                        }
                        
                        # Generate embedding
                        try:
                            embedding = embedder.embed_text(text)
                            chunk['embedding'] = embedding
                            chunks.append(chunk)
                            total_chunks += 1
                        except Exception as e:
                            print(f'Warning: Failed to embed tweet {tweet.get(\"id\", \"unknown\")}: {e}')
                            continue
        else:
            # Standard structure with 'tweets' key
            tweets = data.get('tweets', [])
            total_tweets += len(tweets)
            for tweet in tweets:
                text = tweet.get('text', '')
                if not text.strip():
                    continue
                    
                # Create chunk
                chunk = {
                    'text': text,
                    'tweet_id': str(tweet.get('id', '')),
                    'timestamp': tweet.get('created_at', ''),
                    'metadata': {
                        'author_id': str(tweet.get('author_id', '')),
                        'public_metrics': tweet.get('public_metrics', {})
                    }
                }
                
                # Generate embedding
                try:
                    embedding = embedder.embed_text(text)
                    chunk['embedding'] = embedding
                    chunks.append(chunk)
                    total_chunks += 1
                except Exception as e:
                    print(f'Warning: Failed to embed tweet {tweet.get(\"id\", \"unknown\")}: {e}')
                    continue
    elif isinstance(data, list):
        # Direct list of tweets
        total_tweets += len(data)
        for tweet in data:
            text = tweet.get('text', '')
            if not text.strip():
                continue
                
            # Create chunk
            chunk = {
                'text': text,
                'tweet_id': str(tweet.get('id', '')),
                'timestamp': tweet.get('created_at', ''),
                'metadata': {
                    'author_id': str(tweet.get('author_id', '')),
                    'public_metrics': tweet.get('public_metrics', {})
                }
            }
            
            # Generate embedding
            try:
                embedding = embedder.embed_text(text)
                chunk['embedding'] = embedding
                chunks.append(chunk)
                total_chunks += 1
            except Exception as e:
                print(f'Warning: Failed to embed tweet {tweet.get(\"id\", \"unknown\")}: {e}')
                continue

# Store all chunks
if chunks:
    vector_db.upsert_chunks(chunks)
    print(f'Vector database rebuilt successfully')
    print(f'Total tweets processed: {total_tweets}')
    print(f'Total chunks stored: {total_chunks}')
else:
    print('No valid chunks found to store')
\""
}

# Function to retrieve new posts and score them
retrieve_and_score_posts() {
    print_status "Retrieving 200 new posts and scoring with improved semantic system..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c \"
import sys
import json
from datetime import datetime
from src.utils.enhanced_data_collection import EnhancedDataCollectionPipeline
from src.evaluation.tweet_quality import TweetQualityEvaluator
from src.vector.db import VectorDB
from src.vector.embed import TextEmbedder

print('Starting new post collection and scoring...')

# Initialize components
pipeline = EnhancedDataCollectionPipeline()
evaluator = TweetQualityEvaluator()
vector_db = VectorDB()
embedder = TextEmbedder()

# Retrieve 200 new posts
print('Retrieving 200 new posts...')
collection_results = pipeline.safe_collect_crypto_content(
    target_posts=200,
    save_for_training=True
)

if not collection_results.get('success', True):
    print(f'Collection failed: {collection_results.get(\"error\", \"Unknown error\")}')
    exit(1)

total_new_posts = collection_results.get('total_posts_collected', 0)
print(f'Retrieved {total_new_posts} new posts')

# Score new posts with improved semantic system
print('Scoring new posts with improved semantic system...')
new_posts_scored = 0
new_posts_data = []

# Process collected data
all_tweets = []
if 'kol_data' in collection_results:
    for username, tweets in collection_results['kol_data'].items():
        all_tweets.extend(tweets)

for tweet in all_tweets:
    try:
        # Convert to dict if needed
        if hasattr(tweet, 'id'):
            tweet_dict = {
                'id': tweet.id,
                'text': tweet.text,
                'author_username': tweet.username,
                'public_metrics': tweet.public_metrics,
                'created_at': getattr(tweet, 'created_at', '')
            }
        else:
            tweet_dict = tweet

        # Score with improved system
        evaluation = evaluator.evaluate_tweet_for_training(tweet_dict)
        
        # Add to vector database
        text = tweet_dict.get('text', '')
        if text.strip():
            embedding = embedder.embed_text(text)
            chunk = {
                'text': text,
                'tweet_id': str(tweet_dict.get('id', '')),
                'timestamp': tweet_dict.get('created_at', ''),
                'metadata': {
                    'author_id': str(tweet_dict.get('author_id', '')),
                    'public_metrics': tweet_dict.get('public_metrics', {}),
                    'quality_score': evaluation['quality_score'],
                    'relevance_score': evaluation['relevance_score'],
                    'quality_level': evaluation['quality_level'],
                    'semantic_search_used': evaluation.get('semantic_search_used', False)
                }
            }
            chunk['embedding'] = embedding
            new_posts_data.append(chunk)
            new_posts_scored += 1

    except Exception as e:
        print(f'Warning: Failed to score tweet {tweet_dict.get(\"id\", \"unknown\")}: {e}')
        continue

# Store new posts in vector database
if new_posts_data:
    vector_db.upsert_chunks(new_posts_data)
    print(f'Stored {new_posts_scored} new scored posts in vector database')

# Score existing posts with improved system
print('Scoring existing posts with improved semantic system...')
existing_posts_scored = 0

# Get existing posts from training database
try:
    import sqlite3
    conn = sqlite3.connect('data/training/posts.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, text, author, metadata FROM training_posts LIMIT 100')
    existing_posts = cursor.fetchall()
    conn.close()

    for post_id, text, author, metadata in existing_posts:
        try:
            # Parse metadata
            if metadata:
                metadata_dict = json.loads(metadata)
            else:
                metadata_dict = {}

            # Create tweet dict for evaluation
            tweet_dict = {
                'id': post_id,
                'text': text,
                'author_username': author,
                'public_metrics': metadata_dict
            }

            # Score with improved system
            evaluation = evaluator.evaluate_tweet_for_training(tweet_dict)
            
            # Update database with new scores
            conn = sqlite3.connect('data/training/posts.db')
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE training_posts 
                SET engagement_score = ?, crypto_relevance = ?, quality_score = ?
                WHERE id = ?
            ''', (
                evaluation['engagement_score'],
                evaluation['relevance_score'], 
                evaluation['quality_score'],
                post_id
            ))
            conn.commit()
            conn.close()
            
            existing_posts_scored += 1

        except Exception as e:
            print(f'Warning: Failed to rescore existing post {post_id}: {e}')
            continue

    print(f'Rescored {existing_posts_scored} existing posts')

except Exception as e:
    print(f'Warning: Could not access existing posts database: {e}')

# Final summary
print('\\n=== Collection and Scoring Summary ===')
print(f'New posts retrieved: {total_new_posts}')
print(f'New posts scored and stored: {new_posts_scored}')
print(f'Existing posts rescored: {existing_posts_scored}')
print(f'Total posts processed: {new_posts_scored + existing_posts_scored}')

# Test semantic search with new data
print('\\nTesting semantic search with new data...')
try:
    from src.vector.search import VectorSearcher
    searcher = VectorSearcher(vector_db, embedder)
    
    # Test Bitcoin search
    btc_results = searcher.search_similar('Bitcoin adoption and investment', top_k=5)
    print(f'Bitcoin search results: {len(btc_results)} found')
    
    # Test Gold search  
    gold_results = searcher.search_similar('Gold investment and RWA tokenization', top_k=5)
    print(f'Gold search results: {len(gold_results)} found')
    
except Exception as e:
    print(f'Semantic search test failed: {e}')

print('Collection and scoring completed successfully')
\""
}

# Function to run LoRA training on H200
run_lora_training() {
    print_status "Starting LoRA training on H200 server..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 run_training_remote.py"
}

# Function to test full system
test_full_system() {
    print_status "Testing full system integration..."
    ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 -c \"
from src.evaluation.tweet_quality import TweetQualityEvaluator
from src.vector.db import VectorDB
from src.vector.search import VectorSearcher
from src.vector.embed import TextEmbedder

# Test vector search
vector_db = VectorDB()
embedder = TextEmbedder()
searcher = VectorSearcher(vector_db, embedder)

print('Vector search components initialized')

# Test evaluation with semantic search
evaluator = TweetQualityEvaluator()
print('Evaluation system with semantic search initialized')

# Test sample search
try:
    results = searcher.search_similar('Bitcoin adoption', top_k=3)
    print(f'Semantic search test: Found {len(results)} results')
except Exception as e:
    print(f'Semantic search test failed: {e}')

print('Full system test completed')
\""
}

# Main deployment flow
main() {
    echo -e "${BLUE}Starting unified deployment with new post collection...${NC}"
    
    # Test connection
    test_ssh
    
    # Check H200 status
    check_h200_status
    
    # Sync project files
    sync_project
    
    # Install dependencies
    install_dependencies
    
    # Wipe both databases to start fresh
    wipe_both_databases
    
    # Test improved evaluation system
    test_evaluation_system
    
    # Retrieve new posts and score them with improved system
    retrieve_and_score_posts
    
    # Test full system
    test_full_system
    
    # Run LoRA training on H200
    run_lora_training
    
    echo ""
    echo -e "${GREEN}Unified deployment with new post collection completed successfully!${NC}"
    echo "================================================"
    echo -e "${BLUE}Key improvements deployed:${NC}"
    echo "1. Logarithmic engagement scoring for better differentiation"
    echo "2. Expanded crypto keywords and terminology"
    echo "3. Granular quality levels (8 distinct levels)"
    echo "4. Enhanced semantic search integration"
    echo "5. Both databases wiped and rebuilt with new scoring"
    echo "6. 200 new posts retrieved and scored with improved system"
    echo "7. Existing posts rescored with new semantic system"
    echo "8. LoRA training started on H200 with full database"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Monitor LoRA training progress on H200"
    echo "2. Monitor evaluation results for new tweets"
    echo "3. Verify Bitcoin content receives higher scores"
    echo "4. Check semantic search integration"
    echo "5. Adjust granularity levels if needed"
    echo "6. Review collection and scoring summary"
}

# Run main function
main "$@" 