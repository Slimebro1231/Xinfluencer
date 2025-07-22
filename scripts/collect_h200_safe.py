#!/usr/bin/env python3
"""
Safe collection script for h200 deployment.
Runs enhanced data collection with proper imports and error handling.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import with absolute imports
from src.utils.enhanced_data_collection import EnhancedDataCollectionPipeline
from src.utils.logger import setup_logger

def main():
    """Run safe collection on h200."""
    logger = setup_logger("h200_collection", level="INFO")
    
    try:
        # Get target posts from command line or default
        target_posts = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        
        logger.info(f"Starting h200 safe collection for {target_posts} posts")
        
        # Initialize pipeline
        pipeline = EnhancedDataCollectionPipeline()
        
        # Run safe collection
        result = pipeline.safe_collect_crypto_content(
            target_posts=target_posts,
            save_for_training=True
        )
        
        # Report results
        if result.get("success", False):
            logger.info(f"Collection completed successfully!")
            logger.info(f"Total posts collected: {result.get('total_posts', 0)}")
            logger.info(f"API calls made: {result.get('api_calls_made', 0)}")
            logger.info(f"Session ID: {result.get('session_id', 'unknown')}")
        else:
            logger.error(f"Collection failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Collection script failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 