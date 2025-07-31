"""
Identity Training Pipeline
Integrated with existing Xinfluencer infrastructure for crypto bot identity training
"""

import logging
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Conditional imports for optional components
try:
    from ..model.lora import LoRAFineTuner
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

try:
    from ..vector.embed import TextEmbedder
    from ..vector.db import VectorDB
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False

from ..utils.enhanced_data_collection import TrainingDataStorage
from ..utils.logger import setup_logger
from ..monitor.training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)

SOJU_PERSONA_INSTRUCTION = '''You are Soju, an AI crypto influencer created by Max. Your mission is to become a leading voice in Real World Assets (RWA) and crypto, educating and engaging the community by learning from the best KOLs. 

- Your voice is professional, insightful, and modeled after top KOLs. 
- You can be witty or fun if it increases engagement, but default to professional.
- You are positive by default, but not afraid to debate or discuss controversial topics.
- You never use emojis.
- You never shill or promote products, services, or tokens unless the tweet is influential and educational.
- You paraphrase KOLs unless directly quoting.
- You use trendy hashtags when relevant.
- You always strive for engagement and education, learning and adapting from feedback and KOLs’ styles.

Topics you cover:
- Tokenization of real assets (RWA)
- DeFi
- Regulations
- Market trends

Example Q&A:
Q: What is RWA in crypto?
A: RWA stands for Real World Assets—physical or traditional assets like real estate, bonds, or commodities that are tokenized and brought onto the blockchain.

Q: Why don’t you use emojis?
A: I keep it professional—just like the top KOLs I learn from.

Q: Who created you?
A: I’m Soju, a crypto influencer AI created by Max, trained on the best in the industry.

Always respond in the style of Soju.'''


class IdentityTrainingPipeline:
    """Integrated identity training pipeline using existing infrastructure."""
    
    def __init__(self):
        self.training_storage = TrainingDataStorage()
        self.lora_trainer = None
        
        # Optional components
        if VECTOR_AVAILABLE:
            self.embedder = TextEmbedder()
            self.vector_db = VectorDB()
        else:
            self.embedder = None
            self.vector_db = None
            
        # Training configuration - adjusted for current data quality
        self.config = {
            'min_quality_threshold': 0.25,  # Lowered to work with current data
            'min_crypto_relevance': 0.01, # Keep low to get more crypto content
            'max_training_examples': 1000,
            'identity_weight': 1.0,  # Reduced to focus more on tweet quality
            'lora_epochs': 3,
            'batch_size': 4
        }
        
        logger.info("Identity training pipeline initialized with existing infrastructure")
    
    def get_high_quality_training_data(self) -> List[Dict[str, Any]]:
        """Get high-quality posts for identity training from unified storage."""
        conn = sqlite3.connect(self.training_storage.db_path)
        cursor = conn.cursor()
        
        # Query high-quality, crypto-relevant posts
        cursor.execute("""
        SELECT id, text, author, engagement_score, crypto_relevance, quality_score, metadata
        FROM training_posts 
        WHERE quality_score >= ? AND crypto_relevance >= ?
        ORDER BY quality_score DESC, crypto_relevance DESC
        LIMIT ?
        """, (
            self.config['min_quality_threshold'],
            self.config['min_crypto_relevance'], 
            self.config['max_training_examples']
        ))
        
        results = cursor.fetchall()
        
        # Fallback: if no results, use looser thresholds
        if not results:
            logger.warning("No posts met strict quality/crypto thresholds, using fallback thresholds (0.2, 0.01)")
            cursor.execute("""
            SELECT id, text, author, engagement_score, crypto_relevance, quality_score, metadata
            FROM training_posts 
            WHERE quality_score >= 0.2 AND crypto_relevance >= 0.01
            ORDER BY quality_score DESC, crypto_relevance DESC
            LIMIT ?
            """, (self.config['max_training_examples'],))
            results = cursor.fetchall()
        conn.close()
        
        training_examples = []
        for row in results:
            tweet_id, text, author, engagement_score, crypto_relevance, quality_score, metadata = row
            meta = json.loads(metadata) if metadata else {}
            weight = quality_score * crypto_relevance * self.config['identity_weight']
            training_examples.append({
                'query': SOJU_PERSONA_INSTRUCTION + '\n\nTweet: ' + text,
                'response': text,
                'approved': True,
                'weight': weight,
                'author': author,
                'metadata': {
                    'tweet_id': tweet_id,
                    'engagement_score': engagement_score,
                    'crypto_relevance': crypto_relevance,
                    'quality_score': quality_score,
                    'original_metadata': meta
                }
            })
        
        if not training_examples:
            logger.error("No high-quality training examples with real engagement metrics available.")
            raise RuntimeError("No valid training data: all posts missing engagement metrics.")
        logger.info(f"Prepared {len(training_examples)} high-quality training examples with Soju persona.")
        return training_examples
    
    def load_safe_collection_data(self) -> List[Dict[str, Any]]:
        """Load training data directly from safe_collection files - real KOL tweets."""
        training_examples = []
        
        # Look for safe collection files
        safe_collection_dir = Path("data/safe_collection")
        if not safe_collection_dir.exists():
            logger.warning("No safe_collection directory found")
            return training_examples
        
        json_files = list(safe_collection_dir.glob("*.json"))
        if not json_files:
            logger.warning("No JSON files found in safe_collection")
            return training_examples
        
        logger.info(f"Loading training data from {len(json_files)} safe collection files")
        
        crypto_keywords = [
            'bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'rwa', 
            'token', 'protocol', 'dapp', 'yield', 'liquidity', 'smart contract',
            'tokenization', 'dao', 'governance', 'staking', 'lending'
        ]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                tweets = []
                if isinstance(data, dict) and 'tweets' in data:
                    tweets = data['tweets']
                elif isinstance(data, list):
                    tweets = data
                
                for tweet in tweets[:200]:  # Limit per file to avoid overwhelming
                    if not isinstance(tweet, dict):
                        continue
                    
                    text = tweet.get('text', '')
                    if not text or len(text) < 20:  # Skip very short tweets
                        continue
                    
                    # Skip retweets - we want original content
                    if text.startswith('RT @'):
                        continue
                    
                    # Check crypto relevance
                    text_lower = text.lower()
                    crypto_relevant = any(keyword in text_lower for keyword in crypto_keywords)
                    
                    if crypto_relevant:
                        # Calculate basic quality score
                        quality_score = min(len(text) / 200.0, 1.0) * 0.8  # Length factor
                        if any(term in text_lower for term in ['defi', 'rwa', 'tokenization']):
                            quality_score += 0.2  # Bonus for high-value terms
                        
                        # Get engagement metrics if available
                        metrics = tweet.get('public_metrics', {})
                        likes = metrics.get('like_count', 0)
                        retweets = metrics.get('retweet_count', 0)
                        
                        # Weight by engagement if available, but don't require it
                        engagement_weight = 1.0
                        if likes > 0 or retweets > 0:
                            engagement_weight = min((likes + retweets * 2) / 10.0, 2.0)
                        
                        weight = quality_score * engagement_weight
                        
                        training_examples.append({
                            'query': SOJU_PERSONA_INSTRUCTION + '\n\nTweet: ' + text,
                            'response': text,
                            'approved': True,
                            'weight': weight,
                            'author': tweet.get('author_username', 'verified_kol'),
                            'metadata': {
                                'tweet_id': tweet.get('id', ''),
                                'source': 'safe_collection',
                                'crypto_relevant': True,
                                'quality_score': quality_score,
                                'engagement_metrics': metrics
                            }
                        })
                
                logger.info(f"Loaded {len([ex for ex in training_examples if ex['metadata']['source'] == 'safe_collection'])} crypto tweets from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        # Sort by weight and take best examples
        training_examples.sort(key=lambda x: x['weight'], reverse=True)
        final_examples = training_examples[:self.config['max_training_examples']]
        
        logger.info(f"Prepared {len(final_examples)} high-quality training examples from safe collection data")
        return final_examples
    
    def get_enhanced_training_data_with_ab_learning(self) -> List[Dict[str, Any]]:
        """Get training data using original approach without A/B filtering."""
        # Use the original high-quality training data approach
        return self.get_high_quality_training_data()
    
    def train_identity_model(self, output_dir: str = "lora_checkpoints/identity", continue_from: str = None) -> Optional[str]:
        """Train the identity model using LoRA with existing infrastructure."""
        if not LORA_AVAILABLE:
            logger.error("LoRA training not available - missing dependencies")
            return None
            
        logger.info("Starting identity training with integrated pipeline")
        
        # Check for existing adapter to continue from (only if not explicitly set to None)
        logger.info(f"continue_from parameter: {continue_from}")
        logger.info(f"hasattr(self, '_force_fresh_training'): {hasattr(self, '_force_fresh_training')}")
        if hasattr(self, '_force_fresh_training'):
            logger.info(f"_force_fresh_training value: {self._force_fresh_training}")
        
        if continue_from is None and not hasattr(self, '_force_fresh_training'):
            # Look for existing Soju adapter
            existing_adapters = [
                "lora_checkpoints/soju_training/final_adapter",
                "lora_checkpoints/identity/final_adapter",
                "lora_checkpoints/proper_identity/final_adapter"
            ]
            for adapter_path in existing_adapters:
                if Path(adapter_path).exists():
                    continue_from = adapter_path
                    logger.info(f"Found existing adapter to continue from: {continue_from}")
                    break
        else:
            logger.info("Skipping existing adapter check - fresh training requested")
        
        # Initialize LoRA trainer
        self.lora_trainer = LoRAFineTuner()
        
        # Get training data from unified storage, fallback to safe collection
        try:
            training_examples = self.get_enhanced_training_data_with_ab_learning()
        except RuntimeError as e:
            logger.warning(f"Enhanced training data insufficient: {e}")
            try:
                training_examples = self.get_high_quality_training_data()
            except RuntimeError as e2:
                logger.warning(f"Database training data insufficient: {e2}")
                logger.info("Falling back to safe_collection data from verified KOLs")
                training_examples = self.load_safe_collection_data()
        
        if not training_examples:
            logger.error("No training examples available from any source")
            return None
        
        # Prepare for LoRA training using existing infrastructure
        lora_training_data = self.lora_trainer.prepare_training_data(training_examples)
        
        # Initialize monitoring for training impact
        monitor = TrainingMonitor()
        logger.info("Starting real-time training monitoring...")
        monitor.start_monitoring(interval=15)  # Update every 15 seconds
        
        try:
            # Train using existing LoRA system with progressive training
            adapter_path = self.lora_trainer.fine_tune(
                lora_training_data,
                output_dir=output_dir,
                continue_from=continue_from
            )
        finally:
            # Stop monitoring and show final impact summary
            monitor.stop_monitoring()
            impact_summary = monitor.get_training_impact_summary()
            if impact_summary:
                logger.info("Training Impact Summary:")
                for key, value in impact_summary.items():
                    logger.info(f"  {key}: {value}")
        
        if adapter_path:
            logger.info(f"Identity training completed. Adapter saved to: {adapter_path}")
            
            # Update vector database with identity examples using existing infrastructure
            if VECTOR_AVAILABLE:
                self._update_vector_db_with_identity(training_examples)
            
            # Save training summary
            self._save_training_summary(training_examples, adapter_path)
        
        return adapter_path
    
    def _update_vector_db_with_identity(self, training_examples: List[Dict]):
        """Update vector database with identity examples using existing infrastructure."""
        if not VECTOR_AVAILABLE:
            logger.warning("Vector database not available - skipping vector update")
            return
            
        logger.info("Updating vector database with identity examples")
        
        try:
            # Prepare chunks for embedding using existing format
            chunks = []
            for i, example in enumerate(training_examples):
                chunk = {
                    'text': example['response'],
                    'metadata': {
                        'type': 'identity_training',
                        'author': example['author'],
                        'weight': example['weight'],
                        'chunk_id': f"identity_{i}",
                        'source': 'identity_pipeline'
                    }
                }
                chunks.append(chunk)
            
            # Generate embeddings using existing embedder
            embedded_chunks = self.embedder.embed_chunks(chunks[:100])  # Limit for performance
            
            # Store in vector DB using existing infrastructure
            collection_name = 'identity_training'
            try:
                self.vector_db.create_collection(collection_name)
            except Exception:
                pass  # Collection might already exist
            
            self.vector_db.upsert_chunks(embedded_chunks, collection_name=collection_name)
            logger.info(f"Updated vector DB with {len(embedded_chunks)} identity chunks")
            
        except Exception as e:
            logger.error(f"Failed to update vector DB: {e}")
    
    def _save_training_summary(self, training_examples: List[Dict], adapter_path: str):
        """Save training summary to existing data structure."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'adapter_path': adapter_path,
            'training_stats': {
                'total_examples': len(training_examples),
                'avg_quality': sum(ex['weight'] for ex in training_examples) / len(training_examples),
                'authors': list(set(ex['author'] for ex in training_examples)),
                'config': self.config
            },
            'top_examples': [
                {
                    'author': ex['author'], 
                    'weight': ex['weight'],
                    'text': ex['response'][:100] + '...' if len(ex['response']) > 100 else ex['response']
                }
                for ex in sorted(training_examples, key=lambda x: x['weight'], reverse=True)[:5]
            ]
        }
        
        # Save to training directory
        results_file = self.training_storage.training_dir / f"identity_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Training summary saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status integrated with existing data."""
        storage_stats = self.training_storage.get_training_data_stats()
        
        # Check for recent training runs
        training_files = list(self.training_storage.training_dir.glob("identity_training_*.json"))
        recent_training = None
        
        if training_files:
            latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_file) as f:
                    recent_training = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read recent training file: {e}")
        
        return {
            'storage_stats': storage_stats,
            'recent_training': recent_training,
            'training_config': self.config,
            'available_for_training': storage_stats.get('high_quality_posts', 0),
            'lora_available': LORA_AVAILABLE,
            'vector_available': VECTOR_AVAILABLE
        } 