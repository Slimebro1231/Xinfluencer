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
            
        # Training configuration
        self.config = {
            'min_quality_threshold': 0.7,
            'min_crypto_relevance': 0.8,
            'max_training_examples': 1000,
            'identity_weight': 1.5,
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
        conn.close()
        
        training_examples = []
        for row in results:
            tweet_id, text, author, engagement_score, crypto_relevance, quality_score, metadata = row
            
            meta = json.loads(metadata) if metadata else {}
            # Require real engagement metrics
            if not (meta.get('like_count', 0) or meta.get('retweet_count', 0) or meta.get('reply_count', 0)):
                continue  # skip if no engagement
            
            # Weight by quality and crypto relevance
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
    
    def train_identity_model(self, output_dir: str = "lora_checkpoints/identity") -> Optional[str]:
        """Train the identity model using LoRA with existing infrastructure."""
        if not LORA_AVAILABLE:
            logger.error("LoRA training not available - missing dependencies")
            return None
            
        logger.info("Starting identity training with integrated pipeline")
        
        # Initialize LoRA trainer
        self.lora_trainer = LoRAFineTuner()
        
        # Get training data from unified storage
        training_examples = self.get_high_quality_training_data()
        
        if not training_examples:
            logger.error("No high-quality training examples available")
            return None
        
        # Prepare for LoRA training using existing infrastructure
        lora_training_data = self.lora_trainer.prepare_training_data(training_examples)
        
        # Train using existing LoRA system
        adapter_path = self.lora_trainer.fine_tune(
            lora_training_data,
            output_dir=output_dir
        )
        
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