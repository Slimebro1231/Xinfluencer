#!/usr/bin/env python3
"""Run manual training using the custom trainer that bypasses HuggingFace Trainer."""

import os
import sys
import logging
import json
from pathlib import Path

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.identity_pipeline import IdentityTrainingPipeline
from src.training.manual_trainer import ManualTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_manual_training():
    """Run manual training using the custom trainer."""
    
    logger.info("Starting manual training with custom trainer...")
    
    try:
        # Initialize identity pipeline to get training data
        pipeline = IdentityTrainingPipeline()
        
        # Get high-quality training data
        logger.info("Loading training data...")
        training_examples = pipeline.get_high_quality_training_data()
        
        if not training_examples:
            logger.warning("No high-quality data found, trying safe collection data...")
            training_examples = pipeline.load_safe_collection_data()
        
        if not training_examples:
            logger.error("No training data available")
            return None
        
        logger.info(f"Loaded {len(training_examples)} training examples")
        
        # Show data statistics
        quality_scores = [ex.get('weight', 1.0) for ex in training_examples]
        avg_quality = sum(quality_scores) / len(quality_scores)
        logger.info(f"Training data statistics:")
        logger.info(f"  Average weight: {avg_quality:.3f}")
        logger.info(f"  Max weight: {max(quality_scores):.3f}")
        logger.info(f"  Min weight: {min(quality_scores):.3f}")
        
        # Use the pipeline's manual training method
        logger.info("Starting manual training using pipeline...")
        adapter_path = pipeline.train_identity_model_manual(
            output_dir="lora_checkpoints/manual_training"
        )
        
        if adapter_path:
            logger.info(f"Training completed successfully!")
            logger.info(f"Adapter saved to: {adapter_path}")
            
            # Test the trained model
            logger.info("Testing trained model...")
            manual_trainer = ManualTrainer()
            manual_trainer.load_adapter(adapter_path)
            
            test_prompts = [
                "Create a crypto tweet about Bitcoin",
                "What is RWA in crypto?",
                "Explain DeFi to a beginner"
            ]
            
            for prompt in test_prompts:
                try:
                    response = manual_trainer.generate(prompt, max_length=100)
                    logger.info(f"\nPrompt: {prompt}")
                    logger.info(f"Response: {response}")
                except Exception as e:
                    logger.error(f"Generation failed for prompt '{prompt}': {e}")
            
            return adapter_path
        else:
            logger.error("Training failed")
            return None
            
    except Exception as e:
        logger.error(f"Manual training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        adapter_path = run_manual_training()
        if adapter_path:
            print(f"\nTraining completed successfully! Adapter: {adapter_path}")
        else:
            print("\nTraining failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1) 