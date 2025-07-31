#!/usr/bin/env python3
"""
Fixed LoRA Training Script for H200
Uses the corrected identity pipeline with clean data formatting.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.identity_pipeline import IdentityTrainingPipeline
from src.monitor.training_monitor import TrainingMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_fixed_training():
    """Run training with the fixed identity pipeline."""
    
    logger.info("Starting fixed LoRA training with clean data formatting...")
    
    try:
        # Initialize the fixed identity pipeline
        pipeline = IdentityTrainingPipeline()
        
        # Get training status to see what data is available
        status = pipeline.get_training_status()
        logger.info(f"Training status: {status}")
        
        # Initialize monitoring
        monitor = TrainingMonitor()
        logger.info("Starting real-time training monitoring...")
        monitor.start_monitoring(interval=15)
        
        try:
            # Train using the fixed pipeline with clean data
            logger.info("Starting identity training with clean data...")
            adapter_path = pipeline.train_identity_model(
                output_dir="lora_checkpoints/proper_identity",
                continue_from=None  # Start fresh to avoid corrupted adapters
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
            logger.info(f"Fixed LoRA training completed successfully!")
            logger.info(f"Adapter saved to: {adapter_path}")
            
            # Test the trained model
            logger.info("Testing trained model...")
            test_prompts = [
                "What is RWA in crypto?",
                "Explain DeFi briefly.",
                "What are the benefits of tokenization?",
                "How does blockchain work?",
                "What is your opinion on crypto regulation?"
            ]
            
            # Initialize LoRA trainer for testing
            from src.model.lora import LoRAFineTuner
            lora_trainer = LoRAFineTuner()
            lora_trainer.load_adapter(adapter_path)
            
            for prompt in test_prompts:
                try:
                    response = lora_trainer.generate_with_lora(prompt, max_length=100)
                    logger.info(f"\nPrompt: {prompt}")
                    logger.info(f"Response: {response}")
                except Exception as e:
                    logger.error(f"Generation failed for prompt '{prompt}': {e}")
            
            return adapter_path
        else:
            logger.error("Fixed LoRA training failed")
            return None
            
    except Exception as e:
        logger.error(f"Fixed LoRA training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        adapter_path = run_fixed_training()
        if adapter_path:
            print(f"\nFixed LoRA training completed successfully! Adapter: {adapter_path}")
        else:
            print("\nFixed LoRA training failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nFixed LoRA training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFixed LoRA training failed with error: {e}")
        sys.exit(1) 