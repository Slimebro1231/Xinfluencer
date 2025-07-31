#!/usr/bin/env python3
"""
Remote LoRA Training Script for H200
Runs on the H200 server with full GPU memory and proper training setup.
Uses existing training data from lora_checkpoints.
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

from src.model.lora import LoRAFineTuner
from src.monitor.training_monitor import TrainingMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_existing_training_data():
    """Load training data from existing lora_checkpoints."""
    training_data = []
    
    # Check for existing training data files
    data_sources = [
        "lora_checkpoints/proper_identity/training_data.json",
        "lora_checkpoints/manual_training/training_data.json"
    ]
    
    for data_file in data_sources:
        if Path(data_file).exists():
            logger.info(f"Loading training data from {data_file}")
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to the format expected by LoRA trainer
                for item in data:
                    if 'text' in item:
                        training_data.append({
                            'text': item['text'],
                            'weight': item.get('weight', 1.0),
                            'type': 'existing_training'
                        })
                
                logger.info(f"Loaded {len(data)} examples from {data_file}")
                
            except Exception as e:
                logger.error(f"Failed to load {data_file}: {e}")
    
    return training_data

def run_h200_training():
    """Run LoRA training on H200 with full resources."""
    
    logger.info("Starting H200 LoRA training with existing data...")
    
    try:
        # Load existing training data
        training_data = load_existing_training_data()
        
        if not training_data:
            logger.error("No training data found in lora_checkpoints")
            return None
        
        logger.info(f"Loaded {len(training_data)} training examples")
        
        # Show data statistics
        weights = [ex.get('weight', 1.0) for ex in training_data]
        avg_weight = sum(weights) / len(weights)
        logger.info(f"Training data statistics:")
        logger.info(f"  Total examples: {len(training_data)}")
        logger.info(f"  Average weight: {avg_weight:.3f}")
        logger.info(f"  Max weight: {max(weights):.3f}")
        logger.info(f"  Min weight: {min(weights):.3f}")
        
        # Initialize LoRA trainer with H200-optimized settings
        logger.info("Initializing LoRA trainer for H200...")
        lora_trainer = LoRAFineTuner()
        
        # Save prepared training data
        output_dir = "lora_checkpoints/h200_training"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/training_data.json", 'w') as f:
            json.dump(training_data[:20], f, indent=2)  # Save first 20 examples
        
        logger.info(f"Using {len(training_data)} training examples for LoRA")
        
        # Initialize monitoring
        monitor = TrainingMonitor()
        logger.info("Starting real-time training monitoring...")
        monitor.start_monitoring(interval=15)
        
        try:
            # Check for existing adapter to continue from
            existing_adapters = [
                "lora_checkpoints/soju_training/final_adapter",
                "lora_checkpoints/identity/final_adapter", 
                "lora_checkpoints/proper_identity/final_adapter",
                "lora_checkpoints/manual_training/final_adapter"
            ]
            
            continue_from = None
            for adapter_path in existing_adapters:
                if Path(adapter_path).exists():
                    continue_from = adapter_path
                    logger.info(f"Found existing adapter to continue from: {continue_from}")
                    break
            
            # Train with LoRA using existing data format
            logger.info("Starting LoRA fine-tuning on H200...")
            adapter_path = lora_trainer.fine_tune(
                training_data,
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
            logger.info(f"H200 LoRA training completed successfully!")
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
            
            for prompt in test_prompts:
                try:
                    response = lora_trainer.generate_with_lora(prompt, max_length=100)
                    logger.info(f"\nPrompt: {prompt}")
                    logger.info(f"Response: {response}")
                except Exception as e:
                    logger.error(f"Generation failed for prompt '{prompt}': {e}")
            
            return adapter_path
        else:
            logger.error("H200 LoRA training failed")
            return None
            
    except Exception as e:
        logger.error(f"H200 LoRA training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        adapter_path = run_h200_training()
        if adapter_path:
            print(f"\nH200 LoRA training completed successfully! Adapter: {adapter_path}")
        else:
            print("\nH200 LoRA training failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nH200 LoRA training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nH200 LoRA training failed with error: {e}")
        sys.exit(1) 